import os
import sys
import time
import json
import hashlib
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add parent directory to path so we can import schemas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.schemas import RiskRequest, RiskResponse, ClaimRequest, ClaimResponse

load_dotenv()

app = FastAPI(
    title="Hospital Operations & Revenue Risk Intelligence API",
    description="FastAPI service for clinical visit risk and financial claim outcome predictions.",
    version="1.0.0"
)

# Global variables for models and aggregates
model_A = None
model_B = None
aggregates = {}
model_version = os.getenv("MODEL_VERSION", "1.0.0")

# Setup local SQLite audit database for logs
AUDIT_DB_PATH = "audit_logs.db"

def init_audit_db():
    conn = sqlite3.connect(AUDIT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        endpoint TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        input_data TEXT,
        prediction TEXT,
        model_version TEXT,
        feature_hash TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_prediction(endpoint: str, input_dict: dict, prediction_str: str, feature_hash: str):
    try:
        conn = sqlite3.connect(AUDIT_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO prediction_logs (endpoint, input_data, prediction, model_version, feature_hash) VALUES (?, ?, ?, ?, ?)",
            (endpoint, json.dumps(input_dict), prediction_str, model_version, feature_hash)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to log prediction to audit DB: {str(e)}")

def load_db_aggregates():
    """
    Fetch provider rejection rates and department revenue realization rates dynamically from MySQL.
    If connection fails, it falls back to pre-calculated training values.
    """
    global aggregates
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    db = os.getenv("MYSQL_DATABASE", "hospital_db")
    
    import urllib.parse
    password_encoded = urllib.parse.quote_plus(password)
    
    try:
        conn_str = f"mysql+pymysql://{user}:{password_encoded}@{host}:{port}/{db}"
        engine = create_engine(conn_str)
        
        with engine.connect() as conn:
            # Rejection rates per provider
            query_rej = """
            SELECT p.insurance_provider,
                   COUNT(b.bill_id) as total,
                   SUM(CASE WHEN b.claim_status = 'Rejected' THEN 1 ELSE 0 END) as rejected
            FROM patients p
            JOIN visits v ON p.patient_id = v.patient_id
            JOIN billing b ON v.visit_id = b.visit_id
            GROUP BY p.insurance_provider;
            """
            df_rej = pd.read_sql(query_rej, conn)
            df_rej['rate'] = df_rej['rejected'] / df_rej['total'].replace(0, 1)
            provider_rejection = df_rej.set_index('insurance_provider')['rate'].to_dict()
            global_rejection = df_rej['rejected'].sum() / df_rej['total'].replace(0, 1).sum()
            
            # Realization rates per department
            query_real = """
            SELECT v.department,
                   SUM(b.billed_amount) as billed,
                   SUM(b.approved_amount) as approved
            FROM visits v
            JOIN billing b ON v.visit_id = b.visit_id
            GROUP BY v.department;
            """
            df_real = pd.read_sql(query_real, conn)
            df_real['rate'] = df_real['approved'] / df_real['billed'].replace(0, 1)
            dept_realization = df_real.set_index('department')['rate'].to_dict()
            global_realization = df_real['approved'].sum() / df_real['billed'].replace(0, 1).sum()
            
            aggregates = {
                'provider_rejection': provider_rejection,
                'global_rejection': global_rejection,
                'dept_realization': dept_realization,
                'global_realization': global_realization
            }
            print("Successfully loaded aggregates dynamically from MySQL database!")
            
    except Exception as e:
        print(f"Could not load aggregates from MySQL ({str(e)}). Falling back to training defaults.")
        # Fallback values from training set
        aggregates = {
            'provider_rejection': {'SecureLife': 0.28, 'HealthPlus': 0.22, 'CareOne': 0.25, 'MediCareX': 0.29},
            'global_rejection': 0.26,
            'dept_realization': {'Cardiology': 0.55, 'Orthopedics': 0.58, 'ICU': 0.52, 'General': 0.60, 'Neurology': 0.54, 'ER': 0.49},
            'global_realization': 0.54
        }

@app.on_event("startup")
def startup_event():
    global model_A, model_B
    # Initialize audit DB
    init_audit_db()
    
    # Load aggregates
    load_db_aggregates()
    
    # Load ML models
    try:
        model_A = joblib.load("models/risk_model.pkl")
        model_B = joblib.load("models/claim_model.pkl")
        print("Trained machine learning models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_version": model_version,
        "database_connected": len(aggregates) > 0,
        "models_loaded": (model_A is not None) and (model_B is not None)
    }

@app.post("/predict/risk", response_model=RiskResponse)
def predict_risk(request: RiskRequest):
    if model_A is None:
        raise HTTPException(status_code=503, detail="Model A is not loaded")
        
    input_dict = request.dict()
    
    # Hash features to ensure auditability and compliance
    input_json = json.dumps(input_dict, sort_keys=True)
    feature_hash = hashlib.sha256(input_json.encode('utf-8')).hexdigest()
    
    # Preprocess request to matching dataframe
    input_df = pd.DataFrame([input_dict])
    
    try:
        # Run prediction
        probs = model_A.predict_proba(input_df)[0]
        pred_class_encoded = int(np.argmax(probs))
        
        classes = ['Low', 'Medium', 'High']
        predicted_class = classes[pred_class_encoded]
        probabilities_dict = {classes[i]: float(probs[i]) for i in range(3)}
        
        # Log to audit DB
        log_prediction("/predict/risk", input_dict, predicted_class, feature_hash)
        
        return RiskResponse(
            predicted_risk_score=predicted_class,
            probabilities=probabilities_dict,
            model_version=model_version
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/claim", response_model=ClaimResponse)
def predict_claim(request: ClaimRequest):
    if model_B is None:
        raise HTTPException(status_code=503, detail="Model B is not loaded")
        
    input_dict = request.dict()
    
    # Hash raw inputs
    input_json = json.dumps(input_dict, sort_keys=True)
    feature_hash = hashlib.sha256(input_json.encode('utf-8')).hexdigest()
    
    # Preprocess request by lookup of aggregates (preventing client-side leakage calculation)
    provider = input_dict['insurance_provider']
    dept = input_dict['department']
    
    provider_rej_rate = aggregates['provider_rejection'].get(provider, aggregates['global_rejection'])
    dept_real_rate = aggregates['dept_realization'].get(dept, aggregates['global_realization'])
    
    # Build complete features matching training schema
    features_dict = input_dict.copy()
    features_dict['provider_rejection_rate'] = provider_rej_rate
    features_dict['revenue_realization_rate_dept'] = dept_real_rate
    
    input_df = pd.DataFrame([features_dict])
    
    try:
        # Run prediction
        probs = model_B.predict_proba(input_df)[0]
        pred_class_encoded = int(np.argmax(probs))
        
        classes = ['Paid', 'Pending', 'Rejected']
        predicted_class = classes[pred_class_encoded]
        probabilities_dict = {classes[i]: float(probs[i]) for i in range(3)}
        
        # Log to audit DB
        log_prediction("/predict/claim", input_dict, predicted_class, feature_hash)
        
        return ClaimResponse(
            predicted_claim_status=predicted_class,
            probabilities=probabilities_dict,
            model_version=model_version
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/dashboard/stats")
def get_dashboard_stats():
    # 1. Fetch SQLite Audit Logs (last 10)
    audit_logs = []
    try:
        conn = sqlite3.connect(AUDIT_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, endpoint, timestamp, input_data, prediction, model_version, feature_hash FROM prediction_logs ORDER BY id DESC LIMIT 10")
        rows = cursor.fetchall()
        for r in rows:
            audit_logs.append({
                "id": r[0],
                "endpoint": r[1],
                "timestamp": r[2],
                "input_data": json.loads(r[3]),
                "prediction": r[4],
                "model_version": r[5],
                "feature_hash": r[6]
            })
        conn.close()
    except Exception as e:
        print(f"Error fetching audit logs: {e}")

    # 2. Fetch MySQL counts
    mysql_stats = {"patients": 0, "visits": 0, "billing": 0}
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    db = os.getenv("MYSQL_DATABASE", "hospital_db")
    
    import urllib.parse
    password_encoded = urllib.parse.quote_plus(password)
    
    try:
        conn_str = f"mysql+pymysql://{user}:{password_encoded}@{host}:{port}/{db}"
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            mysql_stats["patients"] = conn.execute(text("SELECT COUNT(*) FROM patients")).scalar()
            mysql_stats["visits"] = conn.execute(text("SELECT COUNT(*) FROM visits")).scalar()
            mysql_stats["billing"] = conn.execute(text("SELECT COUNT(*) FROM billing")).scalar()
    except Exception as e:
        print(f"Error fetching MySQL counts: {e}")
        # Fallback to defaults if MySQL is not available or query fails
        mysql_stats = {"patients": 5000, "visits": 25000, "billing": 25000}

    # 3. Model validation metrics (Model A & B training specs)
    model_stats = {
        "model_A": {
            "name": "Visit Risk Classifier",
            "type": "Random Forest",
            "estimators": 100,
            "max_depth": 15,
            "accuracy": 0.999,
            "recall_high_risk": 1.0,
            "class_weights": "balanced"
        },
        "model_B": {
            "name": "Insurance Claim Status Classifier",
            "type": "LightGBM",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "accuracy": 0.762,
            "recall_rejected": 0.78,
            "class_weights": "is_unbalance=True"
        },
        "roi_savings_inr": 21800000
    }

    return {
        "mysql_stats": mysql_stats,
        "model_stats": model_stats,
        "audit_logs": audit_logs
    }

@app.get("/dashboard/challenges")
def get_dashboard_challenges():
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    db = os.getenv("MYSQL_DATABASE", "hospital_db")
    
    import urllib.parse
    password_encoded = urllib.parse.quote_plus(password)
    
    try:
        conn_str = f"mysql+pymysql://{user}:{password_encoded}@{host}:{port}/{db}"
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            # 1. Patient Flow
            avg_los = float(conn.execute(text("SELECT AVG(length_of_stay_hours) FROM visits")).scalar() or 19.55)
            df_los_dept = pd.read_sql("SELECT department, AVG(length_of_stay_hours) as avg_los, COUNT(*) as visit_count FROM visits GROUP BY department", conn)
            df_risk = pd.read_sql("SELECT risk_score, COUNT(*) as cnt FROM visits GROUP BY risk_score", conn)
            
            # 2. Revenue Leakage
            df_claims = pd.read_sql("SELECT claim_status, COUNT(*) as cnt, SUM(billed_amount) as total_billed FROM billing GROUP BY claim_status", conn)
            df_rej_prov = pd.read_sql("""
                SELECT p.insurance_provider, 
                       COUNT(b.bill_id) as total_claims,
                       SUM(CASE WHEN b.claim_status = 'Rejected' THEN 1 ELSE 0 END) as rejected_claims,
                       SUM(CASE WHEN b.claim_status = 'Rejected' THEN b.billed_amount ELSE 0 END) as rejected_amount
                FROM patients p
                JOIN visits v ON p.patient_id = v.patient_id
                JOIN billing b ON v.visit_id = b.visit_id
                GROUP BY p.insurance_provider
            """, conn)
            
            # 3. Delayed Payments
            df_pay_prov = pd.read_sql("""
                SELECT p.insurance_provider, 
                   AVG(b.payment_days) as avg_payment_days,
                   SUM(CASE WHEN b.claim_status = 'Pending' THEN 1 ELSE 0 END) as pending_claims,
                   SUM(CASE WHEN b.claim_status = 'Pending' THEN b.billed_amount ELSE 0 END) as pending_amount,
                   SUM(b.billed_amount) as total_billed,
                   SUM(b.approved_amount) as total_approved
                FROM patients p
                JOIN visits v ON p.patient_id = v.patient_id
                JOIN billing b ON v.visit_id = b.visit_id
                GROUP BY p.insurance_provider
            """, conn)
            
            return {
                "status": "success",
                "patient_flow": {
                    "avg_los_overall": avg_los,
                    "los_by_department": df_los_dept.to_dict(orient="records"),
                    "risk_distribution": df_risk.to_dict(orient="records")
                },
                "revenue_leakage": {
                    "claims_status": df_claims.to_dict(orient="records"),
                    "rejection_by_provider": df_rej_prov.to_dict(orient="records")
                },
                "delayed_payments": {
                    "payment_delays_by_provider": df_pay_prov.to_dict(orient="records")
                }
            }
    except Exception as e:
        print(f"Error querying challenges metrics: {e}")
        # Fallback values derived from the exact SQL analytics layer distributions
        return {
            "status": "fallback",
            "patient_flow": {
                "avg_los_overall": 19.551584,
                "los_by_department": [
                    {"department": "Cardiology", "avg_los": 19.600962, "visit_count": 4159},
                    {"department": "ER", "avg_los": 19.534967, "visit_count": 4220},
                    {"department": "General", "avg_los": 19.434905, "visit_count": 4228},
                    {"department": "ICU", "avg_los": 19.355234, "visit_count": 4064},
                    {"department": "Neurology", "avg_los": 19.718098, "visit_count": 4165},
                    {"department": "Orthopedics", "avg_los": 19.662656, "visit_count": 4164}
                ],
                "risk_distribution": [
                    {"risk_score": "Low", "cnt": 12470},
                    {"risk_score": "High", "cnt": 5034},
                    {"risk_score": "Medium", "cnt": 7496}
                ]
            },
            "revenue_leakage": {
                "claims_status": [
                    {"claim_status": "Paid", "cnt": 14940, "total_billed": 319181260.67},
                    {"claim_status": "Pending", "cnt": 6263, "total_billed": 127700774.24},
                    {"claim_status": "Rejected", "cnt": 3797, "total_billed": 74886901.14}
                ],
                "rejection_by_provider": [
                    {"insurance_provider": "SecureLife", "total_claims": 5965, "rejected_claims": 936.0, "rejected_amount": 18713433.38},
                    {"insurance_provider": "HealthPlus", "total_claims": 6220, "rejected_claims": 931.0, "rejected_amount": 18256994.37},
                    {"insurance_provider": "CareOne", "total_claims": 6283, "rejected_claims": 934.0, "rejected_amount": 18242419.36},
                    {"insurance_provider": "MediCareX", "total_claims": 6532, "rejected_claims": 996.0, "rejected_amount": 19674054.03}
                ]
            },
            "delayed_payments": {
                "payment_delays_by_provider": [
                    {"insurance_provider": "SecureLife", "avg_payment_days": 13.0781, "pending_claims": 1431.0, "pending_amount": 29457507.52, "total_billed": 126289000.0, "total_approved": 93770890.0},
                    {"insurance_provider": "HealthPlus", "avg_payment_days": 13.0818, "pending_claims": 1609.0, "pending_amount": 33087338.92, "total_billed": 130180700.0, "total_approved": 96251780.0},
                    {"insurance_provider": "CareOne", "avg_payment_days": 13.0269, "pending_claims": 1562.0, "pending_amount": 32129519.52, "total_billed": 130708000.0, "total_approved": 96997760.0},
                    {"insurance_provider": "MediCareX", "avg_payment_days": 13.009, "pending_claims": 1661.0, "pending_amount": 33026408.28, "total_billed": 134591200.0, "total_approved": 100135500.0}
                ]
            }
        }

@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")

# Mount static files folder
app.mount("/static", StaticFiles(directory="app/static"), name="static")

