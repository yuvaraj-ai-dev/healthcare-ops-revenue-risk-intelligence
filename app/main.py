import os
import sys
import time
import json
import hashlib
import sqlite3
from fastapi import FastAPI, HTTPException
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
