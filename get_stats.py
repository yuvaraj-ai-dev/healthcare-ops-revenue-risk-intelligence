import os
import json
import urllib.parse
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load env file
load_dotenv()

host = os.getenv("MYSQL_HOST", "localhost")
port = os.getenv("MYSQL_PORT", "3306")
user = os.getenv("MYSQL_USER", "root")
password = os.getenv("MYSQL_PASSWORD", "")
db = os.getenv("MYSQL_DATABASE", "hospital_db")

password_encoded = urllib.parse.quote_plus(password)
conn_str = f"mysql+pymysql://{user}:{password_encoded}@{host}:{port}/{db}"
engine = create_engine(conn_str)

try:
    with engine.connect() as conn:
        print("--- Basic Counts ---")
        patients = conn.execute(text("SELECT COUNT(*) FROM patients")).scalar()
        visits = conn.execute(text("SELECT COUNT(*) FROM visits")).scalar()
        billing = conn.execute(text("SELECT COUNT(*) FROM billing")).scalar()
        print(f"Patients: {patients}, Visits: {visits}, Billing: {billing}")
        
        print("\n--- Length of Stay by Department ---")
        df_los = pd.read_sql("SELECT department, AVG(length_of_stay_hours) as avg_los, COUNT(*) as cnt FROM visits GROUP BY department", conn)
        print(df_los)
        
        print("\n--- Claims Status ---")
        df_claims = pd.read_sql("SELECT claim_status, COUNT(*) as cnt, SUM(billed_amount) as sum_billed, SUM(approved_amount) as sum_appr FROM billing GROUP BY claim_status", conn)
        print(df_claims)
        
        print("\n--- Claims by Provider ---")
        df_prov = pd.read_sql("""
            SELECT p.insurance_provider, 
                   COUNT(*) as total_claims,
                   SUM(CASE WHEN b.claim_status = 'Rejected' THEN 1 ELSE 0 END) as rejected_count,
                   AVG(b.payment_days) as avg_pay_days,
                   SUM(b.billed_amount) as total_billed,
                   SUM(b.approved_amount) as total_approved
            FROM patients p
            JOIN visits v ON p.patient_id = v.patient_id
            JOIN billing b ON v.visit_id = b.visit_id
            GROUP BY p.insurance_provider
        """, conn)
        print(df_prov)

except Exception as e:
    print("Database Query Error:", e)
