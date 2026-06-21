import pandas as pd
import numpy as np

def load_joined_data(engine):
    """
    Load patients, visits, and billing tables joined from MySQL.
    """
    query = """
    SELECT v.visit_id, v.patient_id, v.visit_date, v.department, v.visit_type,
           v.length_of_stay_hours, v.risk_score, v.doctor_id,
           p.age, p.gender, p.city, p.insurance_provider, p.chronic_flag, p.registration_date,
           b.bill_id, b.billed_amount, b.approved_amount, b.claim_status, b.payment_days, b.billing_date
    FROM visits v
    LEFT JOIN patients p ON v.patient_id = p.patient_id
    LEFT JOIN billing b ON v.visit_id = b.visit_id
    ORDER BY v.visit_date ASC;
    """
    df = pd.read_sql(query, engine)
    
    # Convert dates to datetime
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    df['registration_date'] = pd.to_datetime(df['registration_date'])
    df['billing_date'] = pd.to_datetime(df['billing_date'])
    
    # Cast numerical types
    df['length_of_stay_hours'] = pd.to_numeric(df['length_of_stay_hours'], errors='coerce')
    df['billed_amount'] = pd.to_numeric(df['billed_amount'], errors='coerce')
    df['approved_amount'] = pd.to_numeric(df['approved_amount'], errors='coerce')
    df['payment_days'] = pd.to_numeric(df['payment_days'], errors='coerce')
    
    # Clean string data
    if 'gender' in df.columns:
        df['gender'] = df['gender'].str.strip()
    if 'city' in df.columns:
        df['city'] = df['city'].str.strip()
    if 'insurance_provider' in df.columns:
        df['insurance_provider'] = df['insurance_provider'].str.strip()
    
    return df

def fit_feature_aggregates(train_df):
    """
    Calculate aggregates on training data to avoid data leakage.
    Returns dictionaries containing the mapping values.
    """
    # 1. Insurance Provider Rejection Rate (Rejected claims / Total claims per provider)
    provider_stats = train_df.groupby('insurance_provider').agg(
        total_claims=('claim_status', 'count'),
        rejected_claims=('claim_status', lambda x: (x == 'Rejected').sum())
    )
    # Handle division by zero
    provider_stats['rejection_rate'] = provider_stats['rejected_claims'] / provider_stats['total_claims'].replace(0, 1)
    provider_rejection_map = provider_stats['rejection_rate'].to_dict()
    
    # Default for unseen providers: global average rejection rate in training
    global_rejection = train_df['claim_status'].eq('Rejected').mean()
    
    # 2. Department Revenue Realization Rate (approved / billed per department)
    dept_stats = train_df.groupby('department').agg(
        total_billed=('billed_amount', 'sum'),
        total_approved=('approved_amount', 'sum')
    )
    dept_stats['realization_rate'] = dept_stats['total_approved'] / dept_stats['total_billed'].replace(0, 1)
    dept_realization_map = dept_stats['realization_rate'].to_dict()
    
    # Default for unseen departments: global average realization rate in training
    global_realization = train_df['approved_amount'].sum() / train_df['billed_amount'].replace(0, 1).sum()
    
    return {
        'provider_rejection': provider_rejection_map,
        'global_rejection': global_rejection,
        'dept_realization': dept_realization_map,
        'global_realization': global_realization
    }

def transform_features(df, aggregates=None):
    """
    Engineer features for the dataset.
    If aggregates is provided, it uses the mapped rates to prevent leakage.
    """
    df = df.copy()
    
    # --- 1. Temporal Features ---
    # Days from registration to visit date
    df['days_since_registration'] = (df['visit_date'] - df['registration_date']).dt.days
    # Clip negative values if registration was recorded after visit by anomaly
    df['days_since_registration'] = df['days_since_registration'].clip(lower=0)
    
    df['visit_month'] = df['visit_date'].dt.month
    df['visit_day_of_week'] = df['visit_date'].dt.dayofweek
    
    # --- 2. Patient-level Features ---
    # Visit frequency (cumulative count up to current visit or total count)
    df['visit_frequency'] = df.groupby('patient_id')['visit_id'].transform('count')
    # Average length of stay per patient in the dataset
    df['avg_length_of_stay_patient'] = df.groupby('patient_id')['length_of_stay_hours'].transform('mean')
    # Fill NaN for patients without LoS
    df['avg_length_of_stay_patient'] = df['avg_length_of_stay_patient'].fillna(df['length_of_stay_hours'].mean())
    
    # --- 3. Insurance & Department Historical Aggregates (Leakage Controlled) ---
    if aggregates is not None:
        # Map provider rejection rate
        df['provider_rejection_rate'] = df['insurance_provider'].map(aggregates['provider_rejection'])
        df['provider_rejection_rate'] = df['provider_rejection_rate'].fillna(aggregates['global_rejection'])
        
        # Map department realization rate
        df['revenue_realization_rate_dept'] = df['department'].map(aggregates['dept_realization'])
        df['revenue_realization_rate_dept'] = df['revenue_realization_rate_dept'].fillna(aggregates['global_realization'])
    else:
        # In case we run it globally without train/test split (e.g. during initial profiling)
        # Provider rejection rate
        prov_rej = df.groupby('insurance_provider')['claim_status'].apply(lambda x: (x == 'Rejected').mean())
        df['provider_rejection_rate'] = df['insurance_provider'].map(prov_rej).fillna(0.0)
        
        # Department realization rate
        dept_billed = df.groupby('department')['billed_amount'].transform('sum')
        dept_appr = df.groupby('department')['approved_amount'].transform('sum')
        df['revenue_realization_rate_dept'] = (dept_appr / dept_billed.replace(0, 1)).fillna(0.0)
        
    return df
