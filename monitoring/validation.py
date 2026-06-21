import pandas as pd
import numpy as np

def validate_patient_data(df):
    """
    Validate patients dataframe inputs.
    Returns a dictionary with validation success boolean and list of error details.
    """
    errors = []
    
    # 1. Check age range
    if 'age' in df.columns:
        invalid_ages = df[(df['age'] < 0) | (df['age'] > 120)]
        if not invalid_ages.empty:
            errors.append(f"Age validation failed: Found {len(invalid_ages)} records out of range [0, 120].")
            
    # 2. Check gender categories
    if 'gender' in df.columns:
        invalid_genders = df[~df['gender'].isin(['M', 'F'])]
        if not invalid_genders.empty:
            errors.append(f"Gender validation failed: Found {len(invalid_genders)} invalid values (expected 'M' or 'F').")
            
    # 3. Check chronic flag values
    if 'chronic_flag' in df.columns:
        invalid_flags = df[~df['chronic_flag'].isin([0, 1])]
        if not invalid_flags.empty:
            errors.append(f"Chronic flag validation failed: Found {len(invalid_flags)} invalid values (expected 0 or 1).")
            
    success = len(errors) == 0
    return {"success": success, "errors": errors}

def validate_visit_data(df):
    """
    Validate visits dataframe inputs.
    """
    errors = []
    
    # 1. Check length of stay hours
    if 'length_of_stay_hours' in df.columns:
        invalid_los = df[df['length_of_stay_hours'] <= 0]
        if not invalid_los.empty:
            errors.append(f"Length of stay validation failed: Found {len(invalid_los)} records with stay <= 0 hours.")
            
    # 2. Check risk score categories
    if 'risk_score' in df.columns:
        invalid_scores = df[~df['risk_score'].isin(['Low', 'Medium', 'High'])]
        if not invalid_scores.empty:
            errors.append(f"Risk score validation failed: Found {len(invalid_scores)} invalid classes (expected Low, Medium, High).")
            
    success = len(errors) == 0
    return {"success": success, "errors": errors}

def validate_billing_data(df):
    """
    Validate billing dataframe inputs.
    """
    errors = []
    
    # 1. Check billed amount
    if 'billed_amount' in df.columns:
        invalid_bills = df[df['billed_amount'] < 0]
        if not invalid_bills.empty:
            errors.append(f"Billed amount validation failed: Found {len(invalid_bills)} records with negative billed amounts.")
            
    # 2. Check approved amount vs billed amount
    if 'approved_amount' in df.columns and 'billed_amount' in df.columns:
        over_approved = df[df['approved_amount'] > df['billed_amount']]
        if not over_approved.empty:
            errors.append(f"Approved amount validation failed: Found {len(over_approved)} records where approved_amount > billed_amount.")
            
    # 3. Check payment days range
    if 'payment_days' in df.columns:
        invalid_days = df[df['payment_days'] < 0]
        if not invalid_days.empty:
            errors.append(f"Payment days validation failed: Found {len(invalid_days)} records with negative payment days.")
            
    success = len(errors) == 0
    return {"success": success, "errors": errors}
