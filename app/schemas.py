from pydantic import BaseModel, Field

class RiskRequest(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age of the patient", example=45)
    gender: str = Field(..., description="Gender: M or F", example="M")
    city: str = Field(..., description="Patient city", example="Hyderabad")
    chronic_flag: int = Field(..., ge=0, le=1, description="Chronic condition flag (0 or 1)", example=1)
    length_of_stay_hours: float = Field(..., ge=0.0, description="Length of stay in hours", example=14.5)
    department: str = Field(..., description="Clinical department", example="Cardiology")
    visit_type: str = Field(..., description="Visit type (OPD, ER, ICU)", example="ER")
    days_since_registration: int = Field(..., ge=0, description="Days since registration", example=120)
    visit_month: int = Field(..., ge=1, le=12, description="Month of visit", example=6)
    visit_day_of_week: int = Field(..., ge=0, le=6, description="Weekday of visit (0=Monday)", example=2)
    visit_frequency: int = Field(..., ge=1, description="Historical visits count for this patient", example=3)
    avg_length_of_stay_patient: float = Field(..., ge=0.0, description="Average length of stay for this patient", example=12.2)

class RiskResponse(BaseModel):
    predicted_risk_score: str = Field(..., description="Predicted risk score: Low, Medium, or High")
    probabilities: dict = Field(..., description="Class probabilities")
    model_version: str = Field(..., description="Model version")

class ClaimRequest(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age of the patient", example=45)
    gender: str = Field(..., description="Gender: M or F", example="M")
    city: str = Field(..., description="Patient city", example="Hyderabad")
    chronic_flag: int = Field(..., ge=0, le=1, description="Chronic condition flag (0 or 1)", example=1)
    length_of_stay_hours: float = Field(..., ge=0.0, description="Length of stay in hours", example=14.5)
    department: str = Field(..., description="Clinical department", example="Cardiology")
    visit_type: str = Field(..., description="Visit type (OPD, ER, ICU)", example="ER")
    days_since_registration: int = Field(..., ge=0, description="Days since registration", example=120)
    visit_month: int = Field(..., ge=1, le=12, description="Month of visit", example=6)
    visit_day_of_week: int = Field(..., ge=0, le=6, description="Weekday of visit (0=Monday)", example=2)
    visit_frequency: int = Field(..., ge=1, description="Historical visits count for this patient", example=3)
    avg_length_of_stay_patient: float = Field(..., ge=0.0, description="Average length of stay for this patient", example=12.2)
    risk_score: str = Field(..., description="Visit risk score (Low, Medium, High)", example="Medium")
    billed_amount: float = Field(..., ge=0.0, description="Billed amount in INR", example=25000.0)
    insurance_provider: str = Field(..., description="Insurance provider name", example="SecureLife")

class ClaimResponse(BaseModel):
    predicted_claim_status: str = Field(..., description="Predicted status: Paid, Pending, or Rejected")
    probabilities: dict = Field(..., description="Class probabilities")
    model_version: str = Field(..., description="Model version")
