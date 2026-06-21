import time
import requests
import sys

BASE_URL = "http://127.0.0.1:8000"

def wait_for_api(retries=10, delay=2):
    print("Waiting for API server to become healthy...")
    for i in range(retries):
        try:
            resp = requests.get(f"{BASE_URL}/health")
            if resp.status_code == 200:
                print("API server is UP and healthy!")
                print("Response:", resp.json())
                return True
        except Exception:
            pass
        time.sleep(delay)
    print("API server failed to start in time.")
    return False

def test_health():
    print("\n--- Testing /health Endpoint ---")
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert "status" in data
    assert "models_loaded" in data
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True
    print("Health Check Test: PASSED")

def test_predict_risk():
    print("\n--- Testing /predict/risk Endpoint ---")
    payload = {
        "age": 45,
        "gender": "M",
        "city": "Hyderabad",
        "chronic_flag": 1,
        "length_of_stay_hours": 14.5,
        "department": "Cardiology",
        "visit_type": "ER",
        "days_since_registration": 120,
        "visit_month": 6,
        "visit_day_of_week": 2,
        "visit_frequency": 3,
        "avg_length_of_stay_patient": 12.2
    }
    resp = requests.post(f"{BASE_URL}/predict/risk", json=payload)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}. Response: {resp.text}"
    data = resp.json()
    print("Prediction Response:", data)
    assert "predicted_risk_score" in data
    assert "probabilities" in data
    assert data["predicted_risk_score"] in ["Low", "Medium", "High"]
    print("Predict Risk Test: PASSED")

def test_predict_claim():
    print("\n--- Testing /predict/claim Endpoint ---")
    payload = {
        "age": 45,
        "gender": "M",
        "city": "Hyderabad",
        "chronic_flag": 1,
        "length_of_stay_hours": 14.5,
        "department": "Cardiology",
        "visit_type": "ER",
        "days_since_registration": 120,
        "visit_month": 6,
        "visit_day_of_week": 2,
        "visit_frequency": 3,
        "avg_length_of_stay_patient": 12.2,
        "risk_score": "Medium",
        "billed_amount": 25000.0,
        "insurance_provider": "SecureLife"
    }
    resp = requests.post(f"{BASE_URL}/predict/claim", json=payload)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}. Response: {resp.text}"
    data = resp.json()
    print("Prediction Response:", data)
    assert "predicted_claim_status" in data
    assert "probabilities" in data
    assert data["predicted_claim_status"] in ["Paid", "Pending", "Rejected"]
    print("Predict Claim Test: PASSED")

if __name__ == "__main__":
    if not wait_for_api():
        sys.exit(1)
        
    try:
        test_health()
        test_predict_risk()
        test_predict_claim()
        print("\n==============================")
        print("ALL API INTEGRATION TESTS PASSED!")
        print("==============================")
    except AssertionError as e:
        print("\nTEST FAILURE!")
        print("AssertionError:", str(e))
        sys.exit(1)
    except Exception as e:
        print("\nUNEXPECTED FAILURE!")
        print("Error:", str(e))
        sys.exit(1)
