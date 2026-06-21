# Model Card: Hospital Risk and Revenue Intelligence Platform

This model card documents the performance, architecture, assumptions, limitations, and operational boundaries of the predictive models deployed in the Hospital Operations & Revenue Risk Intelligence Platform.

---

## 1. Model Details
- **Developer:** AI and ML Capstone Team (Expert Decision-Support Platforms)
- **Model Version:** `1.0.0`
- **Release Date:** June 21, 2026
- **Model Type:** 
  - **Model A (Visit Risk Classifier):** Random Forest Classifier (multiclass: Low, Medium, High risk)
  - **Model B (Claim Status Classifier):** LightGBM Classifier (multiclass: Paid, Pending, Rejected)
- **Frameworks:** Python, Scikit-Learn, LightGBM, SQLAlchemy, PyMySQL, Joblib

---

## 2. Intended Use
- **Primary Users:**
  - **Hospital Operations & Clinical Teams:** Use Model A predictions to forecast visit risk levels, optimize bed allocations, and plan staffing needs (focusing on ICU and ER patient flow).
  - **Hospital Finance & Billing Teams:** Use Model B predictions to identify insurance claims that are highly likely to be Rejected *before* they are submitted, enabling billing corrections and reducing revenue leakage.
- **Out of Scope:**
  - Model A is **not** a diagnostic tool and does not replace clinical triage by physicians. It acts as an operational decision-support tool.
  - Model B does not guarantee payment from insurance providers but highlights risk profiles.

---

## 3. Training & Evaluation Data
- **Source Database:** Local MySQL database `hospital_db` holding 3 combined datasets:
  - `patients.csv` (5,000 patient demographic profiles)
  - `visits.csv` (25,000 operational visit encounters)
  - `billing.csv` (25,000 claim transactions)
- **Train/Test Split Strategy:**
  - **Temporal Split:** Sorted chronologically by `visit_date` to prevent target leakage. The earliest **80%** (20,000 records) was used for training and aggregate feature fitting, and the latest **20%** (5,000 records) was reserved for testing.

---

## 4. Performance Summary

### Model A: Visit Risk Classifier (Test Set Evaluation)
- **Accuracy:** ~99.9% (Perfect boundary separation achieved based on length of stay, clinical department, and patient frequency metrics)
- **High-Risk Recall:** ~100% (Crucial for clinical staffing validation)

### Model B: Claim Status Classifier (Test Set Evaluation)
- **Accuracy:** ~76.2%
- **Rejection Recall:** ~78.0% (Captures 78% of rejected claims prior to filing)
- **Pending Recall:** ~69.0%
- **Paid Recall:** ~81.0%

---

## 5. Business Impact & ROI
- **Operational Value (Model A):** The model correctly identifies 100% of high-risk patient encounters, allowing hospital administration to proactively allocate critical resources (ICU beds, specialists) and reduce clinical bottlenecks.
- **Financial Protection (Model B):** 
  - The model achieves a **78.0% recall on Rejected claims**. 
  - In our test set (latest 20% of data), this flags approximately **INR 21,800,000** in rejected billing transactions, allowing billing teams to intercept and correct claims before submission.

---

## 6. Fairness & Demographic Audits
The models were audited across key patient segments in the test set to ensure equal treatment and prevent bias:
- **Gender:** Accuracy and F1-score are balanced between Male (~76.1%) and Female (~76.3%) patient records.
- **City:** Consistent performance across major cities (Bangalore, Chennai, Delhi, Hyderabad, Mumbai, Pune) with no more than 1.5% variance.
- **Insurance Provider:** Claim status predictions remain stable across different insurance networks (CareOne, HealthPlus, MediCareX, SecureLife), verifying that the model doesn't disproportionately target any specific insurer.

---

## 7. Assumptions & Limitations
- **Data Completeness:** The model assumes billing records are generated in sequence.
- **Cold-Start Problem:** For new clinical departments or newly onboarded insurance providers, the model will fall back to global average rates until sufficient transaction history is gathered.
- **Label Consistency:** Changes in coding standards (e.g. ICD-10 to ICD-11) or insurance provider guidelines may shift baseline rejection rates.

---

## 8. Retraining & Governance Trigger
To maintain model reliability, the following governance protocol is established:
1. **Performance Retraining Trigger:** If overall accuracy drops below **70%** or the Rejected claim recall falls below **70%** on weekly audits, a retraining cycle must be initiated.
2. **Drift Detection Retraining Trigger:** Weekly checks run the Kolmogorov-Smirnov test on numerical features and the Population Stability Index (PSI) on categorical predictions. A PSI value exceeding **0.25** indicates significant data drift and triggers automated model retraining.
