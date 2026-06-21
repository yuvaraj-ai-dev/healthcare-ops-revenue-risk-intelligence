# Model Governance, Compliance, and Retraining Guidelines

This document outlines the governance standards, regulatory compliance boundaries, and long-term maintenance protocols for the Hospital Operations & Revenue Risk Intelligence Platform (incorporating Model A: Visit Risk Classifier and Model B: Claim Status Classifier).

---

## 1. Regulatory Compliance & Fairness Standards
To align with healthcare governance and ethical AI principles:
- **Patient Anonymization:** No Direct Identifiers (PII) such as Patient Name, Social Security Numbers, or exact street addresses are stored in the relational database or passed to the inference endpoints. Patient records are referenced strictly via anonymized system keys (`patient_id`).
- **Fairness Guarantee:** Predictions must be audited periodically across demographics (`gender`, `city`) and insurers to ensure no systematic bias. As demonstrated in Phase 4 evaluation, prediction accuracy remains consistent across male/female records (~76%) and cities (within a 1.5% accuracy range).
- **Audit Logging:** Every prediction request and response, along with its input feature hash and model version, is written to an immutable local SQLite database (`audit_logs.db`). This ensures that predictions can be fully audited and reconstructed in the event of disputed claims or compliance reviews.

---

## 2. Platform Assumptions & Limitations
- **Sequence Logging:** We assume billing transactions are logged sequentially after patient visits are completed. Incomplete encounters or billing records entered out-of-order will lead to temporary records and fallback predictions.
- **Cold-Start Providers/Departments:** When the hospital network expands to new departments or signs agreements with new insurance providers, the system falls back to global averages for rejection and realization rates. These fallback values are updated automatically as transactions accumulate.

---

## 3. Retraining Trigger Strategy
The predictive models must be retrained to mitigate performance degradation caused by concept drift, seasonality, or changing insurance policies.

We establish three retraining triggers:
1. **Performance Drop Trigger:**
   - **Metric:** Overall accuracy or Rejected class recall on the claim classifier.
   - **Trigger:** If weekly audit metrics drop below **70%** (baseline is 76% accuracy and 78% recall).
2. **Data & Prediction Drift Trigger:**
   - **Metric:** Population Stability Index (PSI) computed on inputs and predictions.
   - **Trigger:** If the PSI for key features (e.g. `billed_amount`, `age`) or predicted classes exceeds **0.25** on the weekly audit summary, indicating a significant change in patient cohort distributions.
3. **Scheduled Retraining Trigger:**
   - **Interval:** Semi-annually (every 6 months), regardless of performance, to capture long-term seasonality and incorporate updated insurance contracts.

---

## 4. Retraining Procedure
When a trigger is activated, the retraining pipeline follows these steps:
1. **Data Pull:** Extract the latest 6 months of verified visits and resolved billing transactions from MySQL.
2. **Verification & Clean:** Run validation checks via `monitoring/validation.py` to ensure only clean records are used (excluding active/pending transactions).
3. **Feature Engineering:** Execute the feature builder script `build_features.py` using time-based splitting to fit new provider-rejection and department-realization rates.
4. **Model Fit & Tune:** Run hyperparameter search to fit new Random Forest and LightGBM models.
5. **Validation Check:** Compare the new model against the active model on a held-out evaluation set. The new model must achieve a performance increase or remain within a 0.5% margin of the active model before deployment.
6. **Promote:** Save the serialized pipeline to `models/` with an incremented patch version and log the promotion event.
