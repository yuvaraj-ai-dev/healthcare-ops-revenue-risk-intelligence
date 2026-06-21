import os
import zipfile

# Files/folders checklist to pack
checklist = [
    "notebooks/Phase1_SQL.ipynb",
    "notebooks/Phase2_EDA.ipynb",
    "notebooks/Phase3_Modeling.ipynb",
    "notebooks/Phase4_Evaluation.ipynb",
    "app/main.py",
    "app/schemas.py",
    "app/static/index.html",
    "monitoring/validation.py",
    "monitoring/drift_detection.py",
    "models/risk_model.pkl",
    "models/claim_model.pkl",
    "data_outputs/model_table.csv",
    "data_outputs/drift_summary.csv",
    "docs/Model_Card.md",
    "docs/Governance_Compliance.md",
    "Healthcare_Insights_Report.docx",
    "requirements.txt",
    ".env.example",
    "walkthrough.md",
    "README.md",
    "build_features.py",
    "pack_project.py"
]

zip_name = "Capstone_Graded_Project.zip"

print(f"Starting workspace packager for Capstone project submission...")
print(f"Creating archive: {zip_name}\n")

missing_files = []
packed_count = 0

with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for item in checklist:
        if os.path.exists(item):
            print(f"  [+] Packing: {item} ({os.path.getsize(item):,} bytes)")
            zipf.write(item)
            packed_count += 1
        else:
            print(f"  [x] WARNING: Missing file: {item}")
            missing_files.append(item)

print(f"\nPackager completed.")
print(f"Total packed files: {packed_count}/{len(checklist)}")

if missing_files:
    print(f"\nWARNING: {len(missing_files)} files were missing and not packed!")
    for f in missing_files:
        print(f"  - {f}")
else:
    print(f"SUCCESS: All checklisted files successfully packed into {zip_name}!")
