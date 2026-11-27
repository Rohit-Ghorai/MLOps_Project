# SCSE3040 MLOps Mini Project — Group G192

## Problem: MP063 — City Bus Delay Prediction (Nagpur)

**Faculty:** Saurabh Shrivastava | **Batch:** 28

### Group Members

| Enrolment No. | Name | Assigned Role | Responsibility Domain |
|---|---|---|---|
| **S24CSEU0374** | Rohit Ghorai | Data Engineering Lead | DVC setup, 7 hidden data problems detection, data cleaning pipeline, findings.yaml data section |
| **S24CSEU0441** | Sagar Goyal | ML Engineering Lead | ML feature pipeline, 3 candidate models training, MLflow tracking (`sqlite:///mlflow.db`), model evaluation (MAE), model registration, champion alias, holdout predictions |
| **S24CSEU0483** | Saujanya Kajal Baniyan | Service & Operations Lead | FastAPI service (`POST /predict`, `GET /health`), Pydantic schema validation, Docker containerization, Kubernetes manifests, CI/CD GitHub Actions |

---

## Project Architecture & Directory Layout

```text
MP063_City_Bus_Delay/
├── README.md                              # Group ID, problem ID, enrolment numbers, documentation
├── requirements.txt                       # Pinned production and test dependencies
├── params.yaml                            # Centralized project parameters and model configurations
├── pytest.ini                             # Pytest testpath and pythonpath configuration
├── Dockerfile                             # Production API image (mp063-api:1.0)
├── .dockerignore
├── .github/workflows/ci.yml               # GitHub Actions test workflow
├── app/
│   └── main.py                            # FastAPI API, Pydantic validation and JSON logs
├── k8s/
│   ├── deployment.yaml                    # Three-replica API Deployment
│   └── service.yaml                       # ClusterIP Service
├── mlflow.db                              # SQLite MLflow tracking database
├── data/
│   ├── raw/                               # Raw datasets tracked by DVC
│   │   ├── train.csv                      # Original raw training dataset (SHA-256: ebf413b2...)
│   │   ├── train.csv.dvc                  # DVC tracker for train.csv
│   │   ├── later.csv                      # Drift monitoring dataset (SHA-256: efe02164...)
│   │   ├── later.csv.dvc                  # DVC tracker for later.csv
│   │   └── holdout_features.csv           # Holdout test features (SHA-256: feb6d831...)
│   └── processed/
│       ├── train_clean.csv                # 1,000 cleaned training records
│       └── later_clean.csv                # 300 cleaned monitoring records
├── models/
│   ├── model.joblib                       # Registered champion inference pipeline (v2 promoted)
│   ├── model_v1.joblib                    # Baseline champion model (v1)
│   └── model_v2.joblib                    # Retrained model (v2)
├── predictions/
│   └── holdout_predictions.csv            # Final predictions on holdout_features.csv (250 records)
├── reports/
│   ├── data_exploration.md                # Exploratory data analysis report
│   ├── data_issues_report.md              # Seven hidden data problems audit
│   ├── data_cleaning_report.md            # Data cleaning validation report
│   ├── model_training_summary.json        # Machine learning experiment results
│   ├── model_evaluation_report.md         # Model comparison & validation metrics
│   └── findings.yaml                      # Project evaluation form (automated marking)
├── src/
│   ├── data_pipeline/                     # Data engineering modules (Rohit Ghorai)
│   │   ├── __init__.py
│   │   ├── cleaning.py                    # Deduplication, target typo fix, leakage protection
│   │   ├── detect_issues.py               # Statistical audits for 7 data issues
│   │   ├── exploration.py                 # Feature profiling and summaries
│   │   └── validation.py                  # Cryptographic and business rule validation suite
│   └── model_pipeline/                    # ML engineering modules (Sagar Goyal)
│       ├── __init__.py
│       ├── pipeline.py                    # Unified feature transformer and safety guards
│       ├── train.py                       # 3 candidate models training & MLflow tracking
│       ├── evaluate.py                    # Model evaluation & schema robustness checks
│       ├── predict.py                     # Batch prediction generator for holdout set
│       └── monitoring.py                  # PSI drift calculation, v2 retraining & promotion
└── tests/
    ├── test_model_pipeline.py             # Pytest suite validating end-to-end ML pipeline
    └── test_service.py                    # FastAPI endpoint and request validation tests
```

---

## Quickstart & Reproduction

### 1. Environment Setup
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Data Validation
```bash
python -m src.data_pipeline.validation
```

### 3. Run Model Training & Tracking
```bash
python -m src.model_pipeline.train
```

### 4. Run Model Evaluation
```bash
python -m src.model_pipeline.evaluate
```

### 5. Run Monitoring, Drift Detection & Retraining
```bash
python -m src.model_pipeline.monitoring
```

### 6. Generate Holdout Predictions
```bash
python -m src.model_pipeline.predict
```

### 7. Run Test Suite
```bash
python -m pytest tests/ -v
```

### 8. Run the Prediction API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API provides `GET /health` and `POST /predict`. Requests require exactly
the eight model features; `route_km` must be at least 1.5 and `stops` cannot
exceed 70. Prediction events are written as JSON lines to stdout.

### 9. Build and deploy
```bash
docker build -t mp063-api:1.0 .
kubectl apply -f k8s/
```

The monitoring command computes PSI, evaluates group MAE, retrains v2, and
applies the 10% promotion rule:
```bash
python -m src.model_pipeline.monitoring
```
