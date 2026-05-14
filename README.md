# Vehicle Insurance Cross-Sell Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red?style=flat-square&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerised-blue?style=flat-square&logo=docker)
![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20S3%20%7C%20ECR-orange?style=flat-square&logo=amazonaws)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?style=flat-square&logo=githubactions)

A production-grade end-to-end machine learning system that predicts whether a health insurance customer is likely to purchase vehicle insurance. Built with a fully automated ML pipeline, REST API, and cloud deployment on AWS.

**Live Demo:** [http://44.205.214.105:8501](http://44.205.214.105:8501)

---

## The Business Problem

An insurance company wants to cross-sell vehicle insurance to existing health insurance customers. Reaching out to every customer is expensive and inefficient. This system identifies which customers are most likely to be interested — enabling the sales team to focus their efforts and maximise revenue.

**Why Recall?**
A missed interested customer means lost revenue (False Negative). A wasted sales call to an uninterested customer is a minor cost (False Positive). For this business problem, maximising recall is the right objective — the model achieves **~95% recall** on the test set.

---

## Architecture

```
MongoDB Atlas          AWS S3
(Raw Dataset)          (Model Registry)
      │                      │
      ▼                      ▼
┌─────────────────────────────────────┐
│         Training Pipeline           │
│                                     │
│  Data Ingestion → Data Validation   │
│       → Data Transformation         │
│       → Model Trainer               │
│       → Model Evaluation            │
│       → Model Pusher                │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│         Prediction Pipeline         │
│                                     │
│  Raw Input → Preprocessing          │
│           → LogisticRegression      │
│           → Prediction + Confidence │
└─────────────────────────────────────┘
                │
                ▼
┌──────────────────────┐    ┌──────────────────────┐
│   FastAPI Backend    │◄───│  Streamlit Frontend  │
│   Port 5000          │    │  Port 8501           │
└──────────────────────┘    └──────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│         AWS EC2 (Docker Compose)    │
│   GitHub Actions CI/CD → ECR → EC2  │
└─────────────────────────────────────┘
```

---

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.11 |
| **ML** | Scikit-learn, Imbalanced-learn (SMOTEENN), Pandas, NumPy |
| **Feature Engineering** | Custom scikit-learn transformer (target encoding, frequency encoding, interaction features) |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Frontend** | Streamlit |
| **Database** | MongoDB Atlas (pymongo) |
| **Cloud** | AWS EC2, S3, ECR, IAM |
| **Containerisation** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **MLOps** | Custom ML pipeline, model versioning via S3, artifact management |
| **Utilities** | dill, PyYAML, python-dotenv, certifi |

---

## ML Pipeline

The training pipeline runs end-to-end automatically via the `/train` endpoint or triggered by CI/CD.

### 1. Data Ingestion
- Fetches the raw dataset from MongoDB Atlas
- Splits into train (75%) and test (25%) sets
- Saves artifacts to a timestamped folder

### 2. Data Validation
- Validates column count and column names against `config/schema.yaml`
- Pipeline stops immediately if validation fails — fail fast principle

### 3. Data Transformation
A custom scikit-learn `Pipeline` with two steps:

**Step 1 — Custom Feature Engineer:**
- Target encoding for `Policy_Sales_Channel` and `Region_Code`
- Frequency encoding for high-cardinality columns
- Ordinal encoding for `Vehicle_Age`
- Interaction features: `Age × Vehicle_Age`, `Age × Channel`, `Damage × NotInsured`
- Binary features for damage and insurance status combinations

**Step 2 — ColumnTransformer:**
- `StandardScaler` on numerical features
- `MinMaxScaler` on `Annual_Premium` (handles wide range and outliers)
- `OneHotEncoder(drop='first')` on categorical features (avoids multicollinearity)

**Class Imbalance:** SMOTEENN applied to training data only — oversamples minority class and removes noisy majority samples.

### 4. Model Training
- LogisticRegression with L1 regularisation (`penalty='l1'`, `solver='saga'`)
- `class_weight='balanced'` for additional imbalance handling
- `max_iter=3000` for convergence on large dataset
- Evaluates: accuracy, F1, precision, recall, ROC-AUC
- Quality gate: rejects models below 0.6 accuracy threshold

### 5. Model Evaluation
- Compares new model recall score against production model in S3
- Accepts new model only if recall improves — prevents deploying regressions

### 6. Model Pusher
- Pushes accepted model to AWS S3 as the new production model
- Model is a `MyModel` object — wraps preprocessing pipeline + trained model together

---

## Project Structure

```
vehicle-insurance-prediction/
│
├── src/vehicle_insurance/
│   ├── components/          # ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── feature_engineering.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   ├── pipeline/
│   │   ├── training_pipeline.py    # Orchestrates all components
│   │   └── prediction_pipeline.py  # Serves predictions
│   ├── entity/
│   │   ├── config_entity.py        # Input configs (dataclasses)
│   │   ├── artifact_entity.py      # Output artifacts (dataclasses)
│   │   ├── estimator.py            # MyModel wrapper
│   │   └── s3_estimator.py         # S3 model operations
│   ├── configuration/
│   │   ├── mongo_db_connection.py  # MongoDB Singleton client
│   │   └── aws_connection.py       # AWS S3 Singleton client
│   ├── cloud_storage/
│   │   └── aws_storage.py          # S3 operations
│   ├── schema/
│   │   ├── user_input.py           # Pydantic request validation
│   │   └── prediction_response.py  # Pydantic response schema
│   ├── constants/__init__.py        # All project constants
│   ├── logger/__init__.py           # Custom logger
│   ├── exception/__init__.py        # Custom exception handler
│   └── utils/main_utils.py          # Shared utility functions
│
├── config/
│   ├── schema.yaml                  # Dataset schema definition
│   └── model.yaml                   # Model configuration
│
├── app.py                           # FastAPI application
├── frontend.py                      # Streamlit UI
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yaml
├── requirements.backend.txt
├── requirements.frontend.txt
└── .github/workflows/cicd.yaml      # GitHub Actions CI/CD
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — confirms API is running |
| `GET` | `/health` | Returns `{"status": "OK"}` |
| `GET` | `/train` | Triggers the full ML training pipeline |
| `POST` | `/predict` | Returns prediction for a customer |

### POST /predict — Request Body

```json
{
  "Gender": "Male",
  "Age": 35,
  "Driving_License": 1,
  "Region_Code": 28,
  "Previously_Insured": 0,
  "Vehicle_Age": "1-2 Year",
  "Vehicle_Damage": "Yes",
  "Annual_Premium": 40000.0,
  "Policy_Sales_Channel": 26,
  "Vintage": 200
}
```

### POST /predict — Response

```json
{
  "prediction": "Interested",
  "confidence": 0.8741,
  "class_probabilities": {
    "Not Interested": 0.1259,
    "Interested": 0.8741
  }
}
```

Interactive API documentation available at: [http://44.205.214.105:5000/docs](http://44.205.214.105:5000/docs)

---

## Deployment

### CI/CD Pipeline (GitHub Actions)

Every push to `main` automatically:

1. **CI Job** (GitHub-hosted ubuntu runner)
   - Builds Docker images for backend and frontend
   - Pushes images to AWS ECR

2. **CD Job** (EC2 self-hosted runner)
   - Pulls latest images from ECR
   - Stops existing containers
   - Starts updated containers via Docker Compose

### Infrastructure
- **AWS EC2** — t2.micro instance running both containers
- **AWS ECR** — private Docker image registry
- **AWS S3** — model versioning and storage
- **MongoDB Atlas** — raw dataset storage
- **Docker Compose** — orchestrates backend (port 5000) and frontend (port 8501)

---

## Running Locally

### Prerequisites
- Python 3.11
- Docker Desktop
- MongoDB Atlas account with dataset loaded
- AWS account with S3 bucket and IAM credentials

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Hello-Mitra/vehicle-insurance-prediction.git
cd vehicle-insurance-prediction

# 2. Create and activate virtual environment
uv venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Fill in MONGODB_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
```

### Run with Docker Compose

```bash
# Build and start both containers
docker-compose up --build

# Access the app
# Frontend: http://localhost:8501
# API docs:  http://localhost:5000/docs
```

### Run locally without Docker

```bash
# Terminal 1 — Start FastAPI backend
uvicorn app:app --host 0.0.0.0 --port 5000 --reload

# Terminal 2 — Start Streamlit frontend
# First update API_URL in frontend.py to http://localhost:5000/predict
streamlit run frontend.py
```

### Run the training pipeline

```bash
python demo.py
# Add TrainPipeline().run_pipeline() to demo.py
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```
MONGODB_URL=mongodb+srv://<username>:<password>@cluster.mongodb.net/
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=ap-south-1
```

---

## Model Performance

| Metric | Score |
|---|---|
| **Recall** | ~0.95 |
| **ROC-AUC** | ~0.85-0.88 |
| **F1 Score** | ~0.79 |
| **Precision** | ~0.81 |
| **Accuracy** | ~0.82 |

Recall is the primary business metric — maximising identification of interested customers minimises lost revenue from missed cross-sell opportunities.

---

## Key Design Decisions

- **Singleton pattern** for MongoDB and S3 connections — one connection shared across all usages
- **MyModel wrapper** — packages preprocessing pipeline and trained model into one object for single-file deployment
- **Fail fast** — pipeline stops at data validation and model evaluation if quality gates are not met
- **No data leakage** — preprocessing fit on training data only, statistics applied to test data
- **SMOTEENN on training data only** — test data preserves real-world class distribution for reliable evaluation
- **Separate Docker images** — backend and frontend have separate requirements files, keeping images lean

---

## Author

**Arijit Mitra**
[LinkedIn](https://www.linkedin.com/in/arijit-mitra-131423145/) · [GitHub](https://github.com/Hello-Mitra)
