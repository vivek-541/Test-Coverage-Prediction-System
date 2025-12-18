# 🎯 Test Coverage Prediction System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green)
![Status](https://img.shields.io/badge/Status-Production-success)

A Machine Learning & Deep Learning system designed to predict **Test Case Coverage Percentage** based on feature descriptions and test plans. 

This system analyzes the semantic relationship between a software feature's description and its associated test cases to estimate how well the feature is covered (e.g., "40% coverage" vs "95% coverage"). It helps QA and Engineering teams identify gaps in their testing strategy before writing code.

---

## 🏗️ Project Architecture

The project follows a production-grade MLOps structure, separating experimentation (notebooks) from deployment (app).

```text
├── app/                 # 🚀 Production API Application
│   ├── api/             # API Route definitions (endpoints)
│   ├── core/            # Config and logging settings
│   ├── models/          # Model wrappers (DL vs ML logic)
│   └── schemas/         # Pydantic data validation schemas
├── artifacts/           # 💾 Saved Models & Evaluation Plots
│   ├── dl/              # PyTorch LSTM Model & Artifacts
│   └── ml/              # Scikit-Learn Gradient Boosting Model
├── data/                # 📂 Dataset Storage
│   ├── raw/             # Original CSV uploads
│   └── processed/       # Cleaned 'Goldilocks' datasets
├── notebooks/           # 📓 Jupyter Notebooks for Experiments
├── training/            # 🏋️ Training Scripts & Documentation
└── requirements.txt     # Python dependencies
```
### 🧠 Models Overview

The system implements two distinct approaches. You can switch between them based on your need for speed vs. complexity.
🤖 1. ML Model (Gradient Boosting)

    Algorithm: Gradient Boosting Regressor (Scikit-Learn).

    Location: artifacts/ml/test_coverage_model_balanced.pkl

    Features: TF-IDF vectors, Domain Encoding, Engineered numeric features (counts, lengths).

    Use Case: Fast inference, highly interpretable baseline.

🧠 2. DL Model (Deep Learning)

    Algorithm: LSTM (Long Short-Term Memory) with PyTorch.

    Location: artifacts/dl/test_coverage_pytorch_working.pkl

    Architecture: Embedding Layer (96 dim) → LSTM (192 hidden) → Dense Layers.

    Use Case: Captures complex sequential dependencies in text; better for deep semantic understanding.

🔌 API Documentation
POST /predict
<img width="1442" height="621" alt="image" src="https://github.com/user-attachments/assets/69709900-e306-4c3c-96bc-7a8d1c1dc109" />

Generates a coverage prediction score.
<img width="1442" height="621" alt="image" src="https://github.com/user-attachments/assets/d0462156-b5d1-4ce9-bee8-e974e84508f3" />

Endpoint: http://localhost:8000/predict
Request Body (JSON)
```{
  "feature_description": "Implement a user authentication system allowing login via email and password with a 'Forgot Password' flow.",
  "input_test_cases": [
    "Verify user can login with valid credentials",
    "Verify error message on invalid password",
    "Check empty field validation",
    "Verify password reset link is sent"
  ],
  "domain": "security"
}```

Field	Type	Description
feature_description	string	Detailed text describing the feature to be built (10-5000 chars).
input_test_cases	list	A list of strings, each describing a specific test case.
domain	string	The business domain. Allowed values: security, compliance, healthcare, finance, other.
Response (JSON)
JSON

{
  "coverage_score": 85.5,
  "confidence": "High",
  "missing_aspects_detected": false,
  "model_used": "GradientBoostingRegressor"
}

📊 Supported Domains
The model is trained to recognize context within specific industry domains. Ensure your requests use one of the standardized domain tags:
💰 Finance: Transactions, payments, currency, ledgers.

🏥 Healthcare: HIPAA compliance, patient data, prescriptions.

Ecommerece 

social media 

🛠️ Development & Training

To retrain the models, navigate to the notebooks/ directory.

    ML.ipynb: Handles data preprocessing, feature engineering, and training the Scikit-Learn Gradient Boosting model.

    DL.ipynb: Handles vocabulary building, tokenization, and training the PyTorch LSTM model.

Both notebooks automatically save the best-performing models to the artifacts/ folder.
Bash

# To launch Jupyter
jupyter notebook notebooks/
