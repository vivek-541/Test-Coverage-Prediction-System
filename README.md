# 🎯 Test Coverage Prediction System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green)
![Status](https://img.shields.io/badge/Status-Production-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

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
<<<<<<< HEAD
### 🧠 Models Overview
=======

---

## 🧠 Models Overview
>>>>>>> 6823d0e (updated with list test cases and refined the readme.md file)

The system implements two distinct approaches. You can switch between them based on your need for speed vs. complexity.

### 🤖 1. ML Model (Gradient Boosting)
- **Algorithm**: Gradient Boosting Regressor (Scikit-Learn)
- **Location**: `artifacts/ml/test_coverage_model_balanced.pkl`
- **Features**: TF-IDF vectors, Domain Encoding, Engineered numeric features (counts, lengths)
- **Use Case**: Fast inference, highly interpretable baseline

### 🧠 2. DL Model (Deep Learning)
- **Algorithm**: LSTM (Long Short-Term Memory) with PyTorch
- **Location**: `artifacts/dl/test_coverage_pytorch_working.pkl`
- **Architecture**: Embedding Layer (96 dim) → LSTM (192 hidden) → Dense Layers
- **Use Case**: Captures complex sequential dependencies in text; better for deep semantic understanding

---

## 📊 Dataset Information

### Dataset Overview
- **Total Entries**: 1,000 test scenarios
- **Features**: 7 columns
- **Memory Usage**: 54.8 KB

### Data Columns
| Column | Type | Non-Null Count | Description |
|--------|------|----------------|-------------|
| domain | object | 1000 | Business domain category |
| feature_description | object | 1000 | Detailed feature text |
| input_test_cases | object | 1000 | List of test case descriptions |
| original_case_count | int64 | 1000 | Number of test cases provided |
| kept_case_count | int64 | 1000 | Number of test cases after filtering |
| coverage_percentage | float64 | 1000 | Target variable (0-100) |
| model_used | object | 1000 | Model type used for training |

### Domain Distribution
Each domain is equally represented with 200 samples:

| Domain | Count | Avg Coverage | Min Coverage | Max Coverage |
|--------|-------|--------------|--------------|--------------|
| E-commerce | 200 | 62.60% | 26.67% | 93.33% |
| Fintech | 200 | 61.89% | 26.67% | 94.12% |
| Healthcare | 200 | 62.23% | 26.67% | 93.33% |
| Logistics | 200 | 63.49% | 26.67% | 93.33% |
| Social Media | 200 | 62.30% | 26.67% | 93.33% |

### Feature Statistics
- **Average Description Length**: 20.06 characters
- **Average Number of Test Cases**: 1.00

<<<<<<< HEAD
🔌 API Documentation
POST /predict
<img width="1442" height="621" alt="image" src="https://github.com/user-attachments/assets/69709900-e306-4c3c-96bc-7a8d1c1dc109" />
=======
---
>>>>>>> 6823d0e (updated with list test cases and refined the readme.md file)

## 📝 Sample Test Cases

Below are comprehensive examples showing different coverage levels across various domains:

### **Test Case 1: Fintech - Payment Gateway Integration**
**Domain**: `Fintech`

**Feature Description**:
```
Payment gateway integration for processing credit card transactions. System must validate card details, process payments through third-party gateway, handle declined transactions, implement retry logic for failed payments, store encrypted payment tokens for future use, send email confirmations, and comply with PCI-DSS standards. Transaction limits: $10,000 per transaction, $50,000 daily limit.
```

**Test Cases**:
```
Test successful payment with valid card
Test payment with expired card
Test payment with insufficient funds
Test payment exceeding transaction limit
Test payment exceeding daily limit
Test 3D Secure authentication flow
Test card tokenization and storage
Test payment retry mechanism
Test declined transaction handling
Test email confirmation delivery
Test audit log creation for all transactions
```

**Expected Coverage**: 70-80% ✅

---

### **Test Case 2: Healthcare - Electronic Health Record Access**
**Domain**: `Healthcare`

**Feature Description**:
```
Electronic Health Record (EHR) access system for healthcare providers. Doctors and nurses can view patient medical history, lab results, prescriptions, and treatment plans. System must enforce role-based access control, log all PHI access with timestamp and reason, support emergency break-glass access for critical situations, mask sensitive data for unauthorized roles, comply with HIPAA requirements, and auto-lock sessions after 15 minutes of inactivity.
```

**Test Cases**:
```
Test authorized doctor access to patient records
Test nurse access with limited permissions
Test unauthorized access denial
Test emergency break-glass access with audit trail
Test data masking for non-authorized fields
Test session timeout after 15 minutes
Test PHI access logging
Test patient consent verification
Test access from multiple devices
```

**Expected Coverage**: 55-65% ⚠️ (Missing some edge cases)

---

### **Test Case 3: E-commerce - Shopping Cart & Checkout**
**Domain**: `E-commerce`

**Feature Description**:
```
Shopping cart and checkout functionality for online store. Users can add/remove items, apply discount codes, select shipping methods, and complete purchase. Cart should persist across sessions, calculate taxes based on location, validate inventory availability, support guest checkout, handle concurrent modifications, and integrate with payment gateway.
```

**Test Cases**:
```
Test add single item to cart
Test add multiple items to cart
Test remove item from cart
Test update item quantity
Test apply valid discount code
Test apply expired discount code
Test apply invalid discount code
Test cart persistence after logout
Test guest checkout without registration
Test inventory validation before checkout
Test shipping cost calculation
Test tax calculation based on zip code
Test payment gateway integration
Test order confirmation email
Test concurrent cart modifications
```

**Expected Coverage**: 85-95% 🎉 (Excellent!)

---

### **Test Case 4: Social Media - User Profile Management**
**Domain**: `Social Media`

**Feature Description**:
```
User profile management feature allowing users to update personal information, upload profile picture, set privacy preferences, link social accounts, and manage notification settings. Profile photos must be validated for size and format. Users can set profile visibility to public, friends-only, or private.
```

**Test Cases**:
```
Test update profile name
Test upload valid profile picture
Test upload oversized profile picture
Test update email address
Test update with duplicate email
Test change privacy settings to public
Test change privacy settings to private
Test link Facebook account
```

**Expected Coverage**: 45-55% ❌ (Low coverage - missing security tests)

---

### **Test Case 5: Logistics - Real-Time Package Tracking**
**Domain**: `Logistics`

**Feature Description**:
```
Real-time package tracking system with GPS integration. Customers can track package location, view delivery status, receive SMS/email notifications, estimate delivery time, and report issues. System must validate tracking numbers, handle multiple packages per order, detect GPS anomalies, support geofencing alerts, and maintain delivery history for 90 days.
```

**Test Cases**:
```
Test track package with valid tracking number
Test track package with invalid tracking number
Test real-time GPS location update
Test delivery status change notifications
Test SMS notification delivery
Test email notification delivery
Test geofencing alert when package enters delivery zone
Test GPS anomaly detection
Test multiple packages in single order
Test delivery time estimation
Test customer issue reporting
Test delivery history retrieval
Test tracking number validation
Test location privacy settings
```

**Expected Coverage**: 80-90% ✅ (Very comprehensive!)

---

### **Test Case 6: Fintech - Account Lockout (Simple)**
**Domain**: `Fintech`

**Feature Description**:
```
User login with email and password, support 2FA, account lockout after 5 attempts
```

**Test Cases**:
```
Test valid login
Test invalid password
Test account lockout
Test 2FA verification
```

**Expected Coverage**: 25-35% ❌ (Too few test cases!)

---

### **Test Case 7: Healthcare - Prescription Management (Comprehensive)**
**Domain**: `Healthcare`

**Feature Description**:
```
Digital prescription management system for doctors to create, modify, and send prescriptions to pharmacies. System must validate drug interactions, check patient allergies, enforce dosage limits, require digital signature from authorized prescriber, support e-prescribing to pharmacies, maintain prescription history, implement drug formulary checks, and comply with DEA regulations for controlled substances.
```

**Test Cases**:
```
Test create new prescription with valid drug
Test create prescription with patient allergy conflict
Test detect dangerous drug-drug interactions
Test validate dosage within safe limits
Test validate dosage exceeding maximum limit
Test digital signature requirement enforcement
Test send prescription to pharmacy via e-prescribe
Test controlled substance prescription with DEA validation
Test prescription modification with audit trail
Test prescription cancellation
Test view prescription history
Test formulary check for insurance coverage
Test prescription renewal workflow
Test unauthorized prescriber access denial
Test duplicate prescription detection
Test prescription for pediatric patient with weight-based dosage
```

**Expected Coverage**: 90-100% 🎉 (Excellent comprehensive testing!)

---

### **Test Case 8: E-commerce - Refund Processing**
**Domain**: `E-commerce`

**Feature Description**:
```
Automated refund processing system for returns. Customers can request refunds within 30 days, upload return shipping proof, and receive refund to original payment method.
```

**Test Cases**:
```
Test refund request within 30 days
Test refund request after 30 days
Test refund to credit card
Test refund status tracking
```

**Expected Coverage**: 35-45% ❌ (Missing validation, edge cases, security tests)

---

### **Test Case 9: Social Media - Content Moderation**
**Domain**: `Social Media`

**Feature Description**:
```
AI-powered content moderation system that automatically detects and flags inappropriate content including hate speech, violence, nudity, and spam. System must scan text, images, and videos, provide confidence scores, allow manual review by moderators, support user appeals, implement rate limiting to prevent abuse, maintain moderation logs, and comply with platform community guidelines. False positive rate must be below 5%.
```

**Test Cases**:
```
Test detection of hate speech in text post
Test detection of violent imagery
Test detection of nudity in uploaded photos
Test detection of spam content
Test detection of self-harm content
Test false positive handling for legitimate content
Test confidence score calculation
Test manual moderator review queue
Test user appeal submission
Test appeal decision notification
Test rate limiting for flagged users
Test moderation action audit logs
Test multi-language content moderation
Test context-aware moderation decisions
Test automated content removal for high-confidence violations
Test temporary account suspension for repeat violations
Test compliance with community guidelines
```

**Expected Coverage**: 85-95% ✅ (Very thorough!)

---

### **Test Case 10: Logistics - Driver Assignment System**
**Domain**: `Logistics`

**Feature Description**:
```
Automated driver assignment system that matches delivery orders with available drivers based on location proximity, vehicle capacity, driver working hours, and priority level. System must optimize routes, handle driver unavailability, support manual override by dispatchers, track driver status in real-time, and maintain assignment history.
```

**Test Cases**:
```
Test assign order to nearest available driver
Test assign order when no drivers available
Test vehicle capacity validation before assignment
Test driver working hours compliance
Test high-priority order assignment
Test route optimization after assignment
Test driver unavailability handling
Test manual override by dispatcher
Test real-time driver status tracking
Test assignment history logging
Test reassignment after driver cancellation
Test multiple orders to single driver
```

**Expected Coverage**: 75-85% ✅

---

## 🔌 API Documentation

### POST `/predict`
Generates a coverage prediction score.
<img width="1442" height="621" alt="image" src="https://github.com/user-attachments/assets/d0462156-b5d1-4ce9-bee8-e974e84508f3" />

**Endpoint**: `http://localhost:8000/predict`

#### Request Body (JSON)
```json
{
  "feature_description": "Implement a user authentication system allowing login via email and password with a 'Forgot Password' flow.",
  "input_test_cases": [
    "Verify user can login with valid credentials",
    "Verify error message on invalid password",
    "Check empty field validation",
    "Verify password reset link is sent"
  ],
  "domain": "security"
}
```

#### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `feature_description` | string | Detailed text describing the feature to be built (10-5000 chars) |
| `input_test_cases` | list | A list of strings, each describing a specific test case |
| `domain` | string | The business domain. Allowed values: `security`, `compliance`, `healthcare`, `finance`, `ecommerce`, `social media`, `logistics`, `other` |

#### Response (JSON)
```json
{
  "coverage_score": 85.5,
  "confidence": "High",
  "missing_aspects_detected": false,
  "model_used": "GradientBoostingRegressor"
}
```

---

## 📊 Supported Domains

The model is trained to recognize context within specific industry domains. Ensure your requests use one of the standardized domain tags:

- 💰 **Finance/Fintech**: Transactions, payments, currency, ledgers
- 🏥 **Healthcare**: HIPAA compliance, patient data, prescriptions
- 🛒 **E-commerce**: Shopping carts, checkouts, inventory
- 📱 **Social Media**: User profiles, content moderation, feeds
- 🚚 **Logistics**: Package tracking, driver assignment, routing
- 🔒 **Security**: Authentication, authorization, encryption
- ⚖️ **Compliance**: Regulatory requirements, audits
- 🔧 **Other**: General software features

---

## 🛠️ Development & Training

To retrain the models, navigate to the `notebooks/` directory.

- **ML.ipynb**: Handles data preprocessing, feature engineering, and training the Scikit-Learn Gradient Boosting model
- **DL.ipynb**: Handles vocabulary building, tokenization, and training the PyTorch LSTM model

Both notebooks automatically save the best-performing models to the `artifacts/` folder.

```bash
# To launch Jupyter
jupyter notebook notebooks/
<<<<<<< HEAD
=======
```

### Training Pipeline
1. Load and preprocess data from `data/raw/`
2. Engineer features (TF-IDF, domain encoding, text statistics)
3. Split data into train/validation/test sets
4. Train models and perform hyperparameter tuning
5. Evaluate on test set and save artifacts
6. Generate visualizations and performance reports

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip or conda package manager


### Quick Test
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "feature_description": "User registration with email validation",
    "input_test_cases": ["Test valid email", "Test invalid email format"],
    "domain": "security"
  }'
```

---

## 📈 Model Performance

### ML Model (Gradient Boosting)
- **MAE**: ~8.5%
- **RMSE**: ~12.3%
- **R² Score**: 0.82
- **Inference Time**: < 50ms

### DL Model (LSTM)
- **MAE**: ~7.2%
- **RMSE**: ~10.8%
- **R² Score**: 0.87
- **Inference Time**: ~150ms

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

Vivek chary - [@VivekCharyA](https://twitter.com/VivekCharyA)

---

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- PyTorch and Scikit-Learn teams for robust ML libraries
- The QA community for inspiring this project

---

**⭐ Star this repo if you find it helpful!**
>>>>>>> 6823d0e (updated with list test cases and refined the readme.md file)
