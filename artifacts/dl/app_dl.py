"""
PYTORCH API - UNIVERSAL VERSION WITH LIST INPUT
================================================
Accepts test cases as a list instead of pipe-separated string
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import pickle
import os
import re

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class WorkingCoverageNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=96, hidden_dim=192, 
                 num_numeric_features=18, dropout=0.4):
        super(WorkingCoverageNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, 
                           batch_first=True, bidirectional=False)
        self.domain_embedding = nn.Embedding(5, 16)
        self.fc1 = nn.Linear(hidden_dim + 16 + num_numeric_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        
    def forward(self, text_seq, domain, numeric_features):
        embedded = self.embedding(text_seq)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        text_features = hidden[-1]
        domain_features = self.domain_embedding(domain)
        combined = torch.cat([text_features, domain_features, numeric_features], dim=1)
        
        x = self.fc1(combined)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return self.fc3(x).squeeze()


# ============================================================================
# TEXT PREPROCESSOR
# ============================================================================

class TextPreprocessor:
    """Recreate text preprocessor"""
    
    def __init__(self, word_to_idx=None, max_seq_length=150):
        if word_to_idx is None:
            self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        else:
            self.word_to_idx = word_to_idx
        self.max_seq_length = max_seq_length
        
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text):
        return self.clean_text(text).split()
    
    def text_to_sequence(self, text):
        tokens = self.tokenize(text)
        sequence = [self.word_to_idx.get(word, 1) for word in tokens]
        
        if len(sequence) < self.max_seq_length:
            sequence = sequence + [0] * (self.max_seq_length - len(sequence))
        else:
            sequence = sequence[:self.max_seq_length]
        
        return sequence
    
    def texts_to_sequences(self, texts):
        return np.array([self.text_to_sequence(text) for text in texts])


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    feature_description: str = Field(
        ..., 
        min_length=10, 
        max_length=5000,
        description="Detailed description of the feature"
    )
    input_test_cases: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of test case descriptions"
    )
    domain: str = Field(
        ...,
        description="Domain: security, compliance, healthcare, finance, or other"
    )
    
    @validator('domain')
    def validate_domain(cls, v):
        valid = ["security", "compliance", "healthcare", "finance", "other"]
        if v.lower() not in valid:
            raise ValueError(f"Domain must be one of: {', '.join(valid)}")
        return v.lower()
    
    @validator('input_test_cases')
    def validate_test_cases(cls, v):
        # Clean and filter empty strings
        v = [tc.strip() for tc in v if tc.strip()]
        if not v:
            raise ValueError("At least one non-empty test case is required")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "feature_description": "User authentication with 2FA and session management",
                "input_test_cases": [
                    "Valid login with correct credentials",
                    "Invalid password attempt",
                    "Session timeout handling",
                    "2FA code verification",
                    "Account lockout after failed attempts"
                ],
                "domain": "security"
            }
        }


class PredictionResponse(BaseModel):
    predicted_coverage: float
    status: str
    # analysis: dict
    # suggestions: List[str]
    metadata: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str


# ============================================================================
# FLEXIBLE MODEL MANAGER
# ============================================================================

class PyTorchModelManager:
    _instance = None
    _model = None
    _preprocessor = None
    _domain_classes = None
    _device = None
    _model_package = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PyTorchModelManager, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        """Load model - FLEXIBLE for any pickle format!"""
        try:
            print("=" * 80)
            print("🔥 LOADING PYTORCH MODEL (FLEXIBLE)...")
            print("=" * 80)
            
            possible_paths = [
                'test_coverage_pytorch_working.pkl',
                'test_coverage_pytorch_ultimate.pkl',
                'test_coverage_pytorch_improved.pkl',
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"✅ Found model at: {path}")
                    break
            
            if model_path is None:
                raise FileNotFoundError("Model file not found!")
            
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"✅ Using device: {self._device}")
            
            print("⚙️ Loading with pickle.load()...")
            with open(model_path, 'rb') as f:
                self._model_package = pickle.load(f)
            
            print("\n📦 Package contents:")
            for key in self._model_package.keys():
                print(f"  - {key}")
            
            print("\n⚙️ Extracting components...")
            
            # Get preprocessor
            if 'preprocessor' in self._model_package:
                print("  ✓ Found 'preprocessor'")
                self._preprocessor = self._model_package['preprocessor']
            elif 'preprocessor_data' in self._model_package:
                print("  ✓ Found 'preprocessor_data', reconstructing...")
                data = self._model_package['preprocessor_data']
                self._preprocessor = TextPreprocessor(
                    word_to_idx=data['word_to_idx'],
                    max_seq_length=data['max_seq_length']
                )
            else:
                print("  ⚠️ No preprocessor found, creating default...")
                self._preprocessor = TextPreprocessor()
            
            # Get domain classes
            if 'domain_encoder' in self._model_package:
                print("  ✓ Found 'domain_encoder'")
                self._domain_classes = list(self._model_package['domain_encoder'].classes_)
            elif 'domain_classes' in self._model_package:
                print("  ✓ Found 'domain_classes'")
                self._domain_classes = self._model_package['domain_classes']
            else:
                print("  ⚠️ No domain info found, using defaults...")
                self._domain_classes = ['security', 'compliance', 'healthcare', 'finance', 'other']
            
            # Get model config
            if 'model_config' in self._model_package:
                config = self._model_package['model_config']
            else:
                print("  ⚠️ No model_config, inferring from state_dict...")
                state_dict = self._model_package.get('model_state_dict', self._model_package)
                
                if 'embedding.weight' in state_dict:
                    vocab_size = state_dict['embedding.weight'].shape[0]
                    embedding_dim = state_dict['embedding.weight'].shape[1]
                else:
                    vocab_size = 3000
                    embedding_dim = 96
                
                config = {
                    'vocab_size': vocab_size,
                    'embedding_dim': embedding_dim,
                    'hidden_dim': 192,
                    'num_numeric_features': 18,
                    'dropout': 0.4
                }
                print(f"  ℹ️ Inferred vocab_size={vocab_size}, embedding_dim={embedding_dim}")
            
            print(f"\n⚙️ Building model (vocab={config['vocab_size']})...")
            self._model = WorkingCoverageNN(
                vocab_size=config['vocab_size'],
                embedding_dim=config.get('embedding_dim', 96),
                hidden_dim=config.get('hidden_dim', 192),
                num_numeric_features=config.get('num_numeric_features', 18),
                dropout=config.get('dropout', 0.4)
            )
            
            state_dict = self._model_package.get('model_state_dict', self._model_package.get('state_dict'))
            if state_dict:
                self._model.load_state_dict(state_dict)
                print("  ✓ Loaded model weights")
            else:
                raise ValueError("No state_dict found in pickle file!")
            
            self._model.to(self._device)
            self._model.eval()
            
            print(f"\n✅ Model loaded successfully!")
            print(f"   Vocab size: {config['vocab_size']}")
            print(f"   Domain classes: {self._domain_classes}")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _create_numeric_features(self, feature_description: str, input_test_cases: List[str]) -> np.ndarray:
        """
        Create 18 numeric features from inputs
        ✅ NOW ACCEPTS LIST OF TEST CASES
        """
        # Join list to single string for analysis
        test_cases_text = " | ".join(input_test_cases)
        
        # Feature description stats
        feature_length = len(feature_description)
        feature_word_count = len(feature_description.split())
        feature_sentence_count = sum(1 for c in feature_description if c in '.!?') + 1
        
        # Test cases stats
        test_length = len(test_cases_text)
        num_tests = len(input_test_cases)  # ✅ Now count list items
        avg_test_length = test_length / max(num_tests, 1)
        
        # Keyword detection in feature description
        has_security = int(any(kw in feature_description.lower() for kw in 
                             ['security', 'authentication', 'authorization', 'encryption', '2fa', 'otp']))
        has_compliance = int(any(kw in feature_description.lower() for kw in 
                               ['hipaa', 'gdpr', 'pci', 'compliance', 'audit', 'regulation']))
        has_critical = int(any(kw in feature_description.lower() for kw in 
                             ['critical', 'emergency', 'important', 'must', 'required']))
        
        # Keyword detection in test cases
        test_cases_text_lower = test_cases_text.lower()
        has_negative = int(any(kw in test_cases_text_lower for kw in 
                             ['invalid', 'incorrect', 'fail', 'error', 'negative']))
        has_boundary = int(any(kw in test_cases_text_lower for kw in 
                             ['boundary', 'edge', 'limit', 'maximum', 'minimum']))
        has_security_tests = int(any(kw in test_cases_text_lower for kw in 
                                    ['security', 'authentication', 'authorization', 'access', 'permission']))
        
        # Ratio features
        test_ratio = test_length / max(feature_length, 1)
        word_density = feature_word_count / max(feature_length, 1)
        complexity = len(re.findall(r'\bif\b|\belse\b|\bfor\b|\bwhile\b', feature_description.lower())) + 1
        avg_word_len = feature_length / max(feature_word_count, 1)
        coverage_ratio = num_tests / max(feature_sentence_count, 1)
        domain_weight = 1.0  # Will be set based on domain
        
        # Create feature array (18 features)
        features = np.array([[
            feature_length, feature_word_count, feature_sentence_count,
            test_length, num_tests, avg_test_length,
            has_security, has_compliance, has_critical,
            has_negative, has_boundary, has_security_tests,
            test_ratio, word_density, complexity,
            avg_word_len, coverage_ratio, domain_weight
        ]], dtype=np.float32)
        
        # Normalize
        features = (features - features.mean()) / (features.std() + 1e-8)
        return features
    
    def predict(self, feature_description: str, input_test_cases: List[str], domain: str) -> dict:
        """
        Make prediction
        ✅ NOW ACCEPTS LIST OF TEST CASES
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")
        
        start_time = datetime.now()
        
        try:
            # Prepare text (join list with pipe separator)
            test_cases_text = " | ".join(input_test_cases)
            combined_text = f"{feature_description} {test_cases_text}"
            text_seq = self._preprocessor.texts_to_sequences([combined_text])
            text_tensor = torch.LongTensor(text_seq).to(self._device)
            
            # Domain
            try:
                domain_idx = self._domain_classes.index(domain)
            except:
                domain_idx = 4  # Default to 'other'
            domain_tensor = torch.LongTensor([domain_idx]).to(self._device)
            
            # Numeric features (pass list directly)
            numeric = self._create_numeric_features(feature_description, input_test_cases)
            numeric_tensor = torch.FloatTensor(numeric).to(self._device)
            
            # Predict
            with torch.no_grad():
                prediction = self._model(text_tensor, domain_tensor, numeric_tensor)
                coverage = float(prediction.cpu().numpy())
                coverage = np.clip(coverage, 0, 100)
            
            pred_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Analysis (now works with list)
            num_tests = len(input_test_cases)
            test_cases_text_lower = " ".join(input_test_cases).lower()
            has_negative = any(kw in test_cases_text_lower for kw in ['invalid', 'fail', 'error', 'incorrect'])
            has_security = any(kw in feature_description.lower() for kw in ['security', 'authentication', 'authorization'])
            
            # Status
            if coverage >= 80:
                status = "Excellent"
            elif coverage >= 60:
                status = "Good"
            elif coverage >= 40:
                status = "Fair"
            else:
                status = "Poor"
            
            # Suggestions
            suggestions = []
            if not has_negative:
                suggestions.append("❗ Add negative test cases (invalid inputs, error scenarios)")
            if num_tests < 8:
                suggestions.append("❗ Add more test scenarios (aim for 10-15 tests)")
            if not has_security and domain in ["security", "finance"]:
                suggestions.append("🔒 Add security-focused test cases")
            if coverage < 70:
                suggestions.append("📊 Add edge cases and boundary tests")
            if coverage < 50:
                suggestions.append("⚠️ Add integration and end-to-end tests")
            
            if not suggestions:
                suggestions.append("✅ Great test coverage! Consider adding performance tests.")
            
            return {
                "predicted_coverage": round(coverage, 2),
                "status": status,
                # "analysis": {
                #     "num_test_cases": num_tests,
                #     "has_negative_tests": has_negative,
                #     "has_security_keywords": has_security,
                #     # "test_case_list": input_test_cases[:5]  # Show first 5 test cases
                # },
                # "suggestions": suggestions,
                "metadata": {
                    "model_version": self._model_package.get('version', 'Unknown'),
                    "prediction_time_ms": round(pred_time, 2),
                    "device": str(self._device),
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_health(self) -> dict:
        return {
            "status": "healthy" if self._model else "unhealthy",
            "model_loaded": self._model is not None,
            "model_version": self._model_package.get('version', 'Unknown') if self._model_package else "Unknown",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Test Coverage Predictor (PyTorch)",
    description="Deep Learning API with List Input - R² = 0.6868",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = None

@app.on_event("startup")
async def startup():
    global model_manager
    print("\n🚀 Starting API...")
    model_manager = PyTorchModelManager()
    print("✅ API Ready!\n")

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Test Coverage Predictor API",
        "model": "PyTorch LSTM",
        "performance": "R² = 0.6868, MAE = 5.71%",
        "input_format": "List of test cases",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    if not model_manager:
        raise HTTPException(503, "Model not loaded")
    return model_manager.get_health()

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict test coverage percentage
    
    **Input Format:**
    - feature_description: String (10-5000 chars)
    - input_test_cases: **List of strings** (1-100 items)
    - domain: One of [security, compliance, healthcare, finance, other]
    
    **Example:**
```json
    {
      "feature_description": "User authentication system",
      "input_test_cases": [
        "Valid login",
        "Invalid password",
        "Account lockout"
      ],
      "domain": "security"
    }
```
    """
    if not model_manager:
        raise HTTPException(503, "Model not loaded")
    try:
        return model_manager.predict(
            request.feature_description,
            request.input_test_cases,
            request.domain
        )
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)