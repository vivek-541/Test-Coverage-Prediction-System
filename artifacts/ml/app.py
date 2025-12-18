from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Union
import numpy as np
from datetime import datetime
import pickle
import traceback
import os

# ============================================================================
# PYDANTIC MODELS (STANDARDIZED)
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for coverage prediction - STANDARDIZED"""
    
    feature_description: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Detailed description of the feature to be tested"
    )
    
    input_test_cases: List[str] = Field(  # ✅ Changed from test_cases
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
        # ✅ STANDARDIZED domain values (lowercase)
        valid_domains = ["security", "compliance", "healthcare", "finance", "other"]
        v = v.lower()
        if v not in valid_domains:
            raise ValueError(f"Domain must be one of: {', '.join(valid_domains)}")
        return v
    
    @validator('input_test_cases')
    def validate_test_cases(cls, v):
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
    """Response model for coverage prediction - STANDARDIZED"""
    predicted_coverage: float  # ✅ Changed from coverage_percentage
    status: str
    # analysis: dict
    # suggestions: List[str]
    metadata: dict


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str
    model_name: str
    timestamp: str


# ============================================================================
# MODEL MANAGER FOR SKLEARN MODELS (STANDARDIZED)
# ============================================================================

class SklearnModelManager:
    """Singleton model manager for sklearn models"""
    
    _instance = None
    _model = None
    _model_package = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SklearnModelManager, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        """Load sklearn model package"""
        try:
            print("=" * 80)
            print("🔥 LOADING SKLEARN MODEL...")
            print("=" * 80)
            
            # Try different paths
            possible_paths = [
                'test_coverage_model_balanced.pkl',
                'test_coverage_model_production.pkl',
                'test_coverage_model_fixed.pkl',
                'ml_models/test_coverage_model_balanced.pkl',
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"✅ Found model at: {path}")
                    break
            
            if model_path is None:
                raise FileNotFoundError(
                    f"Model not found! Tried:\n" + 
                    "\n".join(f"  - {p}" for p in possible_paths)
                )
            
            # Load model
            with open(model_path, 'rb') as f:
                self._model_package = pickle.load(f)
            
            self._model = self._model_package['model']
            
            print(f"✅ Model loaded successfully!")
            print(f"   Model Type: {self._model_package.get('model_name', 'Unknown')}")
            print(f"   Version: {self._model_package.get('version', 'Unknown')}")
            print(f"   TF-IDF Features: {len(self._model_package['tfidf_vectorizer'].vocabulary_)}")
            print("=" * 80)
            
        except FileNotFoundError as e:
            print(f"❌ {str(e)}")
            raise RuntimeError("Model file not found.")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict(self, feature_description: str, input_test_cases: List[str], domain: str) -> dict:
        """
        Make prediction using sklearn model
        ✅ STANDARDIZED to accept List[str] for input_test_cases
        """
        
        if self._model is None:
            raise RuntimeError("Model not loaded")
        
        start_time = datetime.now()
        
        # Get components from model package
        tfidf_vectorizer = self._model_package['tfidf_vectorizer']
        domain_encoder = self._model_package['domain_encoder']
        scaler = self._model_package['scaler']
        
        # 1. COMBINE TEXT (join list with pipe)
        test_cases_text = " | ".join(input_test_cases)
        combined_text = f"{feature_description} {test_cases_text}"
        
        # 2. TF-IDF FEATURES
        tfidf_features = tfidf_vectorizer.transform([combined_text]).toarray()
        
        # 3. DOMAIN FEATURES (one-hot encoding)
        # Map standardized domains to original training domains
        domain_mapping = {
            'security': 'Fintech',      # Map security to closest match
            'compliance': 'Healthcare',  # Compliance often healthcare-related
            'healthcare': 'Healthcare',
            'finance': 'Fintech',
            'other': 'E-commerce'        # Default
        }
        
        mapped_domain = domain_mapping.get(domain, 'E-commerce')
        
        try:
            domain_idx = domain_encoder.transform([mapped_domain])[0]
        except:
            domain_idx = 0  # Default to first domain
        
        # Create one-hot vector
        num_domains = len(domain_encoder.classes_)
        domain_features = np.zeros((1, num_domains))
        domain_features[0, domain_idx] = 1
        
        # 4. ENGINEERED FEATURES
        feature_length = len(feature_description)
        feature_word_count = len(feature_description.split())
        num_test_cases = len(input_test_cases)  # ✅ Count list items
        
        # Keyword detection
        has_security = int(any(
            kw in feature_description.lower() 
            for kw in ['security', 'authentication', 'authorization', 'encryption', '2fa', 'otp']
        ))
        
        has_compliance = int(any(
            kw in feature_description.lower() 
            for kw in ['hipaa', 'gdpr', 'pci', 'compliance', 'audit', 'regulation']
        ))
        
        has_negative_tests = int(any(
            kw in test_cases_text.lower() 
            for kw in ['invalid', 'incorrect', 'fail', 'error', 'negative']
        ))
        
        # Create engineered features array
        engineered_features = np.array([[
            feature_length,
            feature_word_count,
            num_test_cases,
            has_security,
            has_compliance,
            has_negative_tests
        ]])
        
        # Scale numeric features
        engineered_features = scaler.transform(engineered_features)
        
        # 5. COMBINE ALL FEATURES
        X = np.hstack([tfidf_features, domain_features, engineered_features])
        
        # 6. PREDICT
        coverage = float(self._model.predict(X)[0])
        coverage = np.clip(coverage, 0, 100)
        
        # Calculate prediction time
        prediction_time = (datetime.now() - start_time).total_seconds() * 1000
        

        
        # 8. DETERMINE STATUS
        if coverage >= 80:
            status = "Excellent"
        elif coverage >= 60:
            status = "Good"
        elif coverage >= 40:
            status = "Fair"
        else:
            status = "Poor"
        

        return {
            "predicted_coverage": round(coverage, 2),  # ✅ Standardized name
            "status": status,
            "metadata": {
                "model_version": self._model_package.get('version', 'Unknown'),
                "model_name": self._model_package.get('model_name', 'Unknown'),
                "model_type": "Sklearn Gradient Boosting",
                "prediction_time_ms": round(prediction_time, 2),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "num_features": X.shape[1]
            }
        }
    
    def get_health(self) -> dict:
        """Get model health status"""
        return {
            "status": "healthy" if self._model is not None else "unhealthy",
            "model_loaded": self._model is not None,
            "model_version": self._model_package.get('version', 'Unknown') if self._model_package else "Unknown",
            "model_name": self._model_package.get('model_name', 'Unknown') if self._model_package else "Unknown",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Test Coverage Predictor API (Sklearn)",
    description="Machine Learning API with Standardized Format - R² = 0.641",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model manager
model_manager = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model_manager
    try:
        print("\n🚀 Starting FastAPI application...")
        model_manager = SklearnModelManager()
        print("✅ FastAPI app started successfully\n")
    except Exception as e:
        print(f"❌ Failed to start app: {str(e)}")
        raise


# ============================================================================
# ENDPOINTS (STANDARDIZED)
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Test Coverage Predictor API (Sklearn)",
        "version": "2.0.0",
        "model": "Sklearn Gradient Boosting",
        "performance": "R² = 0.641, MAE = 6.21%",
        "input_format": "List of test cases",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "domains": "/domains",
            "model-info": "/model-info"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_manager.get_health()


@app.post("/predict", response_model=Union[PredictionResponse, List[PredictionResponse]], tags=["Prediction"])
async def predict_coverage(
    request: Union[PredictionRequest, List[PredictionRequest]]
):
    """
    Predict test coverage percentage
    
    **Input Format:**
    - feature_description: String (10-5000 chars)
    - input_test_cases: **List of strings** (1-100 items)
    - domain: One of [security, compliance, healthcare, finance, other]
    
    **Example:**
```json
    {
      "feature_description": "Payment processing system",
      "input_test_cases": [
        "Valid payment",
        "Expired card",
        "Insufficient funds"
      ],
      "domain": "finance"
    }
```
    
    Supports both single prediction and batch predictions (up to 10 at once)
    """
    
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check if single or batch request
    is_batch = isinstance(request, list)
    requests = request if is_batch else [request]
    
    # Limit batch size
    if len(requests) > 10:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 10 requests per batch. Please split into smaller batches."
        )
    
    # Process all requests
    results = []
    for req in requests:
        try:
            result = model_manager.predict(
                req.feature_description,
                req.input_test_cases,  # ✅ Standardized name
                req.domain
            )
            results.append(result)
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Prediction failed: {str(e)}"
            )
    
    # Return single result or batch results
    return results[0] if not is_batch else results


@app.get("/domains", tags=["General"])
async def get_domains():
    """Get list of supported domains (STANDARDIZED)"""
    return {
        "domains": [
            {
                "name": "security",
                "description": "Authentication, Authorization, Encryption, 2FA",
                "keywords": ["security", "authentication", "authorization", "encryption"]
            },
            {
                "name": "compliance",
                "description": "HIPAA, GDPR, PCI-DSS, Audit, Regulations",
                "keywords": ["hipaa", "gdpr", "pci", "compliance", "audit"]
            },
            {
                "name": "healthcare",
                "description": "EMR, EHR, Patient data, Medical records",
                "keywords": ["patient", "medical", "health", "clinical"]
            },
            {
                "name": "finance",
                "description": "Banking, Payments, Trading, Transactions",
                "keywords": ["payment", "transaction", "banking", "financial"]
            },
            {
                "name": "other",
                "description": "E-commerce, Social Media, Logistics, General",
                "keywords": ["shopping", "cart", "order", "delivery"]
            }
        ],
        "note": "Domains are mapped internally to training data categories"
    }


@app.get("/model-info", tags=["General"])
async def get_model_info():
    """Get detailed model information"""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_package = model_manager._model_package
    
    return {
        "model_name": model_package.get('model_name', 'Unknown'),
        "version": model_package.get('version', 'Unknown'),
        "model_type": str(type(model_package['model']).__name__),
        "algorithm": "Gradient Boosting Regressor",
        "framework": "Scikit-learn",
        "features": {
            "tfidf_features": len(model_package['tfidf_vectorizer'].vocabulary_),
            "domain_features": len(model_package['domain_encoder'].classes_),
            "engineered_features": 6,
            "total_features": len(model_package['tfidf_vectorizer'].vocabulary_) + len(model_package['domain_encoder'].classes_) + 6
        },
        "training_domains": list(model_package['domain_encoder'].classes_),
        "standardized_domains": ["security", "compliance", "healthcare", "finance", "other"],
        "performance": {
            "test_r2": 0.641,
            "test_mae": 6.21,
            "description": "Baseline model - fast and reliable"
        }
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # ✅ Changed port to 8001