"""
FastAPI web application for smartwatch sentiment analysis
"""
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="Smartwatch Sentiment Analyzer")

# Setup templates
templates = Jinja2Templates(directory="app/templates")

# Sentiment labels
SENTIMENT_LABELS = {
    0: "Negative 😞",
    1: "Neutral 😐",
    2: "Positive 😊"
}


class PredictionRequest(BaseModel):
    text: str
    model_type: str = "transformer"


class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    model_used: str


# Global model storage
models = {
    'classical': None,
    'transformer': None,
    'ensemble': None,
    'advanced_transformer': None
}


def load_models():
    """Load pre-trained models"""
    global models
    
    try:
        # Load classical model
        from src.classical_ml import ClassicalMLPipeline
        classical_pipeline = ClassicalMLPipeline(model_type='logistic')
        classical_pipeline.load_model('models/')
        models['classical'] = classical_pipeline
        print("✅ Classical model loaded")
    except Exception as e:
        print(f"⚠️ Classical model not loaded: {e}")
    
    try:
        # Load ensemble model
        from src.advanced_models import EnsembleClassicalModel
        ensemble_pipeline = EnsembleClassicalModel()
        ensemble_pipeline.load_model('models/ensemble/')
        models['ensemble'] = ensemble_pipeline
        print("✅ Ensemble model loaded")
    except Exception as e:
        print(f"⚠️ Ensemble model not loaded: {e}")
    
    try:
        # Load transformer model
        from src.transformer_model import TransformerPipeline
        transformer_pipeline = TransformerPipeline()
        transformer_pipeline.load_model('models/transformer')
        models['transformer'] = transformer_pipeline
        print("✅ Transformer model loaded")
    except Exception as e:
        print(f"⚠️ Transformer model not loaded: {e}")
    
    try:
        # Load advanced transformer model
        from src.advanced_models import AdvancedTransformerModel
        advanced_pipeline = AdvancedTransformerModel()
        advanced_pipeline.load_model('models/advanced_transformer')
        models['advanced_transformer'] = advanced_pipeline
        print("✅ Advanced transformer model loaded")
    except Exception as e:
        print(f"⚠️ Advanced transformer model not loaded: {e}")


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render home page"""
    return templates.TemplateResponse("professional.html", {"request": request})

@app.get("/advanced", response_class=HTMLResponse)
async def advanced_home(request: Request):
    """Render advanced home page"""
    return templates.TemplateResponse("advanced.html", {"request": request})

@app.get("/classic", response_class=HTMLResponse)
async def classic_home(request: Request):
    """Render classic home page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(text: str = Form(...), model_type: str = Form("transformer")):
    """Make sentiment prediction"""
    
    if not text.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Please provide review text"}
        )
    
    # Preprocess text
    from src.preprocessing import clean_text
    cleaned_text = clean_text(text)
    
    # Select model
    model = models.get(model_type)
    
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": f"{model_type} model not available"}
        )
    
    try:
        # Make prediction
        if model_type == "transformer":
            predictions = model.predict([cleaned_text])
            probabilities = model.predict_proba([cleaned_text])
            sentiment_idx = predictions[0]
            confidence = float(probabilities[0][sentiment_idx])
        else:
            predictions = model.predict([cleaned_text])
            probabilities = model.predict_proba([cleaned_text])
            sentiment_idx = predictions[0]
            confidence = float(max(probabilities[0]))
        
        sentiment_label = SENTIMENT_LABELS[sentiment_idx]
        
        return {
            "text": text,
            "sentiment": sentiment_label,
            "sentiment_class": int(sentiment_idx),
            "confidence": round(confidence * 100, 2),
            "model_used": model_type
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "classical": models['classical'] is not None,
            "transformer": models['transformer'] is not None
        }
    }


@app.get("/api/models")
async def get_models():
    """Get available models"""
    available = []
    
    if models['classical']:
        available.append({"name": "classical", "type": "Logistic Regression + TF-IDF", "category": "Classical ML"})
    if models['ensemble']:
        available.append({"name": "ensemble", "type": "Ensemble (LR + SVM + NB + GB)", "category": "Classical ML"})
    if models['transformer']:
        available.append({"name": "transformer", "type": "DistilBERT", "category": "Transformer"})
    if models['advanced_transformer']:
        available.append({"name": "advanced_transformer", "type": "DistilBERT (Advanced)", "category": "Transformer"})
    
    return {"available_models": available}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
