"""
Quick test script to verify model predictions
"""
import sys
sys.path.append('.')

from src.classical_ml import ClassicalMLPipeline
from src.transformer_model import TransformerPipeline
from src.preprocessing import clean_text

# Test reviews
test_reviews = [
    "This smartwatch is absolutely amazing! Best purchase ever!",
    "Terrible product. Waste of money. Very disappointed.",
    "It's okay, nothing special. Does the job.",
    "Love the fitness tracking features! Highly recommend!",
    "Battery died after one week. Poor quality."
]

sentiment_map = {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}

print("="*80)
print("SMARTWATCH SENTIMENT ANALYZER - PREDICTION TEST")
print("="*80)

# Load Classical ML model
print("\n🤖 Loading Classical ML Model (Logistic Regression)...")
classical_model = ClassicalMLPipeline(model_type='logistic')
classical_model.load_model('models/')

# Load Transformer model
print("🚀 Loading Transformer Model (DistilBERT)...")
transformer_model = TransformerPipeline()
transformer_model.load_model('models/transformer')

print("\n" + "="*80)
print("PREDICTIONS")
print("="*80)

for i, review in enumerate(test_reviews, 1):
    print(f"\n📝 Review {i}: \"{review}\"")
    
    # Clean text
    cleaned = clean_text(review)
    
    # Classical ML prediction
    classical_pred = classical_model.predict([cleaned])[0]
    classical_proba = classical_model.predict_proba([cleaned])[0]
    classical_conf = max(classical_proba) * 100
    
    # Transformer prediction
    transformer_pred = transformer_model.predict([cleaned])[0]
    transformer_proba = transformer_model.predict_proba([cleaned])[0]
    transformer_conf = transformer_proba[transformer_pred] * 100
    
    print(f"   Classical ML: {sentiment_map[classical_pred]} (Confidence: {classical_conf:.1f}%)")
    print(f"   Transformer:  {sentiment_map[transformer_pred]} (Confidence: {transformer_conf:.1f}%)")
    
    if classical_pred != transformer_pred:
        print(f"   ⚠️  Models disagree!")

print("\n" + "="*80)
print("✅ Test complete!")
print("="*80)
