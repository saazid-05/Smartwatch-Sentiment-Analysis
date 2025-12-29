# 🎬 Smartwatch Sentiment Analyzer - Demo Guide

## 🌐 Web Application Demo

### Access the Application
The web application is currently running at:
**http://localhost:8000**

### Features to Try

#### 1. Positive Review Test
```
This smartwatch is absolutely amazing! The battery life lasts for days and the fitness tracking is incredibly accurate. Best purchase I've made this year!
```
Expected: **Positive 😊** sentiment

#### 2. Negative Review Test
```
Terrible product. Stopped working after just one week. The screen is unresponsive and customer service was no help. Complete waste of money.
```
Expected: **Negative 😞** sentiment

#### 3. Neutral Review Test
```
It's okay for the price. Nothing special but does the basic functions. Battery could be better.
```
Expected: **Neutral 😐** sentiment

#### 4. Model Comparison
Try the same review with both models:
- Select "Classical ML (Logistic Regression)"
- Analyze the review
- Then select "Transformer (DistilBERT)"
- Analyze the same review
- Compare confidence scores and predictions

### Sample Reviews to Test

**Positive Examples:**
- "Love the design and features! Heart rate monitoring is spot on."
- "Absolutely perfect! The health metrics are detailed and accurate."
- "Excellent smartwatch! The call quality is clear and convenient."

**Negative Examples:**
- "Poor battery life and the charging is slow. Would not recommend."
- "The worst smartwatch I've ever owned. Constantly disconnects."
- "Horrible experience. The watch randomly restarts frequently."

**Neutral Examples:**
- "Pretty good overall. Some features are missing but acceptable."
- "It's decent. Does the basics well but lacks advanced features."
- "Average at best. The watch is functional but nothing impresses."

## 🔌 API Demo

### Using cURL

#### Predict with Classical ML
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "text=This smartwatch is fantastic!" \
  -F "model_type=classical"
```

#### Predict with Transformer
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "text=This smartwatch is fantastic!" \
  -F "model_type=transformer"
```

#### Health Check
```bash
curl http://localhost:8000/api/health
```

#### List Models
```bash
curl http://localhost:8000/api/models
```

### Using Python Requests
```python
import requests

# Predict sentiment
response = requests.post(
    "http://localhost:8000/predict",
    data={
        "text": "This smartwatch is amazing!",
        "model_type": "transformer"
    }
)

result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}%")
```

## 📊 Model Comparison Demo

### View Comparison Chart
The comparison chart is saved at:
`results/model_comparison.png`

Open it to see:
- Bar chart comparing all models
- Accuracy scores for each model
- Visual distinction between Classical ML (blue) and Transformers (red)

### Run Full Comparison
```bash
python notebooks/train_and_compare.py
```

This will:
1. Load and preprocess data
2. Train all classical ML models
3. Train transformer model
4. Generate comparison report
5. Create visualization

## 🧪 Testing Scripts

### Quick Prediction Test
```bash
python test_predictions.py
```

Shows predictions from both model types on sample reviews.

### Custom Test
Create your own test:
```python
from src.classical_ml import ClassicalMLPipeline

model = ClassicalMLPipeline(model_type='logistic')
model.load_model('models/')

review = "Your custom review here"
prediction = model.predict([review])
print(f"Sentiment: {prediction[0]}")  # 0=Negative, 1=Neutral, 2=Positive
```

## 📱 Mobile Demo

The web interface is responsive! Try accessing from:
- Smartphone browser
- Tablet
- Different screen sizes

## 🎯 Key Demonstrations

### 1. Real-time Analysis
- Type a review in the text area
- Click "Analyze Sentiment"
- See instant results with confidence scores

### 2. Model Switching
- Analyze the same review with different models
- Compare results and confidence levels
- Observe how models handle different types of text

### 3. Confidence Scores
- Notice how confidence varies by review clarity
- Clear positive/negative reviews = higher confidence
- Ambiguous reviews = lower confidence

### 4. Visual Feedback
- Positive results show in green
- Negative results show in red
- Neutral results show in yellow
- Smooth animations and transitions

## 🔍 What to Observe

### Classical ML Strengths
- Fast inference (< 100ms)
- Consistent predictions
- Works well with clear sentiment words

### Transformer Strengths
- Better contextual understanding
- Handles complex sentences
- More nuanced predictions
- (Note: Requires larger dataset to show full potential)

## 📈 Performance Metrics

Current performance on test set:
- **Accuracy**: 80%
- **F1 Score**: 0.71 (Transformer)
- **Inference Time**: 
  - Classical ML: ~50ms
  - Transformer: ~200ms

## 🎓 Educational Value

This demo shows:
1. **End-to-end ML pipeline**: From data to deployment
2. **Model comparison**: Classical vs Modern approaches
3. **Web deployment**: Production-ready application
4. **API design**: RESTful endpoints
5. **User experience**: Clean, intuitive interface

## 🚀 Next Steps

After the demo:
1. Try with your own smartwatch reviews
2. Experiment with different model types
3. Modify the code to add features
4. Train on larger datasets
5. Deploy to cloud platforms

## 💡 Tips

- Use clear, descriptive reviews for best results
- Try edge cases (sarcasm, mixed sentiment)
- Compare model confidence scores
- Test with different review lengths
- Experiment with technical vs casual language

---

**Enjoy exploring the Smartwatch Sentiment Analyzer!** 🎉
