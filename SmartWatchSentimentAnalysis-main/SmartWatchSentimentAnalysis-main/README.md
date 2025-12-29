# ⌚ Smartwatch Sentiment Analyzer

A complete sentiment analysis system comparing Classical Machine Learning with Transformer models for smartwatch reviews.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/Transformers-4.35+-orange.svg)](https://huggingface.co/transformers/)

## 🎯 Project Overview

**Domain**: Wearable Tech  
**Dataset**: Amazon Gadget Reviews (Smartwatch focus)  
**Deployment**: Web Application (FastAPI)  
**Goal**: Compare Classical ML vs Transformer models to demonstrate improvement in accuracy with contextual understanding

## ✨ Features

- 📊 **Data Pipeline**: Automated data loading, cleaning, and preprocessing
- 🤖 **Classical ML**: Logistic Regression, SVM, Naive Bayes with TF-IDF
- 🚀 **Transformers**: DistilBERT fine-tuned for sentiment analysis
- 📈 **Comparison**: Comprehensive evaluation and visualization
- 🌐 **Web App**: Beautiful, responsive interface for real-time predictions
- 🔌 **REST API**: Production-ready endpoints for integration
- 📱 **Mobile-Friendly**: Responsive design for all devices

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python notebooks/train_and_compare.py
```

### 3. Launch Web Application
```bash
python app/main.py
```

### 4. Access the App
Open your browser to: **http://localhost:8000**

## 📊 Results

### Model Performance
| Model | Accuracy | Type |
|-------|----------|------|
| Logistic Regression | 80.0% | Classical ML |
| SVM | 80.0% | Classical ML |
| Naive Bayes | 80.0% | Classical ML |
| DistilBERT | 80.0% | Transformer |

**Note**: With larger datasets (1000+ reviews), transformers typically achieve 5-15% higher accuracy due to contextual understanding.

## 🏗️ Project Structure

```
smartwatch-sentiment-analyzer/
├── 📁 app/
│   ├── main.py                     # FastAPI web application
│   └── templates/
│       ├── index.html              # Classic web interface
│       ├── professional.html       # Professional web interface
│       └── advanced.html           # Advanced web interface
├── 📁 src/
│   ├── data_collection.py          # Data loading & preparation
│   ├── preprocessing.py            # Text cleaning & tokenization
│   ├── classical_ml.py             # Classical ML models
│   ├── transformer_model.py        # Transformer models
│   ├── advanced_models.py          # Advanced ML models
│   ├── web_scraper.py              # Web scraping utilities
│   └── evaluation.py               # Model comparison & visualization
├── 📁 data/
│   ├── amazon_reviews.csv          # Dataset (sample reviews)
│   └── README.md                   # Data documentation
├── 📁 notebooks/
│   ├── train_and_compare.py        # Complete training pipeline
│   └── train_realworld.py          # Real-world training script
├── 📁 models/                      # Saved models (created after training)
├── 📁 results/                     # Comparison charts (created after training)
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
├── USAGE.md                        # Detailed usage guide
├── DEMO.md                         # Demo instructions
├── QUICKSTART.md                   # Quick start guide
├── test_api.py                     # API testing script
└── test_predictions.py             # Prediction testing script
```

## 🎨 Web Interface

The web application features:
- 🎯 Real-time sentiment analysis
- 🔄 Model selection (Classical ML vs Transformer)
- 📊 Confidence scores
- 🎨 Color-coded results (Green/Red/Yellow)
- 📱 Responsive design
- ⚡ Fast inference (< 1 second)

## 🔌 API Endpoints

### Predict Sentiment
```bash
POST /predict
Form Data:
  - text: "Review text here"
  - model_type: "classical" or "transformer"
```

### Health Check
```bash
GET /api/health
```

### List Models
```bash
GET /api/models
```

## 💻 Usage Examples

### Python API
```python
import requests

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

### Command Line
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "text=This smartwatch is fantastic!" \
  -F "model_type=transformer"
```

### Python Script
```python
from src.classical_ml import ClassicalMLPipeline

model = ClassicalMLPipeline(model_type='logistic')
model.load_model('models/')

prediction = model.predict(["Great smartwatch!"])
print(f"Sentiment: {prediction[0]}")  # 0=Negative, 1=Neutral, 2=Positive
```

## 🧪 Testing

Run the test script:
```bash
python test_predictions.py
```

## 📚 Documentation

- **[USAGE.md](USAGE.md)** - Detailed usage instructions
- **[DEMO.md](DEMO.md)** - Demo guide with examples
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide

## 🔬 Technical Details

### Classical ML
- **Vectorization**: TF-IDF (5000 features, bigrams)
- **Models**: Logistic Regression, SVM, Naive Bayes
- **Training Time**: < 1 minute
- **Inference**: ~50ms per prediction

### Transformers
- **Model**: DistilBERT (66M parameters)
- **Fine-tuning**: 3 epochs on smartwatch reviews
- **Training Time**: ~30 seconds (CPU)
- **Inference**: ~200ms per prediction

### Preprocessing
- Text normalization
- URL/HTML removal
- Special character handling
- NLTK tokenization
- Stopword removal (optional)

## 🎓 Key Insights

### Why Transformers Excel
1. **Contextual Understanding**: Captures word meaning based on surrounding context
2. **Pre-trained Knowledge**: Leverages understanding from billions of words
3. **Semantic Relationships**: Better handles negations, sarcasm, and nuance
4. **Transfer Learning**: Benefits from large-scale pre-training

### Model Comparison
- **Classical ML**: Fast, interpretable, good for simple patterns
- **Transformers**: Superior accuracy, handles complexity, requires more compute

## 🚀 Future Enhancements

- [ ] Larger dataset (10K+ reviews)
- [ ] Additional transformer models (BERT, RoBERTa)
- [ ] Aspect-based sentiment analysis
- [ ] Multi-language support
- [ ] Docker containerization
- [ ] Cloud deployment
- [ ] Model monitoring dashboard

## 📦 Dependencies

- FastAPI - Web framework
- Transformers - Hugging Face transformers
- PyTorch - Deep learning framework
- Scikit-learn - Classical ML
- NLTK - Text processing
- Pandas - Data manipulation
- Matplotlib/Seaborn - Visualization

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add more models
- Improve preprocessing
- Enhance the UI
- Add features
- Fix bugs

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Hugging Face for Transformers library
- Scikit-learn for ML tools
- FastAPI for web framework
- NLTK for NLP utilities

## 📞 Support

For questions or issues:
1. Check [USAGE.md](USAGE.md) for detailed instructions
2. Review [DEMO.md](DEMO.md) for examples
3. Read [QUICKSTART.md](QUICKSTART.md) for quick setup

---

**Status**: ✅ Complete and Functional  
**Last Updated**: December 2, 2025  
**Version**: 1.0.0

Made with ❤️ for demonstrating ML model comparison
