# ⚡ Quick Start Guide

Get the Smartwatch Sentiment Analyzer running in 3 minutes!

## ✅ Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 2GB free disk space
- Internet connection (for first-time model download)

## 🚀 Installation & Setup

### Step 1: Install Dependencies (1 minute)
```bash
pip install -r requirements.txt
```

### Step 2: Train Models (Already Done! ✅)
The models are already trained and saved in the `models/` directory.

If you want to retrain:
```bash
python notebooks/train_and_compare.py
```

### Step 3: Start the Web App (10 seconds)
```bash
python app/main.py
```

### Step 4: Open Your Browser
Navigate to: **http://localhost:8000**

## 🎯 First Test

1. **Enter a review** in the text box:
   ```
   This smartwatch is amazing! Great battery life and accurate tracking.
   ```

2. **Select a model**: Choose "Transformer (DistilBERT)"

3. **Click "Analyze Sentiment"**

4. **See the result**: Should show "Positive 😊" with confidence score

## 📊 Try Different Reviews

### Positive Example
```
Love this smartwatch! The fitness features are excellent and very accurate.
```

### Negative Example
```
Terrible product. Stopped working after one week. Very disappointed.
```

### Neutral Example
```
It's okay. Does the basic functions but nothing special.
```

## 🔄 Compare Models

1. Analyze a review with "Classical ML"
2. Analyze the same review with "Transformer"
3. Compare the confidence scores!

## 🧪 Test via Command Line

```bash
python test_predictions.py
```

This will show predictions for 5 sample reviews.

## 🔌 Test the API

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "text=This smartwatch is fantastic!" \
  -F "model_type=transformer"
```

## 📈 View Comparison Chart

Open: `results/model_comparison.png`

## 🎓 What's Next?

- Read [USAGE.md](USAGE.md) for detailed instructions
- Check [DEMO.md](DEMO.md) for more examples
- Review [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for technical details

## ❓ Troubleshooting

### Port Already in Use
```bash
# Use a different port
uvicorn app.main:app --port 8001
```

### Models Not Found
```bash
# Retrain models
python notebooks/train_and_compare.py
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 🎉 You're Ready!

The Smartwatch Sentiment Analyzer is now running and ready to analyze reviews!

---

**Need Help?** Check the documentation files or review the code comments.
