# Usage Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your Amazon smartwatch reviews CSV file in the `data/` directory:
- File name: `amazon_reviews.csv`
- Required columns: `review_text`, `rating`, `product_name`

### 3. Train Models

Run the complete training pipeline:

```bash
python notebooks/train_and_compare.py
```

This will:
- Load and preprocess the data
- Train classical ML models (Logistic Regression, SVM, Naive Bayes)
- Train transformer model (DistilBERT)
- Generate comparison report and visualizations
- Save all models

### 4. Launch Web Application

```bash
python app/main.py
```

Then open your browser to: `http://localhost:8000`

## Individual Components

### Data Collection

```python
from src.data_collection import load_and_prepare_data

df = load_and_prepare_data("data/amazon_reviews.csv")
print(f"Loaded {len(df)} reviews")
```

### Preprocessing

```python
from src.preprocessing import preprocess_dataframe, split_data

df = preprocess_dataframe(df)
train_df, val_df, test_df = split_data(df)
```

### Train Classical ML Model

```python
from src.classical_ml import ClassicalMLPipeline

# Train Logistic Regression
pipeline = ClassicalMLPipeline(model_type='logistic')
pipeline.train(train_df['cleaned_text'], train_df['sentiment'])
results = pipeline.evaluate(test_df['cleaned_text'], test_df['sentiment'])
pipeline.save_model()
```

### Train Transformer Model

```python
from src.transformer_model import TransformerPipeline

# Train DistilBERT
pipeline = TransformerPipeline(model_name='distilbert-base-uncased')
pipeline.train(train_df, val_df)
results = pipeline.evaluate(test_df)
pipeline.save_model()
```

### Make Predictions

```python
# Load saved model
pipeline = ClassicalMLPipeline(model_type='logistic')
pipeline.load_model('models/')

# Predict
text = "This smartwatch is amazing!"
prediction = pipeline.predict([text])
print(f"Sentiment: {prediction[0]}")  # 0=Negative, 1=Neutral, 2=Positive
```

## API Endpoints

### Web Interface
- `GET /` - Home page with sentiment analyzer

### API Endpoints
- `POST /predict` - Make sentiment prediction
  - Form data: `text` (review text), `model_type` (classical/transformer)
  
- `GET /api/health` - Health check
- `GET /api/models` - List available models

### Example API Usage

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "text=This smartwatch is fantastic!" \
  -F "model_type=transformer"
```

## Model Comparison

The project compares:

**Classical ML Models:**
- Logistic Regression with TF-IDF
- Support Vector Machine (SVM)
- Naive Bayes
- Random Forest

**Transformer Models:**
- DistilBERT (default)
- Can be extended to BERT, RoBERTa, etc.

Expected improvement: Transformers typically achieve 5-15% higher accuracy due to contextual understanding.

## Customization

### Use Different Transformer Model

```python
pipeline = TransformerPipeline(model_name='bert-base-uncased')
# or
pipeline = TransformerPipeline(model_name='roberta-base')
```

### Adjust Training Parameters

Edit `src/transformer_model.py`:
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size
- `learning_rate`: Learning rate

### Change Sentiment Labels

Edit `src/data_collection.py` in `create_sentiment_labels()` function to adjust rating-to-sentiment mapping.

## Troubleshooting

### Out of Memory Error
- Reduce batch size in transformer training
- Use smaller model (distilbert instead of bert)
- Reduce max_length in tokenization

### Model Not Loading
- Ensure models are trained and saved first
- Check file paths in load_model() calls

### Low Accuracy
- Ensure sufficient training data (>1000 samples recommended)
- Check data quality and balance
- Increase training epochs
- Try different models
