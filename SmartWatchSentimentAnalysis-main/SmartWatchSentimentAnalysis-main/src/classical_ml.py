"""
Classical ML models for sentiment classification
"""
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import pickle
import os


class ClassicalMLPipeline:
    def __init__(self, model_type='logistic', vectorizer_type='tfidf'):
        self.model_type = model_type
        self.vectorizer_type = vectorizer_type
        self.vectorizer = None
        self.model = None
        
    def _create_vectorizer(self):
        """Create text vectorizer"""
        if self.vectorizer_type == 'tfidf':
            return TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        else:
            return CountVectorizer(max_features=5000, ngram_range=(1, 2))
    
    def _create_model(self):
        """Create ML model"""
        models = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'svm': LinearSVC(random_state=42, max_iter=2000),
            'naive_bayes': MultinomialNB(),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        return models.get(self.model_type, models['logistic'])
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"Training {self.model_type} with {self.vectorizer_type}...")
        
        # Create and fit vectorizer
        self.vectorizer = self._create_vectorizer()
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train_vec, y_train)
        
        print("Training completed!")
        
    def predict(self, X):
        """Make predictions"""
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_vec = self.vectorizer.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_vec)
        else:
            return self.model.decision_function(X_vec)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, 
                                       labels=[0, 1, 2],
                                       target_names=['Negative', 'Neutral', 'Positive'],
                                       zero_division=0)
        
        print(f"\n{self.model_type.upper()} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'report': report
        }
    
    def save_model(self, path='models/'):
        """Save model and vectorizer"""
        os.makedirs(path, exist_ok=True)
        
        model_path = os.path.join(path, f'{self.model_type}_model.pkl')
        vectorizer_path = os.path.join(path, f'{self.model_type}_vectorizer.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, path='models/'):
        """Load model and vectorizer"""
        model_path = os.path.join(path, f'{self.model_type}_model.pkl')
        vectorizer_path = os.path.join(path, f'{self.model_type}_vectorizer.pkl')
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        print(f"Model loaded from {model_path}")


def train_all_classical_models(train_df, test_df):
    """Train and evaluate all classical ML models"""
    results = {}
    
    X_train = train_df['cleaned_text']
    y_train = train_df['sentiment']
    X_test = test_df['cleaned_text']
    y_test = test_df['sentiment']
    
    model_types = ['logistic', 'svm', 'naive_bayes', 'random_forest']
    
    for model_type in model_types:
        pipeline = ClassicalMLPipeline(model_type=model_type)
        pipeline.train(X_train, y_train)
        results[model_type] = pipeline.evaluate(X_test, y_test)
        pipeline.save_model()
    
    return results


if __name__ == "__main__":
    from data_collection import load_and_prepare_data
    from preprocessing import preprocess_dataframe, split_data
    
    # Load and prepare data
    df = load_and_prepare_data("data/amazon_reviews.csv")
    df = preprocess_dataframe(df)
    
    # Split data
    train_df, val_df, test_df = split_data(df)
    
    # Train models
    results = train_all_classical_models(train_df, test_df)
