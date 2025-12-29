"""
Advanced models with better performance and real-world features
"""
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
import pickle
import os


class EnsembleClassicalModel:
    """
    Ensemble of classical ML models for better performance
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        self.model = None
    
    def train(self, X_train, y_train):
        """Train ensemble model"""
        print("Training Ensemble Classical Model...")
        
        # Vectorize
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        # Create ensemble
        lr = LogisticRegression(max_iter=1000, C=2.0, random_state=42)
        svm = LinearSVC(C=1.0, random_state=42, max_iter=2000)
        nb = MultinomialNB(alpha=0.1)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        self.model = VotingClassifier(
            estimators=[
                ('lr', lr),
                ('svm', svm),
                ('nb', nb),
                ('gb', gb)
            ],
            voting='hard'
        )
        
        self.model.fit(X_train_vec, y_train)
        print("Training completed!")
    
    def predict(self, X):
        """Make predictions"""
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_vec = self.vectorizer.transform(X)
        # Get predictions from each classifier
        predictions = []
        for name, clf in self.model.named_estimators_.items():
            if hasattr(clf, 'predict_proba'):
                predictions.append(clf.predict_proba(X_vec))
            else:
                # For SVM, use decision function
                decision = clf.decision_function(X_vec)
                # Convert to probabilities
                proba = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
                predictions.append(proba)
        
        # Average probabilities
        return np.mean(predictions, axis=0)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(
            y_test, y_pred,
            labels=[0, 1, 2],
            target_names=['Negative', 'Neutral', 'Positive'],
            zero_division=0
        )
        
        print(f"\nENSEMBLE MODEL Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(report)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'predictions': y_pred,
            'report': report
        }
    
    def save_model(self, path='models/ensemble/'):
        """Save model"""
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(os.path.join(path, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"Ensemble model saved to {path}")
    
    def load_model(self, path='models/ensemble/'):
        """Load model"""
        with open(os.path.join(path, 'model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        
        with open(os.path.join(path, 'vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        print(f"Ensemble model loaded from {path}")


class AdvancedTransformerModel:
    """
    Advanced transformer model with better training strategies
    """
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=3):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None
    
    def prepare_dataset(self, df, text_column='cleaned_text', label_column='sentiment'):
        """Prepare dataset with better tokenization"""
        from datasets import Dataset
        
        dataset = Dataset.from_pandas(df[[text_column, label_column]])
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                padding='max_length',
                truncation=True,
                max_length=256,  # Increased for better context
                return_tensors='pt'
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.rename_column(label_column, 'labels')
        tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return tokenized_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute comprehensive metrics"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_df, val_df, output_dir='models/advanced_transformer', epochs=5):
        """Train with better hyperparameters"""
        print(f"Training Advanced {self.model_name}...")
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_df)
        val_dataset = self.prepare_dataset(val_df)
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        
        # Better training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            report_to='none',
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        self.trainer.train()
        
        print("Training completed!")
    
    def evaluate(self, test_df):
        """Evaluate model"""
        test_dataset = self.prepare_dataset(test_df)
        results = self.trainer.evaluate(test_dataset)
        
        print(f"\n{self.model_name.upper()} Results:")
        print(f"Accuracy: {results['eval_accuracy']:.4f}")
        print(f"F1 Score: {results['eval_f1']:.4f}")
        print(f"Precision: {results['eval_precision']:.4f}")
        print(f"Recall: {results['eval_recall']:.4f}")
        
        return results
    
    def predict(self, texts):
        """Make predictions"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        return predictions.numpy()
    
    def predict_proba(self, texts):
        """Get prediction probabilities"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        return probs.numpy()
    
    def save_model(self, path='models/advanced_transformer'):
        """Save model"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='models/advanced_transformer'):
        """Load model"""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    print("Advanced models module loaded successfully!")
