"""
Transformer-based models for sentiment classification
"""
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os


class TransformerPipeline:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=3):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None
        
    def prepare_dataset(self, df, text_column='cleaned_text', label_column='sentiment'):
        """Convert dataframe to HuggingFace dataset"""
        dataset = Dataset.from_pandas(df[[text_column, label_column]])
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                padding='max_length',
                truncation=True,
                max_length=128
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.rename_column(label_column, 'labels')
        tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return tokenized_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_df, val_df, output_dir='models/transformer'):
        """Train transformer model"""
        print(f"Training {self.model_name}...")
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_df)
        val_dataset = self.prepare_dataset(val_df)
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            report_to='none',  # Disable wandb
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
        
        return results
    
    def predict(self, texts):
        """Make predictions on new texts"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
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
            max_length=128,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        return probs.numpy()
    
    def save_model(self, path='models/transformer'):
        """Save model and tokenizer"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='models/transformer'):
        """Load model and tokenizer"""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    from data_collection import load_and_prepare_data
    from preprocessing import preprocess_dataframe, split_data
    
    # Load and prepare data
    df = load_and_prepare_data("data/amazon_reviews.csv")
    df = preprocess_dataframe(df)
    
    # Split data
    train_df, val_df, test_df = split_data(df)
    
    # Train DistilBERT
    pipeline = TransformerPipeline(model_name='distilbert-base-uncased')
    pipeline.train(train_df, val_df)
    results = pipeline.evaluate(test_df)
    pipeline.save_model()
