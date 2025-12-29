"""
Complete training and comparison pipeline
Run this script to train all models and generate comparison
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection import load_and_prepare_data
from src.preprocessing import preprocess_dataframe, split_data
from src.classical_ml import ClassicalMLPipeline
from src.transformer_model import TransformerPipeline
from src.evaluation import compare_models, plot_model_comparison, generate_comparison_report
import pandas as pd


def main():
    print("="*80)
    print("SMARTWATCH SENTIMENT ANALYZER - TRAINING PIPELINE")
    print("="*80)
    
    # Step 1: Load data
    print("\n📂 Step 1: Loading data...")
    df = load_and_prepare_data("data/amazon_reviews.csv")
    
    if df.empty:
        print("⚠️ No data found. Please add amazon_reviews.csv to data/ folder")
        print("Creating sample data for demonstration...")
        
        # Create sample data
        sample_data = {
            'review_text': [
                'This smartwatch is amazing! Best purchase ever.',
                'Battery life is terrible. Very disappointed.',
                'It works okay, nothing special.',
                'Love the fitness tracking features!',
                'Waste of money, stopped working after a week.',
                'Decent smartwatch for the price.',
            ],
            'rating': [5, 1, 3, 5, 1, 3],
            'product_name': ['Apple Watch'] * 6
        }
        df = pd.DataFrame(sample_data)
        from src.data_collection import create_sentiment_labels
        df = create_sentiment_labels(df)
    
    print(f"✅ Loaded {len(df)} reviews")
    
    # Step 2: Preprocess
    print("\n🔧 Step 2: Preprocessing data...")
    df = preprocess_dataframe(df)
    print(f"✅ Preprocessed {len(df)} reviews")
    
    # Step 3: Split data
    print("\n✂️ Step 3: Splitting data...")
    train_df, val_df, test_df = split_data(df)
    print(f"✅ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Step 4: Train Classical ML models
    print("\n🤖 Step 4: Training Classical ML models...")
    classical_results = {}
    
    model_types = ['logistic', 'svm', 'naive_bayes']
    
    for model_type in model_types:
        print(f"\nTraining {model_type}...")
        pipeline = ClassicalMLPipeline(model_type=model_type)
        pipeline.train(train_df['cleaned_text'], train_df['sentiment'])
        results = pipeline.evaluate(test_df['cleaned_text'], test_df['sentiment'])
        classical_results[model_type] = results
        pipeline.save_model()
    
    # Step 5: Train Transformer model
    print("\n🚀 Step 5: Training Transformer model...")
    transformer_results = {}
    
    print("\nTraining DistilBERT...")
    transformer_pipeline = TransformerPipeline(model_name='distilbert-base-uncased')
    transformer_pipeline.train(train_df, val_df)
    results = transformer_pipeline.evaluate(test_df)
    transformer_results['distilbert'] = results
    transformer_pipeline.save_model()
    
    # Step 6: Generate comparison
    print("\n📊 Step 6: Generating comparison report...")
    generate_comparison_report(classical_results, transformer_results)
    
    # Create comparison dataframe
    comparison_data = {}
    for name, res in classical_results.items():
        comparison_data[name] = {'accuracy': res['accuracy']}
    for name, res in transformer_results.items():
        comparison_data[name] = {'accuracy': res['eval_accuracy']}
    
    comparison_df = pd.DataFrame(comparison_data).T
    plot_model_comparison(comparison_df)
    
    print("\n✅ Training and comparison complete!")
    print("📁 Models saved to models/ directory")
    print("📊 Comparison plot saved to results/model_comparison.png")
    print("\n🌐 Run 'python app/main.py' to start the web application")


if __name__ == "__main__":
    main()
