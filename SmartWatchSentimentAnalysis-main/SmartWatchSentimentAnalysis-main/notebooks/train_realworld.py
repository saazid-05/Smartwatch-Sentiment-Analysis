"""
Real-world training pipeline with data collection and advanced models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.web_scraper import collect_and_prepare_data
from src.data_collection import create_sentiment_labels
from src.preprocessing import preprocess_dataframe, split_data
from src.classical_ml import ClassicalMLPipeline
from src.advanced_models import EnsembleClassicalModel, AdvancedTransformerModel
from src.transformer_model import TransformerPipeline
from src.evaluation import compare_models, plot_model_comparison, generate_comparison_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    print("="*80)
    print("REAL-WORLD SMARTWATCH SENTIMENT ANALYZER - TRAINING PIPELINE")
    print("="*80)
    
    # Step 1: Collect real-world data
    print("\n📂 Step 1: Collecting real-world data...")
    print("Options:")
    print("  1. Use existing dataset (fastest)")
    print("  2. Collect from Hugging Face API (recommended)")
    print("  3. Scrape from Amazon (requires setup)")
    
    choice = input("\nSelect option (1/2/3) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        df = pd.read_csv("data/amazon_reviews.csv")
        print(f"✅ Loaded {len(df)} reviews from existing dataset")
    elif choice == "2":
        try:
            df = collect_and_prepare_data(source='api', max_reviews=1000)
        except:
            print("⚠️ API collection failed. Using existing dataset.")
            df = pd.read_csv("data/amazon_reviews.csv")
    else:
        print("⚠️ Web scraping requires additional setup. Using existing dataset.")
        df = pd.read_csv("data/amazon_reviews.csv")
    
    # Ensure sentiment labels
    if 'sentiment' not in df.columns:
        df = create_sentiment_labels(df)
    
    print(f"✅ Total reviews: {len(df)}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    
    # Step 2: Preprocess
    print("\n🔧 Step 2: Preprocessing data...")
    df = preprocess_dataframe(df)
    print(f"✅ Preprocessed {len(df)} reviews")
    
    # Step 3: Split data
    print("\n✂️ Step 3: Splitting data...")
    train_df, val_df, test_df = split_data(df, test_size=0.2, val_size=0.1)
    print(f"✅ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Step 4: Train Classical ML models
    print("\n🤖 Step 4: Training Classical ML models...")
    classical_results = {}
    
    # Standard models
    print("\n--- Standard Classical Models ---")
    model_types = ['logistic', 'svm', 'naive_bayes']
    
    for model_type in model_types:
        print(f"\nTraining {model_type}...")
        pipeline = ClassicalMLPipeline(model_type=model_type)
        pipeline.train(train_df['cleaned_text'], train_df['sentiment'])
        results = pipeline.evaluate(test_df['cleaned_text'], test_df['sentiment'])
        classical_results[model_type] = results
        pipeline.save_model()
    
    # Ensemble model
    print("\n--- Ensemble Model (Advanced) ---")
    ensemble_model = EnsembleClassicalModel()
    ensemble_model.train(train_df['cleaned_text'], train_df['sentiment'])
    ensemble_results = ensemble_model.evaluate(test_df['cleaned_text'], test_df['sentiment'])
    classical_results['ensemble'] = ensemble_results
    ensemble_model.save_model()
    
    # Step 5: Train Transformer models
    print("\n🚀 Step 5: Training Transformer models...")
    transformer_results = {}
    
    # Standard DistilBERT
    print("\n--- DistilBERT (Standard) ---")
    distilbert = TransformerPipeline(model_name='distilbert-base-uncased')
    distilbert.train(train_df, val_df)
    distilbert_results = distilbert.evaluate(test_df)
    transformer_results['distilbert'] = distilbert_results
    distilbert.save_model('models/transformer')
    
    # Advanced DistilBERT with better hyperparameters
    print("\n--- DistilBERT (Advanced) ---")
    advanced_distilbert = AdvancedTransformerModel(model_name='distilbert-base-uncased')
    advanced_distilbert.train(train_df, val_df, epochs=5)
    advanced_results = advanced_distilbert.evaluate(test_df)
    transformer_results['distilbert_advanced'] = advanced_results
    advanced_distilbert.save_model('models/advanced_transformer')
    
    # Step 6: Generate comprehensive comparison
    print("\n📊 Step 6: Generating comparison report...")
    generate_comparison_report(classical_results, transformer_results)
    
    # Create detailed comparison
    comparison_data = {}
    
    # Classical models
    for name, res in classical_results.items():
        comparison_data[f'Classical_{name}'] = {
            'accuracy': res['accuracy'],
            'type': 'Classical ML'
        }
    
    # Transformer models
    for name, res in transformer_results.items():
        comparison_data[f'Transformer_{name}'] = {
            'accuracy': res['eval_accuracy'],
            'type': 'Transformer'
        }
    
    # Create visualization
    create_detailed_comparison_plot(comparison_data)
    
    # Step 7: Generate insights
    print("\n💡 Step 7: Generating insights...")
    generate_insights(classical_results, transformer_results)
    
    print("\n✅ Training and comparison complete!")
    print("📁 Models saved to models/ directory")
    print("📊 Visualizations saved to results/ directory")
    print("\n🌐 Run 'python app/main.py' to start the web application")


def create_detailed_comparison_plot(comparison_data):
    """Create detailed comparison visualization"""
    import os
    os.makedirs('results', exist_ok=True)
    
    # Prepare data
    models = list(comparison_data.keys())
    accuracies = [comparison_data[m]['accuracy'] for m in models]
    types = [comparison_data[m]['type'] for m in models]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bar chart
    colors = ['#3498db' if t == 'Classical ML' else '#e74c3c' for t in types]
    bars = ax1.bar(range(len(models)), accuracies, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right', fontsize=9)
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend([plt.Rectangle((0,0),1,1, fc='#3498db', alpha=0.7),
                plt.Rectangle((0,0),1,1, fc='#e74c3c', alpha=0.7)],
               ['Classical ML', 'Transformer'], loc='lower right')
    
    # Plot 2: Box plot by type
    classical_accs = [acc for acc, t in zip(accuracies, types) if t == 'Classical ML']
    transformer_accs = [acc for acc, t in zip(accuracies, types) if t == 'Transformer']
    
    bp = ax2.boxplot([classical_accs, transformer_accs],
                      labels=['Classical ML', 'Transformer'],
                      patch_artist=True,
                      notch=True)
    
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Distribution by Model Type', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add mean lines
    ax2.axhline(y=sum(classical_accs)/len(classical_accs), color='#3498db', 
                linestyle='--', linewidth=2, alpha=0.7, label=f'Classical Mean: {sum(classical_accs)/len(classical_accs):.3f}')
    ax2.axhline(y=sum(transformer_accs)/len(transformer_accs), color='#e74c3c', 
                linestyle='--', linewidth=2, alpha=0.7, label=f'Transformer Mean: {sum(transformer_accs)/len(transformer_accs):.3f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results/detailed_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Detailed comparison plot saved to results/detailed_comparison.png")


def generate_insights(classical_results, transformer_results):
    """Generate insights from model comparison"""
    print("\n" + "="*80)
    print("🔍 KEY INSIGHTS & ANALYSIS")
    print("="*80)
    
    # Calculate statistics
    classical_accs = [r['accuracy'] for r in classical_results.values()]
    transformer_accs = [r.get('eval_accuracy', r.get('accuracy')) for r in transformer_results.values()]
    
    best_classical = max(classical_accs)
    best_transformer = max(transformer_accs)
    avg_classical = sum(classical_accs) / len(classical_accs)
    avg_transformer = sum(transformer_accs) / len(transformer_accs)
    
    improvement = ((best_transformer - best_classical) / best_classical) * 100
    
    print(f"\n📊 Performance Summary:")
    print(f"   Classical ML:")
    print(f"      Best Accuracy: {best_classical:.4f}")
    print(f"      Average Accuracy: {avg_classical:.4f}")
    print(f"   Transformer:")
    print(f"      Best Accuracy: {best_transformer:.4f}")
    print(f"      Average Accuracy: {avg_transformer:.4f}")
    print(f"   Improvement: {improvement:+.2f}%")
    
    print(f"\n💡 Why Transformers Perform Better:")
    print(f"   1. Contextual Understanding: Captures word meaning based on surrounding context")
    print(f"   2. Pre-trained Knowledge: Leverages understanding from billions of words")
    print(f"   3. Attention Mechanism: Focuses on relevant parts of the text")
    print(f"   4. Better Handling of:")
    print(f"      • Negations ('not good' vs 'good')")
    print(f"      • Sarcasm and irony")
    print(f"      • Long-range dependencies")
    print(f"      • Nuanced sentiment")
    
    print(f"\n⚡ Real-World Applications:")
    print(f"   • E-commerce: Product review analysis")
    print(f"   • Customer Service: Sentiment monitoring")
    print(f"   • Market Research: Brand perception")
    print(f"   • Social Media: Opinion mining")
    
    print("="*80)


if __name__ == "__main__":
    main()
