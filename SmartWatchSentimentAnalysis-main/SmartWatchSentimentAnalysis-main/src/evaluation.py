"""
Model comparison and evaluation visualization
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def compare_models(results_dict):
    """
    Compare multiple models and create visualization
    results_dict: {'model_name': {'accuracy': 0.85, 'f1': 0.84, ...}}
    """
    comparison_df = pd.DataFrame(results_dict).T
    
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(comparison_df.to_string())
    print("="*60)
    
    return comparison_df


def plot_model_comparison(comparison_df, save_path='results/model_comparison.png'):
    """Create bar plot comparing model accuracies"""
    import os
    os.makedirs('results', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = comparison_df.index
    accuracies = comparison_df['accuracy']
    
    colors = ['#3498db' if 'transformer' not in model.lower() else '#e74c3c' 
              for model in models]
    
    bars = ax.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Classical ML vs Transformer Models - Accuracy Comparison', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


def generate_comparison_report(classical_results, transformer_results):
    """
    Generate comprehensive comparison report
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON REPORT")
    print("="*80)
    
    print("\n📊 CLASSICAL ML MODELS:")
    print("-" * 80)
    for model_name, results in classical_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
    
    print("\n\n🤖 TRANSFORMER MODELS:")
    print("-" * 80)
    for model_name, results in transformer_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {results.get('eval_accuracy', results.get('accuracy')):.4f}")
        print(f"  F1 Score: {results.get('eval_f1', results.get('f1', 'N/A')):.4f}")
    
    # Calculate improvement
    best_classical = max([r['accuracy'] for r in classical_results.values()])
    best_transformer = max([r.get('eval_accuracy', r.get('accuracy')) 
                           for r in transformer_results.values()])
    
    improvement = ((best_transformer - best_classical) / best_classical) * 100
    
    print("\n\n📈 KEY INSIGHTS:")
    print("-" * 80)
    print(f"Best Classical ML Accuracy: {best_classical:.4f}")
    print(f"Best Transformer Accuracy: {best_transformer:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    print("\n✅ Transformer models show superior performance due to:")
    print("   • Contextual understanding of text")
    print("   • Pre-trained language representations")
    print("   • Better handling of semantic relationships")
    print("="*80)


if __name__ == "__main__":
    # Example usage
    classical_results = {
        'logistic': {'accuracy': 0.82},
        'svm': {'accuracy': 0.81},
        'naive_bayes': {'accuracy': 0.78},
        'random_forest': {'accuracy': 0.80}
    }
    
    transformer_results = {
        'distilbert': {'eval_accuracy': 0.89, 'eval_f1': 0.88},
        'bert': {'eval_accuracy': 0.91, 'eval_f1': 0.90}
    }
    
    generate_comparison_report(classical_results, transformer_results)
