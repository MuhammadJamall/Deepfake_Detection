"""
Model comparison and evaluation utilities
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from pathlib import Path
import json


class ModelComparator:
    """
    Compare multiple trained models
    """
    
    def __init__(self, results_dict):
        """
        Args:
            results_dict: Dictionary with model names as keys and results as values
                         {
                             'efficientnet_b4': {results from Evaluator},
                             'xception': {results from Evaluator},
                             ...
                         }
        """
        self.results_dict = results_dict
        self.comparison_metrics = {}
    
    def compute_metrics(self):
        """
        Compute all comparison metrics for each model
        """
        for model_name, results in self.results_dict.items():
            predictions = results['predictions']
            labels = results['labels']
            probabilities = results['probabilities']
            
            # Calculate metrics
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(labels, predictions, average='weighted', zero_division=0)
            f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
            
            # ROC-AUC for binary classification
            if len(np.unique(labels)) == 2:
                roc_auc = roc_auc_score(labels, probabilities[:, 1])
            else:
                roc_auc = None
            
            self.comparison_metrics[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            }
        
        return self.comparison_metrics
    
    def get_comparison_dataframe(self):
        """
        Get comparison metrics as pandas DataFrame
        
        Returns:
            DataFrame with models as rows and metrics as columns
        """
        if not self.comparison_metrics:
            self.compute_metrics()
        
        df = pd.DataFrame(self.comparison_metrics).T
        df = df.round(4)
        
        return df
    
    def print_comparison_table(self):
        """
        Print formatted comparison table
        """
        if not self.comparison_metrics:
            self.compute_metrics()
        
        print("\n" + "="*80)
        print("MODEL COMPARISON - TEST SET METRICS")
        print("="*80)
        
        df = self.get_comparison_dataframe()
        print(df.to_string())
        
        print("\n" + "="*80)
        print("BEST MODELS BY METRIC:")
        print("="*80)
        
        for metric in df.columns:
            best_model = df[metric].idxmax()
            best_score = df[metric].max()
            print(f"  {metric.upper():<12}: {best_model:<20} ({best_score:.4f})")
        
        print("="*80 + "\n")
    
    def plot_comparison_bars(self, save_path='results/plots/model_comparison_bars.png'):
        """
        Plot comparison metrics as bar chart
        """
        if not self.comparison_metrics:
            self.compute_metrics()
        
        df = self.get_comparison_dataframe()
        
        # Remove ROC-AUC if it has NaN values
        if df['roc_auc'].isna().any():
            df = df.drop('roc_auc', axis=1)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(df.columns))
        width = 0.25
        
        for i, model_name in enumerate(df.index):
            offset = (i - len(df.index)/2 + 0.5) * width
            ax.bar(x + offset, df.loc[model_name], width, label=model_name, alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Comparison - Test Set Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df.columns)
        ax.legend()
        ax.set_ylim([0.7, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison bar plot saved: {save_path}")
        plt.close()
    
    def plot_comparison_radar(self, save_path='results/plots/model_comparison_radar.png'):
        """
        Plot comparison metrics as radar chart
        """
        if not self.comparison_metrics:
            self.compute_metrics()
        
        df = self.get_comparison_dataframe()
        
        # Remove ROC-AUC if it has NaN values
        if df['roc_auc'].isna().any():
            df = df.drop('roc_auc', axis=1)
        
        categories = list(df.columns)
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
        
        for idx, (model_name, color) in enumerate(zip(df.index, colors)):
            values = df.loc[model_name].tolist()
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0.7, 1.0)
        ax.set_title('Model Comparison - Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison radar plot saved: {save_path}")
        plt.close()
    
    def plot_per_class_comparison(self, class_names=None, save_path='results/plots/per_class_comparison.png'):
        """
        Plot per-class metrics comparison
        """
        if not class_names:
            class_names = [f"Class {i}" for i in range(len(self.results_dict[list(self.results_dict.keys())[0]]['confusion_matrix']))]
        
        fig, axes = plt.subplots(1, len(class_names), figsize=(15, 4))
        
        if len(class_names) == 1:
            axes = [axes]
        
        for class_idx, class_name in enumerate(class_names):
            metrics_per_model = {}
            
            for model_name, results in self.results_dict.items():
                class_report = results['classification_report']
                
                if class_name in class_report:
                    metrics_per_model[model_name] = {
                        'Precision': class_report[class_name]['precision'],
                        'Recall': class_report[class_name]['recall'],
                        'F1-Score': class_report[class_name]['f1-score']
                    }
            
            df_class = pd.DataFrame(metrics_per_model).T
            
            x = np.arange(len(df_class.columns))
            width = 0.35
            
            for i, model_name in enumerate(df_class.index):
                offset = (i - len(df_class.index)/2 + 0.5) * width
                axes[class_idx].bar(x + offset, df_class.loc[model_name], width, label=model_name, alpha=0.8)
            
            axes[class_idx].set_title(f'{class_name}', fontweight='bold')
            axes[class_idx].set_xticks(x)
            axes[class_idx].set_xticklabels(df_class.columns, rotation=45)
            axes[class_idx].set_ylim([0, 1.1])
            axes[class_idx].grid(axis='y', alpha=0.3)
            
            if class_idx == 0:
                axes[class_idx].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Per-class comparison plot saved: {save_path}")
        plt.close()
    
    def export_to_json(self, save_path='results/metrics/model_comparison.json'):
        """
        Export comparison metrics to JSON
        """
        if not self.comparison_metrics:
            self.compute_metrics()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            model_name: {k: float(v) if v is not None else None for k, v in metrics.items()}
            for model_name, metrics in self.comparison_metrics.items()
        }
        
        with open(save_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Comparison metrics exported to: {save_path}")
    
    def get_winner(self, metric='f1'):
        """
        Get the best model for a specific metric
        
        Args:
            metric (str): Metric to compare ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')
            
        Returns:
            Tuple: (model_name, score)
        """
        if not self.comparison_metrics:
            self.compute_metrics()
        
        scores = {name: metrics[metric] for name, metrics in self.comparison_metrics.items()}
        best_model = max(scores, key=scores.get)
        best_score = scores[best_model]
        
        return best_model, best_score


def create_comparison_summary(results_dict, class_names=None):
    """
    Create a comprehensive comparison summary
    
    Args:
        results_dict: Dictionary with model results
        class_names: List of class names
        
    Returns:
        ModelComparator instance
    """
    comparator = ModelComparator(results_dict)
    comparator.compute_metrics()
    
    return comparator