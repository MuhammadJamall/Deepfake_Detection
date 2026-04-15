"""
Model evaluation and metrics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    accuracy_score
)
from tqdm import tqdm
from pathlib import Path


class Evaluator:
    """
    Evaluator class for model evaluation
    """
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @staticmethod
    def _ensure_parent_dir(save_path):
        """Ensure parent directory exists before saving plots."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def evaluate(self, test_loader, class_names=None):
        """
        Full evaluation on test set
        
        Args:
            test_loader: Test dataloader
            class_names: List of class names
            
        Returns:
            Dictionary with metrics and predictions
        """
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Evaluating")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                predicted = outputs.argmax(dim=1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        if all_labels.size == 0:
            raise ValueError("test_loader produced no samples for evaluation")

        labels = None
        if class_names:
            labels = list(range(len(class_names)))
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions, labels=labels)
        
        # Classification report
        class_report = classification_report(
            all_labels, all_predictions,
            labels=labels,
            target_names=class_names if class_names else None,
            output_dict=True,
            zero_division=0
        )
        
        # ROC-AUC (for binary classification)
        if len(np.unique(all_labels)) == 2 and all_probs.ndim == 2 and all_probs.shape[1] >= 2:
            fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = None, None, None
        
        results = {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probs,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'class_names': class_names
        }
        
        return results
    
    def plot_confusion_matrix(self, results, save_path='results/plots/confusion_matrix.png'):
        """Plot confusion matrix"""
        save_path = self._ensure_parent_dir(save_path)
        cm = results['confusion_matrix']
        class_names = results['class_names'] or [f"Class {i}" for i in range(len(cm))]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix - Test Set')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {save_path}")
        plt.close()
    
    def plot_roc_curve(self, results, save_path='results/plots/roc_curve.png'):
        """Plot ROC curve"""
        save_path = self._ensure_parent_dir(save_path)
        if results['fpr'] is None:
            print("⚠ ROC curve only available for binary classification")
            return
        
        plt.figure(figsize=(10, 8))
        plt.plot(results['fpr'], results['tpr'], color='darkorange', lw=2,
                label=f"ROC curve (AUC = {results['roc_auc']:.4f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Test Set')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved: {save_path}")
        plt.close()
    
    def plot_metrics_per_class(self, results, save_path='results/plots/metrics_per_class.png'):
        """Plot precision, recall, F1 per class"""
        save_path = self._ensure_parent_dir(save_path)
        report = results['classification_report']
        class_names = results['class_names'] or [f"Class {i}" for i in range(len(results['confusion_matrix']))]
        
        metrics_dict = {}
        for class_name in class_names:
            if class_name in report:
                metrics_dict[class_name] = {
                    'Precision': report[class_name]['precision'],
                    'Recall': report[class_name]['recall'],
                    'F1-Score': report[class_name]['f1-score']
                }

        if not metrics_dict:
            print("⚠ No class metrics available to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(metrics_dict))
        width = 0.25
        
        precisions = [metrics_dict[c]['Precision'] for c in metrics_dict]
        recalls = [metrics_dict[c]['Recall'] for c in metrics_dict]
        f1_scores = [metrics_dict[c]['F1-Score'] for c in metrics_dict]
        
        ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_ylabel('Score')
        ax.set_title('Metrics per Class - Test Set')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_dict.keys())
        ax.legend()
        ax.set_ylim(0.0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Metrics per class plot saved: {save_path}")
        plt.close()
    
    def print_results(self, results):
        """Print evaluation results"""
        print("\n" + "="*70)
        print("TEST SET EVALUATION RESULTS")
        print("="*70)
        print(f"Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        
        if results['roc_auc'] is not None:
            print(f"ROC-AUC Score: {results['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])
        
        print("\nClassification Report:")
        report = results['classification_report']
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        
        class_names = results['class_names'] or [f"Class {i}" for i in range(len(results['confusion_matrix']))]
        for class_name in class_names:
            if class_name in report:
                print(f"{class_name:<20} {report[class_name]['precision']:<12.4f} "
                      f"{report[class_name]['recall']:<12.4f} {report[class_name]['f1-score']:<12.4f} "
                      f"{int(report[class_name]['support']):<10}")
        
        print("="*70 + "\n")