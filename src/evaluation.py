"""
Week 2: Comprehensive Evaluation Pipeline
Binary Classification Metrics: Confusion Matrix, ROC Curve, Classification Report
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score
)
from pathlib import Path
import json
from typing import Any, cast


class BinaryClassificationEvaluator:
    """Evaluate binary classification model on test set"""
    
    def __init__(
        self,
        model,
        device,
        class_names: list[str] | None = None,
        model_name: str = 'Model'
    ):
        self.model = model
        self.device = device
        self.class_names = class_names if class_names is not None else ['REAL', 'FAKE']
        self.model_name = model_name
        self.metrics: dict[str, float | None] = {}

        if len(self.class_names) != 2:
            raise ValueError("BinaryClassificationEvaluator requires exactly 2 class names.")

    def _get_probabilities_and_predictions(self, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert model outputs to binary class probabilities and predictions."""
        if outputs.ndim == 1:
            fake_probs = torch.sigmoid(outputs)
            probs = torch.stack((1.0 - fake_probs, fake_probs), dim=1)
            predicted = (fake_probs >= 0.5).long()
            return probs, predicted

        if outputs.ndim == 2 and outputs.shape[1] == 1:
            fake_probs = torch.sigmoid(outputs.squeeze(1))
            probs = torch.stack((1.0 - fake_probs, fake_probs), dim=1)
            predicted = (fake_probs >= 0.5).long()
            return probs, predicted

        if outputs.ndim == 2 and outputs.shape[1] >= 2:
            probs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probs, dim=1)
            return probs[:, :2], predicted

        raise ValueError(
            f"Unexpected output shape {tuple(outputs.shape)}. "
            "Expected [N], [N,1], or [N,2+] for binary classification."
        )

    def _compute_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, dict[str, Any]]:
        """Return text and dict versions of the classification report."""
        label_indices = list(range(len(self.class_names)))

        report_text = cast(
            str,
            classification_report(
                y_true,
                y_pred,
                labels=label_indices,
                target_names=self.class_names,
                digits=4,
                zero_division=0,
                output_dict=False
            )
        )

        report_dict = cast(
            dict[str, Any],
            classification_report(
                y_true,
                y_pred,
                labels=label_indices,
                target_names=self.class_names,
                digits=4,
                zero_division=0,
                output_dict=True
            )
        )

        return report_text, report_dict
    
    def evaluate(self, test_loader, save_path='results/'):
        """
        Full evaluation on test set
        
        Args:
            test_loader: Test dataloader
            save_path: Path to save results
            
        Returns:
            Dictionary with all metrics
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probs: list[np.ndarray] = []
        
        print("\n" + "="*70)
        print("EVALUATION: Computing Predictions on Test Set")
        print("="*70 + "\n")
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probs, predicted = self._get_probabilities_and_predictions(outputs)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        all_probs_np = np.array(all_probs)

        if all_labels_np.size == 0:
            raise ValueError("Test dataloader is empty. Cannot evaluate on zero samples.")

        unique_labels = np.unique(all_labels_np)
        
        # Compute metrics
        self.metrics['accuracy'] = float(accuracy_score(all_labels_np, all_preds_np))
        self.metrics['precision'] = float(precision_score(all_labels_np, all_preds_np, zero_division=0))
        self.metrics['recall'] = float(recall_score(all_labels_np, all_preds_np, zero_division=0))
        self.metrics['f1'] = float(f1_score(all_labels_np, all_preds_np, zero_division=0))

        if unique_labels.size == 2:
            self.metrics['auc_roc'] = float(roc_auc_score(all_labels_np, all_probs_np[:, 1]))
        else:
            self.metrics['auc_roc'] = None
            print("! AUC-ROC skipped: only one class present in y_true.")
        
        print(f"✓ Test Accuracy: {self.metrics['accuracy']:.4f} ({self.metrics['accuracy']*100:.2f}%)")
        print(f"✓ Precision: {self.metrics['precision']:.4f}")
        print(f"✓ Recall: {self.metrics['recall']:.4f}")
        print(f"✓ F1 Score: {self.metrics['f1']:.4f}")

        if self.metrics['auc_roc'] is not None:
            print(f"✓ AUC-ROC: {self.metrics['auc_roc']:.4f}\n")
        else:
            print("✓ AUC-ROC: N/A (single-class test labels)\n")

        cm = confusion_matrix(all_labels_np, all_preds_np, labels=[0, 1])
        report_text, report_dict = self._compute_classification_report(all_labels_np, all_preds_np)
        
        # Generate plots
        self._plot_confusion_matrix(cm, save_path)
        self._plot_roc_curve(all_labels_np, all_probs_np, save_path)
        self._plot_precision_recall(all_labels_np, all_probs_np, save_path)
        self._print_classification_report(report_text, save_path)
        
        # Save metrics to JSON
        self._save_metrics(self.metrics, save_path)

        evaluation_results = {
            **self.metrics,
            'predictions': all_preds_np.tolist(),
            'labels': all_labels_np.tolist(),
            'probabilities': all_probs_np.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': report_dict
        }
        
        return evaluation_results
    
    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: str):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'}, ax=ax)
        
        ax.set_title(f'Confusion Matrix - {self.model_name} Binary Classification', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        save_file = Path(save_path) / 'plots' / 'confusion_matrix.png'
        save_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_file), dpi=300, bbox_inches='tight')
        print(f"✓ Confusion Matrix saved: {save_file}")
        plt.close(fig)
    
    def _plot_roc_curve(self, y_true, y_probs, save_path):
        """Plot ROC curve"""
        if len(np.unique(y_true)) < 2:
            print("! ROC curve skipped: only one class present in y_true.")
            return

        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2.5, 
               label=f'ROC Curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {self.model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_file = Path(save_path) / 'plots' / 'roc_curve.png'
        save_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_file), dpi=300, bbox_inches='tight')
        print(f"✓ ROC Curve saved: {save_file}")
        plt.close(fig)
    
    def _plot_precision_recall(self, y_true, y_probs, save_path):
        """Plot Precision-Recall curve"""
        if len(np.unique(y_true)) < 2:
            print("! Precision-Recall curve skipped: only one class present in y_true.")
            return

        precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='green', lw=2.5,
               label=f'Precision-Recall Curve (AUC = {pr_auc:.4f})')
        ax.axhline(y=np.mean(y_true), color='gray', linestyle='--', 
                  label='Random Classifier', lw=2)
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve - {self.model_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best", fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_file = Path(save_path) / 'plots' / 'precision_recall_curve.png'
        save_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_file), dpi=300, bbox_inches='tight')
        print(f"✓ Precision-Recall Curve saved: {save_file}")
        plt.close(fig)
    
    def _print_classification_report(self, report_text: str, save_path: str):
        """Print and save classification report"""
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(report_text)
        print("="*70 + "\n")
        
        # Save report to file
        report_file = Path(save_path) / 'metrics' / 'classification_report.txt'
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"✓ Classification Report saved: {report_file}")
    
    def _save_metrics(self, metrics, save_path):
        """Save metrics to JSON"""
        metrics_file = Path(save_path) / 'metrics' / 'metrics.json'
        metrics_file.parent.mkdir(parents=True, exist_ok=True)

        serializable_metrics = {
            key: float(value) if value is not None else None
            for key, value in metrics.items()
        }
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=4)
        print(f"✓ Metrics saved: {metrics_file}")