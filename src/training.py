"""
Week 2: EfficientNet-B4 Trainer with Two-Phase Transfer Learning
Binary Classification: Real vs Fake
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from pathlib import Path
import importlib
import importlib.util
from typing import Any

wandb: Any | None
if importlib.util.find_spec("wandb") is not None:
    wandb = importlib.import_module("wandb")
else:
    wandb = None


class EfficientNetTrainer:
    """
    Trainer class for EfficientNet-B4 binary classification
    Implements two-phase training: Phase 1 (frozen backbone) + Phase 2 (full fine-tune)
    """
    
    def __init__(self, model, device, model_name='efficientnet_b4'):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.best_accuracy = 0
        self.best_loss = float('inf')
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'phase': []
        }
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            num_batches += 1
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            accuracy = correct / total
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.4f}'})

        if num_batches == 0:
            raise ValueError("train_loader is empty. Cannot train on zero batches.")
        
        epoch_loss = total_loss / num_batches
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating", leave=False)
            for images, labels in pbar:
                num_batches += 1
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                accuracy = correct / total
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.4f}'})

        if num_batches == 0:
            raise ValueError("val_loader is empty. Cannot validate on zero batches.")

        epoch_loss = total_loss / num_batches
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self, train_loader, val_loader, num_epochs=15, phase1_epochs=3, 
              save_path='results/models/', use_wandb=True):
        """
        Two-phase training strategy:
        Phase 1: Freeze backbone, train classifier only (3 epochs)
        Phase 2: Unfreeze and fine-tune entire model (12 epochs)
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Total epochs (default 15)
            phase1_epochs: Number of Phase 1 epochs (default 3)
            save_path: Path to save best model
            use_wandb: Log to Weights & Biases
        """
        
        if num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if phase1_epochs < 0:
            raise ValueError("phase1_epochs must be >= 0")
        if phase1_epochs > num_epochs:
            raise ValueError("phase1_epochs cannot be greater than num_epochs")

        log_to_wandb = use_wandb and wandb is not None
        wandb_logger = wandb if log_to_wandb else None
        if use_wandb and wandb is None:
            print("! Weights & Biases (wandb) is not installed. Continuing without wandb logging.")

        Path(save_path).mkdir(parents=True, exist_ok=True)
        criterion = nn.CrossEntropyLoss()
        
        # ==================== PHASE 1: Frozen Backbone ====================
        print("\n" + "="*70)
        print("PHASE 1: Training Classifier Only (Backbone FROZEN)")
        print("="*70)
        print(f"Duration: {phase1_epochs} epochs")
        print(f"Learning Rate: 1e-3 (Classifier Head)")
        print("="*70 + "\n")
        
        # Freeze backbone
        self.model.freeze_backbone()
        
        # Optimizer and scheduler for Phase 1
        optimizer_phase1 = Adam(self.model.get_head_params(), lr=1e-3)
        scheduler_phase1 = None
        if phase1_epochs > 0:
            scheduler_phase1 = CosineAnnealingLR(optimizer_phase1, T_max=phase1_epochs)
        
        for epoch in range(phase1_epochs):
            print(f"\n{'─'*70}")
            print(f"PHASE 1 - Epoch {epoch+1}/{phase1_epochs}")
            print(f"{'─'*70}")
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer_phase1, criterion)
            val_loss, val_acc, _, _ = self.validate(val_loader, criterion)
            
            if scheduler_phase1 is not None:
                scheduler_phase1.step()
            current_lr = optimizer_phase1.param_groups[0]['lr']
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            self.history['phase'].append('Phase 1')
            
            # Log to Weights & Biases
            if wandb_logger is not None:
                wandb_logger.log({
                    'phase': 'Phase 1',
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': current_lr
                })
            
            # Save best model
            if (val_acc > self.best_accuracy) or (
                val_acc == self.best_accuracy and val_loss < self.best_loss
            ):
                self.best_accuracy = val_acc
                self.best_loss = val_loss
                save_file = Path(save_path) / f"{self.model_name}_best.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer_phase1.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, save_file)
                print(f"✓ Best model saved: {save_file} (Acc: {val_acc:.4f})")
        
        # ==================== PHASE 2: Fine-tune Full Model ====================
        print("\n" + "="*70)
        print("PHASE 2: Fine-tuning Entire Model (Backbone UNFROZEN)")
        print("="*70)
        print(f"Duration: {num_epochs - phase1_epochs} epochs")
        print(f"Learning Rates: 1e-4 (Backbone) | 1e-3 (Classifier Head)")
        print("="*70 + "\n")
        
        # Unfreeze backbone
        self.model.unfreeze_backbone()
        
        # Different learning rates for backbone and head
        param_groups = [
            {'params': self.model.get_backbone_params(), 'lr': 1e-4},
            {'params': self.model.get_head_params(), 'lr': 1e-3}
        ]
        optimizer_phase2 = Adam(param_groups)
        phase2_epochs = num_epochs - phase1_epochs
        scheduler_phase2 = None
        if phase2_epochs > 0:
            scheduler_phase2 = CosineAnnealingLR(optimizer_phase2, T_max=phase2_epochs)
        
        for epoch in range(phase1_epochs, num_epochs):
            print(f"\n{'─'*70}")
            print(f"PHASE 2 - Epoch {epoch+1}/{num_epochs}")
            print(f"{'─'*70}")
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer_phase2, criterion)
            val_loss, val_acc, _, _ = self.validate(val_loader, criterion)
            
            if scheduler_phase2 is not None:
                scheduler_phase2.step()
            backbone_lr = optimizer_phase2.param_groups[0]['lr']
            head_lr = optimizer_phase2.param_groups[1]['lr']
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Backbone LR: {backbone_lr:.6f} | Head LR: {head_lr:.6f}")
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(backbone_lr)
            self.history['phase'].append('Phase 2')
            
            # Log to Weights & Biases
            if wandb_logger is not None:
                wandb_logger.log({
                    'phase': 'Phase 2',
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'backbone_lr': backbone_lr,
                    'head_lr': head_lr
                })
            
            # Save best model
            if (val_acc > self.best_accuracy) or (
                val_acc == self.best_accuracy and val_loss < self.best_loss
            ):
                self.best_accuracy = val_acc
                self.best_loss = val_loss
                save_file = Path(save_path) / f"{self.model_name}_best.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer_phase2.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, save_file)
                print(f"✓ Best model saved: {save_file} (Acc: {val_acc:.4f})")
        
        print("\n" + "="*70)
        print(f"✓ TRAINING COMPLETE!")
        print(f"  Best Validation Accuracy: {self.best_accuracy:.4f} ({self.best_accuracy*100:.2f}%)")
        print("="*70 + "\n")
        
        return self.history
    
    def get_history(self):
        """Return training history"""
        return self.history