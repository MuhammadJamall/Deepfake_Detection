"""
Training utilities and functions
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from pathlib import Path


class Trainer:
    """
    Trainer class for model training and evaluation
    """
    
    def __init__(self, model, device, model_name='efficientnet_b4'):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.best_accuracy = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            accuracy = correct / total
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.4f}'})
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                accuracy = correct / total
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.4f}'})
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs=15, phase1_epochs=3, save_path='results/models/'):
        """
        Two-phase training:
        Phase 1: Freeze backbone, train classifier only
        Phase 2: Unfreeze and fine-tune entire model
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Total epochs
            phase1_epochs: Epochs for phase 1
            save_path: Path to save best model
        """
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        criterion = nn.CrossEntropyLoss()
        
        print("\n" + "="*60)
        print("PHASE 1: Training Classifier Only (Backbone Frozen)")
        print("="*60)
        
        # Phase 1: Freeze backbone
        self.model.freeze_backbone()
        optimizer_phase1 = Adam(self.model.get_head_params(), lr=1e-3)
        scheduler_phase1 = CosineAnnealingLR(optimizer_phase1, T_max=phase1_epochs)
        
        for epoch in range(phase1_epochs):
            print(f"\nPhase 1 - Epoch {epoch+1}/{phase1_epochs}")
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer_phase1, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            scheduler_phase1.step()
            current_lr = optimizer_phase1.param_groups[0]['lr']
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                save_file = f"{save_path}{self.model_name}_best.pth"
                torch.save(self.model.state_dict(), save_file)
                print(f"✓ Best model saved: {save_file}")
        
        print("\n" + "="*60)
        print("PHASE 2: Fine-tuning Entire Model (Backbone Unfrozen)")
        print("="*60)
        
        # Phase 2: Unfreeze backbone
        self.model.unfreeze_backbone()
        
        # Different learning rates for backbone and head
        param_groups = [
            {'params': self.model.get_backbone_params(), 'lr': 1e-4},
            {'params': self.model.get_head_params(), 'lr': 1e-3}
        ]
        optimizer_phase2 = Adam(param_groups)
        scheduler_phase2 = CosineAnnealingLR(optimizer_phase2, T_max=(num_epochs - phase1_epochs))
        
        for epoch in range(phase1_epochs, num_epochs):
            print(f"\nPhase 2 - Epoch {epoch+1}/{num_epochs}")
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer_phase2, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            scheduler_phase2.step()
            current_lr = optimizer_phase2.param_groups[0]['lr']
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                save_file = f"{save_path}{self.model_name}_best.pth"
                torch.save(self.model.state_dict(), save_file)
                print(f"✓ Best model saved: {save_file}")
        
        print("\n" + "="*60)
        print(f"Training Complete! Best Accuracy: {self.best_accuracy:.4f}")
        print("="*60 + "\n")
        
        return self.history