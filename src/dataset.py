"""
Dataset loading and preprocessing module
Binary Classification: REAL (Original) vs FAKE (All Deepfakes)
"""

import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from collections import defaultdict


# Class mapping: Original classes -> Binary labels
CLASS_MAPPING = {
    'Original': 0,          # REAL
    'Deepfakes': 1,         # FAKE
    'Face2Face': 1,         # FAKE
    'FaceShifter': 1,       # FAKE
    'FaceSwap': 1,          # FAKE
    'NeuralTextures': 1     # FAKE
}


class DeepfakeDataset(Dataset):
    """
    Custom PyTorch Dataset for Deepfake detection (Binary Classification)
    """
    
    def __init__(self, root_dir, indices=None, transform=None):
        """
        Args:
            root_dir (str): Directory containing class folders
            indices (list, optional): Specific indices to use (for train/val/test splits)
            transform (callable, optional): Transformations to apply
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Collect all images and their labels
        self.images = []
        self.labels = []
        
        # Load images from 6 classes and map to binary labels
        for class_name in sorted(CLASS_MAPPING.keys()):
            class_path = self.root_dir / class_name
            if class_path.exists():
                image_files = sorted(list(class_path.glob('*.png')) + list(class_path.glob('*.jpg')))
                binary_label = CLASS_MAPPING[class_name]
                
                for img_file in image_files:
                    self.images.append(img_file)
                    self.labels.append(binary_label)
        
        # Apply indices if provided
        if indices is not None:
            self.images = [self.images[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=224):
    """Create training and validation transforms"""
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform


def get_dataloaders(dataset_dir, batch_size=32, train_split=0.8, val_split=0.1, num_workers=2):
    """
    Create train, validation, and test dataloaders with BINARY classification
    """
    
    train_transform, val_transform = get_transforms()
    
    # Load full dataset
    full_dataset = DeepfakeDataset(dataset_dir, transform=None)
    total_size = len(full_dataset)
    
    print(f"\n✓ Dataset loaded from: {dataset_dir}")
    print(f"✓ Total images: {total_size}")
    print(f"✓ Number of classes (Binary): 2 (REAL=0, FAKE=1)")
    
    # Calculate split sizes
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    # Create indices and split ONCE
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(total_size).tolist()
    
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size:]
    
    # Create 3 datasets using same images but different indices
    train_dataset = DeepfakeDataset(dataset_dir, indices=train_indices, transform=train_transform)
    val_dataset = DeepfakeDataset(dataset_dir, indices=val_indices, transform=val_transform)
    test_dataset = DeepfakeDataset(dataset_dir, indices=test_indices, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print("\n" + "="*50)
    print("DATALOADER SPLIT (Binary Classification)")
    print("="*50)
    print(f"Training set: {train_size} images ({train_split*100:.0f}%)")
    print(f"Validation set: {val_size} images ({val_split*100:.0f}%)")
    print(f"Test set: {test_size} images ({(1-train_split-val_split)*100:.0f}%)")
    print(f"Batch size: {batch_size}")
    print("="*50 + "\n")
    
    binary_class_to_idx = {'REAL': 0, 'FAKE': 1}
    
    return train_loader, val_loader, test_loader, binary_class_to_idx