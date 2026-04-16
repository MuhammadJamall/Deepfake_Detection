"""
Dataset loading and preprocessing module
Binary Classification: REAL (Original) vs FAKE (All Deepfakes)
"""

import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
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
    
    Loads images from organized class folders and provides binary labels.
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
                # Find all images (.png and .jpg)
                image_files = sorted(list(class_path.glob('*.png')) + list(class_path.glob('*.jpg')))
                binary_label = CLASS_MAPPING[class_name]
                
                for img_file in image_files:
                    self.images.append(img_file)
                    self.labels.append(binary_label)
        
        # Apply indices if provided (for train/val/test split)
        if indices is not None:
            self.images = [self.images[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
    
    def __len__(self):
        """Return total number of images"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Return image and label
        
        Args:
            idx (int): Index of the item
            
        Returns:
            tuple: (image tensor, binary label)
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and convert image to RGB
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_stats(self):
        """Print dataset statistics"""
        class_counts = defaultdict(int)
        for label in self.labels:
            class_name = 'REAL' if label == 0 else 'FAKE'
            class_counts[class_name] += 1
        
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Total images: {len(self)}")
        print("\nClass distribution (Binary):")
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            percentage = (count / len(self)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        print("="*50 + "\n")


def get_transforms(img_size=224):
    """
    Create training and validation transforms
    
    Args:
        img_size (int): Image size for resizing
        
    Returns:
        tuple: (train_transform, val_transform)
    """
    
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
    Create train, validation, and test dataloaders with proper splitting
    
    IMPORTANT: Data is split ONCE at the beginning to avoid data leakage.
    Each image appears in only ONE set (train/val/test).
    
    Args:
        dataset_dir (str): Path to dataset directory
        batch_size (int): Batch size for dataloaders
        train_split (float): Proportion for training (default: 0.8)
        val_split (float): Proportion for validation (default: 0.1)
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, binary_class_to_idx)
    """
    
    train_transform, val_transform = get_transforms()
    
    # Load full dataset ONCE (without transforms or indices)
    full_dataset = DeepfakeDataset(dataset_dir, transform=None)
    total_size = len(full_dataset)
    
    print(f"\n✓ Dataset loaded from: {dataset_dir}")
    print(f"✓ Total images: {total_size}")
    print(f"✓ Number of classes (Binary): 2 (REAL, FAKE)")
    
    # Calculate split sizes (80/10/10)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    # Create indices and split ONCE
    all_indices = list(range(total_size))
    torch.manual_seed(42)  # For reproducibility
    shuffled_indices = torch.randperm(total_size).tolist()
    
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size:]
    
    # Create 3 datasets using the same full_dataset but with different indices
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
    print("DATALOADER SPLIT (No Data Leakage)")
    print("="*50)
    print(f"Training set: {train_size} images ({train_split*100:.0f}%)")
    print(f"Validation set: {val_size} images ({val_split*100:.0f}%)")
    print(f"Test set: {test_size} images ({(1-train_split-val_split)*100:.0f}%)")
    print(f"Batch size: {batch_size}")
    print("="*50 + "\n")
    
    # Binary class mapping
    binary_class_to_idx = {'REAL': 0, 'FAKE': 1}
    
    return train_loader, val_loader, test_loader, binary_class_to_idx


if __name__ == "__main__":
    # Test code
    dataset_path = "./data"
    
    if os.path.exists(dataset_path):
        train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
            dataset_path,
            batch_size=32
        )
        
        # Verify a batch
        images, labels = next(iter(train_loader))
        print(f"✓ Batch shape: {images.shape}")
        print(f"✓ Labels: {labels}")
        print(f"✓ Unique labels in batch: {torch.unique(labels).tolist()}")
    else:
        print(f"⚠ Dataset not found at {dataset_path}")