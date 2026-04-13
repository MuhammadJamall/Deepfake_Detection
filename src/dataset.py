"""
Dataset loading and preprocessing module
"""

import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from collections import defaultdict


class DeepfakeDataset(Dataset):
    """
    Custom PyTorch Dataset for Deepfake detection
    
    Loads images from organized class folders and provides labels.
    """
    
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        """
        Args:
            root_dir (str): Directory containing class folders
            transform (callable, optional): Transformations to apply
            class_to_idx (dict, optional): Predefined class to index mapping
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Build class mapping if not provided
        if class_to_idx is None:
            classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Collect all images and their labels
        self.images = []
        self.labels = []
        
        for class_name, class_idx in self.class_to_idx.items():
            class_path = self.root_dir / class_name
            if class_path.exists():
                # Find all images (.png and .jpg)
                image_files = list(class_path.glob('*.png')) + list(class_path.glob('*.jpg'))
                for img_file in image_files:
                    self.images.append(img_file)
                    self.labels.append(class_idx)
    
    def __len__(self):
        """Return total number of images"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Return image and label
        
        Args:
            idx (int): Index of the item
            
        Returns:
            tuple: (image tensor, label)
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and convert image to RGB
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, idx):
        """Get class name from index"""
        return self.idx_to_class[idx]
    
    def get_stats(self):
        """Print dataset statistics"""
        class_counts = defaultdict(int)
        for label in self.labels:
            class_name = self.idx_to_class[label]
            class_counts[class_name] += 1
        
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Total images: {len(self)}")
        print("\nClass distribution:")
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
    Create train, validation, and test dataloaders
    
    Args:
        dataset_dir (str): Path to dataset directory
        batch_size (int): Batch size for dataloaders
        train_split (float): Proportion for training (default: 0.8)
        val_split (float): Proportion for validation (default: 0.1)
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_to_idx)
    """
    
    train_transform, val_transform = get_transforms()
    
    # Create datasets with appropriate transforms
    train_dataset = DeepfakeDataset(dataset_dir, transform=train_transform)
    val_dataset = DeepfakeDataset(dataset_dir, transform=val_transform)
    test_dataset = DeepfakeDataset(dataset_dir, transform=val_transform)
    
    class_to_idx = train_dataset.class_to_idx
    
    print(f"\n✓ Dataset loaded from: {dataset_dir}")
    print(f"✓ Total images: {len(train_dataset)}")
    print(f"✓ Number of classes: {len(class_to_idx)}")
    print(f"✓ Classes: {list(class_to_idx.keys())}")
    
    # Calculate split sizes
    total_size = len(train_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    # Split datasets
    train_dataset, _ = random_split(
        train_dataset, 
        [train_size, total_size - train_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset, _ = random_split(
        val_dataset, 
        [val_size, total_size - val_size],
        generator=torch.Generator().manual_seed(42)
    )
    test_dataset, _ = random_split(
        test_dataset, 
        [test_size, total_size - test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
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
    print("DATALOADER SPLIT")
    print("="*50)
    print(f"Training set: {train_size} images ({train_split*100:.0f}%)")
    print(f"Validation set: {val_size} images ({val_split*100:.0f}%)")
    print(f"Test set: {test_size} images ({(1-train_split-val_split)*100:.0f}%)")
    print(f"Batch size: {batch_size}")
    print("="*50 + "\n")
    
    return train_loader, val_loader, test_loader, class_to_idx


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
    else:
        print(f"⚠ Dataset not found at {dataset_path}")