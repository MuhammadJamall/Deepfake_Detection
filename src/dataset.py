from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import platform

class DeepfakeDataset(Dataset):
    """Custom dataset for Deepfake detection"""
    
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Build class mapping
        if class_to_idx is None:
            classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Collect all images
        self.images = []
        self.labels = []
        
        for class_name, class_idx in self.class_to_idx.items():
            class_path = self.root_dir / class_name
            if class_path.exists():
                for img_file in list(class_path.glob('*.png')) + list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg')):
                    self.images.append(img_file)
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, idx):
        return self.idx_to_class[idx]


def get_transforms(img_size=224):
    """Return training and validation transforms"""
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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


def get_dataloaders(dataset_dir, batch_size=32, train_split=0.8, val_split=0.1, num_workers=None):
    """Create train, val, test dataloaders"""
    
    train_transform, val_transform = get_transforms()
    if num_workers is None:
        num_workers = 0 if platform.system() == "Windows" else 2
    
    # Create full dataset
    full_dataset = DeepfakeDataset(dataset_dir, transform=None)
    class_to_idx = full_dataset.class_to_idx
    
    print(f"Total images: {len(full_dataset)}")
    print(f"Classes: {list(class_to_idx.keys())}")
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size]
    )
    
    # Apply transforms per split without sharing state
    train_images = [full_dataset.images[i] for i in train_dataset.indices]
    train_labels = [full_dataset.labels[i] for i in train_dataset.indices]
    val_images = [full_dataset.images[i] for i in val_dataset.indices]
    val_labels = [full_dataset.labels[i] for i in val_dataset.indices]
    test_images = [full_dataset.images[i] for i in test_dataset.indices]
    test_labels = [full_dataset.labels[i] for i in test_dataset.indices]

    train_dataset = DeepfakeDataset(dataset_dir, transform=train_transform, class_to_idx=class_to_idx)
    val_dataset = DeepfakeDataset(dataset_dir, transform=val_transform, class_to_idx=class_to_idx)
    test_dataset = DeepfakeDataset(dataset_dir, transform=val_transform, class_to_idx=class_to_idx)

    train_dataset.images, train_dataset.labels = train_images, train_labels
    val_dataset.images, val_dataset.labels = val_images, val_labels
    test_dataset.images, test_dataset.labels = test_images, test_labels
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"\nDataset Split:")
    print(f"  Train: {train_size} images")
    print(f"  Val: {val_size} images")
    print(f"  Test: {test_size} images")
    
    return train_loader, val_loader, test_loader, class_to_idx


if __name__ == "__main__":
    # Test the dataset
    dataset = DeepfakeDataset("./data")
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.class_to_idx}")
    
    # Test dataloaders
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders("./data")
    print("Dataloaders created successfully!")