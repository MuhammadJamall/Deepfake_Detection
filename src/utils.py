"""
Utility functions for the project
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def create_results_dirs():
    """Create results directories if they don't exist"""
    dirs = [
        'results/plots',
        'results/metrics',
        'results/models'
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✓ Results directories created")


def check_gpu():
    """Check GPU availability and print details"""
    print("\n" + "="*50)
    print("GPU INFORMATION")
    print("="*50)
    
    if torch.cuda.is_available():
        print(f"✓ GPU Available: YES")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("✗ GPU Available: NO (CPU will be used)")
    print("="*50 + "\n")
    
    return torch.cuda.is_available()


def get_device():
    """Get device (GPU or CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor for visualization
    
    Args:
        image: Normalized image tensor (C, H, W)
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Denormalized image tensor
    """
    mean = torch.tensor(mean).reshape(3, 1, 1)
    std = torch.tensor(std).reshape(3, 1, 1)
    
    image = image * std + mean
    return torch.clamp(image, 0, 1)


def plot_sample_batch(images, labels, class_to_idx, save_path='results/plots/batch_sample.png'):
    """
    Visualize a batch of images
    
    Args:
        images: Batch of images
        labels: Batch of labels
        class_to_idx: Class to index mapping
        save_path: Path to save the plot
    """
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()
    
    for idx in range(min(8, len(images))):
        img = images[idx].cpu()
        img = denormalize_image(img)
        img = img.permute(1, 2, 0).numpy()
        
        label_idx = labels[idx].item()
        label_name = idx_to_class[label_idx]
        
        axes[idx].imshow(img)
        axes[idx].set_title(label_name, fontsize=11, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Batch visualization saved to {save_path}")
    plt.close()


def count_parameters(model):
    """Count total trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model, model_name):
    """Print model architecture and parameter count"""
    print(f"\n{'='*50}")
    print(f"MODEL: {model_name}")
    print(f"{'='*50}")
    print(f"Total Parameters: {count_parameters(model):,}")
    print(f"Model Summary:\n{model}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    check_gpu()
    create_results_dirs()