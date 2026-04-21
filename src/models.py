"""
Model definitions for deepfake detection (Binary Classification)
Updated for Week 2: EfficientNet-B4 Binary Classifier
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B4_Weights
from timm import create_model as timm_create_model
from typing import cast


class EfficientNetBinaryClassifier(nn.Module):
    """EfficientNet-B4 for binary classification (Real vs Fake)"""
    
    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()
        # Load pretrained EfficientNet-B4 using torchvision weights API.
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)
        
        # Get the number of features from the backbone
        classifier_in = cast(nn.Linear, self.backbone.classifier[1])
        num_features = classifier_in.in_features
        
        # Replace classifier for binary classification
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone parameters (everything except classifier)"""
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_backbone_params(self):
        """Get backbone parameters (excluding classifier)"""
        return [p for name, p in self.backbone.named_parameters() 
                if 'classifier' not in name and p.requires_grad]
    
    def get_head_params(self):
        """Get classifier head parameters"""
        return self.backbone.classifier.parameters()


class XceptionClassifier(nn.Module):
    """Xception-based classifier using timm."""

    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()
        self.backbone = timm_create_model(
            'legacy_xception',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze backbone parameters (everything except final fc head)."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_backbone_params(self):
        """Get backbone parameters (excluding head)."""
        return [
            p for name, p in self.backbone.named_parameters()
            if 'fc' not in name and p.requires_grad
        ]

    def get_head_params(self):
        """Get classifier head parameters."""
        return self.backbone.fc.parameters()


def create_binary_classifier(pretrained=True, model_name='efficientnet_b4'):
    """
    Create a binary classifier for deepfake detection
    
    Args:
        pretrained (bool): Use ImageNet pretrained weights
        model_name (str): Model architecture name
    
    Returns:
        nn.Module: Model instance
    """
    if model_name == 'efficientnet_b4':
        return EfficientNetBinaryClassifier(pretrained=pretrained, num_classes=2)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_model(model_name='efficientnet_b4', num_classes=2, pretrained=True):
    """Backward-compatible model factory for app/training scripts."""
    if model_name == 'efficientnet_b4':
        return EfficientNetBinaryClassifier(pretrained=pretrained, num_classes=num_classes)
    if model_name == 'xception':
        return XceptionClassifier(pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f"Unknown model: {model_name}")