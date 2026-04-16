"""
Model definitions for deepfake detection (Binary Classification)
"""

import torch
import torch.nn as nn
from timm import create_model as timm_create_model


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B4 for binary classification"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = timm_create_model('efficientnet_b4', pretrained=pretrained)
        
        # Get the number of features from the backbone
        num_features = self.backbone.classifier.in_features
        
        # Replace classifier for binary classification
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.features.parameters():
            param.requires_grad = True
    
    def get_backbone_params(self):
        """Get backbone parameters"""
        return self.backbone.features.parameters()
    
    def get_head_params(self):
        """Get classifier head parameters"""
        return self.backbone.classifier.parameters()


class XceptionClassifier(nn.Module):
    """Xception for binary classification"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = timm_create_model('legacy_xception', pretrained=pretrained)
        
        # Get the number of features from the backbone
        num_features = self.backbone.fc.in_features
        
        # Replace classifier for binary classification
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_backbone_params(self):
        """Get backbone parameters (excluding head)"""
        params = []
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                params.append(param)
        return params
    
    def get_head_params(self):
        """Get classifier head parameters"""
        return self.backbone.fc.parameters()


def create_model(model_name, num_classes=2, pretrained=True):
    """
    Create a model for binary classification
    
    Args:
        model_name (str): 'efficientnet_b4' or 'xception'
        num_classes (int): Number of classes (default: 2 for binary)
        pretrained (bool): Use pretrained weights
    
    Returns:
        nn.Module: Model instance
    """
    if model_name == 'efficientnet_b4':
        return EfficientNetClassifier(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'xception':
        return XceptionClassifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing model creation for binary classification...")
    
    efficientnet = create_model('efficientnet_b4', num_classes=2, pretrained=True)
    efficientnet = efficientnet.to(device)
    print(f"✓ EfficientNet-B4 created: {efficientnet}")
    
    xception = create_model('xception', num_classes=2, pretrained=True)
    xception = xception.to(device)
    print(f"✓ Xception created: {xception}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        eff_out = efficientnet(dummy_input)
        xep_out = xception(dummy_input)
    
    print(f"\n✓ EfficientNet output shape: {eff_out.shape}")
    print(f"✓ Xception output shape: {xep_out.shape}")
    print("✓ Both models ready for binary classification!")