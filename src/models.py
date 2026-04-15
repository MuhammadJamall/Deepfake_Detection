"""
Model definitions and architectures
"""

from typing import cast

import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B4 for binary classification (Real vs Fake)
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        
        # Load pretrained EfficientNet-B4
        self.backbone = models.efficientnet_b4(pretrained=pretrained)
        
        # Get number of input features for classifier
        classifier_layer = cast(nn.Linear, self.backbone.classifier[1])
        num_features = classifier_layer.in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all backbone layers except classifier"""
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_backbone_params(self):
        """Get backbone parameters for separate learning rates"""
        backbone_params = []
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name and param.requires_grad:
                backbone_params.append(param)
        return backbone_params
    
    def get_head_params(self):
        """Get classifier head parameters"""
        head_params = []
        for name, param in self.backbone.named_parameters():
            if 'classifier' in name and param.requires_grad:
                head_params.append(param)
        return head_params


def create_model(model_name='efficientnet_b4', num_classes=2, pretrained=True):
    """
    Factory function to create models
    
    Args:
        model_name (str): Model name
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights
        
    Returns:
        Model instance
    """
    
    if model_name == 'efficientnet_b4':
        return EfficientNetClassifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported")


if __name__ == "__main__":
    # Test model creation
    model = create_model('efficientnet_b4', num_classes=2)
    print(f"✓ Model created: {model}")
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")