import torch
import torch.nn as nn
import torchvision.models as models
import timm


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B4 for binary classification (Real vs Fake)
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        
        # Load pretrained EfficientNet-B4
        self.backbone = models.efficientnet_b4(pretrained=pretrained)
        
        # Get number of input features for classifier
        classifier = self.backbone.classifier
        if isinstance(classifier, nn.Sequential):
            classifier_head = classifier[1]
            if isinstance(classifier_head, nn.Linear):
                num_features = classifier_head.in_features
            else:
                raise TypeError("Unexpected EfficientNet classifier head type")
        else:
            raise TypeError("Unexpected EfficientNet classifier structure")
        
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


class XceptionClassifier(nn.Module):
    """
    Xception for binary classification (Real vs Fake)
    
    Xception uses depthwise separable convolutions which are theoretically
    better at catching local artifacts introduced by face-swapping algorithms.
    This architecture was used in the original FaceForensics++ paper.
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super(XceptionClassifier, self).__init__()
        
        # Load pretrained Xception using timm
        self.backbone = timm.create_model('xception', pretrained=pretrained, num_classes=num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all backbone layers except classifier"""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:  # fc is the classifier in Xception
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_backbone_params(self):
        """Get backbone parameters for separate learning rates"""
        backbone_params = []
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name and param.requires_grad:
                backbone_params.append(param)
        return backbone_params
    
    def get_head_params(self):
        """Get classifier head parameters"""
        head_params = []
        for name, param in self.backbone.named_parameters():
            if 'fc' in name and param.requires_grad:
                head_params.append(param)
        return head_params


def create_model(model_name='efficientnet_b4', num_classes=2, pretrained=True):
    """
    Factory function to create models
    
    Args:
        model_name (str): Model name ('efficientnet_b4' or 'xception')
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights
        
    Returns:
        Model instance
    """
    
    if model_name == 'efficientnet_b4':
        return EfficientNetClassifier(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'xception':
        return XceptionClassifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported")


if __name__ == "__main__":
    # Test model creation
    print("Testing EfficientNet-B4...")
    model1 = create_model('efficientnet_b4', num_classes=2)
    print(f"✓ EfficientNet created: {sum(p.numel() for p in model1.parameters()):,} parameters")
    
    print("\nTesting Xception...")
    model2 = create_model('xception', num_classes=2)
    print(f"✓ Xception created: {sum(p.numel() for p in model2.parameters()):,} parameters")