import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class ChestXRayEfficientNet(nn.Module):
    def __init__(self, num_classes=3):
        super(ChestXRayEfficientNet, self).__init__()
        
        # Pretrained EfficientNet-B0 Backbone
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # We extract up to the features (conv blocks)
        self.features = self.backbone.features
        
        # The number of output channels from EfficientNet-B0 features is 1280
        num_features = 1280
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Custom Classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Increased Dropout to 0.5 to prevent overfitting
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Pass through backbone features
        # Input shape expected: (Batch, 3, 224, 224)
        x = self.features(x)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # Flatten spatial dimensions
        
        # Fully Connected Classifier
        logits = self.classifier(x)
        
        # Note: In PyTorch, we usually output raw logits and use nn.CrossEntropyLoss
        # which applies Softmax internally. If you need probabilities, use predict()
        return logits
        
    def predict(self, x):
        """Returns softmax probabilities for the 3 classes"""
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities
