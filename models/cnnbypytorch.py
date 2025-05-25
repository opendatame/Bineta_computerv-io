# model.py
import torch
import torch.nn as nn
from torchvision import models

class ResNet50FineTune(nn.Module):
    def __init__(self, num_classes=4, freeze_base=True):
        super(ResNet50FineTune, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # remplacer la dernière couche fully connected
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)


if __name__ == "__main__":
    model = ResNet50FineTune(num_classes=4, freeze_base=True)
    print(model)
    input_tensor = torch.randn(1, 3, 224, 224)  
    output = model(input_tensor)
    print("Output shape:", output.shape)

