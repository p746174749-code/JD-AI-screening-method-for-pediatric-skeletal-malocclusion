import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ConvNeXt-Tiny
class ConvNeXt_T(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXt_T, self).__init__()
        self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        num_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ConvNeXt-Small
class ConvNeXt_S(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXt_S, self).__init__()
        self.model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        num_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ConvNeXt-Base
class ConvNeXt_B(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXt_B, self).__init__()
        self.model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        num_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    x1 = torch.randn(2, 3, 224, 224)
    model = ConvNeXt_T(num_classes=3)
    output = model(x1)
    print("Output shape:", output.shape)
