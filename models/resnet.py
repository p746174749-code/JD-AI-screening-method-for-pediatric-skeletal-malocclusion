import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet152, ResNet152_Weights


class Resnet_152(nn.Module):
    def __init__(self, num_classes):
        super(Resnet_152, self).__init__()
        self.resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


class Resnet_101(nn.Module):
    def __init__(self, num_classes):
        super(Resnet_101, self).__init__()
        self.resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


if __name__ == "__main__":

    x1 = torch.randn(2, 3, 224, 224)
    model = Resnet_152(num_classes=3)
    output = model(x1)
    print("Output shape:", output.shape)


