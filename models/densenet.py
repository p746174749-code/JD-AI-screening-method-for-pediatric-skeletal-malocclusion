import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision.models import densenet121, DenseNet121_Weights


class Densenet_121(nn.Module):
    def __init__(self, num_classes):
        super(Densenet_121, self).__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.densenet(x)
        return x

if __name__ == "__main__":

    x1 = torch.randn(2, 3, 224, 224)
    model = Densenet_121(num_classes=3)
    output = model(x1)
    print("Output shape:", output.shape)



