import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Swin-Tiny
class Swin_T(nn.Module):
    def __init__(self, num_classes):
        super(Swin_T, self).__init__()
        self.swin = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        num_features = self.swin.head.in_features
        self.swin.head = nn.Linear(num_features, num_classes)

    def forward(self, x):

        x = self.swin(x)
        return x


# Swin-Small
class Swin_S(nn.Module):
    def __init__(self, num_classes):
        super(Swin_S, self).__init__()
        self.model = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)
        num_features = self.model.head.in_features
        self.model.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Swin-Base
class Swin_B(nn.Module):
    def __init__(self, num_classes):
        super(Swin_B, self).__init__()
        self.model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
        num_features = self.model.head.in_features
        self.model.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    x1 = torch.randn(2, 3, 224, 224)
    model = Swin_T(num_classes=3)
    output = model(x1)
    print("Output shape:", output.shape)
