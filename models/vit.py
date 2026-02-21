import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ViT(nn.Module):
    def __init__(self, num_classes):
        super(ViT, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

        # 替换分类头最后一层 Linear
        in_features = self.vit.heads[-1].in_features
        self.vit.heads[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vit(x)


if __name__ == "__main__":

    x1 = torch.randn(2, 3, 224, 224)
    model = ViT(num_classes=3)
    output = model(x1)
    print("Output shape:", output.shape)
