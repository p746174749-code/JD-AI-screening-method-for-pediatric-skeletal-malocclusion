import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torchvision import models
from torch.nn.init import trunc_normal_


class MIAModule(nn.Module):
    def __init__(
            self,
            in_channels: int = 768,  # Input channels (consistent with your feature dimension)
            d_model: int = 768,  # Keep output dimension unchanged
            n_heads: int = 12,  # 768/12=64 meets multi-head attention requirements
            num_others: int = 1,  # Number of other inputs (adjust based on actual case)
            spatial_dim: int = 7,  # Spatial dimension 7x7
            dropout: float = 0.1,
    ):
        super().__init__()
        # Parameter validation
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.spatial_dim = spatial_dim
        self.num_others = num_others
        self.N = spatial_dim * spatial_dim

        # ----------------- Input Projection Layers (Keep dimension unchanged) -----------------
        self.input_proj_x1 = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.input_proj_others = nn.ModuleList([
            nn.Conv2d(in_channels, d_model, kernel_size=1)
            for _ in range(num_others)
        ])

        # ----------------- Position Encoding -----------------
        self.pos_embed_x1 = nn.Parameter(torch.zeros(1, self.N, d_model))
        self.pos_embed_others = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.N, d_model))
            for _ in range(num_others)
        ])
        # Initialization
        trunc_normal_(self.pos_embed_x1, std=0.02)
        for pos in self.pos_embed_others:
            trunc_normal_(pos, std=0.02)

        # ----------------- Attention Modules -----------------
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)

        # ----------------- Fusion Attention -----------------
        self.fusion_query = nn.Parameter(torch.randn(1, d_model))
        self.fusion_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # ----------------- Gating Mechanism -----------------
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Sigmoid()
        )

        # ----------------- Output Layer -----------------
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x1, *others):
        B, C, H, W = x1.shape
        # Validate input dimensions
        assert (H, W) == (self.spatial_dim, self.spatial_dim), f"Input size should be {self.spatial_dim}x{self.spatial_dim}"
        assert len(others) == self.num_others, f"Requires {self.num_others} other inputs"

        # ----------------- Input Projection (Keep channel count at 768) -----------------
        x1 = self.input_proj_x1(x1)  # (B, 768, 7, 7)
        x1_flat = x1.flatten(2).permute(0, 2, 1) + self.pos_embed_x1  # (B, 49, 768)

        # Process other inputs
        others_proj = []
        for i, x_o in enumerate(others):
            xo_proj = self.input_proj_others[i](x_o)  # (B, 768, 7, 7)
            xo_flat = xo_proj.flatten(2).permute(0, 2, 1) + self.pos_embed_others[i]
            others_proj.append(xo_flat)

        # ----------------- Attention Processing -----------------
        # Self-Attention
        x1_self, _ = self.self_attn(x1_flat, x1_flat, x1_flat)
        x1_self = self.norm_self(x1_flat + x1_self)

        # Cross-Attention
        cross_feats = []
        for xo in others_proj:
            x_cross, _ = self.cross_attn(x1_flat, xo, xo)
            x_cross = self.norm_cross(x1_flat + x_cross)
            cross_feats.append(x_cross)

        # ----------------- Feature Fusion -----------------
        features = [x1_self] + cross_feats
        stack = torch.stack(features, dim=2)  # (B, 49, num_features, 768)
        B, N, L, D = stack.shape
        stack = stack.view(B * N, L, D)

        # Fusion Attention
        fusion_q = self.fusion_query.repeat(B * N, 1, 1)
        fused, _ = self.fusion_attn(fusion_q, stack, stack)
        fused_token = fused.view(B, N, D)

        # ----------------- Gated Output -----------------
        gate_input = torch.cat([x1_flat, fused_token], dim=-1)
        gate_weight = self.gate(gate_input)
        output = gate_weight * fused_token + (1 - gate_weight) * x1_flat

        # ----------------- Restore Spatial Dimensions -----------------
        output = self.norm_out(output)
        output = output.permute(0, 2, 1).view(B, D, H, W)  # (B, 768, 7, 7)

        return output


# --------------------- Bidirectional Cross Attention ---------------------
class BiCrossAttention(nn.Module):
    def __init__(self, dim, heads=8, ffn_hidden=None, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        ffn_hidden = ffn_hidden or dim * 4

        self.q1 = nn.Linear(dim, dim)
        self.k1 = nn.Linear(dim, dim)
        self.v1 = nn.Linear(dim, dim)

        self.q2 = nn.Linear(dim, dim)
        self.k2 = nn.Linear(dim, dim)
        self.v2 = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        B, N, C = x1.shape

        # x1->x2
        q1 = self.q1(x1).reshape(B, N, self.heads, -1).transpose(1, 2)
        k2 = self.k2(x2).reshape(B, N, self.heads, -1).transpose(1, 2)
        v2 = self.v2(x2).reshape(B, N, self.heads, -1).transpose(1, 2)
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        out1 = (attn1.softmax(dim=-1) @ v2).transpose(1, 2).reshape(B, N, C)
        out1 = self.dropout(out1)

        # x2->x1
        q2 = self.q2(x2).reshape(B, N, self.heads, -1).transpose(1, 2)
        k1 = self.k1(x1).reshape(B, N, self.heads, -1).transpose(1, 2)
        v1 = self.v1(x1).reshape(B, N, self.heads, -1).transpose(1, 2)
        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        out2 = (attn2.softmax(dim=-1) @ v1).transpose(1, 2).reshape(B, N, C)
        out2 = self.dropout(out2)

        # Fusion + Residual + FFN
        attn_out = self.proj(torch.cat([out1, out2], dim=-1))
        x = self.norm1(x1 + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        out = self.norm2(x + self.dropout(ffn_out))
        return out


# --------------------- Fusion Attention Module ---------------------
class FusionAttention(nn.Module):
    def __init__(self, dim, heads=8, ffn_hidden=None, dropout=0.1):
        super().__init__()
        ffn_hidden = ffn_hidden or dim * 4
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, feats, return_attn=False):
        x = torch.stack(feats, dim=1)  # [B, T, N, C]
        B, T, N, C = x.shape
        x = x.view(B * N, T, C)

        attn_out, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        x = self.norm1(x + self.dropout(attn_out))

        fusion_token = x[:, 0, :].view(B, N, C)
        ffn_out = self.ffn(fusion_token)
        out = self.norm2(fusion_token + self.dropout(ffn_out))

        if return_attn:
            attn_map = attn_weights.mean(dim=1)  # [B*N, T, T]
            attn_map = attn_map.view(B, N, T, T).mean(dim=1)  # [B, T, T]
            return out, attn_map
        else:
            return out


# --------------------- Total Fusion Module ---------------------
class MVCAF(nn.Module):
    def __init__(self, dim=768, heads=8, ffn_hidden=None, dropout=0.1):
        super().__init__()
        self.cross_attn = BiCrossAttention(dim, heads, ffn_hidden, dropout)
        self.fusion_attn = FusionAttention(dim, heads, ffn_hidden, dropout)
        self.dim = dim

    def forward(self, x_list, return_attn=False):
        # Input is (B, C, H, W), converted to (B, H*W, C)
        x_list = [x.flatten(2).transpose(1, 2) for x in x_list]  # [B, 49, C]
        x1 = x_list[0]
        cross_feats = [self.cross_attn(x1, x) for x in x_list[1:]]

        if return_attn:
            fused, attn_map = self.fusion_attn(cross_feats, return_attn=True)
            fused = fused.transpose(1, 2).view(-1, self.dim, 7, 7)
            return fused, attn_map
        else:
            fused = self.fusion_attn(cross_feats)
            return fused.transpose(1, 2).view(-1, self.dim, 7, 7)


class ConvNeXt_T2(nn.Module):
    def __init__(self, num_classes=3, share_backbone=True, fusion_heads=8, fusion_ffn_hidden=None):
        super(ConvNeXt_T2, self).__init__()
        self.share_backbone = share_backbone

        # Create feature extractor
        def create_feature_extractor():
            model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            model.avgpool = nn.Identity()
            model.classifier = nn.Identity()
            return model

        # backbone
        if self.share_backbone:
            self.shared_backbone = create_feature_extractor()
        else:
            self.branch1 = create_feature_extractor()
            self.branch2 = create_feature_extractor()

        self.fusion = MVCAF(dim=768, heads=fusion_heads, ffn_hidden=fusion_ffn_hidden)

        # Classifier head
        self.classifier = nn.Linear(768, num_classes)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x1, x2):
        if self.share_backbone:
            feat1 = self.shared_backbone(x1)  # (B, 768, 7, 7)
            feat2 = self.shared_backbone(x2)  # (B, 768, 7, 7)
        else:
            feat1 = self.branch1(x1)
            feat2 = self.branch2(x2)

        fused_feat = self.fusion([feat1, feat2])  # (B, 768, 7, 7)
        pooled_feat = fused_feat.mean(dim=[2, 3])  # Global Average Pooling

        output = self.classifier(pooled_feat)

        # print(f"feat1 shape: {feat1.shape}")
        # print(f"feat2 shape: {feat2.shape}")
        # print(f"fused_feat shape: {fused_feat.shape}")
        # print(f"pooled_feat shape: {pooled_feat.shape}")
        return output


class ConvNeXt_T3(nn.Module):
    def __init__(self, num_classes=3, share_backbone=True, fusion_heads=8, fusion_ffn_hidden=None):
        super(ConvNeXt_T3, self).__init__()
        self.share_backbone = share_backbone

        def create_feature_extractor():
            model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            # Remove classifier head and global pooling layer
            model.avgpool = nn.Identity()
            model.classifier = nn.Identity()
            return model

        if self.share_backbone:
            self.shared_backbone = create_feature_extractor()
        else:
            self.branch1 = create_feature_extractor()
            self.branch2 = create_feature_extractor()
            self.branch3 = create_feature_extractor()

        self.fusion = MVCAF(dim=768, heads=fusion_heads, ffn_hidden=fusion_ffn_hidden)

        # Classifier output
        self.classifier = nn.Linear(768, num_classes)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x1, x2, x3):
        if self.share_backbone:
            feat1 = self.shared_backbone(x1)  # (B, 768, 7, 7)
            feat2 = self.shared_backbone(x2)  # (B, 768, 7, 7)
            feat3 = self.shared_backbone(x3)  # (B, 768, 7, 7)
        else:
            feat1 = self.branch1(x1)
            feat2 = self.branch2(x2)
            feat3 = self.branch3(x3)

        # Use MIAModule to fuse features; primary input feat1, others are feat2, feat3
        fused_feat = self.fusion([feat1, feat2, feat3])  # (B, 768, 7, 7)

        # Global Average Pooling (B, 768, 7, 7) -> (B, 768)
        pooled_feat = fused_feat.mean(dim=[2, 3])  # Global average pooling

        output = self.classifier(pooled_feat)
        return output


class ConvNeXt_T4(nn.Module):
    def __init__(self, num_classes=3, share_backbone=True, fusion_heads=8, fusion_ffn_hidden=None):
        super(ConvNeXt_T4, self).__init__()
        self.share_backbone = share_backbone

        def create_feature_extractor():
            model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            # Remove classifier head and global pooling layer
            model.avgpool = nn.Identity()
            model.classifier = nn.Identity()
            return model

        if self.share_backbone:
            self.shared_backbone = create_feature_extractor()
        else:
            self.branch1 = create_feature_extractor()
            self.branch2 = create_feature_extractor()
            self.branch3 = create_feature_extractor()
            self.branch4 = create_feature_extractor()

        self.fusion = MVCAF(dim=768, heads=fusion_heads, ffn_hidden=fusion_ffn_hidden)

        # Classifier output
        self.classifier = nn.Linear(768, num_classes)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x1, x2, x3, x4):
        if self.share_backbone:
            feat1 = self.shared_backbone(x1)  # (B, 768, 7, 7)
            feat2 = self.shared_backbone(x2)  # (B, 768, 7, 7)
            feat3 = self.shared_backbone(x3)  # (B, 768, 7, 7)
            feat4 = self.shared_backbone(x4)  # (B, 768, 7, 7)
        else:
            feat1 = self.branch1(x1)
            feat2 = self.branch2(x2)
            feat3 = self.branch3(x3)
            feat4 = self.branch4(x4)
        # Use MIAModule to fuse features; primary input feat1, others are feat2, feat3, feat4
        fused_feat = self.fusion([feat1, feat2, feat3, feat4])  # (B, 768, 7, 7)

        # Global Average Pooling (B, 768, 7, 7) -> (B, 768)
        pooled_feat = fused_feat.mean(dim=[2, 3])  # Global average pooling

        output = self.classifier(pooled_feat)
        return output


if __name__ == "__main__":

    # Test simulation
    B, C, H, W = 16, 3, 224, 224

    x1 = torch.randn(B, C, H, W)  
    x2 = torch.randn(B, C, H, W)  
    x3 = torch.randn(B, C, H, W)  
    x4 = torch.randn(B, C, H, W)  

    # Assume classification task has 3 classes
    num_classes = 3

    # Instantiate 4-input model
    model = ConvNeXt_T4(num_classes=3, share_backbone=True)

    # Forward pass
    output = model(x1, x2, x3, x4)

    # Check output shape: Expected [16, 3]
    print("Output shape:", output.shape)  