import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pamap2_models.components import (
    MultiScaleDecomposition, TemporalDecomposition,
    FeatureEncoder, TemporalDecomposedAttention
)


class TDN_PAMAP2(nn.Module):
    """Temporal Decomposition Network for PAMAP2 HAR"""

    def __init__(self, window_size=128, input_channels=18, num_classes=12,
                 embed_dim=64, feature_dim=32, use_multi_scale=True):
        super().__init__()

        self.window_size = window_size
        self.input_channels = input_channels
        self.num_classes = num_classes

        # 1. Input projection
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_channels, embed_dim, 3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # 2. Temporal decomposition
        if use_multi_scale:
            self.decomposition = MultiScaleDecomposition(window_size)
        else:
            self.decomposition = TemporalDecomposition(window_size)

        # 3. Dual-branch feature encoders
        self.trend_encoder = FeatureEncoder(embed_dim, feature_dim, kernel_size=15)
        self.seasonal_encoder = FeatureEncoder(embed_dim, feature_dim, kernel_size=3)

        # 4. Temporal decomposed attention
        self.trend_attention = TemporalDecomposedAttention(
            feature_dim=feature_dim, num_heads=4,
            use_freq=False, seq_len=window_size,
            freq_bins=min(64, window_size // 2 + 1), dropout=0.1
        )
        self.seasonal_attention = TemporalDecomposedAttention(
            feature_dim=feature_dim, num_heads=4,
            use_freq=True, seq_len=window_size,
            freq_bins=min(64, window_size // 2 + 1), dropout=0.1
        )

        # 5. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        # Layer normalizations
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 1)
        batch_size = x.size(0)

        # 1. Input projection
        x_proj = x.permute(0, 2, 1)
        x_proj = self.input_projection(x_proj)
        x_proj = x_proj.permute(0, 2, 1)
        x_proj = self.norm1(x_proj)

        # 2. Temporal decomposition
        trend, seasonal = self.decomposition(x_proj)

        # 3. Dual-branch encoding
        trend_features = self.trend_encoder(trend)
        trend_features = self.norm2(trend_features)

        seasonal_features = self.seasonal_encoder(seasonal)
        seasonal_features = self.norm3(seasonal_features)

        # 4. Attention enhancement
        trend_attended = self.trend_attention(trend_features)
        seasonal_attended = self.seasonal_attention(seasonal_features)

        # 5. Temporal pooling
        trend_pool = F.adaptive_avg_pool1d(trend_features.permute(0, 2, 1), 1).view(batch_size, -1)
        seasonal_pool = F.adaptive_avg_pool1d(seasonal_features.permute(0, 2, 1), 1).view(batch_size, -1)
        trend_att_pool = F.adaptive_avg_pool1d(trend_attended.permute(0, 2, 1), 1).view(batch_size, -1)
        seasonal_att_pool = F.adaptive_avg_pool1d(seasonal_attended.permute(0, 2, 1), 1).view(batch_size, -1)

        # 6. Classification
        combined = torch.cat([trend_pool, seasonal_pool, trend_att_pool, seasonal_att_pool], dim=1)
        output = self.classifier(combined)

        return output

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)