import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyEnhancer(nn.Module):
    """Frequency domain enhancement module"""

    def __init__(self, seq_len: int, out_dim: int, k_bins: int = 64, log_amp: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.k_bins = min(k_bins, seq_len // 2 + 1)
        self.log_amp = log_amp
        self.proj = nn.Linear(self.k_bins, out_dim, bias=False)
        self.register_buffer("hann", torch.hann_window(seq_len), persistent=False)

    def forward(self, x):
        B, L, D = x.shape

        # Channel averaging
        xm = x.mean(dim=2)

        # Hann window
        if (not hasattr(self, "hann")) or self.hann is None or self.hann.shape[0] != L:
            hann = torch.hann_window(L, device=x.device, dtype=x.dtype)
        else:
            hann = self.hann.to(x.device, x.dtype)

        Xf = torch.fft.rfft(xm * hann, dim=1)
        mag = Xf.abs()

        # Take first k bins
        k = min(self.k_bins, mag.shape[1])
        mag = mag[:, :k]

        if self.log_amp:
            mag = torch.log1p(mag)

        # Zero padding if needed
        if k < self.k_bins:
            pad = (0, self.k_bins - k)
            mag = F.pad(mag, pad, mode='constant', value=0.0)

        ctx = self.proj(mag)
        ctx = ctx.unsqueeze(1).expand(B, L, -1)
        return ctx


class TemporalDecomposition(nn.Module):
    """Single-scale temporal decomposition"""

    def __init__(self, seq_len, kernel_size=25):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = (kernel_size - 1) // 2
        self.smoothing = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False
        )

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        trend = self.smoothing(x_permuted)
        trend = trend.permute(0, 2, 1)
        seasonal = x - trend
        return trend, seasonal


class MultiScaleDecomposition(nn.Module):
    """Multi-scale temporal decomposition"""

    def __init__(self, seq_len, kernel_sizes=[13, 25, 37]):
        super().__init__()
        self.decomposers = nn.ModuleList([
            TemporalDecomposition(seq_len, ks) for ks in kernel_sizes
        ])

    def forward(self, x):
        multi_trend = []
        multi_seasonal = []

        for decomposer in self.decomposers:
            trend, seasonal = decomposer(x)
            multi_trend.append(trend)
            multi_seasonal.append(seasonal)

        trend = torch.mean(torch.stack(multi_trend), dim=0)
        seasonal = torch.mean(torch.stack(multi_seasonal), dim=0)
        return trend, seasonal


class FeatureEncoder(nn.Module):
    """Feature encoding module"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_net(x)
        return x.permute(0, 2, 1)


class TemporalDecomposedAttention(nn.Module):
    """Temporal decomposed attention with frequency enhancement"""

    def __init__(self, feature_dim, num_heads=4, use_freq=True, seq_len=128,
                 freq_bins=64, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_freq = use_freq
        self.trend_dim = 8
        self.seasonal_dim = feature_dim - self.trend_dim

        # Trend branch
        self.trend_net = nn.Sequential(
            nn.Linear(feature_dim, self.trend_dim),
            nn.LayerNorm(self.trend_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.trend_dim, self.trend_dim)
        )

        # Seasonal branch
        self.seasonal_proj = nn.Linear(feature_dim, self.seasonal_dim, bias=False)

        if use_freq:
            self.freq_enhancer = FrequencyEnhancer(
                seq_len=seq_len, out_dim=self.seasonal_dim, k_bins=freq_bins, log_amp=True
            )
            self.freq_gate = nn.Sequential(
                nn.Linear(self.seasonal_dim * 2, self.seasonal_dim),
                nn.GELU(),
                nn.Linear(self.seasonal_dim, self.seasonal_dim),
                nn.Sigmoid()
            )

        self.seasonal_attention = nn.MultiheadAttention(
            embed_dim=self.seasonal_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        self.fusion = nn.Linear(self.trend_dim + self.seasonal_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _moving_avg(x, kernel_size=5):
        xT = x.permute(0, 2, 1)
        pad = kernel_size // 2
        trend = F.avg_pool1d(xT, kernel_size=kernel_size, stride=1, padding=pad)
        return trend.permute(0, 2, 1)

    def forward(self, x):
        residual = x
        B, L, _ = x.shape

        # Temporal decomposition
        trend_comp = self._moving_avg(x)
        seas_comp = x - trend_comp

        # Trend branch
        t = self.trend_net(trend_comp.reshape(-1, self.feature_dim)).reshape(B, L, self.trend_dim)

        # Seasonal branch
        s = self.seasonal_proj(seas_comp)

        if self.use_freq:
            fctx = self.freq_enhancer(seas_comp)
            gate = self.freq_gate(torch.cat([s, fctx], dim=-1))
            s = s + gate * fctx

        s_att, _ = self.seasonal_attention(s, s, s)

        out = torch.cat([t, s_att], dim=-1)
        out = self.dropout(self.fusion(out))
        return self.norm(residual + out)