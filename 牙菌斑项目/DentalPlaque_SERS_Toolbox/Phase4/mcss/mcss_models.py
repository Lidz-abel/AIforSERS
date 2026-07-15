"""Phase 4C MCSS MIL models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, groups: int = 4):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        gn_groups = min(groups, out_ch)
        self.norm = nn.GroupNorm(gn_groups, out_ch)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, groups: int = 4):
        super().__init__()
        self.block1 = ConvBlock(channels, channels, kernel_size, groups)
        self.block2 = ConvBlock(channels, channels, kernel_size, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block2(self.block1(x))


class SpectrumFeatureEncoder(nn.Module):
    """Multi-scale 1D encoder returning one embedding per spectrum."""

    def __init__(
        self,
        in_channels: int = 1,
        kernel_sizes: list[int] | None = None,
        base_channels: int = 32,
        n_res_blocks: int = 2,
        embedding_dim: int = 96,
        dropout_rate: float = 0.35,
        group_norm_groups: int = 4,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 7, 15]

        self.branches = nn.ModuleList(
            [ConvBlock(in_channels, base_channels, k, group_norm_groups) for k in kernel_sizes]
        )
        merged_ch = len(kernel_sizes) * base_channels
        res_ch = base_channels * 2
        self.proj = ConvBlock(merged_ch, res_ch, 1, group_norm_groups)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(res_ch, 3, group_norm_groups) for _ in range(n_res_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(res_ch, embedding_dim)
        self.out_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.proj(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.out_norm(F.relu(x))


class GatedAttentionPooling(nn.Module):
    """Gated attention pooling for bag-level MIL."""

    def __init__(self, embedding_dim: int, attention_hidden: int = 64, dropout_rate: float = 0.0):
        super().__init__()
        self.att_v = nn.Linear(embedding_dim, attention_hidden)
        self.att_u = nn.Linear(embedding_dim, attention_hidden)
        self.att_w = nn.Linear(attention_hidden, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = torch.tanh(self.att_v(z))
        u = torch.sigmoid(self.att_u(z))
        scores = self.att_w(self.dropout(v * u)).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(z * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class MeanPooling(nn.Module):
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.full(
            (z.size(0), z.size(1)),
            fill_value=1.0 / z.size(1),
            dtype=z.dtype,
            device=z.device,
        )
        return z.mean(dim=1), weights


class MCSSMILNet(nn.Module):
    """Bag-level classifier.

    Input shape: [B, K, 1, L]
    Output logits: [B, n_classes]
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        kernel_sizes: list[int] | None = None,
        base_channels: int = 32,
        n_res_blocks: int = 2,
        embedding_dim: int = 96,
        dropout_rate: float = 0.35,
        group_norm_groups: int = 4,
        pooling: str = "gated_attention",
        attention_hidden: int = 64,
    ):
        super().__init__()
        self.encoder = SpectrumFeatureEncoder(
            in_channels=in_channels,
            kernel_sizes=kernel_sizes,
            base_channels=base_channels,
            n_res_blocks=n_res_blocks,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            group_norm_groups=group_norm_groups,
        )
        if pooling == "gated_attention":
            self.pooling = GatedAttentionPooling(embedding_dim, attention_hidden, dropout_rate)
        elif pooling == "mean":
            self.pooling = MeanPooling()
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim // 2, n_classes),
        )

    def forward(self, bag: torch.Tensor, return_attention: bool = False):
        bsz, bag_size, channels, length = bag.shape
        flat = bag.reshape(bsz * bag_size, channels, length)
        z = self.encoder(flat).reshape(bsz, bag_size, -1)
        pooled, weights = self.pooling(z)
        logits = self.classifier(pooled)
        if return_attention:
            return logits, weights
        return logits

    def mc_dropout_sample(self, bag: torch.Tensor, n_samples: int = 50) -> torch.Tensor:
        original_mode = self.training
        self.train()
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                samples.append(self.forward(bag).unsqueeze(0))
        self.train(original_mode)
        return torch.cat(samples, dim=0)
