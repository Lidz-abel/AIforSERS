"""Multi-scale AttentionMIL with a binary heteroscedastic output head."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, groups: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(min(groups, out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, 3, groups),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(groups, channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class SpectrumEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        kernels = list(cfg["kernel_sizes"])
        channels = int(cfg["base_channels"])
        groups = int(cfg["group_norm_groups"])
        hidden = channels * 2
        in_channels = int(cfg.get("in_channels", 1))
        self.branches = nn.ModuleList([ConvNormAct(in_channels, channels, k, groups) for k in kernels])
        self.projection = ConvNormAct(channels * len(kernels), hidden, 1, groups)
        self.residual = nn.Sequential(*[ResidualBlock(hidden, groups) for _ in range(int(cfg["n_res_blocks"]))])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Sequential(
            nn.Dropout(float(cfg["dropout_rate"])),
            nn.Linear(hidden, int(cfg["embedding_dim"])),
            nn.GELU(),
            nn.LayerNorm(int(cfg["embedding_dim"])),
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        features = torch.cat([branch(spectra) for branch in self.branches], dim=1)
        features = self.residual(self.projection(features))
        return self.output(self.pool(features).squeeze(-1))


class GatedAttention(nn.Module):
    def __init__(self, embedding_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.v = nn.Linear(embedding_dim, hidden)
        self.u = nn.Linear(embedding_dim, hidden)
        self.w = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.w(self.dropout(torch.tanh(self.v(embeddings)) * torch.sigmoid(self.u(embeddings))))
        weights = torch.softmax(scores.squeeze(-1), dim=-1)
        return torch.sum(embeddings * weights.unsqueeze(-1), dim=-2), weights


class HeteroscedasticMCSSMIL(nn.Module):
    """Input [B, M, K, 1, L], output one mu/log-variance pair per bag."""

    def __init__(self, cfg: dict):
        super().__init__()
        embedding_dim = int(cfg["embedding_dim"])
        dropout = float(cfg["dropout_rate"])
        self.encoder = SpectrumEncoder(cfg)
        self.attention = GatedAttention(embedding_dim, int(cfg["attention_hidden"]), dropout)
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(embedding_dim, 1)
        self.log_var_head = nn.Linear(embedding_dim, 1)
        self.log_var_min = float(cfg["log_var_min"])
        self.log_var_max = float(cfg["log_var_max"])

    def forward(self, patient_bags: torch.Tensor, return_attention: bool = False):
        batch, n_bags, bag_size, channels, length = patient_bags.shape
        flat = patient_bags.reshape(batch * n_bags * bag_size, channels, length)
        embeddings = self.encoder(flat).reshape(batch * n_bags, bag_size, -1)
        attended, weights = self.attention(embeddings)
        mean_pooled = embeddings.mean(dim=1)
        fused = self.fusion(torch.cat([attended, mean_pooled], dim=-1))
        mu = self.mu_head(fused).reshape(batch, n_bags)
        log_var = self.log_var_head(fused).reshape(batch, n_bags)
        log_var = torch.clamp(log_var, self.log_var_min, self.log_var_max)
        if return_attention:
            return mu, log_var, weights.reshape(batch, n_bags, bag_size)
        return mu, log_var
