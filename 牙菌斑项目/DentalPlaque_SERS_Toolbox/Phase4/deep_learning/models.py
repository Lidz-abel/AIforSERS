"""Phase 4A: CC-SERSNet v1 — Multi-scale 1D CNN with MC Dropout.

Architecture:
  Input [B, 1, 732]
  → 3 parallel Conv1D branches (k=3, 7, 15)
  → Concat → Residual Blocks ×2
  → Global Average Pooling
  → Dropout → Linear → logits [B, 2]

Uses GroupNorm (not BatchNorm) for clean MC Dropout posterior sampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv1d → GroupNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, groups: int = 4):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        # Use GroupNorm; fall back to LayerNorm if channels < groups
        gn_groups = min(groups, out_ch)
        self.norm = nn.GroupNorm(gn_groups, out_ch) if out_ch >= groups else nn.LayerNorm(out_ch)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if isinstance(self.norm, nn.LayerNorm):
            x = x.transpose(1, 2)  # [B, C, L] → [B, L, C]
            x = self.norm(x)
            x = x.transpose(1, 2)  # [B, L, C] → [B, C, L]
        else:
            x = self.norm(x)
        return self.activation(x)


class ResidualBlock(nn.Module):
    """Residual block: ConvBlock → ConvBlock + skip connection."""

    def __init__(self, channels: int, kernel_size: int = 3, groups: int = 4):
        super().__init__()
        self.block1 = ConvBlock(channels, channels, kernel_size, groups)
        self.block2 = ConvBlock(channels, channels, kernel_size, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.block1(x)
        x = self.block2(x)
        return x + residual


class MultiScaleEncoder(nn.Module):
    """Multi-scale 1D CNN encoder.

    Three parallel Conv1D branches with different kernel sizes,
    followed by residual blocks, GAP, dropout, and linear head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_wavenumber: int = 732,
        n_classes: int = 2,
        kernel_sizes: list[int] | None = None,
        base_channels: int = 32,
        n_res_blocks: int = 2,
        dropout_rate: float = 0.3,
        group_norm_groups: int = 4,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 7, 15]

        self.in_channels = in_channels
        self.n_wavenumber = n_wavenumber
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate

        # Multi-scale branches
        branch_out = base_channels
        self.branches = nn.ModuleList([
            ConvBlock(in_channels, branch_out, k, group_norm_groups)
            for k in kernel_sizes
        ])

        # After concat: 3 × branch_out channels
        merged_ch = len(kernel_sizes) * branch_out

        # Projection to base_channels*2 for res blocks
        self.proj = ConvBlock(merged_ch, base_channels * 2, 1, group_norm_groups)

        # Residual blocks
        res_ch = base_channels * 2
        self.res_blocks = nn.ModuleList([
            ResidualBlock(res_ch, 3, group_norm_groups)
            for _ in range(n_res_blocks)
        ])

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifier head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(res_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits [B, n_classes]."""
        # Multi-scale branches
        branch_outs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outs, dim=1)  # [B, 3*base_ch, L]

        # Project & residual blocks
        x = self.proj(x)
        for res_block in self.res_blocks:
            x = res_block(x)

        # GAP + classifier
        x = self.gap(x).squeeze(-1)  # [B, res_ch]
        x = self.dropout(x)
        x = self.fc(x)  # [B, n_classes]

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities [B, n_classes]."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def mc_dropout_sample(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> torch.Tensor:
        """MC Dropout posterior samples.

        Keeps dropout active (model.train()) during T forward passes,
        then restores the original mode.  Returns logits [T, B, n_classes].
        """
        original_mode = self.training  # save current mode
        self.train()  # Enable dropout
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                samples.append(logits.unsqueeze(0))
        self.train(original_mode)  # restore original mode
        return torch.cat(samples, dim=0)  # [T, B, n_classes]

    def mc_predict_proba(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout: return (mean_probs [B, n_classes], samples [T, B, n_classes]).

        mean_probs is the averaged softmax over T MC samples.
        """
        mc_logits = self.mc_dropout_sample(x, n_samples)
        mc_probs = F.softmax(mc_logits, dim=-1)  # [T, B, n_classes]
        mean_probs = mc_probs.mean(dim=0)  # [B, n_classes]
        return mean_probs, mc_probs
