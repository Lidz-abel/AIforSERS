"""Phase 4A: Dataset loading with patient-level split audit.

Training unit: single spectrum (label inherited from patient).
Evaluation unit: patient (spectrum predictions aggregated).
No patient appears in more than one split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_phase4_dataset(cfg: dict) -> dict:
    """Load spectra.npz and split file.  Returns full data dict."""
    from baseline_utils import resolve_path

    dataset_dir = resolve_path(cfg["paths"]["dataset_dir"])
    splits_dir = resolve_path(cfg["paths"]["splits_dir"])

    data = np.load(dataset_dir / "spectra.npz", allow_pickle=True)
    with open(splits_dir / cfg["paths"]["split_file"], "r", encoding="utf-8") as f:
        splits = json.load(f)

    return {
        "X_spectra": data["X_spectra"],
        "X_raw_spectra": data["X_raw_spectra"],
        "labels": data["labels"],
        "patient_index": data["patient_index"],
        "patient_uids": list(data["patient_uids"]),
        "spectrum_ids": list(data["spectrum_ids"]),
        "splits": splits,
    }


def build_split_masks(patient_uids: list[str], splits: dict) -> dict[str, np.ndarray]:
    """Patient-level boolean masks.  0 overlap guaranteed."""
    uid_to_idx = {uid: i for i, uid in enumerate(patient_uids)}
    masks = {}
    for split_name in ["train", "val", "test"]:
        mask = np.zeros(len(patient_uids), dtype=bool)
        for uid in splits[f"{split_name}_patients"]:
            if uid in uid_to_idx:
                mask[uid_to_idx[uid]] = True
        masks[split_name] = mask
    return masks


def build_spectrum_masks(
    patient_index: np.ndarray, patient_masks: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Expand patient-level masks to spectrum-level."""
    spec_masks = {}
    for split_name, p_mask in patient_masks.items():
        spec_masks[split_name] = p_mask[patient_index]
    return spec_masks


def audit_splits(data: dict) -> None:
    """Print split audit and verify no leakage."""
    patient_masks = build_split_masks(data["patient_uids"], data["splits"])
    spec_masks = build_spectrum_masks(data["patient_index"], patient_masks)

    print("=" * 50)
    print("Phase 4A: Split Audit")
    print("=" * 50)

    for name in ["train", "val", "test"]:
        n_p = patient_masks[name].sum()
        n_s = spec_masks[name].sum()
        n_pos = data["labels"][spec_masks[name]].sum()
        n_neg = n_s - n_pos
        print(f"  {name:5s}: {n_p:2d} patients, {n_s:4d} spectra (pos={n_pos}, neg={n_neg})")

    # Leakage check
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = patient_masks[a] & patient_masks[b]
        assert overlap.sum() == 0, f"Patient leakage: {a} ∩ {b} = {overlap.sum()}"
    print("  Patient leakage: NONE (OK)")
    print()


class SpectrumDataset(Dataset):
    """Single-spectrum dataset for training.  Label = patient label.

    Args:
        X: [N, n_wavenumber] spectra.
        y: [N] labels.
        sample_weight: optional [N] per-sample weights for loss balancing.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        self.X = torch.as_tensor(X, dtype=torch.float32).unsqueeze(1)  # [N, 1, 732]
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.sample_weight = (
            torch.as_tensor(sample_weight, dtype=torch.float32)
            if sample_weight is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.sample_weight is not None:
            return self.X[idx], self.y[idx], self.sample_weight[idx]
        return self.X[idx], self.y[idx]
