"""Leakage-safe patient-level grouped MCSS dataset for Phase 4D."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_dataset(dataset_dir: str | Path) -> dict:
    data = np.load(Path(dataset_dir) / "spectra.npz", allow_pickle=True)
    return {
        "X": data["X_spectra"].astype(np.float32),
        "labels": data["labels"].astype(np.int64),
        "patient_index": data["patient_index"].astype(np.int64),
        "patient_uids": [str(x) for x in data["patient_uids"]],
    }


def patient_labels(data: dict) -> np.ndarray:
    patient_ids = np.unique(data["patient_index"])
    if not np.array_equal(patient_ids, np.arange(len(patient_ids))):
        raise ValueError("patient_index must be contiguous and zero based")
    result = np.empty(len(patient_ids), dtype=np.int64)
    for pid in patient_ids:
        values = np.unique(data["labels"][data["patient_index"] == pid])
        if len(values) != 1:
            raise ValueError(f"Patient {pid} has inconsistent labels: {values}")
        result[pid] = values[0]
    return result


class GroupedPatientMCSSDataset(Dataset):
    """One item contains several MCSS bags sampled from exactly one patient."""

    def __init__(
        self,
        data: dict,
        patient_ids: np.ndarray,
        bag_size: int,
        bags_per_group: int,
        groups_per_patient: int,
        seed: int,
        dynamic: bool,
        sample_with_replacement: bool = False,
        max_wavenumber_shift: int = 0,
        gaussian_noise_std: float = 0.0,
    ):
        self.X = data["X"]
        self.labels = data["labels"]
        self.patient_index = data["patient_index"]
        self.patient_ids = np.asarray(patient_ids, dtype=np.int64)
        self.bag_size = int(bag_size)
        self.bags_per_group = int(bags_per_group)
        self.groups_per_patient = int(groups_per_patient)
        self.seed = int(seed)
        self.dynamic = bool(dynamic)
        self.sample_with_replacement = bool(sample_with_replacement)
        self.max_wavenumber_shift = int(max_wavenumber_shift) if dynamic else 0
        self.gaussian_noise_std = float(gaussian_noise_std) if dynamic else 0.0
        self.epoch = 0

        self.patient_to_indices = {}
        self.patient_to_label = {}
        for pid in self.patient_ids:
            indices = np.flatnonzero(self.patient_index == pid)
            if len(indices) == 0:
                raise ValueError(f"Patient {pid} has no spectra")
            values = np.unique(self.labels[indices])
            if len(values) != 1:
                raise ValueError(f"Patient {pid} has inconsistent labels")
            self.patient_to_indices[int(pid)] = indices
            self.patient_to_label[int(pid)] = int(values[0])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.patient_ids) * self.groups_per_patient

    def _augment(self, spectra: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        spectra = spectra.copy()
        if self.max_wavenumber_shift > 0:
            shifts = rng.randint(-self.max_wavenumber_shift, self.max_wavenumber_shift + 1, len(spectra))
            for i, shift in enumerate(shifts):
                if shift > 0:
                    spectra[i, ..., shift:] = spectra[i, ..., :-shift]
                    spectra[i, ..., :shift] = spectra[i, ..., shift : shift + 1]
                elif shift < 0:
                    width = -shift
                    spectra[i, ..., :-width] = spectra[i, ..., width:]
                    spectra[i, ..., -width:] = spectra[i, ..., -width - 1 : -width]
        if self.gaussian_noise_std > 0:
            spectra += rng.normal(0.0, self.gaussian_noise_std, spectra.shape).astype(np.float32)
        return spectra

    def __getitem__(self, idx: int):
        patient_pos = idx % len(self.patient_ids)
        pid = int(self.patient_ids[patient_pos])
        seed = self.seed + 1_000_003 * self.epoch + 9_176 * idx
        rng = np.random.RandomState(seed)
        candidates = self.patient_to_indices[pid]
        replace = self.sample_with_replacement or len(candidates) < self.bag_size

        bags = []
        for _ in range(self.bags_per_group):
            selected = rng.choice(candidates, self.bag_size, replace=replace)
            bags.append(self._augment(self.X[selected], rng))
        bags = torch.from_numpy(np.stack(bags))
        if bags.ndim == 3:
            bags = bags.unsqueeze(2)
        elif bags.ndim != 4:
            raise ValueError(f"Expected [M,K,L] or [M,K,C,L] bags, got {tuple(bags.shape)}")
        return (
            bags,
            torch.tensor(self.patient_to_label[pid], dtype=torch.float32),
            torch.tensor(pid, dtype=torch.long),
        )
