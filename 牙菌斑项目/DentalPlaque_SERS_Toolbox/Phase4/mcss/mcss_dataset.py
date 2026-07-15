"""Phase 4C MCSS bag datasets.

MCSS is applied only after patient-level splitting.  Each returned item is a
bag of spectra from one patient:

    bag: [K, 1, L]
    label: patient label
    patient_id: integer patient index in the full dataset
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PatientSplit:
    train_patients: np.ndarray
    val_patients: np.ndarray
    test_patients: np.ndarray


def load_mcss_dataset(dataset_dir: str | Path) -> dict:
    dataset_dir = Path(dataset_dir)
    data = np.load(dataset_dir / "spectra.npz", allow_pickle=True)
    return {
        "X_spectra": data["X_spectra"].astype(np.float32),
        "labels": data["labels"].astype(np.int64),
        "patient_index": data["patient_index"].astype(np.int64),
        "patient_uids": list(data["patient_uids"]),
        "spectrum_ids": list(data["spectrum_ids"]),
    }


def patient_labels_from_spectra(labels: np.ndarray, patient_index: np.ndarray) -> np.ndarray:
    unique_patients = np.unique(patient_index)
    patient_labels = np.zeros(len(unique_patients), dtype=np.int64)
    for pid in unique_patients:
        mask = patient_index == pid
        values = np.unique(labels[mask])
        if len(values) != 1:
            raise ValueError(f"Patient {pid} has inconsistent spectrum labels: {values}")
        patient_labels[int(pid)] = int(values[0])
    return patient_labels


def make_patient_split(
    patient_labels: np.ndarray,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    stratify: bool = True,
) -> PatientSplit:
    from phase4b_utils import create_patient_split

    split = create_patient_split(
        patient_labels=patient_labels,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify=stratify,
    )
    return PatientSplit(
        train_patients=np.array(split["train_patients"], dtype=np.int64),
        val_patients=np.array(split["val_patients"], dtype=np.int64),
        test_patients=np.array(split["test_patients"], dtype=np.int64),
    )


def audit_patient_split(split: PatientSplit) -> dict:
    train = set(split.train_patients.tolist())
    val = set(split.val_patients.tolist())
    test = set(split.test_patients.tolist())
    overlaps = {
        "train_val": len(train & val),
        "train_test": len(train & test),
        "val_test": len(val & test),
    }
    if any(v != 0 for v in overlaps.values()):
        raise ValueError(f"Patient leakage detected: {overlaps}")
    return {
        "n_train_patients": len(train),
        "n_val_patients": len(val),
        "n_test_patients": len(test),
        "overlaps": overlaps,
    }


class MCSSBagDataset(Dataset):
    """Patient-balanced MCSS bag dataset.

    For training, use ``dynamic=True`` and call ``set_epoch(epoch)`` before
    each epoch.  For validation/test, use ``dynamic=False`` so that bags are
    fixed and evaluation is reproducible.
    """

    def __init__(
        self,
        X_spectra: np.ndarray,
        labels: np.ndarray,
        patient_index: np.ndarray,
        patient_ids: np.ndarray,
        bag_size: int,
        bags_per_patient: int,
        seed: int,
        dynamic: bool,
        sample_with_replacement: bool = False,
    ):
        if bag_size <= 0:
            raise ValueError("bag_size must be positive")
        if bags_per_patient <= 0:
            raise ValueError("bags_per_patient must be positive")

        self.X = X_spectra
        self.labels = labels
        self.patient_index = patient_index
        self.patient_ids = np.array(patient_ids, dtype=np.int64)
        self.bag_size = int(bag_size)
        self.bags_per_patient = int(bags_per_patient)
        self.seed = int(seed)
        self.dynamic = bool(dynamic)
        self.sample_with_replacement = bool(sample_with_replacement)
        self.epoch = 0

        self.patient_to_indices: dict[int, np.ndarray] = {}
        self.patient_to_label: dict[int, int] = {}
        for pid in self.patient_ids:
            idx = np.where(self.patient_index == pid)[0]
            if len(idx) == 0:
                raise ValueError(f"Patient {pid} has no spectra")
            values = np.unique(self.labels[idx])
            if len(values) != 1:
                raise ValueError(f"Patient {pid} has inconsistent labels: {values}")
            self.patient_to_indices[int(pid)] = idx.astype(np.int64)
            self.patient_to_label[int(pid)] = int(values[0])

        self._fixed_bags: list[tuple[int, np.ndarray]] | None = None
        if not self.dynamic:
            self._fixed_bags = self._build_fixed_bags()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.patient_ids) * self.bags_per_patient

    def _sample_indices(self, pid: int, rng: np.random.RandomState) -> np.ndarray:
        candidates = self.patient_to_indices[int(pid)]
        replace = self.sample_with_replacement or len(candidates) < self.bag_size
        return rng.choice(candidates, size=self.bag_size, replace=replace).astype(np.int64)

    def _build_fixed_bags(self) -> list[tuple[int, np.ndarray]]:
        rng = np.random.RandomState(self.seed)
        bags: list[tuple[int, np.ndarray]] = []
        for pid in self.patient_ids:
            for _ in range(self.bags_per_patient):
                bags.append((int(pid), self._sample_indices(int(pid), rng)))
        return bags

    def __getitem__(self, idx: int):
        if self._fixed_bags is not None:
            pid, spec_idx = self._fixed_bags[idx]
        else:
            patient_pos = idx % len(self.patient_ids)
            pid = int(self.patient_ids[patient_pos])
            rng_seed = self.seed + 1000003 * self.epoch + 9176 * idx
            rng = np.random.RandomState(rng_seed)
            spec_idx = self._sample_indices(pid, rng)

        bag = torch.as_tensor(self.X[spec_idx], dtype=torch.float32).unsqueeze(1)
        label = torch.tensor(self.patient_to_label[pid], dtype=torch.long)
        patient_id = torch.tensor(pid, dtype=torch.long)
        return bag, label, patient_id
