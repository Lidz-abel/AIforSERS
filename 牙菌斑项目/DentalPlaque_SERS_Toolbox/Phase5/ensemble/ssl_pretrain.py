"""Fold-local self-supervised pretraining for Raman spectrum encoders."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hetero_mil_model import SpectrumEncoder


class PatientBalancedSSLDataset(Dataset):
    """Equal spectra per patient with two physically plausible corrupted views."""

    def __init__(self, data: dict, patient_ids: np.ndarray, config: dict, seed: int):
        self.X = data["X"]
        self.patient_index = data["patient_index"]
        self.patient_ids = np.asarray(patient_ids, dtype=np.int64)
        self.config = config["pretraining"]
        self.seed = int(seed)
        self.epoch = 0
        self.per_patient = int(self.config["spectra_per_patient_per_epoch"])
        self.indices = {int(pid): np.flatnonzero(self.patient_index == pid) for pid in self.patient_ids}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.patient_ids) * self.per_patient

    @staticmethod
    def _shift_with_edges(spectrum: np.ndarray, shift: int) -> np.ndarray:
        shifted = spectrum.copy()
        if shift > 0:
            shifted[shift:] = spectrum[:-shift]
            shifted[:shift] = spectrum[0]
        elif shift < 0:
            width = -shift
            shifted[:-width] = spectrum[width:]
            shifted[-width:] = spectrum[-1]
        return shifted

    def _view(self, spectrum: np.ndarray, rng: np.random.RandomState):
        max_shift = int(self.config["max_wavenumber_shift"])
        shift = rng.randint(-max_shift, max_shift + 1) if max_shift else 0
        target = self._shift_with_edges(spectrum, shift)
        target = target * rng.uniform(0.95, 1.05)
        corrupted = target + rng.normal(0.0, float(self.config["gaussian_noise_std"]), target.shape)
        mask = np.zeros(len(target), dtype=bool)
        requested = max(1, int(round(float(self.config["mask_fraction"]) * len(target))))
        while int(mask.sum()) < requested:
            width = rng.randint(int(self.config["mask_width_min"]), int(self.config["mask_width_max"]) + 1)
            start = rng.randint(0, max(1, len(target) - width + 1))
            mask[start : start + width] = True
        corrupted[mask] = 0.0
        return corrupted.astype(np.float32), target.astype(np.float32), mask

    def __getitem__(self, index: int):
        patient_position = index % len(self.patient_ids)
        pid = int(self.patient_ids[patient_position])
        rng = np.random.RandomState(self.seed + 1_000_003 * self.epoch + 9_176 * index)
        spectrum_index = rng.choice(self.indices[pid])
        spectrum = self.X[spectrum_index]
        view1, target1, mask1 = self._view(spectrum, rng)
        view2, target2, mask2 = self._view(spectrum, rng)
        return (
            torch.from_numpy(view1).unsqueeze(0),
            torch.from_numpy(target1).unsqueeze(0),
            torch.from_numpy(mask1).unsqueeze(0),
            torch.from_numpy(view2).unsqueeze(0),
            torch.from_numpy(target2).unsqueeze(0),
            torch.from_numpy(mask2).unsqueeze(0),
            torch.tensor(pid, dtype=torch.long),
        )


class SpectralSSLModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_config = config["model"]
        pretrain = config["pretraining"]
        self.encoder = SpectrumEncoder(model_config)
        hidden = int(model_config["base_channels"]) * 2
        self.reconstruction = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, 1, 1),
        )
        embedding = int(model_config["embedding_dim"])
        self.projector = nn.Sequential(
            nn.Linear(embedding, embedding),
            nn.GELU(),
            nn.Linear(embedding, int(pretrain["projection_dim"])),
        )

    def forward(self, spectrum: torch.Tensor):
        feature_map = torch.cat([branch(spectrum) for branch in self.encoder.branches], dim=1)
        feature_map = self.encoder.residual(self.encoder.projection(feature_map))
        reconstruction = self.reconstruction(feature_map)
        embedding = self.encoder.output(self.encoder.pool(feature_map).squeeze(-1))
        projection = F.normalize(self.projector(embedding), dim=1)
        return reconstruction, projection


def patient_masked_nt_xent(z1, z2, patient_ids, temperature: float):
    logits12 = z1 @ z2.T / temperature
    logits21 = z2 @ z1.T / temperature
    same_patient = patient_ids[:, None].eq(patient_ids[None, :])
    diagonal = torch.eye(len(patient_ids), dtype=torch.bool, device=patient_ids.device)
    false_negative = same_patient & ~diagonal
    logits12 = logits12.masked_fill(false_negative, -1e9)
    logits21 = logits21.masked_fill(false_negative, -1e9)
    target = torch.arange(len(patient_ids), device=patient_ids.device)
    return 0.5 * (F.cross_entropy(logits12, target) + F.cross_entropy(logits21, target))


def reconstruction_loss(reconstruction, target, mask, full_weight: float):
    squared = (reconstruction - target).square()
    masked = (squared * mask).sum() / mask.sum().clamp_min(1)
    return masked + full_weight * squared.mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pretrain_encoder(
    data: dict,
    patient_ids: np.ndarray,
    config: dict,
    seed: int,
    device: torch.device,
    checkpoint: Path,
    metadata_path: Path,
    max_epochs: int | None = None,
) -> Path:
    if checkpoint.exists() and metadata_path.exists() and max_epochs is None:
        print(f"skip completed SSL checkpoint={checkpoint.name}", flush=True)
        return checkpoint
    set_seed(seed)
    dataset = PatientBalancedSSLDataset(data, patient_ids, config, seed)
    generator = torch.Generator().manual_seed(seed + 101)
    loader = DataLoader(
        dataset,
        batch_size=int(config["pretraining"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    model = SpectralSSLModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["pretraining"]["learning_rate"]),
        weight_decay=float(config["pretraining"]["weight_decay"]),
    )
    epochs = int(max_epochs or config["pretraining"]["epochs"])
    history = []
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch)
        model.train()
        total, reconstruction_total, contrastive_total, count = 0.0, 0.0, 0.0, 0
        for view1, target1, mask1, view2, target2, mask2, patient_id in loader:
            view1, target1, mask1 = view1.to(device), target1.to(device), mask1.to(device)
            view2, target2, mask2 = view2.to(device), target2.to(device), mask2.to(device)
            patient_id = patient_id.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction1, projection1 = model(view1)
            reconstruction2, projection2 = model(view2)
            reconstruction = 0.5 * (
                reconstruction_loss(
                    reconstruction1, target1, mask1, float(config["pretraining"]["full_reconstruction_weight"])
                )
                + reconstruction_loss(
                    reconstruction2, target2, mask2, float(config["pretraining"]["full_reconstruction_weight"])
                )
            )
            contrastive = patient_masked_nt_xent(
                projection1, projection2, patient_id, float(config["pretraining"]["contrastive_temperature"])
            )
            loss = reconstruction + float(config["pretraining"]["contrastive_weight"]) * contrastive
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["pretraining"]["gradient_clip_norm"]))
            optimizer.step()
            batch = len(patient_id)
            total += float(loss.item()) * batch
            reconstruction_total += float(reconstruction.item()) * batch
            contrastive_total += float(contrastive.item()) * batch
            count += batch
        row = {
            "epoch": epoch,
            "loss": total / count,
            "reconstruction_loss": reconstruction_total / count,
            "contrastive_loss": contrastive_total / count,
        }
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"ssl_epoch={epoch:03d}/{epochs} loss={row['loss']:.5f} "
                f"recon={row['reconstruction_loss']:.5f} contrast={row['contrastive_loss']:.5f}",
                flush=True,
            )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder_state": model.encoder.state_dict(), "seed": seed, "patient_ids": patient_ids.tolist()}, checkpoint)
    metadata_path.write_text(
        json.dumps({"seed": seed, "patient_ids": patient_ids.tolist(), "epochs": epochs, "history": history}, indent=2),
        encoding="utf-8",
    )
    print(f"SSL complete checkpoint={checkpoint}", flush=True)
    return checkpoint


def load_encoder_state(checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return payload["encoder_state"]
