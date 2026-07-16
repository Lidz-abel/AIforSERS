"""Generate leakage-safe OOF and historical-holdout explanations per architecture."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import pairwise_distances

TOOLBOX = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLBOX / "Phase4" / "mcss_hetero"))
sys.path.insert(0, str(TOOLBOX / "Phase5"))

from experiment_utils import load_config, resolve
from hetero_mil_model import HeteroscedasticMCSSMIL
from patient_mcss_dataset import load_dataset, patient_labels


def channels(array: np.ndarray) -> np.ndarray:
    return array[:, None, :] if array.ndim == 2 else array


def sample_bags(data: dict, pid: int, count: int, bag_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.flatnonzero(data["patient_index"] == pid)
    rng = np.random.RandomState(seed)
    selected, bags = [], []
    for _ in range(count):
        indices = rng.choice(candidates, bag_size, replace=len(candidates) < bag_size)
        selected.append(indices)
        bags.append(channels(data["X"][indices]))
    return np.asarray(bags, dtype=np.float32), np.asarray(selected, dtype=int)


def load_model(config: dict, checkpoint: Path, device: torch.device) -> HeteroscedasticMCSSMIL:
    model = HeteroscedasticMCSSMIL(config["model"]).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def model_probability(model, bags: torch.Tensor) -> np.ndarray:
    mu, _ = model(bags.unsqueeze(1))
    return torch.sigmoid(mu).mean(dim=1).cpu().numpy()


@torch.no_grad()
def model_embedding(model, bags: torch.Tensor) -> np.ndarray:
    n_bags, bag_size, n_channels, length = bags.shape
    flat = bags.reshape(n_bags * bag_size, n_channels, length)
    embedding = model.encoder(flat).reshape(n_bags, bag_size, -1)
    attended, _ = model.attention(embedding)
    mean_pooled = embedding.mean(dim=1)
    fused = model.fusion(torch.cat([attended, mean_pooled], dim=-1))
    return fused.mean(dim=0).cpu().numpy()


def expected_integrated_gradients(
    model,
    bags: np.ndarray,
    baseline_spectra: np.ndarray,
    steps: int,
    step_batch: int,
    device,
):
    signed = np.zeros(bags.shape[2:], dtype=np.float64)
    completeness = []
    attention_entropy = []
    probabilities = []
    evaluations = 0
    for bag in bags:
        x = torch.as_tensor(bag, dtype=torch.float32, device=device)
        with torch.no_grad():
            mu, _, attention = model(x[None, None], return_attention=True)
            probability_x = float(torch.sigmoid(mu).mean().item())
            weights = attention.flatten()
            entropy = -(weights * torch.log(weights.clamp_min(1e-10))).sum() / math.log(len(weights))
            attention_entropy.append(float(entropy.item()))
            probabilities.append(probability_x)
        for baseline_spectrum in baseline_spectra:
            baseline = torch.as_tensor(baseline_spectrum, dtype=torch.float32, device=device)
            baseline = baseline.unsqueeze(0).expand_as(x)
            delta = x - baseline
            gradient_sum = torch.zeros_like(x)
            for start in range(0, steps, step_batch):
                stop = min(steps, start + step_batch)
                alpha = torch.arange(start + 1, stop + 1, device=device, dtype=x.dtype) / steps
                interpolated = baseline[None] + alpha[:, None, None, None] * delta[None]
                interpolated = interpolated.detach().requires_grad_(True)
                mu, _ = model(interpolated.unsqueeze(1))
                probability = torch.sigmoid(mu).mean(dim=1)
                gradient = torch.autograd.grad(probability.sum(), interpolated)[0]
                gradient_sum += gradient.sum(dim=0)
            attribution = delta * gradient_sum / steps
            attribution = attribution.mean(dim=0)
            signed += attribution.detach().cpu().numpy()
            with torch.no_grad():
                probability_baseline = float(torch.sigmoid(model(baseline[None, None])[0]).mean().item())
            completeness.append(abs(float(attribution.sum().item()) - (probability_x - probability_baseline)))
            evaluations += 1
    signed /= max(evaluations, 1)
    return signed, float(np.mean(probabilities)), float(np.mean(attention_entropy)), float(np.mean(completeness))


@torch.no_grad()
def faithfulness(
    model,
    bags: np.ndarray,
    baseline_mean: np.ndarray,
    attribution_abs: np.ndarray,
    deletion_fraction: float,
    random_masks: int,
    seed: int,
    device,
):
    tensor = torch.as_tensor(bags, dtype=torch.float32, device=device)
    original = float(model_probability(model, tensor).mean())
    predicted_positive = original >= 0.5
    count = max(1, int(round(bags.shape[-1] * deletion_fraction)))
    top_indices = np.argsort(attribution_abs)[-count:]

    def confidence_drop(indices):
        masked = tensor.clone()
        replacement = torch.as_tensor(baseline_mean[:, indices], dtype=tensor.dtype, device=device)
        masked[:, :, :, indices] = replacement[None, None]
        probability = float(model_probability(model, masked).mean())
        return original - probability if predicted_positive else probability - original

    top_drop = confidence_drop(top_indices)
    rng = np.random.RandomState(seed)
    random_drop = [confidence_drop(rng.choice(bags.shape[-1], count, replace=False)) for _ in range(random_masks)]
    return original, float(top_drop), float(np.mean(random_drop))


def embedding_reference(model, data, train_ids, config, seed, device):
    embeddings, labels = [], []
    for pid in train_ids:
        bags, _ = sample_bags(
            data,
            int(pid),
            int(config["ood"]["embedding_bags_per_patient"]),
            int(config["attribution"]["bag_size"]),
            seed + int(pid) * 97,
        )
        embeddings.append(model_embedding(model, torch.as_tensor(bags, device=device)))
        labels.append(int(np.unique(data["labels"][data["patient_index"] == pid])[0]))
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    distances = pairwise_distances(embeddings)
    np.fill_diagonal(distances, np.inf)
    k = min(int(config["ood"]["knn_neighbors"]), len(embeddings) - 1)
    train_knn = np.sort(distances, axis=1)[:, :k].mean(axis=1)
    return {
        "embeddings": embeddings,
        "labels": labels,
        "knn_mean": float(train_knn.mean()),
        "knn_std": float(train_knn.std() + 1e-8),
    }


def ood_score(embedding: np.ndarray, reference: dict, config: dict):
    distances = np.linalg.norm(reference["embeddings"] - embedding[None], axis=1)
    k = min(int(config["ood"]["knn_neighbors"]), len(distances))
    knn = float(np.sort(distances)[:k].mean())
    knn_z = (knn - reference["knn_mean"]) / reference["knn_std"]
    mahalanobis = []
    floor = float(config["ood"]["variance_floor"])
    for label in np.unique(reference["labels"]):
        group = reference["embeddings"][reference["labels"] == label]
        mean = group.mean(axis=0)
        variance = group.var(axis=0) + floor
        mahalanobis.append(float(np.sqrt(np.mean((embedding - mean) ** 2 / variance))))
    return knn_z, min(mahalanobis)


def explain_patients(
    model,
    data,
    patient_ids,
    train_ids,
    stage,
    fold,
    member,
    architecture,
    config,
    seed,
    device,
    records,
    signed_attributions,
    channel_attributions,
):
    train_indices = np.flatnonzero(np.isin(data["patient_index"], train_ids))
    rng = np.random.RandomState(seed + 701)
    baseline_indices = rng.choice(
        train_indices, int(config["attribution"]["baseline_spectra"]), replace=False
    )
    baseline_spectra = channels(data["X"][baseline_indices])
    baseline_mean = channels(data["X"][train_indices]).mean(axis=0)
    reference = embedding_reference(model, data, train_ids, config, seed + 800, device)
    for pid in patient_ids:
        patient_seed = seed + int(pid) * 10_007
        bags, spectrum_indices = sample_bags(
            data,
            int(pid),
            int(config["attribution"]["mcss_bags_per_patient"]),
            int(config["attribution"]["bag_size"]),
            patient_seed,
        )
        signed_channels, probability, attention_entropy, completeness = expected_integrated_gradients(
            model,
            bags,
            baseline_spectra,
            int(config["attribution"]["integration_steps"]),
            int(config["attribution"]["gradient_step_batch"]),
            device,
        )
        channel_abs = np.abs(signed_channels)
        channel_abs /= np.maximum(channel_abs.sum(axis=1, keepdims=True), 1e-12)
        unified_signed = np.mean(
            signed_channels / np.maximum(np.abs(signed_channels).sum(axis=1, keepdims=True), 1e-12), axis=0
        )
        unified_abs = channel_abs.mean(axis=0)
        unified_abs /= max(unified_abs.sum(), 1e-12)
        _, top_drop, random_drop = faithfulness(
            model,
            bags,
            baseline_mean,
            unified_abs,
            float(config["attribution"]["deletion_fraction"]),
            int(config["attribution"]["random_masks"]),
            patient_seed + 99,
            device,
        )
        embedding = model_embedding(model, torch.as_tensor(bags, device=device))
        knn_z, mahalanobis = ood_score(embedding, reference, config)
        label = int(np.unique(data["labels"][data["patient_index"] == pid])[0])
        record_index = len(signed_attributions)
        signed_attributions.append(unified_signed.astype(np.float32))
        channel_attributions.append(signed_channels.astype(np.float32))
        records.append({
            "record_index": record_index,
            "architecture": architecture,
            "stage": stage,
            "fold": fold,
            "member": member,
            "patient_id": int(pid),
            "true_label": label,
            "model_probability": probability,
            "attention_entropy": attention_entropy,
            "ig_completeness_error": completeness,
            "top_deletion_drop": top_drop,
            "random_deletion_drop": random_drop,
            "faithfulness_gain": top_drop - random_drop,
            "knn_ood_z": knn_z,
            "class_mahalanobis": mahalanobis,
            "sampled_spectrum_indices": json.dumps(spectrum_indices.tolist()),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["intensity", "dual_view"], required=True)
    parser.add_argument("--config", default="Phase5/configs/exp_005_clinical_explainability.yaml")
    parser.add_argument("--limit-patients", type=int, default=None)
    parser.add_argument("--limit-members", type=int, default=None)
    parser.add_argument("--smoke-output", type=str, default=None)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if args.smoke_output:
        config["paths"]["results_dir"] = args.smoke_output
        config["attribution"]["integration_steps"] = 2
        config["attribution"]["gradient_step_batch"] = 1
        config["attribution"]["baseline_spectra"] = 1
        config["attribution"]["mcss_bags_per_patient"] = 1
        config["attribution"]["random_masks"] = 1
        config["ood"]["embedding_bags_per_patient"] = 1
    architecture_config = config["architectures"][args.architecture]
    source_dir = resolve(architecture_config["result_dir"])
    data = load_dataset(resolve(architecture_config["dataset_dir"]))
    labels = patient_labels(data)
    model_config = yaml.safe_load((source_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = min(int(architecture_config["members"]), args.limit_members or 10_000)
    records, signed_attributions, channel_attributions = [], [], []
    base_seed = int(config["attribution"]["deterministic_seed"]) + (0 if args.architecture == "intensity" else 1_000_000)

    print(f"P5-05 architecture={args.architecture} device={device} members={members}", flush=True)
    for fold in range(5):
        split_result = json.loads((source_dir / f"oof_fold_{fold}_member_0.json").read_text(encoding="utf-8"))
        train_ids = np.asarray(split_result["train_patient_ids"], dtype=int)
        val_ids = np.asarray(split_result["val_patient_ids"], dtype=int)
        if args.limit_patients:
            val_ids = val_ids[: args.limit_patients]
        for member in range(members):
            seed = base_seed + fold * 10_000 + member * 1_000
            model = load_model(model_config, source_dir / f"oof_fold_{fold}_member_{member}.pt", device)
            explain_patients(
                model, data, val_ids, train_ids, "oof", fold, member, args.architecture,
                config, seed, device, records, signed_attributions, channel_attributions,
            )
            print(f"complete architecture={args.architecture} fold={fold} member={member}", flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_result = json.loads((source_dir / "final_results.json").read_text(encoding="utf-8"))
    development_ids = np.asarray(final_result["leakage_audit"]["development_patient_ids"], dtype=int) if "development_patient_ids" in final_result.get("leakage_audit", {}) else None
    if development_ids is None:
        locked = set(final_result["test"]["patient_ids"])
        development_ids = np.asarray([pid for pid in range(len(labels)) if pid not in locked], dtype=int)
    test_ids = np.asarray(final_result["test"]["patient_ids"], dtype=int)
    if args.limit_patients:
        test_ids = test_ids[: args.limit_patients]
    for member in range(members):
        seed = base_seed + 900_000 + member * 1_000
        model = load_model(model_config, source_dir / f"final_member_{member}.pt", device)
        explain_patients(
            model, data, test_ids, development_ids, "test", -1, member, args.architecture,
            config, seed, device, records, signed_attributions, channel_attributions,
        )
        print(f"complete architecture={args.architecture} test member={member}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_dir = resolve(config["paths"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_dir / f"member_explanations_{args.architecture}.csv", index=False)
    np.savez_compressed(
        output_dir / f"attributions_{args.architecture}.npz",
        signed=np.asarray(signed_attributions, dtype=np.float32),
        channels=np.asarray(channel_attributions, dtype=np.float32),
    )
    (output_dir / f"{args.architecture}.done").write_text("complete\n", encoding="utf-8")
    print(f"P5-05 {args.architecture} complete records={len(records)}", flush=True)


if __name__ == "__main__":
    main()
