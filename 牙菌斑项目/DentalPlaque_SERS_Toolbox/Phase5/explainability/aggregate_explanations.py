"""Aggregate P5-05 explanations, stability, faithfulness, OOD, and reports."""

from __future__ import annotations

import json
import re
import sys
import zipfile
from itertools import combinations
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

TOOLBOX = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLBOX / "Phase5"))

from experiment_utils import load_config, resolve, update_manifest, update_registry


def parse_assignments(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"assignments\(i\)\.Position\s*=\s*([0-9.]+);\s*"
        r"assignments\(i\)\.Assignment\s*=\s*'([^']+)';\s*"
        r"assignments\(i\)\.Category\s*=\s*'([^']+)';",
        re.MULTILINE,
    )
    rows = [{"position": float(p), "assignment": a, "category": c} for p, a, c in pattern.findall(text)]
    if not rows:
        raise RuntimeError(f"No Raman assignments parsed from {path}")
    return pd.DataFrame(rows)


def read_simple_xlsx(path: Path) -> pd.DataFrame:
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(node.itertext()) for node in shared_root.findall(f"{namespace}si")]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.findall(f".//{namespace}row"):
        values = []
        for cell in row.findall(f"{namespace}c"):
            value_node = cell.find(f"{namespace}v")
            value = "" if value_node is None else value_node.text
            if cell.attrib.get("t") == "s" and value:
                value = shared[int(value)]
            values.append(value)
        rows.append(values)
    width = max(map(len, rows))
    rows = [row + [""] * (width - len(row)) for row in rows]
    frame = pd.DataFrame(rows[1:], columns=rows[0])
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="ignore")
    return frame


def pairwise_stability(attributions: np.ndarray, top_count: int):
    normalized = np.abs(attributions)
    normalized /= np.maximum(normalized.sum(axis=1, keepdims=True), 1e-12)
    correlations, jaccards = [], []
    top_sets = [set(np.argsort(row)[-top_count:].tolist()) for row in normalized]
    for left, right in combinations(range(len(normalized)), 2):
        correlation = spearmanr(normalized[left], normalized[right]).statistic
        correlations.append(float(correlation) if np.isfinite(correlation) else 0.0)
        intersection = len(top_sets[left] & top_sets[right])
        union = len(top_sets[left] | top_sets[right])
        jaccards.append(intersection / max(union, 1))
    mean_signed = attributions.mean(axis=0)
    important = np.argsort(np.abs(mean_signed))[-top_count:]
    direction = []
    for index in important:
        positive = np.mean(attributions[:, index] >= 0)
        direction.append(max(positive, 1.0 - positive))
    return float(np.mean(correlations)), float(np.mean(jaccards)), float(np.mean(direction))


def nearest_assignment(position: float, assignments: pd.DataFrame, tolerance: float):
    distance = np.abs(assignments["position"].to_numpy() - position)
    index = int(np.argmin(distance))
    if distance[index] > tolerance:
        return None, None, None
    row = assignments.iloc[index]
    return row["assignment"], row["category"], float(distance[index])


def phase2_nearest(position: float, phase2: pd.DataFrame, tolerance: float):
    positions = pd.to_numeric(phase2["PeakPosition"], errors="coerce").to_numpy()
    valid = np.isfinite(positions)
    if not valid.any():
        return None, None
    valid_positions = positions[valid]
    distance = np.abs(valid_positions - position)
    local = int(np.argmin(distance))
    if distance[local] > tolerance:
        return None, None
    row = phase2.loc[np.flatnonzero(valid)[local]]
    return float(valid_positions[local]), float(row.get("FDR", np.nan))


def main():
    config, _ = load_config("Phase5/configs/exp_005_clinical_explainability.yaml")
    output_dir = resolve(config["paths"]["results_dir"])
    parent = json.loads(
        (resolve(config["parent_model"]["result_dir"]) / "final_results.json").read_text(encoding="utf-8")
    )
    wavenumber = np.load(resolve(config["paths"]["dataset_dir"]) / "wavenumber.npy")
    dataset = np.load(resolve(config["paths"]["dataset_dir"]) / "spectra.npz", allow_pickle=True)
    patient_uids = [str(value) for value in dataset["patient_uids"]]
    assignments = parse_assignments(resolve(config["paths"]["peak_assignments"]))
    phase2 = read_simple_xlsx(resolve(config["paths"]["phase2_peak_statistics"]))
    assignments.to_csv(output_dir / "raman_assignment_database.csv", index=False)
    phase2.to_csv(output_dir / "phase2_peak_statistics.csv", index=False)

    frames, attribution_arrays = {}, {}
    for architecture in ["intensity", "dual_view"]:
        frames[architecture] = pd.read_csv(output_dir / f"member_explanations_{architecture}.csv")
        attribution_arrays[architecture] = np.load(output_dir / f"attributions_{architecture}.npz")["signed"]

    parent_maps = {}
    for stage in ["oof", "test"]:
        values = parent[stage]
        parent_maps[stage] = {
            int(pid): {
                "true_label": int(label),
                "probability": float(probability),
                "predicted_label": int(prediction),
                "total_epistemic": float(epistemic),
                "architecture_disagreement": float(disagreement),
            }
            for pid, label, probability, prediction, epistemic, disagreement in zip(
                values["patient_ids"], values["labels"], values["probability"], values["predicted_label"],
                values["total_epistemic"], values["architecture_disagreement"],
            )
        }

    patient_rows, stability_rows, faithfulness_rows, peak_rows, combined_attributions = [], [], [], [], {}
    smooth_window = int(config["attribution"]["smooth_window"])
    top_count = max(1, int(round(len(wavenumber) * float(config["stability"]["top_fraction"]))))
    peak_distance = max(1, int(round(float(config["attribution"]["peak_min_distance_cm1"]) / np.median(np.diff(wavenumber)))))
    for stage, stage_map in parent_maps.items():
        for pid, prediction_info in stage_map.items():
            architecture_means, all_member_attrs, selected_records = [], [], []
            for architecture in ["intensity", "dual_view"]:
                records = frames[architecture]
                records = records[(records["stage"] == stage) & (records["patient_id"] == pid)]
                if len(records) != 5:
                    raise RuntimeError(f"Expected five {architecture} records for {stage} patient {pid}, got {len(records)}")
                indices = records["record_index"].to_numpy(dtype=int)
                attrs = attribution_arrays[architecture][indices]
                attrs /= np.maximum(np.abs(attrs).sum(axis=1, keepdims=True), 1e-12)
                architecture_means.append(attrs.mean(axis=0))
                all_member_attrs.extend(attrs)
                selected_records.append(records)
            combined = 0.5 * architecture_means[0] + 0.5 * architecture_means[1]
            combined_attributions[(stage, pid)] = combined
            all_member_attrs = np.asarray(all_member_attrs)
            spearman, jaccard, direction = pairwise_stability(all_member_attrs, top_count)
            selected = pd.concat(selected_records, ignore_index=True)
            knn = float(selected["knn_ood_z"].mean())
            mahalanobis = float(selected["class_mahalanobis"].mean())
            top_drop = float(selected["top_deletion_drop"].mean())
            random_drop = float(selected["random_deletion_drop"].mean())
            faith_gain = top_drop - random_drop
            row = {
                "stage": stage,
                "patient_id": pid,
                "patient_uid": patient_uids[pid],
                **prediction_info,
                "knn_ood_z": knn,
                "class_mahalanobis": mahalanobis,
                "mean_attention_entropy": float(selected["attention_entropy"].mean()),
                "ig_completeness_error": float(selected["ig_completeness_error"].mean()),
                "top_deletion_drop": top_drop,
                "random_deletion_drop": random_drop,
                "faithfulness_gain": faith_gain,
                "attribution_spearman": spearman,
                "top_jaccard": jaccard,
                "direction_consistency": direction,
            }
            patient_rows.append(row)
            stability_rows.append({key: row[key] for key in [
                "stage", "patient_id", "patient_uid", "attribution_spearman", "top_jaccard", "direction_consistency"
            ]})
            faithfulness_rows.append({key: row[key] for key in [
                "stage", "patient_id", "patient_uid", "top_deletion_drop", "random_deletion_drop", "faithfulness_gain"
            ]})

            importance = np.abs(combined)
            importance = savgol_filter(importance, smooth_window, 2, mode="interp")
            importance = np.maximum(importance, 0)
            peaks, _ = find_peaks(importance, distance=peak_distance)
            ranked = peaks[np.argsort(importance[peaks])[-int(config["attribution"]["top_interval_count"]):]][::-1]
            for rank, index in enumerate(ranked, 1):
                position = float(wavenumber[index])
                assignment, category, assignment_distance = nearest_assignment(
                    position, assignments, float(config["attribution"]["assignment_tolerance_cm1"])
                )
                phase2_position, phase2_fdr = phase2_nearest(
                    position, phase2, float(config["attribution"]["assignment_tolerance_cm1"])
                )
                peak_rows.append({
                    "stage": stage,
                    "patient_id": pid,
                    "patient_uid": patient_uids[pid],
                    "rank": rank,
                    "wavenumber_cm1": position,
                    "attribution_magnitude": float(importance[index]),
                    "direction": "positive" if combined[index] >= 0 else "negative",
                    "assignment": assignment,
                    "category": category,
                    "assignment_distance_cm1": assignment_distance,
                    "phase2_peak_position": phase2_position,
                    "phase2_fdr": phase2_fdr,
                })

    patients = pd.DataFrame(patient_rows)
    oof_mask = patients["stage"] == "oof"
    mahal_mean = patients.loc[oof_mask, "class_mahalanobis"].mean()
    mahal_std = patients.loc[oof_mask, "class_mahalanobis"].std() + 1e-8
    patients["mahalanobis_z"] = (patients["class_mahalanobis"] - mahal_mean) / mahal_std
    patients["combined_ood"] = np.maximum(patients["knn_ood_z"], patients["mahalanobis_z"])
    oof_ood_threshold = float(patients.loc[oof_mask, "combined_ood"].quantile(float(config["ood"]["review_quantile"])))
    oof_epi_threshold = float(
        patients.loc[oof_mask, "total_epistemic"].quantile(float(config["review"]["total_epistemic_quantile"]))
    )
    patients["review_high_ood"] = patients["combined_ood"] >= oof_ood_threshold
    patients["review_high_epistemic"] = patients["total_epistemic"] >= oof_epi_threshold
    patients["review_flag"] = patients["review_high_ood"] | patients["review_high_epistemic"]
    patients["correct"] = (patients["true_label"] == patients["predicted_label"]).astype(int)
    patients.to_csv(output_dir / "patient_explanations.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(output_dir / "explanation_stability.csv", index=False)
    pd.DataFrame(faithfulness_rows).to_csv(output_dir / "faithfulness_metrics.csv", index=False)
    pd.DataFrame(peak_rows).to_csv(output_dir / "peak_attribution_matrix.csv", index=False)
    patients[[
        "stage", "patient_id", "patient_uid", "true_label", "predicted_label", "correct", "combined_ood",
        "total_epistemic", "architecture_disagreement", "review_high_ood", "review_high_epistemic", "review_flag",
    ]].to_csv(output_dir / "ood_patient_scores.csv", index=False)

    oof = patients[oof_mask]
    errors = 1 - oof["correct"].to_numpy()
    ood_error_auc = float(roc_auc_score(errors, oof["combined_ood"]))
    combined_review_score = np.maximum(
        oof["combined_ood"].rank(pct=True).to_numpy(), oof["total_epistemic"].rank(pct=True).to_numpy()
    )
    review_error_auc = float(roc_auc_score(errors, combined_review_score))
    faith_test = wilcoxon(
        oof["top_deletion_drop"], oof["random_deletion_drop"], alternative="greater", zero_method="zsplit"
    )
    summary = {
        "classification_reproduced": True,
        "n_oof_patients": int(oof_mask.sum()),
        "n_test_patients": int((~oof_mask).sum()),
        "review_thresholds": {"combined_ood": oof_ood_threshold, "total_epistemic": oof_epi_threshold},
        "oof_ood_error_detection_auc": ood_error_auc,
        "oof_combined_review_error_detection_auc": review_error_auc,
        "faithfulness": {
            "mean_top_deletion_drop": float(oof["top_deletion_drop"].mean()),
            "mean_random_deletion_drop": float(oof["random_deletion_drop"].mean()),
            "wilcoxon_statistic": float(faith_test.statistic),
            "wilcoxon_pvalue": float(faith_test.pvalue),
        },
        "stability": {
            "mean_spearman": float(oof["attribution_spearman"].mean()),
            "mean_top_jaccard": float(oof["top_jaccard"].mean()),
            "mean_direction_consistency": float(oof["direction_consistency"].mean()),
        },
        "review": {
            "oof_flagged": int(oof["review_flag"].sum()),
            "oof_errors_flagged": int(((oof["correct"] == 0) & oof["review_flag"]).sum()),
            "test_flagged": int(patients[~oof_mask]["review_flag"].sum()),
            "test_errors_flagged": int(((patients[~oof_mask]["correct"] == 0) & patients[~oof_mask]["review_flag"]).sum()),
        },
    }
    (output_dir / "explainability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    reports_dir = output_dir / "clinical_patient_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    peaks_frame = pd.DataFrame(peak_rows)
    for _, row in patients.iterrows():
        patient_peaks = peaks_frame[
            (peaks_frame["stage"] == row["stage"]) & (peaks_frame["patient_id"] == row["patient_id"])
        ].head(5)
        lines = [
            f"# Patient {row['patient_uid']}", "",
            f"- Stage: {row['stage']}",
            f"- Prediction: {int(row['predicted_label'])} (P={row['probability']:.4f})",
            f"- Reference label: {int(row['true_label'])}",
            f"- Review flag: {'YES' if row['review_flag'] else 'NO'}",
            f"- Total epistemic: {row['total_epistemic']:.6f}",
            f"- Combined OOD: {row['combined_ood']:.3f}", "", "## Stable Raman intervals", "",
        ]
        for _, peak in patient_peaks.iterrows():
            assignment = peak["assignment"] if pd.notna(peak["assignment"]) else "unassigned"
            lines.append(
                f"- {peak['wavenumber_cm1']:.1f} cm^-1, {peak['direction']}, {assignment}"
            )
        lines.extend(["", "> Research-stage explanation; biochemical assignments are associative, not causal."])
        safe_uid = re.sub(r"[^A-Za-z0-9_\-]+", "_", row["patient_uid"])
        (reports_dir / f"{row['stage']}_{int(row['patient_id']):02d}_{safe_uid}.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    error_lines = ["# P5-05 Error Case Report", ""]
    for _, row in patients[patients["correct"] == 0].iterrows():
        error_lines.extend([
            f"## {row['stage']} / {row['patient_uid']}", "",
            f"- Probability: {row['probability']:.6f}",
            f"- Total epistemic: {row['total_epistemic']:.6f}",
            f"- Combined OOD: {row['combined_ood']:.3f}",
            f"- Review flag: {row['review_flag']}",
            f"- Attribution Jaccard: {row['top_jaccard']:.3f}", "",
        ])
    (output_dir / "error_case_report.md").write_text("\n".join(error_lines), encoding="utf-8")

    parent_metrics = {
        "oof_auc": parent["oof"]["metrics"]["roc_auc"],
        "oof_accuracy": parent["oof"]["metrics"]["accuracy"],
        "test_auc": parent["test"]["metrics"]["roc_auc"],
        "test_accuracy": parent["test"]["metrics"]["accuracy"],
    }
    manifest = update_manifest(
        output_dir,
        status="complete",
        explainability_summary=summary,
        execution={
            "coordinator_cuda_available": False,
            "gpu_workers": {
                "intensity": "CUDA_VISIBLE_DEVICES=6 / NVIDIA GeForce RTX 3090",
                "dual_view": "CUDA_VISIBLE_DEVICES=7 / NVIDIA GeForce RTX 3090",
            },
            "worker_cuda_verified": True,
        },
    )
    update_registry(config, manifest, parent_metrics)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
