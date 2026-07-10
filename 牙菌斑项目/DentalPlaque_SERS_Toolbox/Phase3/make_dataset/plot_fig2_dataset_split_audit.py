from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import ensure_dir, load_config, resolve_path, toolbox_root


COLORS = {
    "neg": "#2F5F9E",
    "pos": "#B43E3A",
    "neg_light": "#8FB3D9",
    "pos_light": "#D98F8B",
    "ink": "#182026",
    "text": "#52606D",
    "grid": "#CBD2D9",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    root = toolbox_root()
    dataset_dir = resolve_path(cfg["paths"]["dataset_dir"], root)
    splits_dir = resolve_path(cfg["paths"]["splits_dir"], root)
    patient_df = pd.read_csv(dataset_dir / "patient_metadata.csv", encoding="utf-8-sig")
    spectrum_df = pd.read_csv(dataset_dir / "spectrum_metadata.csv", encoding="utf-8-sig")
    with open(dataset_dir / "dataset_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(splits_dir / f"split_seed{cfg['seed']}.json", "r", encoding="utf-8") as f:
        split = json.load(f)
    return patient_df, spectrum_df, summary, split


def assign_split(patient_df: pd.DataFrame, spectrum_df: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_map = {}
    for split_name, key in [("train", "train_patients"), ("val", "val_patients"), ("test", "test_patients")]:
        for patient_uid in split[key]:
            if patient_uid in split_map:
                raise RuntimeError(f"Patient appears in multiple splits: {patient_uid}")
            split_map[patient_uid] = split_name

    patient_df = patient_df.copy()
    patient_df["split"] = patient_df["patient_uid"].map(split_map)
    if patient_df["split"].isna().any():
        missing = patient_df.loc[patient_df["split"].isna(), "patient_uid"].tolist()
        raise RuntimeError(f"Patients missing from split file: {missing[:5]}")

    spectrum_df = spectrum_df.copy()
    spectrum_df["split"] = spectrum_df["patient_uid"].map(split_map)
    if spectrum_df["split"].isna().any():
        missing = spectrum_df.loc[spectrum_df["split"].isna(), "patient_uid"].unique().tolist()
        raise RuntimeError(f"Spectra linked to patients missing from split file: {missing[:5]}")

    return patient_df, spectrum_df


def assert_metadata_consistency(patient_df: pd.DataFrame, spectrum_df: pd.DataFrame, summary: dict) -> None:
    if patient_df["patient_uid"].nunique() != int(summary["n_patients"]):
        raise RuntimeError("Patient count mismatch between metadata and summary.")
    if len(spectrum_df) != int(summary["n_spectra"]):
        raise RuntimeError("Spectrum count mismatch between metadata and summary.")
    patient_spectrum_counts = spectrum_df.groupby("patient_uid")["spectrum_id"].count()
    declared_counts = patient_df.set_index("patient_uid")["n_spectra"]
    if not patient_spectrum_counts.sort_index().equals(declared_counts.sort_index()):
        raise RuntimeError("n_spectra in patient metadata does not match spectrum metadata.")


def save_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    ensure_dir(out_dir)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}")


def add_bar_labels(ax: plt.Axes, xs: list[int] | np.ndarray, ys: list[int] | np.ndarray, pad: float) -> None:
    for x, y in zip(xs, ys):
        ax.text(x, y + pad, str(int(y)), ha="center", va="bottom", color=COLORS["ink"])


def add_stacked_segment_labels(ax: plt.Axes, values_by_label: dict[int, np.ndarray], min_height: int = 1) -> None:
    running = np.zeros(len(next(iter(values_by_label.values()))))
    for label, vals in values_by_label.items():
        for i, val in enumerate(vals):
            if int(val) >= min_height:
                ax.text(
                    i,
                    running[i] + val / 2,
                    str(int(val)),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                    weight="bold",
                )
        running += vals


def plot_fig2(patient_df: pd.DataFrame, spectrum_df: pd.DataFrame, summary: dict, split: dict, out_dir: Path) -> None:
    setup_style()
    split_order = ["train", "val", "test"]
    label_order = [0, 1]
    label_names = {0: "Negative", 1: "Positive"}
    label_colors = {0: COLORS["neg"], 1: COLORS["pos"]}
    label_light = {0: COLORS["neg_light"], 1: COLORS["pos_light"]}

    fig = plt.figure(figsize=(12.2, 7.8))
    gs = fig.add_gridspec(2, 2, wspace=0.26, hspace=0.42)
    fig.suptitle("Figure 2. Dataset Composition and Patient-Level Split Audit", y=0.975, fontsize=14, weight="bold")

    ax_a = fig.add_subplot(gs[0, 0])
    patient_counts = patient_df.groupby("label")["patient_uid"].nunique().reindex(label_order)
    x = np.arange(2)
    ax_a.bar(x, patient_counts.values, width=0.58, color=[label_colors[i] for i in label_order])
    ax_a.set_xticks(x, [label_names[i] for i in label_order])
    ax_a.set_ylabel("Patients")
    ax_a.set_title("A. Label Distribution at Patient Level")
    ax_a.set_ylim(0, patient_counts.max() * 1.28)
    add_bar_labels(ax_a, x, patient_counts.values, pad=0.8)
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.grid(axis="y", color=COLORS["grid"], alpha=0.35, linewidth=0.7)

    ax_b = fig.add_subplot(gs[0, 1])
    spectrum_counts = spectrum_df.groupby("label")["spectrum_id"].count().reindex(label_order)
    ax_b.bar(x, spectrum_counts.values, width=0.58, color=[label_light[i] for i in label_order])
    ax_b.set_xticks(x, [label_names[i] for i in label_order])
    ax_b.set_ylabel("Spectra")
    ax_b.set_title("B. Label Distribution at Spectrum Level")
    ax_b.set_ylim(0, spectrum_counts.max() * 1.20)
    add_bar_labels(ax_b, x, spectrum_counts.values, pad=35)
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.grid(axis="y", color=COLORS["grid"], alpha=0.35, linewidth=0.7)

    ax_c = fig.add_subplot(gs[1, 0])
    bottom = np.zeros(len(split_order))
    patient_values_by_label: dict[int, np.ndarray] = {}
    split_patient_totals = []
    for label in label_order:
        vals = np.array(
            [int(((patient_df["split"] == split_name) & (patient_df["label"] == label)).sum()) for split_name in split_order]
        )
        patient_values_by_label[label] = vals
        ax_c.bar(split_order, vals, bottom=bottom, color=label_colors[label], label=label_names[label])
        bottom += vals
    split_patient_totals = bottom.astype(int)
    ax_c.set_ylabel("Patients")
    ax_c.set_title("C. Stratified Split Counts: Patients")
    ax_c.set_ylim(0, split_patient_totals.max() * 1.24)
    for i, total in enumerate(split_patient_totals):
        ax_c.text(i, total + 0.5, str(int(total)), ha="center", va="bottom", color=COLORS["ink"])
    add_stacked_segment_labels(ax_c, patient_values_by_label)
    ax_c.legend(frameon=False, ncol=2, loc="upper right")
    ax_c.spines[["top", "right"]].set_visible(False)
    ax_c.grid(axis="y", color=COLORS["grid"], alpha=0.35, linewidth=0.7)

    ax_d = fig.add_subplot(gs[1, 1])
    bottom = np.zeros(len(split_order))
    spectrum_values_by_label: dict[int, np.ndarray] = {}
    for label in label_order:
        vals = np.array(
            [int(((spectrum_df["split"] == split_name) & (spectrum_df["label"] == label)).sum()) for split_name in split_order]
        )
        spectrum_values_by_label[label] = vals
        ax_d.bar(split_order, vals, bottom=bottom, color=label_light[label], label=label_names[label])
        bottom += vals
    split_spectrum_totals = bottom.astype(int)
    ax_d.set_ylabel("Spectra")
    ax_d.set_title("D. Expanded Sample Counts after Patient Split")
    ax_d.set_ylim(0, split_spectrum_totals.max() * 1.20)
    for i, total in enumerate(split_spectrum_totals):
        ax_d.text(i, total + 25, str(int(total)), ha="center", va="bottom", color=COLORS["ink"])
    add_stacked_segment_labels(ax_d, spectrum_values_by_label, min_height=20)
    ax_d.legend(frameon=False, ncol=2, loc="upper right")
    ax_d.spines[["top", "right"]].set_visible(False)
    ax_d.grid(axis="y", color=COLORS["grid"], alpha=0.35, linewidth=0.7)

    fig.text(
        0.02,
        0.055,
        "Data source: raw CSV files under DentalPlaque_SERS_Toolbox/牙菌斑SERS光谱/阳性+ and 牙菌斑SERS光谱/阴性-. "
        "Excluded: 未知 and 其它数据 folders.",
        color=COLORS["text"],
        fontsize=8.5,
    )
    fig.text(
        0.02,
        0.028,
        f"Dataset: {summary['n_patients']} patients, {summary['n_spectra']} spectra, "
        f"{summary['n_wavenumber']} Raman-shift points per spectrum. "
        "Split is performed by patient before spectrum-level expansion; train/val/test patient overlap = 0.",
        color=COLORS["text"],
        fontsize=8.5,
    )

    save_all(fig, out_dir, "phase3a_fig2_dataset_split_audit")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    patient_df, spectrum_df, summary, split = load_inputs(cfg)
    patient_df, spectrum_df = assign_split(patient_df, spectrum_df, split)
    assert_metadata_consistency(patient_df, spectrum_df, summary)
    out_dir = toolbox_root() / "Figures" / "Phase3"
    plot_fig2(patient_df, spectrum_df, summary, split, out_dir)
    print(f"Saved Figure 2 to {out_dir}")


if __name__ == "__main__":
    main()
