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
    "green": "#4B7F52",
    "gray": "#697386",
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


def load_inputs(cfg: dict) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray, np.ndarray]:
    root = toolbox_root()
    dataset_dir = resolve_path(cfg["paths"]["dataset_dir"], root)
    spectrum_df = pd.read_csv(dataset_dir / "spectrum_metadata.csv", encoding="utf-8-sig")
    with open(dataset_dir / "dataset_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    spectra = np.load(dataset_dir / "spectra.npz", allow_pickle=True)
    wn = np.load(dataset_dir / "wavenumber.npy")
    return spectrum_df, summary, spectra["X_raw_spectra"], spectra["X_spectra"], wn


def assert_preprocessing_consistency(summary: dict, X_raw: np.ndarray, X: np.ndarray, wn: np.ndarray) -> None:
    expected_shape = (int(summary["n_spectra"]), int(summary["n_wavenumber"]))
    if tuple(X_raw.shape) != expected_shape:
        raise RuntimeError(f"Raw spectra shape mismatch: {X_raw.shape} != {expected_shape}")
    if tuple(X.shape) != expected_shape:
        raise RuntimeError(f"SNV spectra shape mismatch: {X.shape} != {expected_shape}")
    if len(wn) != int(summary["n_wavenumber"]):
        raise RuntimeError("Wavenumber vector length mismatch.")
    if not np.isfinite(X_raw).all() or not np.isfinite(X).all():
        raise RuntimeError("Non-finite values detected in spectra arrays.")


def representative_indices(spectrum_df: pd.DataFrame) -> dict[int, int]:
    chosen: dict[int, int] = {}
    for label in [0, 1]:
        sub = spectrum_df[spectrum_df["label"] == label].copy()
        median_mean = sub["mean_intensity"].median()
        sub["distance_to_label_median_mean"] = (sub["mean_intensity"] - median_mean).abs()
        row = sub.sort_values(["distance_to_label_median_mean", "spectrum_index"]).iloc[0]
        chosen[label] = int(row["spectrum_index"])
    return chosen


def patient_level_medians(X: np.ndarray, spectrum_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    patient_rows = []
    patient_labels = []
    for patient_uid, group in spectrum_df.groupby("patient_uid", sort=True):
        indices = group["spectrum_index"].to_numpy(dtype=int)
        patient_rows.append(np.median(X[indices], axis=0))
        patient_labels.append(int(group["label"].iloc[0]))
    return np.asarray(patient_rows), np.asarray(patient_labels)


def save_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    ensure_dir(out_dir)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}")


def plot_fig3(spectrum_df: pd.DataFrame, summary: dict, X_raw: np.ndarray, X: np.ndarray, wn: np.ndarray, out_dir: Path) -> None:
    setup_style()
    fig = plt.figure(figsize=(12.5, 8.9))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.46)
    fig.suptitle("Figure 3. Spectrum Extraction and SNV Preprocessing Quality Control", y=0.975, fontsize=14, weight="bold")

    label_names = {0: "Negative", 1: "Positive"}
    label_colors = {0: COLORS["neg"], 1: COLORS["pos"]}
    chosen = representative_indices(spectrum_df)

    ax_a = fig.add_subplot(gs[0, 0])
    for label in [0, 1]:
        idx = chosen[label]
        ax_a.plot(wn, X_raw[idx], color=label_colors[label], lw=1.1, alpha=0.9, label=f"{label_names[label]} example")
    ax_a.set_title("A. Extracted Raw Spectra")
    ax_a.set_xlabel("Raman shift (cm$^{-1}$)")
    ax_a.set_ylabel("Raw intensity")
    ax_a.legend(frameon=False)
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.grid(axis="y", color=COLORS["grid"], alpha=0.28, linewidth=0.7)

    ax_b = fig.add_subplot(gs[0, 1])
    for label in [0, 1]:
        idx = chosen[label]
        ax_b.plot(wn, X[idx], color=label_colors[label], lw=1.1, alpha=0.9, label=f"{label_names[label]} example")
    ax_b.axhline(0, color=COLORS["gray"], lw=0.8, linestyle=":")
    ax_b.set_title("B. Same Spectra after Per-Spectrum SNV")
    ax_b.set_xlabel("Raman shift (cm$^{-1}$)")
    ax_b.set_ylabel("SNV intensity")
    ax_b.legend(frameon=False)
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.grid(axis="y", color=COLORS["grid"], alpha=0.28, linewidth=0.7)

    ax_c = fig.add_subplot(gs[1, 0])
    snv_means = X.mean(axis=1)
    snv_sds = X.std(axis=1)
    parts = ax_c.violinplot([snv_means, snv_sds], positions=[0, 1], showmeans=True, showextrema=True, widths=0.52)
    for body, color in zip(parts["bodies"], [COLORS["green"], COLORS["green"]]):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.32)
    for key in ["cmeans", "cmins", "cmaxes", "cbars"]:
        parts[key].set_color(COLORS["green"])
        parts[key].set_linewidth(1.0)
    ax_c.scatter(np.zeros_like(snv_means), snv_means, s=5, color=COLORS["gray"], alpha=0.10, linewidths=0)
    ax_c.scatter(np.ones_like(snv_sds), snv_sds, s=5, color=COLORS["gray"], alpha=0.10, linewidths=0)
    ax_c.set_xticks([0, 1], ["SNV mean", "SNV sd"])
    ax_c.set_ylabel("Value per spectrum")
    ax_c.set_title("C. SNV Check across All Spectra")
    ax_c.axhline(0, color=COLORS["gray"], lw=0.8, linestyle=":", alpha=0.8)
    ax_c.axhline(1, color=COLORS["gray"], lw=0.8, linestyle=":", alpha=0.8)
    ax_c.spines[["top", "right"]].set_visible(False)
    ax_c.grid(axis="y", color=COLORS["grid"], alpha=0.28, linewidth=0.7)
    ax_c.text(
        0.04,
        0.94,
        f"n={len(X)} spectra\nmax |mean|={np.max(np.abs(snv_means)):.2e}\nsd range={snv_sds.min():.4f}-{snv_sds.max():.4f}",
        transform=ax_c.transAxes,
        va="top",
        color=COLORS["ink"],
        fontsize=8.5,
    )

    ax_d = fig.add_subplot(gs[1, 1])
    patient_medians, patient_labels = patient_level_medians(X, spectrum_df)
    for label in [0, 1]:
        idx = patient_labels == label
        mean = patient_medians[idx].mean(axis=0)
        sem = patient_medians[idx].std(axis=0, ddof=1) / np.sqrt(idx.sum())
        ax_d.fill_between(wn, mean - sem, mean + sem, color=label_colors[label], alpha=0.16, linewidth=0)
        ax_d.plot(wn, mean, color=label_colors[label], lw=1.35, label=f"{label_names[label]} patient median")
    ax_d.set_title("D. Patient-Level Median Spectra after SNV")
    ax_d.set_xlabel("Raman shift (cm$^{-1}$)")
    ax_d.set_ylabel("SNV intensity")
    ax_d.legend(frameon=False)
    ax_d.spines[["top", "right"]].set_visible(False)
    ax_d.grid(axis="y", color=COLORS["grid"], alpha=0.28, linewidth=0.7)

    neg_row = spectrum_df.loc[spectrum_df["spectrum_index"] == chosen[0]].iloc[0]
    pos_row = spectrum_df.loc[spectrum_df["spectrum_index"] == chosen[1]].iloc[0]
    fig.text(
        0.02,
        0.055,
        "Data source: raw CSV files under DentalPlaque_SERS_Toolbox/牙菌斑SERS光谱/阳性+ and 牙菌斑SERS光谱/阴性-. "
        f"Extraction: D294:D1025 for Raman shift and H294:H1025 for intensity "
        f"({summary['wavenumber_min']:.2f}-{summary['wavenumber_max']:.2f} cm$^{{-1}}$).",
        color=COLORS["text"],
        fontsize=8.5,
    )
    fig.text(
        0.02,
        0.028,
        f"Dataset: {summary['n_spectra']} spectra from {summary['n_patients']} patients, "
        f"{summary['n_wavenumber']} points per spectrum. Example spectra were selected deterministically by "
        f"within-label median raw mean intensity: negative {neg_row['spectrum_id']}; positive {pos_row['spectrum_id']}.",
        color=COLORS["text"],
        fontsize=8.5,
    )

    save_all(fig, out_dir, "phase3a_fig3_preprocessing_qc")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    spectrum_df, summary, X_raw, X, wn = load_inputs(cfg)
    assert_preprocessing_consistency(summary, X_raw, X, wn)
    out_dir = toolbox_root() / "Figures" / "Phase3"
    plot_fig3(spectrum_df, summary, X_raw, X, wn, out_dir)
    print(f"Saved Figure 3 to {out_dir}")


if __name__ == "__main__":
    main()
