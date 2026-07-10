from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from utils import ensure_dir, load_config, resolve_path, toolbox_root


COLORS = {
    "pos": "#B43E3A",
    "neg": "#2F5F9E",
    "train": "#486581",
    "val": "#F0B429",
    "test": "#6A4C93",
    "gray": "#697386",
    "light": "#F5F7FA",
    "ink": "#182026",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_artifacts(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, np.lib.npyio.NpzFile, np.ndarray]:
    dataset_dir = resolve_path(cfg["paths"]["dataset_dir"], toolbox_root())
    splits_dir = resolve_path(cfg["paths"]["splits_dir"], toolbox_root())
    split_path = splits_dir / f"split_seed{cfg['seed']}.json"

    patient_df = pd.read_csv(dataset_dir / "patient_metadata.csv", encoding="utf-8-sig")
    spectrum_df = pd.read_csv(dataset_dir / "spectrum_metadata.csv", encoding="utf-8-sig")
    with open(dataset_dir / "dataset_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    spectra = np.load(dataset_dir / "spectra.npz", allow_pickle=True)
    wn = np.load(dataset_dir / "wavenumber.npy")
    return patient_df, spectrum_df, summary, split, spectra, wn


def save_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    ensure_dir(out_dir)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}")


def add_box(ax: plt.Axes, xy: tuple[float, float], w: float, h: float, title: str, body: str, color: str) -> None:
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h * 0.64, title, ha="center", va="center", weight="bold", color=COLORS["ink"])
    ax.text(xy[0] + w / 2, xy[1] + h * 0.34, body, ha="center", va="center", color=COLORS["gray"], linespacing=1.25)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#9AA5B1") -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.1,
        color=color,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def plot_flowchart(summary: dict, split: dict, out_dir: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(12.0, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle("Phase 3A Dataset Construction: Patient-Level Split, Spectrum-Level Training", y=0.97, fontsize=13, weight="bold")

    top_y = 0.60
    box_w = 0.145
    box_h = 0.20
    xs = [0.035, 0.205, 0.375, 0.545, 0.715]
    boxes = [
        ("Labeled SERS CSV", f"{summary['n_spectra']} spectra\n{summary['n_patients']} patients"),
        ("Parse Spectrum", "D294:D1025 shift\nH294:H1025 intensity"),
        ("Normalize", "Per-spectrum SNV\n732 features"),
        ("Patient Registry", "patient_metadata\nspectrum_metadata"),
        ("Patient Split", f"train {split['counts']['train']} | val {split['counts']['val']} | test {split['counts']['test']}"),
    ]
    colors = [COLORS["gray"], COLORS["gray"], "#4B7F52", "#2F5F9E", "#6A4C93"]
    for x, (title, body), color in zip(xs, boxes, colors):
        add_box(ax, (x, top_y), box_w, box_h, title, body, color)
    for i in range(len(xs) - 1):
        add_arrow(ax, (xs[i] + box_w, top_y + box_h / 2), (xs[i + 1], top_y + box_h / 2))

    split_x = xs[-1] + box_w / 2
    lower = [
        ((0.56, 0.20), "Train Patients", "expand to single spectra\nfit spectrum model", COLORS["train"]),
        ((0.74, 0.20), "Val/Test Patients", "predict each spectrum\naggregate by patient", COLORS["test"]),
    ]
    for xy, title, body, color in lower:
        add_box(ax, xy, 0.18, 0.19, title, body, color)
        add_arrow(ax, (split_x, top_y), (xy[0] + 0.09, xy[1] + 0.19), color=color)

    guard = FancyBboxPatch(
        (0.09, 0.15),
        0.36,
        0.20,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.1,
        edgecolor="#D64545",
        facecolor="#FFF5F5",
    )
    ax.add_patch(guard)
    ax.text(0.27, 0.285, "Leakage Guard", ha="center", va="center", weight="bold", color="#9B1C1C")
    ax.text(
        0.27,
        0.225,
        "No patient appears in more than one split.\nSpectra are expanded only after patient assignment.",
        ha="center",
        va="center",
        color="#5F2120",
        linespacing=1.3,
    )

    ax.text(
        0.035,
        0.072,
        "Data source: labeled positive/negative raw CSV only; excluded folders: unknown and other comparison data.",
        color=COLORS["gray"],
    )
    ax.text(0.035, 0.040, "Output: spectra.npz, wavenumber.npy, patient_metadata.csv, spectrum_metadata.csv, split_seed42.json", color=COLORS["gray"])
    fig.tight_layout()
    save_all(fig, out_dir, "phase3a_make_dataset_flowchart")
    plt.close(fig)


def label_name(label: int) -> str:
    return "Positive" if int(label) == 1 else "Negative"


def split_for_patient(patient_uid: str, split: dict) -> str:
    if patient_uid in set(split["train_patients"]):
        return "train"
    if patient_uid in set(split["val_patients"]):
        return "val"
    if patient_uid in set(split["test_patients"]):
        return "test"
    return "unknown"


def patient_level_medians(spectra: np.ndarray, spectrum_df: pd.DataFrame, patient_df: pd.DataFrame) -> np.ndarray:
    med = np.zeros((len(patient_df), spectra.shape[1]), dtype=np.float32)
    for idx, row in patient_df.iterrows():
        mask = spectrum_df["patient_index"].to_numpy() == row["patient_index"]
        med[idx, :] = np.median(spectra[mask], axis=0)
    return med


def plot_overview(
    patient_df: pd.DataFrame,
    spectrum_df: pd.DataFrame,
    summary: dict,
    split: dict,
    spectra_npz: np.lib.npyio.NpzFile,
    wn: np.ndarray,
    out_dir: Path,
) -> None:
    setup_style()
    X = spectra_npz["X_spectra"]
    X_raw = spectra_npz["X_raw_spectra"]
    patient_df = patient_df.copy()
    patient_df["split"] = patient_df["patient_uid"].map(lambda x: split_for_patient(x, split))
    split_order = ["train", "val", "test"]

    fig = plt.figure(figsize=(13.2, 10.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.86, 1.0, 1.16], wspace=0.42, hspace=0.48)
    fig.suptitle("Phase 3A Dataset Overview: Patient-Level Dataset Construction", y=0.985, fontsize=13, weight="bold")

    ax_a = fig.add_subplot(gs[0, 0])
    group_counts = patient_df.groupby("label")["patient_uid"].nunique().reindex([0, 1])
    x = np.arange(2)
    ax_a.bar(x, group_counts.values, width=0.62, color=[COLORS["neg"], COLORS["pos"]])
    ax_a.set_xticks(x, ["Negative", "Positive"])
    ax_a.set_ylabel("Patients")
    ax_a.set_title("A. Patient Class Distribution")
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.set_ylim(0, max(group_counts.values) * 1.24)
    for i, pc in enumerate(group_counts.values):
        ax_a.text(i, pc + 0.8, str(int(pc)), ha="center", va="bottom")

    ax_b = fig.add_subplot(gs[0, 1])
    spec_counts = spectrum_df.groupby("label")["spectrum_id"].count().reindex([0, 1])
    ax_b.bar(x, spec_counts.values, width=0.62, color=["#8FB3D9", "#D98F8B"])
    ax_b.set_xticks(x, ["Negative", "Positive"])
    ax_b.set_ylabel("Spectra")
    ax_b.set_title("B. Spectrum Count by Label")
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.set_ylim(0, max(spec_counts.values) * 1.18)
    for i, sc in enumerate(spec_counts.values):
        ax_b.text(i, sc + 35, str(int(sc)), ha="center", va="bottom")

    ax_c = fig.add_subplot(gs[0, 2:4])
    bottom = np.zeros(3)
    for label, color in [(0, COLORS["neg"]), (1, COLORS["pos"])]:
        vals = []
        for sp in split_order:
            vals.append(int(((patient_df["split"] == sp) & (patient_df["label"] == label)).sum()))
        ax_c.bar(split_order, vals, bottom=bottom, color=color, label=label_name(label))
        bottom += np.asarray(vals)
    ax_c.set_title("C. Stratified Patient Split")
    ax_c.set_ylabel("Patients")
    ax_c.legend(frameon=False, ncol=2, loc="upper right")
    ax_c.spines[["top", "right"]].set_visible(False)
    ax_c.set_ylim(0, max(bottom) * 1.22)
    for i, total in enumerate(bottom):
        ax_c.text(i, total + 0.5, str(int(total)), ha="center", va="bottom")

    ax_d = fig.add_subplot(gs[1, 0:2])
    sorted_df = patient_df.sort_values(["label", "n_spectra", "patient_uid"]).reset_index(drop=True)
    colors = [COLORS["neg"] if v == 0 else COLORS["pos"] for v in sorted_df["label"]]
    ax_d.bar(np.arange(len(sorted_df)), sorted_df["n_spectra"], color=colors, width=0.85)
    ax_d.axvline((sorted_df["label"] == 0).sum() - 0.5, color="#CBD2D9", lw=1.0)
    ax_d.set_title("D. Spectra per Patient")
    ax_d.set_xlabel("Patients sorted within each class")
    ax_d.set_ylabel("Spectra")
    ax_d.spines[["top", "right"]].set_visible(False)
    ax_d.set_xlim(-0.6, len(sorted_df) - 0.4)
    ax_d.text(0.02, 0.94, "Negative", transform=ax_d.transAxes, color=COLORS["neg"], weight="bold", va="top")
    ax_d.text(0.45, 0.94, "Positive", transform=ax_d.transAxes, color=COLORS["pos"], weight="bold", va="top")

    ax_e = fig.add_subplot(gs[1:, 2:4])
    pat_med = patient_level_medians(X, spectrum_df, patient_df)
    for label, color in [(0, COLORS["neg"]), (1, COLORS["pos"])]:
        idx = patient_df["label"].to_numpy() == label
        mean = pat_med[idx].mean(axis=0)
        sem = pat_med[idx].std(axis=0, ddof=1) / np.sqrt(idx.sum())
        ax_e.fill_between(wn, mean - sem, mean + sem, color=color, alpha=0.18, linewidth=0)
        ax_e.plot(wn, mean, color=color, lw=1.6, label=f"{label_name(label)} patient median")
    ax_e.set_title("E. Patient-Level Median Spectra after SNV")
    ax_e.set_xlabel("Raman shift (cm$^{-1}$)")
    ax_e.set_ylabel("SNV intensity")
    ax_e.legend(frameon=False, loc="upper right")
    ax_e.spines[["top", "right"]].set_visible(False)

    ax_f = fig.add_subplot(gs[2, 0:2])
    pos_sub = spectrum_df[spectrum_df["label"] == 1].sort_values("spectrum_index")
    example_idx = int(pos_sub.iloc[len(pos_sub) // 2]["spectrum_index"])
    raw = X_raw[example_idx].astype(float)
    snv = X[example_idx].astype(float)
    raw_line = ax_f.plot(wn, raw, color="#697386", lw=1.15, label="Raw intensity")[0]
    ax_f.set_title("F. SNV Transformation of One Spectrum")
    ax_f.set_xlabel("Raman shift (cm$^{-1}$)")
    ax_f.set_ylabel("Raw intensity", color="#697386")
    ax_f.tick_params(axis="y", labelcolor="#697386")
    ax_f.spines["top"].set_visible(False)
    ax_f2 = ax_f.twinx()
    snv_line = ax_f2.plot(wn, snv, color="#4B7F52", lw=1.15, linestyle="--", label="SNV intensity")[0]
    ax_f2.set_ylabel("SNV intensity", color="#4B7F52")
    ax_f2.tick_params(axis="y", labelcolor="#4B7F52")
    ax_f2.spines["top"].set_visible(False)
    ax_f.legend([raw_line, snv_line], ["Raw intensity", "SNV intensity"], frameon=False, loc="upper right")
    ax_f.text(
        0.02,
        0.95,
        f"raw mean={raw.mean():.0f}, sd={raw.std():.0f}\nSNV mean={snv.mean():.2f}, sd={snv.std():.2f}",
        transform=ax_f.transAxes,
        va="top",
        color=COLORS["ink"],
        fontsize=7.5,
    )

    fig.text(
        0.015,
        0.02,
        f"Dataset: {summary['n_patients']} patients, {summary['n_spectra']} spectra, {summary['n_wavenumber']} Raman-shift points. "
        "Source is labeled positive/negative raw CSV only. Labels are patient labels; spectra are expanded only after patient-level split.",
        color=COLORS["gray"],
        fontsize=8,
    )

    save_all(fig, out_dir, "phase3a_dataset_overview")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = toolbox_root() / "Figures" / "Phase3"
    patient_df, spectrum_df, summary, split, spectra, wn = load_artifacts(cfg)
    plot_flowchart(summary, split, out_dir)
    plot_overview(patient_df, spectrum_df, summary, split, spectra, wn, out_dir)
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
