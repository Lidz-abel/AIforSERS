from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from utils import ensure_dir, load_config, resolve_path, toolbox_root


COLORS = {
    "ink": "#182026",
    "text": "#52606D",
    "edge": "#6B778C",
    "blue": "#2F5F9E",
    "red": "#B43E3A",
    "green": "#4B7F52",
    "purple": "#6A4C93",
    "warn": "#B91C1C",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    w: float,
    h: float,
    title: str,
    body: str,
    color: str,
    face: str = "white",
) -> None:
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.25,
        edgecolor=color,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h * 0.67, title, ha="center", va="center", weight="bold", color=COLORS["ink"])
    ax.text(
        xy[0] + w / 2,
        xy[1] + h * 0.36,
        body,
        ha="center",
        va="center",
        color=COLORS["text"],
        linespacing=1.25,
    )


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#9AA5B1") -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.15,
        color=color,
        shrinkA=5,
        shrinkB=5,
    )
    ax.add_patch(arrow)


def load_summary_and_split(cfg: dict) -> tuple[dict, dict]:
    root = toolbox_root()
    dataset_dir = resolve_path(cfg["paths"]["dataset_dir"], root)
    splits_dir = resolve_path(cfg["paths"]["splits_dir"], root)
    with open(dataset_dir / "dataset_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(splits_dir / f"split_seed{cfg['seed']}.json", "r", encoding="utf-8") as f:
        split = json.load(f)
    return summary, split


def assert_split_is_patient_independent(split: dict) -> None:
    train = set(split["train_patients"])
    val = set(split["val_patients"])
    test = set(split["test_patients"])
    if train & val or train & test or val & test:
        raise RuntimeError("Patient leakage detected: at least one patient appears in multiple splits.")


def save_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    ensure_dir(out_dir)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}")


def plot_fig1(summary: dict, split: dict, out_dir: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(12.8, 6.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle("Figure 1. Phase 3A Dataset Construction Workflow", y=0.965, fontsize=14, weight="bold")

    source_note = (
        "Data source: raw CSV files under DentalPlaque_SERS_Toolbox/"
        "牙菌斑SERS光谱/阳性+ and 牙菌斑SERS光谱/阴性-"
    )
    ax.text(0.035, 0.905, source_note, color=COLORS["ink"], fontsize=9.5, weight="bold")
    ax.text(
        0.035,
        0.862,
        "Excluded from supervised Phase 3A: 未知 and 其它数据 folders. Labels are patient labels: positive=1, negative=0.",
        color=COLORS["text"],
        fontsize=8.5,
    )

    top_y = 0.575
    box_w = 0.145
    box_h = 0.205
    xs = [0.045, 0.215, 0.385, 0.555, 0.725]
    boxes = [
        (
            "Labeled Raw CSV",
            f"{summary['n_spectra']} spectra\n{summary['n_patients']} patients\npositive 31 | negative 21",
            COLORS["edge"],
        ),
        (
            "Extract Spectrum",
            "wavenumber D294:D1025\nintensity H294:H1025\n732 points",
            COLORS["edge"],
        ),
        ("SNV Normalize", "single spectrum only\nmean=0, sd=1\nno label information", COLORS["green"]),
        ("Metadata Tables", "patient_metadata.csv\nspectrum_metadata.csv\npatient_uid links spectra", COLORS["blue"]),
        (
            "Patient Split",
            f"seed={split['seed']}\ntrain {split['counts']['train']} | val {split['counts']['val']} | test {split['counts']['test']}",
            COLORS["purple"],
        ),
    ]
    for x, (title, body, color) in zip(xs, boxes):
        add_box(ax, (x, top_y), box_w, box_h, title, body, color)
    for i in range(len(xs) - 1):
        add_arrow(ax, (xs[i] + box_w, top_y + box_h / 2), (xs[i + 1], top_y + box_h / 2))

    split_x = xs[-1] + box_w / 2
    lower_y = 0.235
    train_counts = split["label_counts"]["train"]
    val_counts = split["label_counts"]["val"]
    test_counts = split["label_counts"]["test"]
    add_box(
        ax,
        (0.56, lower_y),
        0.18,
        0.205,
        "Training Set",
        f"patient-level split first\nthen expand spectra\npatients: {split['counts']['train']}\nneg {train_counts['0']} | pos {train_counts['1']}",
        COLORS["blue"],
    )
    add_box(
        ax,
        (0.77, lower_y),
        0.18,
        0.205,
        "Validation/Test",
        "kept patient-independent\naggregate spectrum predictions\n"
        f"val neg {val_counts['0']} | pos {val_counts['1']}\n"
        f"test neg {test_counts['0']} | pos {test_counts['1']}",
        COLORS["purple"],
    )
    add_arrow(ax, (split_x, top_y), (0.65, lower_y + 0.205), color=COLORS["blue"])
    add_arrow(ax, (split_x, top_y), (0.86, lower_y + 0.205), color=COLORS["purple"])

    add_box(
        ax,
        (0.06, 0.21),
        0.42,
        0.22,
        "Leakage Control",
        "One patient can appear in only one split.\n"
        "Repeated spectra from the same patient are\n"
        "not independent patient-level samples.\n"
        "Spectrum-level training is allowed only after patient assignment.",
        COLORS["warn"],
        face="#FFF5F5",
    )

    ax.text(
        0.035,
        0.07,
        "Outputs: Results/Phase3/dataset/spectra.npz, wavenumber.npy, patient_metadata.csv, spectrum_metadata.csv; "
        "Results/Phase3/splits/split_seed42.json",
        color=COLORS["text"],
        fontsize=8.4,
    )
    ax.text(
        0.035,
        0.035,
        "Verification: train/val/test patient overlap = 0; total unique split patients = 52.",
        color=COLORS["text"],
        fontsize=8.4,
    )

    save_all(fig, out_dir, "phase3a_fig1_dataset_construction_workflow")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary, split = load_summary_and_split(cfg)
    assert_split_is_patient_independent(split)
    out_dir = toolbox_root() / "Figures" / "Phase3"
    plot_fig1(summary, split, out_dir)
    print(f"Saved Figure 1 to {out_dir}")


if __name__ == "__main__":
    main()
