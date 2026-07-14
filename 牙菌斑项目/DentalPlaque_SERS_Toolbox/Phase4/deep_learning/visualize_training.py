"""Phase 4A: Training Visualization Dashboard.

Reads training_log.csv and generates:
  1. Loss curves (train + val)
  2. Validation patient-level AUC with best-epoch marker
  3. Learning rate schedule
  4. Loss vs AUC correlation
  5. Combined dashboard figure

Also provides TensorBoard event writer for re-running with logging.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

# UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


plt.rcParams.update({
    "font.family": "Arial", "font.size": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})


def load_training_log(csv_path: str) -> list[dict]:
    """Load training_log.csv → list of dicts."""
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(val) -> float:
    """Convert CSV string to float, handling None."""
    if val is None or val == "":
        return float("nan")
    return float(val)


def plot_loss_curves(log: list[dict], save_path: str):
    """Figure 1: Train + Val loss curves."""
    epochs = [int(r["epoch"]) for r in log]
    train_loss = [float(r["train_loss"]) for r in log]
    val_loss = [float(r["val_loss"]) for r in log]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(epochs, train_loss, linewidth=1.5, color="#1f77b4", label="Train Loss")
    ax.plot(epochs, val_loss, linewidth=1.5, color="#ff7f0e", label="Val Loss")
    ax.fill_between(epochs, train_loss, val_loss, alpha=0.1, color="gray")

    # Mark best epoch
    best_idx = np.argmin(val_loss)
    ax.axvline(epochs[best_idx], color="red", linestyle="--", alpha=0.5,
               label=f"Best epoch={epochs[best_idx]} (loss={val_loss[best_idx]:.4f})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss curves: {save_path}")


def plot_auc_curve(log: list[dict], save_path: str):
    """Figure 2: Validation patient-level AUC."""
    epochs = [int(r["epoch"]) for r in log]
    val_auc = [to_float(r.get("val_patient_auc")) for r in log]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(epochs, val_auc, "o-", markersize=4, linewidth=1.5, color="#2ca02c",
            label="Val Patient AUC")

    # Fill between epochs
    valid = ~np.isnan(val_auc)
    ax.fill_between(np.array(epochs)[valid], 0.5, np.array(val_auc)[valid],
                     alpha=0.15, color="#2ca02c")

    best_idx = np.nanargmax(val_auc)
    best_auc = val_auc[best_idx]
    ax.axvline(epochs[best_idx], color="red", linestyle="--", alpha=0.5,
               label=f"Best epoch={epochs[best_idx]} (AUC={best_auc:.4f})")
    ax.axhline(y=best_auc, color="red", linestyle=":", alpha=0.3)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Random (0.5)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Patient-Level AUC")
    ax.set_title("Validation Patient AUC")
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  AUC curve: {save_path}")


def plot_lr_schedule(log: list[dict], save_path: str):
    """Figure 3: Learning rate schedule."""
    epochs = [int(r["epoch"]) for r in log]
    lrs = [float(r["lr"]) for r in log]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.step(epochs, lrs, where="post", linewidth=1.5, color="#9467bd")
    ax.scatter(epochs, lrs, s=15, color="#9467bd", zorder=5)

    # Mark LR drops
    for i in range(1, len(lrs)):
        if lrs[i] < lrs[i - 1]:
            ax.axvline(epochs[i], color="red", linestyle="--", alpha=0.4, linewidth=0.8)
            ax.annotate(f"{lrs[i]:.0e}", (epochs[i], lrs[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (ReduceLROnPlateau)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  LR schedule: {save_path}")


def plot_loss_vs_auc(log: list[dict], save_path: str):
    """Figure 4: Loss vs AUC scatter with epoch coloring."""
    val_loss = np.array([float(r["val_loss"]) for r in log])
    val_auc = np.array([to_float(r["val_patient_auc"]) for r in log])
    epochs = np.array([int(r["epoch"]) for r in log])

    valid = ~np.isnan(val_auc)
    val_loss = val_loss[valid]
    val_auc = val_auc[valid]
    epochs = epochs[valid]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    scatter = ax.scatter(val_loss, val_auc, c=epochs, cmap="viridis",
                          s=40, edgecolors="white", linewidth=0.5, zorder=3)
    plt.colorbar(scatter, ax=ax, label="Epoch")

    # Best point
    best_idx = np.argmax(val_auc)
    ax.scatter([val_loss[best_idx]], [val_auc[best_idx]], s=120,
               facecolors="none", edgecolors="red", linewidth=2, zorder=4)
    ax.annotate(f"Best (epoch={epochs[best_idx]})",
                (val_loss[best_idx], val_auc[best_idx]),
                textcoords="offset points", xytext=(8, -8), fontsize=8, color="red")

    ax.set_xlabel("Validation Loss")
    ax.set_ylabel("Validation Patient AUC")
    ax.set_title("Loss vs AUC Landscape")
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss vs AUC: {save_path}")


def plot_dashboard(log: list[dict], save_path: str):
    """Figure 5: Combined 4-panel dashboard."""
    epochs = [int(r["epoch"]) for r in log]
    train_loss = [float(r["train_loss"]) for r in log]
    val_loss = [float(r["val_loss"]) for r in log]
    val_auc = [to_float(r.get("val_patient_auc")) for r in log]
    lrs = [float(r["lr"]) for r in log]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # Panel 1: Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, linewidth=1.2, label="Train")
    ax.plot(epochs, val_loss, linewidth=1.2, label="Val")
    best_loss_idx = np.argmin(val_loss)
    ax.axvline(epochs[best_loss_idx], color="red", linestyle="--", alpha=0.4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Panel 2: AUC
    ax = axes[0, 1]
    valid = ~np.isnan(val_auc)
    ax.plot(np.array(epochs)[valid], np.array(val_auc)[valid], "o-",
            markersize=3, linewidth=1.2, color="#2ca02c")
    best_auc_idx = np.nanargmax(val_auc)
    ax.axvline(epochs[best_auc_idx], color="red", linestyle="--", alpha=0.4)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUC")
    ax.set_title(f"Val Patient AUC (best={val_auc[best_auc_idx]:.4f})")
    ax.set_ylim(0.4, 1.05); ax.grid(True, alpha=0.3)

    # Panel 3: LR schedule
    ax = axes[1, 0]
    ax.step(epochs, lrs, where="post", linewidth=1.2, color="#9467bd")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    # Panel 4: Train/Val loss gap
    ax = axes[1, 1]
    loss_gap = np.array(train_loss) - np.array(val_loss)
    ax.fill_between(epochs, 0, loss_gap, alpha=0.3, color="gray",
                     label="Overfitting gap")
    ax.plot(epochs, loss_gap, linewidth=1.2, color="purple")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train-Val Loss Gap")
    ax.set_title("Overfitting Signal (Train Loss - Val Loss)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("CC-SERSNet v1 Training Dashboard", fontsize=13, fontweight="bold")
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Dashboard: {save_path}")


def write_tensorboard_events(log: list[dict], log_dir: str):
    """Replay training log into TensorBoard events for interactive viewing."""
    writer = SummaryWriter(log_dir=log_dir)

    for r in log:
        epoch = int(r["epoch"])
        writer.add_scalar("Loss/train", float(r["train_loss"]), epoch)
        writer.add_scalar("Loss/val", float(r["val_loss"]), epoch)
        auc_val = to_float(r.get("val_patient_auc"))
        if not np.isnan(auc_val):
            writer.add_scalar("Metrics/val_patient_auc", auc_val, epoch)
        writer.add_scalar("Params/learning_rate", float(r["lr"]), epoch)

        # Loss gap (overfitting signal)
        gap = float(r["train_loss"]) - float(r["val_loss"])
        writer.add_scalar("Loss/gap", gap, epoch)

    writer.close()
    print(f"  TensorBoard events: {log_dir}")
    print(f"  Run: tensorboard --logdir={log_dir}")


def main():
    # Paths
    toolbox = Path(__file__).resolve().parents[2]
    log_csv = toolbox / "Results" / "Phase4" / "deep_learning" / "training_log.csv"
    viz_dir = toolbox / "Results" / "Phase4" / "deep_learning" / "figures"
    tb_dir = toolbox / "Results" / "Phase4" / "deep_learning" / "tensorboard"

    viz_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Phase 4A: Training Visualization")
    print("=" * 50)

    log = load_training_log(str(log_csv))
    print(f"Loaded {len(log)} epochs from {log_csv}")

    # Generate all figures
    print("\nGenerating figures...")
    plot_loss_curves(log, str(viz_dir / "loss_curves.png"))
    plot_auc_curve(log, str(viz_dir / "auc_curve.png"))
    plot_lr_schedule(log, str(viz_dir / "lr_schedule.png"))
    plot_loss_vs_auc(log, str(viz_dir / "loss_vs_auc.png"))
    plot_dashboard(log, str(viz_dir / "training_dashboard.png"))

    # TensorBoard events
    print("\nWriting TensorBoard events...")
    write_tensorboard_events(log, str(tb_dir))

    print("\nDone. To view TensorBoard:")
    print(f"  tensorboard --logdir={tb_dir}")


if __name__ == "__main__":
    main()
