from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import load_config, resolve_path, toolbox_root, write_json


def make_splits(config_path: Path | None = None) -> dict:
    cfg = load_config(config_path)
    dataset_dir = resolve_path(cfg["paths"]["dataset_dir"], toolbox_root())
    splits_dir = resolve_path(cfg["paths"]["splits_dir"], toolbox_root())
    splits_dir.mkdir(parents=True, exist_ok=True)

    patient_df = pd.read_csv(dataset_dir / "patient_metadata.csv", encoding="utf-8-sig")
    seed = int(cfg["seed"])
    train_ratio = float(cfg["split"]["train_ratio"])
    val_ratio = float(cfg["split"]["val_ratio"])
    test_ratio = float(cfg["split"]["test_ratio"])
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    stratify = patient_df["label"] if cfg["split"].get("stratify", True) else None
    train_df, temp_df = train_test_split(
        patient_df,
        train_size=train_ratio,
        random_state=seed,
        stratify=stratify,
    )

    relative_val = val_ratio / (val_ratio + test_ratio)
    temp_stratify = temp_df["label"] if cfg["split"].get("stratify", True) else None
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val,
        random_state=seed,
        stratify=temp_stratify,
    )

    def records(df: pd.DataFrame) -> list[str]:
        return sorted(df["patient_uid"].tolist())

    split = {
        "seed": seed,
        "unit": "patient",
        "train_patients": records(train_df),
        "val_patients": records(val_df),
        "test_patients": records(test_df),
        "counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "label_counts": {
            "train": train_df["label"].value_counts().sort_index().astype(int).to_dict(),
            "val": val_df["label"].value_counts().sort_index().astype(int).to_dict(),
            "test": test_df["label"].value_counts().sort_index().astype(int).to_dict(),
        },
    }

    all_sets = [
        set(split["train_patients"]),
        set(split["val_patients"]),
        set(split["test_patients"]),
    ]
    if all_sets[0] & all_sets[1] or all_sets[0] & all_sets[2] or all_sets[1] & all_sets[2]:
        raise RuntimeError("Patient leakage detected across splits")

    output = splits_dir / f"split_seed{seed}.json"
    write_json(output, split)
    return split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    split = make_splits(args.config)
    print("Patient-level split created")
    print(split["counts"])
    print(split["label_counts"])


if __name__ == "__main__":
    main()

