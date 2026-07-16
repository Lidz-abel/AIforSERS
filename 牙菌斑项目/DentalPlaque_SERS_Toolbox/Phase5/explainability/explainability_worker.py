"""Resumable two-GPU worker and coordinator for P5-05."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
TOOLBOX = HERE.parents[1]
RESULTS = TOOLBOX / "Results" / "Phase5" / "exp_005_clinical_explainability"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["intensity", "dual_view"], required=True)
    parser.add_argument("--coordinator", action="store_true")
    args = parser.parse_args()
    subprocess.run(
        [sys.executable, "-u", str(HERE / "attribution_worker.py"), "--architecture", args.architecture],
        cwd=TOOLBOX,
        check=True,
    )
    if args.coordinator:
        expected = [RESULTS / "intensity.done", RESULTS / "dual_view.done"]
        while not all(path.exists() for path in expected):
            print(f"waiting for explanation workers: {sum(not path.exists() for path in expected)}", flush=True)
            time.sleep(30)
        subprocess.run(
            [sys.executable, "-u", str(HERE / "aggregate_explanations.py")], cwd=TOOLBOX, check=True
        )
    print(f"P5-05 worker complete: {args.architecture}", flush=True)


if __name__ == "__main__":
    main()
