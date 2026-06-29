"""Launch the Test-1 matrix: each variant config x several seeds.

Runs are sequential (the models are tiny; this keeps VRAM trivial and output
readable). Each run writes to results/<run_name>/seed<N>/.

Usage:
    python experiments/run_seeds.py                     # default 4 variants x seeds 0,1,2
    python experiments/run_seeds.py --seeds 0 1 2 3 4
    python experiments/run_seeds.py --configs configs/v0_shakespeare.yaml ...
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Phase-2 Tier-1 matrix: point anchor + SwiGLU baseline + 4 augmentative variants.
# (Phase-1 configs are still runnable via --configs.)
DEFAULT_CONFIGS = [
    "configs/v0_shakespeare.yaml",        # point — sanity anchor
    "configs/swiglu_shakespeare.yaml",    # the baseline to beat
    "configs/par_dend_shakespeare.yaml",
    "configs/par_dense_shakespeare.yaml",
    "configs/seq_dend_shakespeare.yaml",
    "configs/seq_dense_shakespeare.yaml",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    train = str(ROOT / "src" / "train.py")
    total = len(args.configs) * len(args.seeds)
    i = 0
    for cfg in args.configs:
        for seed in args.seeds:
            i += 1
            print(f"\n========== [{i}/{total}] {cfg}  seed={seed} ==========",
                  flush=True)
            subprocess.run([sys.executable, train, "--config", str(ROOT / cfg),
                            "--seed", str(seed)], check=True)
    print("\n===ALL SEED RUNS DONE===")


if __name__ == "__main__":
    main()
