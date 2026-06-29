"""main_frac and branches sweeps on the sequential augmentative variants.

Question (from the human): is there a 'best of both worlds' point where a small
dendritic add-on adds feature richness WITHOUT starving the main SwiGLU of raw
neurons? We vary one axis at a time (per the project's methodology), reusing the
existing Test-2 runs as the center point (main_frac=0.5, branches=4).

  main_frac sweep : seq_dend & seq_dense over {0.25, 0.75, 0.9}  (0.5 = existing)
  branches  sweep : seq_dend over {2, 8, 16}                     (4   = existing)
both on the PCFG (compositional gap) and BPE (real-text fit/transfer) tasks.

Serial, one process per run (clean VRAM). Run names are suffixed so points don't
collide; aggregate/eval with the printed run-name lists afterwards.

Usage:
    python experiments/run_sweep.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = str(ROOT / "src" / "train.py")

DATASETS = {"pcfg": "pcfg", "bpe": "bpe"}          # suffix -> config infix
MAIN_FRACS = [0.25, 0.75, 0.9]                      # 0.5 already run
BRANCHES = [2, 8, 16]                              # 4 already run


def run(base_cfg: str, run_name: str, seed: int, extra: list[str]):
    cmd = [sys.executable, TRAIN, "--config", str(ROOT / base_cfg),
           "--run-name", run_name, "--seed", str(seed)] + extra
    print(f"\n========== {run_name}  seed={seed} ==========", flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    jobs = []  # (base_cfg, run_name, extra_args)
    # main_frac sweep: both sequential variants, both datasets
    for ds in DATASETS:
        for variant in ("seq_dend", "seq_dense"):
            base = f"configs/{variant}_{ds}.yaml"
            for mf in MAIN_FRACS:
                name = f"{variant}_{ds}_mf{int(round(mf*100)):03d}"
                jobs.append((base, name, ["--main-frac", str(mf)]))
    # branches sweep: seq_dend only, both datasets
    for ds in DATASETS:
        base = f"configs/seq_dend_{ds}.yaml"
        for b in BRANCHES:
            name = f"seq_dend_{ds}_b{b:02d}"
            jobs.append((base, name, ["--branches", str(b)]))

    total = len(jobs) * len(args.seeds)
    print(f"[sweep] {len(jobs)} configs x {len(args.seeds)} seeds = {total} runs")
    i = 0
    for base, name, extra in jobs:
        for seed in args.seeds:
            i += 1
            print(f"\n###### [{i}/{total}] ######", flush=True)
            run(base, name, seed, extra)
    print("\n===ALL SWEEP RUNS DONE===")


if __name__ == "__main__":
    main()
