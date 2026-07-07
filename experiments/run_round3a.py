"""Round 3a — the cheap decisive controls (build -> run -> checkpoint).

Answers three questions before any new architecture is built:
  1. DEPTH CONTROL. Is the phase-2 winner (seq_dense) just "FFN-internal depth"?
     A plain 2-deep SwiGLU stack (deep_swiglu) at matched params, on the XOR PCFG
     + BPE. If deep_swiglu ~= seq_dense, the dendrite story is a depth story.
  2. NON-XOR RULE. Does the effect survive a linearly-separable harmony rule?
     Re-run swiglu / seq_dense / par_dense / deep_swiglu on the OR-like PCFG.
     If the depth advantage collapses on OR, it was "depth helps XOR".
  3. WINNER'S CURSE. Re-seed the grid-minimum cell (seq_dense main_frac=0.25 on
     PCFG, the un-replicated 0.065) with 5 more seeds; expect regression.

Serial, one process per run (clean VRAM), mirroring run_sweep.py. Aggregate/eval
with the printed run-name lists afterwards.

Usage:
    python experiments/run_round3a.py                 # all three blocks, seeds 0 1 2
    python experiments/run_round3a.py --only matrix   # matrix | reseed
    python experiments/run_round3a.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = str(ROOT / "src" / "train.py")

# Block 1+2: depth control (new XOR + OR + BPE) and the three variants on OR.
# (XOR runs for swiglu/seq_dense/par_dense already exist from Test 2.)
MATRIX = [
    "configs/deep_swiglu_pcfg.yaml",
    "configs/deep_swiglu_pcfgor.yaml",
    "configs/deep_swiglu_bpe.yaml",
    "configs/swiglu_pcfgor.yaml",
    "configs/seq_dense_pcfgor.yaml",
    "configs/par_dense_pcfgor.yaml",
]

# Block 3: re-seed the winner's-curse cell. Run name matches run_sweep.py so the
# new seeds land in the SAME dir as the existing seeds 0,1,2.
RESEED_CFG = "configs/seq_dense_pcfg.yaml"
RESEED_NAME = "seq_dense_pcfg_mf025"
RESEED_SEEDS = [3, 4, 5, 6, 7]


def run(cfg: str, seed: int, extra: list[str] | None = None):
    cmd = [sys.executable, TRAIN, "--config", str(ROOT / cfg), "--seed", str(seed)]
    cmd += extra or []
    print(f"\n========== {cfg}  seed={seed}  {' '.join(extra or [])} ==========",
          flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--only", choices=["matrix", "reseed"], default=None,
                    help="run just one block (default: both)")
    args = ap.parse_args()

    do_matrix = args.only in (None, "matrix")
    do_reseed = args.only in (None, "reseed")

    if do_matrix:
        total = len(MATRIX) * len(args.seeds)
        i = 0
        for cfg in MATRIX:
            for seed in args.seeds:
                i += 1
                print(f"\n###### matrix [{i}/{total}] ######", flush=True)
                run(cfg, seed)

    if do_reseed:
        for j, seed in enumerate(RESEED_SEEDS, 1):
            print(f"\n###### reseed [{j}/{len(RESEED_SEEDS)}] ######", flush=True)
            run(RESEED_CFG, seed,
                ["--main-frac", "0.25", "--run-name", RESEED_NAME])

    print("\n===ROUND 3a RUNS DONE===")
    print("Then evaluate:")
    print("  # XOR depth control vs winner")
    print("  python src/eval_generalization.py results/analysis/round3a_pcfg.png "
          "swiglu_pcfg deep_swiglu_pcfg seq_dense_pcfg par_dense_pcfg")
    print("  # OR-rule arm")
    print("  python src/eval_generalization.py results/analysis/round3a_pcfgor.png "
          "swiglu_pcfgor deep_swiglu_pcfgor seq_dense_pcfgor par_dense_pcfgor")
    print("  # BPE fit/transfer")
    print("  python src/eval_generalization.py results/analysis/round3a_bpe.png "
          "swiglu_bpe deep_swiglu_bpe seq_dense_bpe")
    print(f"  # re-seeded winner's-curse cell (8 seeds now): {RESEED_NAME}")


if __name__ == "__main__":
    main()
