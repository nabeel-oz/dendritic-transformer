"""Aggregate multi-seed runs: mean +/- std of val loss per step, table + plot.

A gap inside the noise band is not a result (PROJECT_BRIEF_PHASE1.md section 5). This
script makes the noise band explicit.

Usage:
    python src/aggregate.py results/agg_test1.png \
        v0_shakespeare v2free_shakespeare v1strict_shakespeare v1equal_shakespeare
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load_variant(run_name: str):
    """Return (steps, loss_matrix[n_seeds, n_steps], n_seeds, compute) for one
    variant. `compute` holds final-row tokens / wall-clock / FLOPs (seed means)."""
    seed_dirs = sorted((RESULTS / run_name).glob("seed*"))
    if not seed_dirs:
        raise FileNotFoundError(f"no seed runs under results/{run_name}/")
    steps, rows, fin_tok, fin_wall, fin_flops = None, [], [], [], []
    for sd in seed_dirs:
        s, v, last = [], [], None
        with open(sd / "metrics.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                s.append(int(r["step"]))
                v.append(float(r["val_loss"]))
                last = r
        steps = s if steps is None else steps
        rows.append(v)
        fin_tok.append(int(last["tokens_seen"]))
        fin_wall.append(float(last["wall_clock_s"]))
        fin_flops.append(float(last.get("flops", "nan")))
    compute = {
        "tokens": float(np.mean(fin_tok)),
        "wall_s": float(np.mean(fin_wall)),
        "flops": float(np.mean(fin_flops)),
    }
    return np.array(steps), np.array(rows), len(seed_dirs), compute


def main(out_path: str, run_names: list[str]):
    fig, ax = plt.subplots(figsize=(9, 6))
    print(f"\n{'variant':<26}{'seeds':>6}{'final val (mean+-std)':>24}"
          f"{'Mtok':>8}{'wall_s':>9}{'TFLOP':>9}")
    summary = []
    for name in run_names:
        steps, loss, n, comp = load_variant(name)
        mean, std = loss.mean(0), loss.std(0)
        ax.plot(steps, mean, marker=".", label=f"{name} (n={n})")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
        print(f"{name:<26}{n:>6}{f'{mean[-1]:.3f} +- {std[-1]:.3f}':>24}"
              f"{comp['tokens']/1e6:>8.1f}{comp['wall_s']:>9.0f}"
              f"{comp['flops']/1e12:>9.1f}")
        summary.append((name, mean[-1], std[-1]))

    ax.set_xlabel("steps")
    ax.set_ylabel("val loss (nats/char)")
    ax.set_title("Test 1 — race to coherence (mean +/- std over seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\n[plot] wrote {out_path}")
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python src/aggregate.py out.png <run_name> [<run_name> ...]")
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2:])
