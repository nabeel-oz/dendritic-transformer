"""Plot train/val loss from a run's metrics.csv.

Usage:
    python src/plotting.py results/v0_shakespeare
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, just write a PNG
import matplotlib.pyplot as plt


def load_metrics(run_dir: Path):
    rows = []
    with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def plot_run(run_dir: Path):
    rows = load_metrics(run_dir)
    steps = [int(r["step"]) for r in rows]
    tokens = [int(r["tokens_seen"]) / 1e6 for r in rows]
    train = [float(r["train_loss"]) for r in rows]
    val = [float(r["val_loss"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, xvals, xlabel in ((axes[0], steps, "steps"),
                              (axes[1], tokens, "tokens seen (M)")):
        ax.plot(xvals, train, label="train")
        ax.plot(xvals, val, label="val")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("cross-entropy loss (nats/char)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"{run_dir.name} — loss curves")
    fig.tight_layout()

    out = run_dir / "loss_curve.png"
    fig.savefig(out, dpi=120)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python src/plotting.py results/<run_name>")
        raise SystemExit(1)
    plot_run(Path(sys.argv[1]))
