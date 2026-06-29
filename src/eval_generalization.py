"""Test 2 — compositional generalization (the decisive test).

Plain-language purpose: phases 1 and 2-Test-1 were negative on next-token *fit*.
The dendritic composition prior was always hypothesised to pay off on
*recombining known pieces in unseen ways*. This script measures that.

Two arms (selected by the checkpoint's dataset):
  toy_pcfg            : accuracy of predicting the compositional HARMONY token at
                        positions whose (parent, child) nesting was HELD OUT of
                        training, vs in-distribution positions. The in-dist ->
                        held-out GAP is the signal: a topology that composes
                        better should generalize the harmony rule with a smaller
                        gap. (Length-matched split, so the gap is not a
                        sequence-length artifact.)
  tinyshakespeare_bpe : perplexity on held-out corpus A vs an adjacent unseen
                        corpus B (transfer gap).

Usage:
    python src/eval_generalization.py results/analysis/test2_pcfg.png \
        swiglu_pcfg par_dend_pcfg seq_dend_pcfg par_dense_pcfg seq_dense_pcfg
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RunConfig
from data import load_dataset
from model import GPT

RESULTS = Path(__file__).resolve().parents[1] / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def rebuild_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = RunConfig(**ckpt["config"])
    vocab_size = len(ckpt["vocab"]["stoi"])
    model = GPT(cfg, vocab_size).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


# --- PCFG harmony-rule generalization ----------------------------------------

@torch.no_grad()
def pcfg_harmony_acc(model, cfg, dataset, n_strings: int = 2000):
    """Return (acc_indist, acc_heldout) of harmony-token prediction."""
    x, labels = dataset.gen_eval_batch(n_strings, cfg.context)
    x = x.to(DEVICE)
    logits, _ = model(x)
    pred = logits.argmax(-1).cpu()        # [N, L]: token predicted at each pos
    hit = {True: 0, False: 0}
    tot = {True: 0, False: 0}
    for r, lab in enumerate(labels):
        for pred_idx, target, held in lab:
            tot[held] += 1
            hit[held] += int(pred[r, pred_idx].item() == target)
    acc_in = hit[False] / max(1, tot[False])
    acc_held = hit[True] / max(1, tot[True])
    return acc_in, acc_held, tot[False], tot[True]


def analyze_pcfg(run_name: str, cache: dict):
    seed_dirs = sorted((RESULTS / run_name).glob("seed*"))
    if not seed_dirs:
        raise FileNotFoundError(f"no seed runs under results/{run_name}/")
    ins, helds, gaps = [], [], []
    n_in = n_held = 0
    for sd in seed_dirs:
        model, cfg = rebuild_model(sd / "ckpt.pt")
        ds = cache.setdefault(cfg.dataset, load_dataset(cfg.dataset))
        ai, ah, ni, nh = pcfg_harmony_acc(model, cfg, ds)
        ins.append(ai); helds.append(ah); gaps.append(ai - ah)
        n_in, n_held = ni, nh
    return {
        "n_seeds": len(seed_dirs),
        "in_mean": np.mean(ins), "in_std": np.std(ins),
        "held_mean": np.mean(helds), "held_std": np.std(helds),
        "gap_mean": np.mean(gaps), "gap_std": np.std(gaps),
        "n_in": n_in, "n_held": n_held,
    }


# --- BPE real-text transfer (held-out corpus A vs adjacent corpus B) ---------

@torch.no_grad()
def bpe_transfer_loss(model, cfg, dataset, n_batches: int = 200):
    """Mean cross-entropy (nats/token) on held-out A (val) and on corpus B."""
    out = {}
    for split in ("val", "corpus_b"):
        tot = 0.0
        for _ in range(n_batches):
            x, y = dataset.get_batch(split, cfg.batch_size, cfg.context, DEVICE)
            _, loss = model(x, y)
            tot += loss.item()
        out[split] = tot / n_batches
    return out["val"], out["corpus_b"]


def analyze_bpe(run_name: str, cache: dict):
    seed_dirs = sorted((RESULTS / run_name).glob("seed*"))
    if not seed_dirs:
        raise FileNotFoundError(f"no seed runs under results/{run_name}/")
    a_losses, b_losses, gaps = [], [], []
    for sd in seed_dirs:
        model, cfg = rebuild_model(sd / "ckpt.pt")
        ds = cache.setdefault(cfg.dataset, load_dataset(cfg.dataset))
        la, lb = bpe_transfer_loss(model, cfg, ds)
        a_losses.append(la); b_losses.append(lb); gaps.append(lb - la)
    return {
        "n_seeds": len(seed_dirs),
        "a_mean": np.mean(a_losses), "a_std": np.std(a_losses),
        "b_mean": np.mean(b_losses), "b_std": np.std(b_losses),
        "gap_mean": np.mean(gaps), "gap_std": np.std(gaps),
    }


def main_bpe(out_path: str, run_names: list[str]):
    cache: dict = {}
    rows = []
    print(f"\n{'variant':<22}{'seeds':>6}{'val loss (A)':>16}{'corpusB loss':>16}"
          f"{'transfer gap':>16}")
    for name in run_names:
        r = analyze_bpe(name, cache)
        rows.append((name, r))
        a_s = f"{r['a_mean']:.3f}+-{r['a_std']:.3f}"
        b_s = f"{r['b_mean']:.3f}+-{r['b_std']:.3f}"
        g_s = f"{r['gap_mean']:.3f}+-{r['gap_std']:.3f}"
        print(f"{name:<22}{r['n_seeds']:>6}{a_s:>16}{b_s:>16}{g_s:>16}")
    print("(loss in nats/token; transfer gap = corpusB - A, lower = generalizes better)")

    names = [n for n, _ in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, [r["gap_mean"] for _, r in rows],
           yerr=[r["gap_std"] for _, r in rows], capsize=4, color="steelblue")
    ax.set_ylabel("transfer gap (corpusB − A, nats/token)")
    ax.set_title("Test 2 — BPE real-text transfer to corpus B (mean ± std; lower = better)")
    ax.tick_params(axis="x", rotation=30); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"\n[plot] wrote {out_path}")


def main_pcfg(out_path: str, run_names: list[str]):
    cache: dict = {}
    rows = []
    print(f"\n{'variant':<20}{'seeds':>6}{'in-dist acc':>16}{'held-out acc':>16}"
          f"{'gap (in-held)':>16}")
    for name in run_names:
        r = analyze_pcfg(name, cache)
        rows.append((name, r))
        in_s = f"{r['in_mean']:.3f}+-{r['in_std']:.3f}"
        held_s = f"{r['held_mean']:.3f}+-{r['held_std']:.3f}"
        gap_s = f"{r['gap_mean']:.3f}+-{r['gap_std']:.3f}"
        print(f"{name:<20}{r['n_seeds']:>6}{in_s:>16}{held_s:>16}{gap_s:>16}")
    print(f"(eval positions per seed: in-dist~{rows[0][1]['n_in']}, "
          f"held-out~{rows[0][1]['n_held']}; lower gap = better composition)")

    names = [n for n, _ in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.38
    xs = np.arange(len(names))
    axes[0].bar(xs - width / 2, [r["in_mean"] for _, r in rows], width,
                yerr=[r["in_std"] for _, r in rows], capsize=3, label="in-dist")
    axes[0].bar(xs + width / 2, [r["held_mean"] for _, r in rows], width,
                yerr=[r["held_std"] for _, r in rows], capsize=3, label="held-out")
    axes[0].set_title("harmony-token accuracy (higher = better)")
    axes[0].set_xticks(xs); axes[0].set_xticklabels(names, rotation=30)
    axes[0].legend(); axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(names, [r["gap_mean"] for _, r in rows],
                yerr=[r["gap_std"] for _, r in rows], capsize=4, color="firebrick")
    axes[1].set_title("generalization gap = in-dist − held-out  (lower = better)")
    axes[1].tick_params(axis="x", rotation=30); axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Test 2 — compositional generalization on the toy PCFG (mean ± std)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"\n[plot] wrote {out_path}")


def main(out_path: str, run_names: list[str]):
    """Dispatch on the first run's dataset: PCFG harmony gap or BPE transfer."""
    _, cfg0 = rebuild_model(sorted((RESULTS / run_names[0]).glob("seed*"))[0] / "ckpt.pt")
    if cfg0.dataset == "toy_pcfg":
        main_pcfg(out_path, run_names)
    elif cfg0.dataset == "tinyshakespeare_bpe":
        main_bpe(out_path, run_names)
    else:
        raise ValueError(f"eval_generalization: unsupported dataset {cfg0.dataset!r}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python src/eval_generalization.py out.png <run> [<run> ...]")
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2:])
