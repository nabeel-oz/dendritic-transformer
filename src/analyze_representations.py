"""Test 3 (look inside) — cheap, robust disentanglement proxies.

Plain-language purpose: the central claim is that dendritic structure keeps
internal concepts tidier. This script measures two cheap proxies on the FFN's
internal activations, compared across variants at matched parameters:

  1. mean absolute pairwise correlation between unit activations (lower = units
     carry more distinct things = cleaner);
  2. participation ratio of the activation covariance (effective dimensionality;
     how many directions the representation actually uses).

It rebuilds each trained model from its checkpoint, captures the activation vector
fed into each FFN's final projection (a consistent "unit activation" across all
variants), and aggregates over seeds (mean +/- std). No new data needed.

Usage:
    python src/analyze_representations.py results/analysis/test3_proxies.png \
        v0_shakespeare deepflat_shakespeare v2free_shakespeare \
        v1strict_shakespeare v1equal_shakespeare
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
from ffn import (CompartmentalGatedFFN, DendriticBlock, DendriticFFN,
                 FlatDeepFFN, GatedFFN, PointFFN)
from model import GPT

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_BATCHES = 8          # val batches to pool activations over
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def penultimate_module(ffn):
    """The final projection layer of each FFN variant; its INPUT is the
    'unit activation' representation we probe.

    For the augmentative DendriticBlock we probe the SwiGLU main's `down` input
    (the soma's pre-output hidden — the representation projected toward the
    residual). Note: for the *parallel* topology this captures the main soma
    only, not the add-on path; this is a secondary proxy (Test 2 is decisive)."""
    if isinstance(ffn, PointFFN):
        return ffn.fc2
    if isinstance(ffn, FlatDeepFFN):
        return ffn.fc3
    if isinstance(ffn, DendriticFFN):
        return ffn.soma
    if isinstance(ffn, GatedFFN):
        return ffn.down
    if isinstance(ffn, CompartmentalGatedFFN):
        return ffn.soma
    if isinstance(ffn, DendriticBlock):
        return ffn.main.down
    raise TypeError(type(ffn))


def rebuild_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = RunConfig(**ckpt["config"])
    vocab_size = len(ckpt["vocab"]["stoi"])
    model = GPT(cfg, vocab_size).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def collect_activations(model, cfg, dataset):
    """Return list (per layer) of activation matrices [n_tokens, n_units]."""
    captured = [[] for _ in model.blocks]
    handles = []
    for li, block in enumerate(model.blocks):
        mod = penultimate_module(block.ffn)

        def hook(m, inp, _li=li):
            captured[_li].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).float().cpu())

        handles.append(mod.register_forward_pre_hook(hook))

    with torch.no_grad():
        for _ in range(N_BATCHES):
            x, _ = dataset.get_batch("val", cfg.batch_size, cfg.context, DEVICE)
            model(x)
    for h in handles:
        h.remove()
    return [torch.cat(c, 0).numpy() for c in captured]


def mean_abs_corr(A: np.ndarray) -> float:
    """Mean |correlation| over distinct unit pairs (lower = cleaner)."""
    std = A.std(0)
    A = A[:, std > 1e-8]                 # drop dead units
    if A.shape[1] < 2:
        return float("nan")
    C = np.corrcoef(A, rowvar=False)
    iu = np.triu_indices_from(C, k=1)
    return float(np.nanmean(np.abs(C[iu])))


def participation_ratio(A: np.ndarray) -> float:
    """(sum lambda)^2 / sum(lambda^2) of the activation covariance."""
    cov = np.cov(A, rowvar=False)
    lam = np.clip(np.linalg.eigvalsh(cov), 0, None)
    s1, s2 = lam.sum(), (lam ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else float("nan")


def analyze_variant(run_name: str, dataset_cache: dict):
    seed_dirs = sorted((RESULTS / run_name).glob("seed*"))
    if not seed_dirs:
        raise FileNotFoundError(f"no seed runs under results/{run_name}/")
    corrs, prs = [], []
    for sd in seed_dirs:
        model, cfg = rebuild_model(sd / "ckpt.pt")
        ds = dataset_cache.setdefault(cfg.dataset, load_dataset(cfg.dataset))
        acts = collect_activations(model, cfg, ds)
        corrs.append(np.mean([mean_abs_corr(A) for A in acts]))
        prs.append(np.mean([participation_ratio(A) for A in acts]))
    return (np.mean(corrs), np.std(corrs), np.mean(prs), np.std(prs), len(seed_dirs))


def main(out_path: str, run_names: list[str]):
    cache: dict = {}
    rows = []
    print(f"\n{'variant':<26}{'seeds':>6}{'mean|corr|':>16}{'participation':>18}")
    for name in run_names:
        cm, cs, pm, ps, n = analyze_variant(name, cache)
        rows.append((name, cm, cs, pm, ps))
        print(f"{name:<26}{n:>6}{f'{cm:.4f} +- {cs:.4f}':>16}{f'{pm:.1f} +- {ps:.1f}':>18}")

    names = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(names, [r[1] for r in rows], yerr=[r[2] for r in rows], capsize=4)
    axes[0].set_title("mean |pairwise correlation|  (lower = cleaner)")
    axes[1].bar(names, [r[3] for r in rows], yerr=[r[4] for r in rows], capsize=4)
    axes[1].set_title("participation ratio  (effective dimensionality)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Test 3 — internal disentanglement proxies (mean +/- std over seeds)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"\n[plot] wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python src/analyze_representations.py out.png <run> [<run> ...]")
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2:])
