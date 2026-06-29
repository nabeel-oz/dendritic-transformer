# Dendritic FFN — Experiment Spec (phase 2)

Builds on the existing `src/ffn.py` from phase 1 (which already has `PointFFN`,
`DendriticFFN`, `FlatDeepFFN`, `count_params`, `solve_branch_width`, `solve_flat_deep_width`,
`reference_point_params`, `assert_param_parity`). 

---

## 1. The question

Phase 1 was a clean negative: *substitutive* dendrites (branches that **replace** the direct
path) hurt, because locality starves the unit. We're not revisiting that. The phase-2 question
is narrower and sharper:

> **Do dendrites, added as an *augmentation* on top of a preserved main path, beat a standard
> SwiGLU FFN at matched parameters — and is any gain due to their compartmental structure
> rather than just having an extra path?**

Two honest wirings of the same idea, both preserving the direct signal `x` (so neither can
starve a unit):

```
parallel    out = main(x) + scale·d(x)      # dendritic features added at the end
sequential  out = main(x + scale·d(x))      # dendritic features feed the soma (more expressive)
```

`main` is **SwiGLU** (the production-standard FFN, and the bar to beat). The comparison is a
budget split: the baseline spends its entire parameter budget on SwiGLU; a dendritic block
spends *some* on a smaller SwiGLU main and the rest on a dendritic add-on. If the split wins,
dendrites earned their parameters.

---

## 2. The six variants (the whole Tier-1 run)

All parameter-matched to the phase-1 budget (`reference_point_params`), 3 seeds.

| `cfg.ffn` | what it is | role |
|---|---|---|
| **point** | V0 plain FFN, full budget | sanity anchor (SwiGLU must beat it) |
| **swiglu** | SwiGLU, full budget | **the baseline to beat** |
| **par_dend** | SwiGLU main + dendritic add-on (parallel) | parallel dendritic |
| **par_dense** | SwiGLU main + dense add-on (parallel) | parallel control |
| **seq_dend** | SwiGLU main of `x + dendrites(x)` (sequential) | sequential dendritic |
| **seq_dense** | SwiGLU main of `x + dense(x)` (sequential) | sequential control |

The dendritic add-on is the existing compartmentalized `DendriticFFN` (d→d); the dense add-on
is `FlatDeepFFN` (d→d) at the same budget — same depth/size, no compartments. Default
connectivity `sparse` (capacity-honest / "equal").

### Contrasts and what each isolates

| contrast | answers |
|---|---|
| `par_dend` vs `swiglu` | does the parallel dendritic add-on beat standard SwiGLU? *(primary)* |
| `seq_dend` vs `swiglu` | does the sequential dendritic add-on beat standard SwiGLU? *(primary)* |
| `par_dend` vs `par_dense` | is it the **compartmentalization**, or would any add-on do? *(attribution)* |
| `seq_dend` vs `seq_dense` | same attribution, sequential slot |
| `seq_dend` vs `par_dend` | which topology, if either, works |
| `swiglu` vs `point` | harness sanity (known result: SwiGLU wins) |

### What we deliberately cut (and why)
- **`deep_flat`** — its job (control for extra depth) is now done by the `*_dense` controls in-slot.
- **`masked`/strict connectivity** — folded into a *confirmation* run (Tier 2), used only if the
  fair `sparse` version shows a gain, to rule out a capacity artifact.
- **point-main variants** — the goal is "beat SwiGLU," so point-main is only a Tier-2 *diagnostic*
  (it would distinguish a flat null from "redundant with gating").

---

## 3. Code (new pieces; the rest already exists in `ffn.py`)

```python
import torch, torch.nn as nn, torch.nn.functional as F

class GatedFFN(nn.Module):
    """SwiGLU — the production-standard FFN (LLaMA/PaLM/Mistral). in = out = d_model."""
    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up   = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))
    def allocated_params(self): return count_params(self)
    def active_params(self):    return count_params(self)


def solve_glu_hidden(d: int, target: int) -> int:
    """SwiGLU hidden h (bias-free, params = 3*d*h) best-matching `target`."""
    best_h, best_diff = 1, None
    for h in range(1, 8 * d):
        n = 3 * d * h
        diff = abs(n - target)
        if best_diff is None or diff < best_diff:
            best_h, best_diff = h, diff
        elif n - target > best_diff:
            break
    return best_h


class DendriticBlock(nn.Module):
    """Augmentative dendritic FFN. The soma (main) is PRESERVED; dendrites only ADD.
        parallel   : out = main(x) + scale * addon(x)
        sequential : out = main(x + scale * addon(x))
    addon : d -> d   (dendritic = compartmentalized DendriticFFN ; or dense = FlatDeepFFN control)
    main  : the soma — GatedFFN (SwiGLU).
    scale : small NONZERO init (0.1) so at init the block ~= `main` and the add-on earns its
            way in. Do NOT init at 0 (that freezes the add-on's gradient)."""
    def __init__(self, addon: nn.Module, main: nn.Module, topology: str):
        super().__init__()
        assert topology in ("parallel", "sequential")
        self.addon, self.main, self.topology = addon, main, topology
        self.scale = nn.Parameter(torch.full((1,), 0.1))
    def forward(self, x):
        a = self.scale * self.addon(x)
        return self.main(x) + a if self.topology == "parallel" else self.main(x + a)
    def allocated_params(self): return count_params(self)
    def active_params(self):
        return self.addon.active_params() + self.main.active_params() + self.scale.numel()
```

### Factory wiring (extend `build_ffn`)

```python
    if kind == "swiglu":
        return GatedFFN(d, solve_glu_hidden(d, target), p)

    if kind in ("par_dend", "par_dense", "seq_dend", "seq_dense"):
        topology = "parallel" if kind.startswith("par_") else "sequential"
        is_dend  = kind.endswith("_dend")
        main_frac = getattr(cfg, "main_frac", 0.5)            # budget split; the key Tier-2 dial
        conn = getattr(cfg, "connectivity", "sparse")          # 'sparse' = capacity-honest default

        addon_budget = target - round(target * main_frac)
        if is_dend:
            wb = solve_branch_width(d, B, addon_budget, sparse=(conn == "sparse"))
            addon = DendriticFFN(d, B, wb, wb, conn, p)        # d -> d, compartmentalized
        else:
            addon = FlatDeepFFN(d, solve_flat_deep_width(d, addon_budget), p)  # d -> d, dense control

        main_budget = target - addon.allocated_params() - 1    # -1 for the scale scalar
        main = GatedFFN(d, solve_glu_hidden(d, main_budget), p)
        return DendriticBlock(addon, main, topology)
```

### Parity (generalize `parity_metric`) + init

```python
def parity_metric(cfg, ffn):
    if cfg.ffn in ("par_dend", "seq_dend") and getattr(cfg, "connectivity", "sparse") == "sparse":
        return "active", ffn.active_params()          # capacity-honest variants match on active
    return "allocated", ffn.allocated_params()
```

`assert_param_parity` is unchanged. Apply the shared `init_ffn_weights` (addendum #1) to every
built FFN — including `GatedFFN` and both submodules of `DendriticBlock` — so no variant gets an
init edge. If parity ever trips the 1% tol, nudge `main_frac` (solver rounding leaves a small
remainder).

---

## 4. Decision logic

Primary metric: final val loss (Test 1), mean ± std over ≥3 seeds. Read the two primaries and
the two attribution contrasts together:

- **A — it works.** A dendritic variant is ≤ `swiglu` (better or within noise) *and* < its
  `*_dense` control. → The structure does real work the gate doesn't. Proceed to Test 2 and the
  Tier-2 confirmations.
- **B — clean null.** `*_dend ≈ *_dense ≈ swiglu` (or worse). → Adding a path doesn't help and
  compartmentalization is irrelevant, *even in the faithful augmentative form that can't starve*.
  A strong, reportable null.
- **C — structure helps but not enough.** `*_dend < *_dense` but both worse than `swiglu`. → The
  compartmentalization does something, but the 50/50 split handicaps the main. Sweep `main_frac`
  before concluding either way.

**Test 2 (generalization) is the decisive test regardless of A/B/C** — the whole project thesis
is that any benefit shows up in recombination, not in-distribution fit. Build it next (it needs
new data code) and run it at least on `{swiglu, best *_dend, its *_dense}`.

---

## 5. Run plan — serial batches, mind the 8 GB

Each ~1.8M model uses ~0.3–0.5 GB and finishes in seconds–minutes, **but do not run them in
parallel** — concurrent processes contend for VRAM and thrash. Run a **serial queue, one
process per run** (process exit frees VRAM cleanly), in tiers:

- **Tier 0 — smoke (1 seed, ~800 steps, all six).** Catch bugs; confirm every
  `assert_param_parity` passes. Never judge results on this.
- **Tier 1 — the experiment (3 seeds, 2500 steps, all six).** Plus the cheap Test-3 proxies
  (mean |pairwise corr|, participation ratio) on the checkpoints as a *secondary* readout.
- **Test 2 — generalization (decisive).** Build `data/toy_language.py` (PCFG with a held-out
  compositional split) + a real-text transfer arm; run on the survivors.
- **Tier 2 — only if Tier 1/Test 2 is positive:** `masked`/strict confirmation (rule out capacity
  artifact); `main_frac` sweep; point-main diagnostic (null vs redundant-with-gating); then
  `branches`, `scale` init, `combine=concat`, `branch_act=tanh`, `branch_combine=gate`.

```bash
for v in point swiglu par_dend par_dense seq_dend seq_dense; do
  for s in 0 1 2; do
    python src/train.py --config configs/${v}.yaml --seed ${s}   # one at a time
  done
done
```

---

## 6. Pre-registered predictions (state in README before running)

1. **No starvation.** `x` is preserved, so no phase-1-style collapse; dendritic variants sit near
   their baselines, not +0.13 nats adrift.
2. **Attribution.** If compartmentalization matters: `*_dend < *_dense`.
3. **The decisive bar.** A dendritic variant must reach ≤ `swiglu` to count as a benefit over the
   standard.
4. **Topology.** `sequential ≥ parallel` (the soma computing *on* the dendritic features is
   strictly more expressive).

**Deflationary null, stated up front:** if no dendritic variant beats its dense control *or*
SwiGLU, then compartmental dendritic structure adds nothing even in the most faithful, most
expressive, non-starving form — which, with a null on Test 2, responsibly closes phase 1.

---

## 7. What still applies
- Parameter parity is the floor; log tokens, FLOPs, **and** wall-clock every run (sequential adds
  a layer in series → more wall-clock; report it).
- ≥3 seeds; a gap inside the noise band is not a result.
- Defaults: `branches=4`, `main_frac=0.5`, `connectivity="sparse"`, `branch_act="gelu"`.
  Vary one axis at a time; let the data, not intuition, pick the topology.
- Backbone unchanged from phase 1 (`d_model=192, n_layer=4, n_head=6, context=128, ffn_mult=4`).
