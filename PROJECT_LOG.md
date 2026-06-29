# Dendritic Transformer — Project Log

*A running log of what was built, when, and what each step found. For the accessible
overview see [README.md](README.md); for the design rationale see the project briefs; for the
full results see the phase reports.*

Research idea #1 of **Architectures of Mind** — mapping neuroscience observations to
computational analogues. The full motivation, hypothesis, and experimental design live in
[PROJECT_BRIEF_PHASE1.md](PROJECT_BRIEF_PHASE1.md). This log covers what is built and how to
reproduce it.

## The prediction (stated up front, before results)

We replace only the feed-forward (FFN) neurons inside a transformer with **dendritic** ones
— units that sort their inputs into groups, process each group locally, then combine — and
predict three fingerprints, *all three together*:

1. **Learns from less** — reaches coherent output in fewer training steps.
2. **Travels to the new** — generalizes better to unseen recombinations of familiar pieces.
3. **Cleaner internals** — more monosemantic units (the real claim; the symptoms are
   downstream of it).

**The non-negotiable rule:** every comparison holds *total trainable parameters equal* and
reports compute alongside. Any advantage must come from how a unit is *organized*, never
from extra capacity. Negative results are real results and will be reported plainly.

## Status

| Step | What | State |
|------|------|-------|
| 1 | Shared backbone + **V0 standard point FFN**, char-level TinyShakespeare — sanity gate | ✅ built |
| 2 | V1.strict / V1.equal (structured) + V2 (free) FFNs + parameter-parity assert | ✅ built |
| 3 | Test 1 rig: multi-seed loss curves + samples + mean±std aggregation + compute (tokens/wall/FLOPs) | ✅ built |
| — | `deep_flat` depth-matched control + `analyze_representations.py` (Test 3 proxies) + `branch_act` flag | ✅ built |
| 4–6 | Toy-language PCFG, Test 2 (generalization), Test 3 selectivity/concepts-per-unit (needs toy labels) | ⬜ next |
| P2 | **Phase 2** — augmentative dendrites vs SwiGLU (Tier 1: 6 variants × 3 seeds) | ✅ run — clean null |
| P2.2 | Phase 2 Test 2 — compositional generalization (toy PCFG + BPE transfer) | ✅ run — qualified positive |

The FFN is a single module selected by the `ffn` config string (`point` | `deep_flat` |
`dendritic_free` | `dendritic_structured_strict` | `dendritic_structured_equal`); the
backbone never branches on which it holds, so variants drop in without touching
attention/embeddings/norms.

## Phase 2 — augmentative dendrites vs SwiGLU (pre-registration, before results)

Phase 1 was a clean negative for *substitutive* dendrites (branches that **replace** the
direct path: input locality starves the unit). Phase 2 asks a narrower, fairer question
([PROJECT_BRIEF_PHASE2.md](PROJECT_BRIEF_PHASE2.md)):

> Do dendrites, added as an **augmentation** on top of a *preserved* main path, beat a standard
> SwiGLU FFN at matched parameters — and is any gain due to their compartmental structure rather
> than just having an extra path?

Two non-starving wirings preserve `x`: `parallel out = main(x) + scale·addon(x)` and
`sequential out = main(x + scale·addon(x))`. `main` is **SwiGLU** (the production bar to beat).
To keep the comparison about *topology only*, the dendritic add-on is itself **gated** — a
SwiGLU with compartmentalized (block-diagonal, slice-local) `gate`/`up` and a dense integrating
soma — and the `*_dense` control is a plain SwiGLU add-on of the same budget. So gating, depth,
and integration are held constant and only **compartmentalization** varies. Six Tier-1 variants
(`point`, `swiglu`, `par_dend`, `par_dense`, `seq_dend`, `seq_dense`), 3 seeds, matched params.

**Predictions (stated before running):**
1. **No starvation.** `x` is preserved, so no phase-1-style collapse; dendritic variants sit
   near their baselines, not adrift.
2. **Attribution.** If compartmentalization matters: `*_dend < *_dense`.
3. **The decisive bar.** A dendritic variant must reach ≤ `swiglu` to count as a benefit.
4. **Topology.** `sequential ≥ parallel` (the soma computing *on* the dendrites is more expressive).

**Deflationary null, up front:** if no dendritic variant beats its dense control *or* SwiGLU,
then compartmental dendritic structure adds nothing even in the most faithful, non-starving,
gating-matched form. Test 2 (compositional generalization) remains the decisive follow-up.

### Phase-2 Tier-1 results (char-level TinyShakespeare, 3 seeds, 2500 steps)

**Test 1 — final val loss (mean ± std):**

| variant | final val loss | vs `swiglu` | wall-clock† |
|---|---|---|---|
| par_dense (control) | **1.514 ± 0.002** | −0.005 | ~184 s |
| seq_dense (control) | **1.514 ± 0.007** | −0.005 | ~60 s |
| swiglu (baseline) | 1.519 ± 0.003 | — | ~122 s |
| point (V0) | 1.520 ± 0.004 | +0.001 | ~131 s |
| par_dend | 1.525 ± 0.003 | +0.006 | ~226 s |
| seq_dend | 1.529 ± 0.001 | +0.010 | ~232 s |

†Wall-clock measured under background contention — treat as approximate; the dendritic
einsum variants are clearly ~1.8–1.9× slower than `swiglu` (same penalty as phase 1).

**Test 3 — internal proxies (secondary):**

| variant | mean \|corr\| (↓ cleaner) | participation (eff. dim) |
|---|---|---|
| swiglu | **0.0585** | **92.7** |
| par_dense | 0.0656 | 66.1 |
| par_dend | 0.0673 | 58.0 |
| seq_dend | 0.0702 | 57.9 |
| seq_dense | 0.0744 | 51.6 |
| point | 0.1145 | 47.3 |

**Verdict — Case B clean null (mildly negative):**
1. **No starvation (✅).** Every augmentative variant sits within ~0.01 of `swiglu` — the
   preserved-`x` redesign fixed phase 1's collapse.
2. **Attribution reversed (✗).** `par_dend` (1.525) > `par_dense` (1.514) and `seq_dend` (1.529)
   > `seq_dense` (1.514), both beyond seed noise — compartmentalization does slightly *negative*
   work vs a plain-SwiGLU add-on.
3. **Decisive bar not cleared (✗).** Both dendritic variants land above `swiglu`.
4. **Topology (✗).** `sequential` is not better than `parallel`.

Test 3 agrees: `swiglu` has the cleanest, highest-dimensional internals; dendritic ≈ dense, both
dirtier than plain SwiGLU. So in the most faithful (augmentative, gating- and depth-matched) form,
compartmentalization still buys nothing on in-distribution modeling. One non-robust wrinkle: the
*dense* add-ons edge `swiglu` by −0.005 (~1.4σ) — "an extra path helps a hair," but
compartmentalizing it removes the edge. Full detail in [REPORT_PHASE2.md](REPORT_PHASE2.md).
Plots: `results/agg_phase2.png`, `results/analysis/phase2_proxies.png`.

### Phase-2 Test 2 — compositional generalization (the decisive test)

Char-level flat modeling doesn't stress the FFN. Test 2 moves to tasks that do: a **recursive
PCFG** with a *harmony rule* (`H = parity(color(parent), color(child))`) and a **length-matched
recombination split** (specific parent→child nestings held out of training), plus a **subword-BPE**
real-text arm transferring TinyShakespeare → King James Bible (corpus B). 5 survivors, 3 seeds.

**PCFG — harmony-rule generalization gap (in-dist − held-out, ↓ better):**

| variant | in-dist acc | held-out acc | gap |
|---|---|---|---|
| swiglu | 0.995 | 0.788 | **0.207** |
| par_dend | 0.994 | 0.827 | 0.166 |
| seq_dend | 0.996 | 0.859 | 0.137 |
| par_dense | 0.996 | 0.859 | 0.136 |
| seq_dense | 0.994 | 0.892 | **0.102** |

**BPE — in-dist loss A & transfer gap to corpus B (nats/token, ↓ better):**

| variant | val loss A | transfer gap (B−A) |
|---|---|---|
| swiglu | 5.143 | **2.963** |
| par_dend | 4.878 | 3.050 |
| seq_dend | **4.720** | 3.233 |
| par_dense | 4.980 | 3.136 |
| seq_dense | 4.763 | 3.289 |

**Reading — a qualified positive, but not for the headline claim:**
- **The augmentative gated path earns its parameters.** On both structured tasks every
  augmentative variant beats SwiGLU in-distribution (PCFG −0.04…−0.06, BPE −0.15…−0.43 nats, all
  robust), and the best one roughly **halves** the PCFG compositional gap (seq_dense 0.102 vs
  swiglu 0.207). It was neutral *only* on flat char-level — so the phase-1 null was partly a
  low-power regime.
- **Compartmentalization helps fit, not generalization.** On BPE it improves in-distribution loss
  over its dense twin (`*_dend` < `*_dense`, robust) — a reversal of phase 1 — but it does **not**
  beat the dense control on either generalization measure (PCFG gap, BPE transfer). The plain
  gated path captures the generalization benefit; compartments add capacity-efficiency.
- **Sequential ≥ parallel** holds on the structured tasks.

So the central hypothesis (compartments → better composition) stays **unsupported**, while a real,
parameter-fair secondary result emerges: the **augmentative gated topology** helps once the task
stresses the FFN. Full detail, significance, and threats in [REPORT_PHASE2.md](REPORT_PHASE2.md).
Plots: `results/analysis/test2_pcfg.png`, `results/analysis/test2_bpe.png`.

**`main_frac` / `branches` sweep (seq_dend, seq_dense; both tasks).** Asked whether a "best of both
worlds" exists. It doesn't — via compartments. (1) Fit improves monotonically as *more* budget goes
to the add-on (PCFG 1.62→1.54, BPE 5.15→4.42 at main_frac 0.25), so "preserve the main neurons" is
backwards. (2) The best compositional generalizer is the *dense* large add-on (`seq_dense`,
main_frac 0.25: gap **0.065** vs SwiGLU 0.207, ~3.5σ); compartmentalizing it is worse, and more
branches worse still. (3) Compartments' only robust edge is real-text fit, which does not transfer.
Net: the lever is a *large sequential gated add-on*; compartments trade generalization for a little
fit. Plots: `results/sweep_{pcfg,bpe}_{loss,gap}.png`. See [REPORT_PHASE2.md §6.3](REPORT_PHASE2.md).

## Results so far (honest log)

All variants parameter-matched to V0, char-level TinyShakespeare, 3 seeds, 2500 steps.
`deep_flat` is a depth- and parameter-matched flat control (3 weight layers / 2
nonlinearities, no compartments) added to separate *depth* from *compartmentalization*.

**Test 1 — race to coherence (final val loss, mean ± std):**

| variant | final val loss | gap vs V0 | wall-clock |
|---|---|---|---|
| V0 point | **1.520 ± 0.004** | — | 42 s |
| deep_flat | 1.555 ± 0.003 | +0.035 | 43 s |
| V2 free | 1.550 ± 0.004 | +0.030 | 62 s |
| V1.equal | 1.644 ± 0.011 | +0.124 | 78 s |
| V1.strict | 1.658 ± 0.005 | +0.138 | 63 s |

**Test 3 — internal disentanglement proxies (mean ± std):**

| variant | mean \|pairwise corr\| (↓ cleaner) | participation ratio (eff. dim) |
|---|---|---|
| V0 point | **0.115 ± 0.003** | **46.8 ± 1.8** |
| deep_flat | 0.145 ± 0.001 | 26.6 ± 0.5 |
| V2 free | 0.141 ± 0.004 | 32.3 ± 2.5 |
| V1.equal | 0.179 ± 0.009 | 14.8 ± 2.5 |
| V1.strict | 0.191 ± 0.004 | 13.7 ± 1.3 |

**Reading (negative result, reported plainly):**
- **The V2-vs-V0 loss gap was a depth confound, now isolated.** `deep_flat` (+0.035) ≈ V2
  (+0.030): adding a layer at fixed params is what costs fit; compartmentalization itself is
  roughly loss-neutral (V2 ≈ deep_flat). The **locality prior** (V1, +0.12–0.14) is the real,
  robust loss cost. Symptom 1 ("learns from less") stays **rejected**.
- **Cleaner internals — also not supported.** V0 has the *lowest* unit correlation and
  *highest* effective dimensionality; dendritic variants (esp. V1) are *more* correlated and
  *lower*-dimensional — i.e. more redundant, more entangled, not less. Gaps ≫ seed noise.
- **Compute parity ≠ FLOP/wall-clock parity.** FLOPs are ~equal (~247 TFLOP) but dendritic
  wall-clock is 1.5–1.9× V0's — branched einsums are less hardware-efficient.
- **Saturating branches (`branch_act=tanh`) don't rescue it.** tanh costs further loss
  (+0.04–0.11) and only marginally/inconsistently shifts the proxies (slightly lower
  correlation for V1, worse for V2). Not a fix.
- **Still untested:** compositional generalization (Test 2, toy language) and label-based
  Test 3 (selectivity, concepts-per-unit). The locality prior is *designed* for the
  compositional regime, so that remains the fairest place to look for an upside.

Plots: `results/agg_test1.png`, `results/agg_test1_tanh.png`,
`results/analysis/test3_proxies.png`, `results/analysis/test3_tanh.png`.

## Setup

Python 3.13, NVIDIA GPU recommended (developed on an RTX 4060, 8 GB). Install PyTorch with
CUDA, then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Reproduce the Step-1 sanity run

```bash
python src/train.py --config configs/v0_shakespeare.yaml   # downloads TinyShakespeare on first run
python src/plotting.py results/v0_shakespeare              # writes loss_curve.png
```

Outputs land in `results/v0_shakespeare/`: `metrics.csv` (step, tokens, wall-clock, train/val
loss), printed text samples during training, `ckpt.pt`, and `loss_curve.png`. The run takes a
few minutes on the 4060 and stays well under the 8 GB VRAM budget.

**Pass criteria:** train and val loss fall steadily (char-level loss from ~4.2 toward ~1.5)
and generated samples progress from random characters to plausible Shakespeare-like text.

## Layout

```
configs/        one YAML per run (declares variant + dataset + seed)
src/
  config.py     RunConfig dataclass + YAML loader
  model.py      shared GPT backbone; FFN selected by config
  ffn.py        FFN registry + V0 point FFN + count_params (parity basis)
  data/         real_text.py (TinyShakespeare); toy_language.py (later)
  train.py      training loop; CSV logging; checkpoint; samples
  plotting.py   loss curves
results/        logs, checkpoints, figures (gitignored)
```
