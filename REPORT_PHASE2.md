# Dendritic Transformer — Phase 2 Report (Tier 1: augmentative dendrites vs SwiGLU)

*Architectures of Mind, research idea #1. Phase 2 reframes the phase-1 question after its clean
negative on substitutive dendrites. Status: Tier 1 (6 variants, char-level Test 1 + Test-3
proxies) AND Test 2 (compositional generalization — a recursive PCFG with a length-matched
recombination split, plus a subword-BPE real-text transfer arm) complete; 3 seeds throughout.*

---

## 1. TL;DR

Phase 1 showed *substitutive* dendrites (branches that **replace** the direct path) hurt:
input locality starves the unit. Phase 2 asks the fairer, narrower question:

> Do dendrites, added as an **augmentation** on a *preserved* main path, beat a standard SwiGLU
> FFN at matched parameters — and is any gain due to compartmental structure rather than just
> having an extra path?

To make this a test of *topology only*, the dendritic add-on is itself **gated** (a SwiGLU with
compartmentalized, block-diagonal gate/up + a dense integrating soma), and its control is a plain
SwiGLU add-on of equal budget. Gating, depth, and integration are held constant; only
compartmentalization varies.

**Result (two-part, and more interesting than phase 1's flat null):**

- **Test 1 (char-level loss): a clean null.** The redesign fixed starvation — every augmentative
  variant sits within ~0.01 nats of SwiGLU — but compartmentalization does *not* beat its
  plain-SwiGLU control, and no dendritic variant beats full SwiGLU. Internal proxies agree
  (SwiGLU is cleanest). Flat char-level modeling does not exercise the prior.

- **Test 2 (structured + subword tasks): a qualified, partial positive — but not for the
  hypothesis as stated.** On a compositional PCFG and on subword real text, the **augmentative
  gated path itself** is a robust win over plain SwiGLU (PCFG in-dist −0.04 to −0.06 nats; BPE
  in-dist −0.15 to −0.43 nats; PCFG compositional-generalization gap roughly halved by the best
  augmentative variant). **Compartmentalization** further helps *in-distribution real-text fit*
  (`*_dend` < `*_dense`, robust on BPE) — a reversal of phase 1 — **but does not improve
  generalization**: it does not beat its dense control on the PCFG harmony gap or the BPE
  transfer gap. **Sequential ≥ parallel** holds on the structured tasks.

**Bottom line.** The dendritic *augmentation* earns its parameters on tasks that actually stress
the FFN; the dendritic *compartmentalization* helps fit but is still not the source of better
*generalization* — the plain-gated extra path matches or beats it there. The central hypothesis
(compartments → better composition) remains unsupported; a real and reportable secondary finding
(augmentation, and compartmentalization-for-fit, both help once the regime stresses the FFN)
emerges. The phase-1 char-level null was partly a low-power regime, vindicating that concern.

---

## 2. What changed from phase 1, and why

| | Phase 1 (substitutive) | Phase 2 (augmentative) |
|---|---|---|
| dendrites | **replace** the direct path | **add** to a preserved main path |
| can starve a unit? | yes (locality removes the direct signal) | no (`x` always preserved) |
| baseline | plain GELU point FFN (V0) | **SwiGLU** (production standard) |
| add-on primitive | non-gated GELU MLP | **gated** (SwiGLU) — see below |

**Design decision — gating held constant.** The round-2 brief originally specified the phase-1
non-gated `DendriticFFN` as the add-on. That would confound the primary "beat SwiGLU" contrast:
~half the dendritic block's budget would sit in a non-gated primitive we already know is weaker
than SwiGLU (the `point` vs `swiglu` result), so a loss couldn't separate "compartmentalization
doesn't help" from "half the budget is non-gated." Making the add-on a **compartmentalized
SwiGLU** removes that confound — gating, depth, and the integrating projection are identical
across all arms, so only compartmentalization varies. (Gating also sidesteps the phase-1
collapse trap: `silu(gate)·up` is already non-collapsible, so no two-nonlinearity branch trick is
needed — the module is simpler than `DendriticFFN`.)

```
dend add-on (CompartmentalGatedFFN): block-diagonal gate/up (each branch reads its d/B slice)
                                     → silu·mul → dense soma → d
dense add-on (plain SwiGLU):         dense gate/up (reads all of x)
                                     → silu·mul → dense down → d
```

The two wirings, both preserving `x`:
```
parallel    out = main(x) + scale·addon(x)
sequential  out = main(x + scale·addon(x))      # the soma computes ON the dendrites
```
`scale` is a learnable scalar init 0.1 (non-zero so the add-on receives gradient from step 0).

---

## 3. The six variants and parity

Backbone unchanged from phase 1 (`d_model=192, n_layer=4, n_head=6, context=128, ffn_mult=4`,
~1.8M params). Only the FFN sub-layer changes. Budget split `main_frac=0.5`, `branches=4`,
`connectivity=sparse`.

| `cfg.ffn` | what | role | parity (per-FFN, V0=295,872) |
|---|---|---|---|
| `point` | plain GELU FFN, full budget | sanity anchor | 295,872 alloc (0.00%) |
| `swiglu` | SwiGLU, full budget | **baseline to beat** | 296,064 alloc (0.065%) |
| `par_dend` | SwiGLU main + compartmental-SwiGLU add-on (parallel) | parallel dendritic *(primary)* | 296,065 **active** (0.065%) |
| `par_dense` | SwiGLU main + plain-SwiGLU add-on (parallel) | parallel control | 296,065 alloc (0.065%) |
| `seq_dend` | SwiGLU main of `x + dendrites(x)` | sequential dendritic *(primary)* | 296,065 **active** (0.065%) |
| `seq_dense` | SwiGLU main of `x + swiglu(x)` | sequential control | 296,065 alloc (0.065%) |

Sparse dendritic variants are matched on **active** params (no dead weights, auto-widened — the
phase-1 "V1.equal" convention); everything else on **allocated**. All within 0.065% of V0 (tol 1%).

Three clean contrasts:

| contrast | holds fixed | isolates |
|---|---|---|
| `*_dend` vs `swiglu` | params, gating | budget split + compartmentalization *(the bar)* |
| `*_dend` vs `*_dense` | params, gating, depth, integration, split | **compartmentalization alone** *(attribution)* |
| `seq_*` vs `par_*` | everything but wiring | topology |

---

## 4. Results (char-level TinyShakespeare, 3 seeds, 2500 steps)

### 4.1 Test 1 — final val loss (mean ± std)

| variant | final val loss | gap vs `swiglu` | wall-clock† |
|---|---|---|---|
| par_dense | **1.514 ± 0.002** | −0.005 | ~184 s |
| seq_dense | **1.514 ± 0.007** | −0.005 | ~60 s |
| swiglu | 1.519 ± 0.003 | — | ~122 s |
| point | 1.520 ± 0.004 | +0.001 | ~131 s |
| par_dend | 1.525 ± 0.003 | +0.006 | ~226 s |
| seq_dend | 1.529 ± 0.001 | +0.010 | ~232 s |

†Timing ran under background contention — approximate. The dendritic einsum variants are clearly
~1.8–1.9× slower than `swiglu` (FLOPs matched at ~248 TFLOP; same hardware-efficiency penalty as
phase 1). Plot: `results/agg_phase2.png`.

### 4.2 Test 3 — internal disentanglement proxies (mean ± std)

| variant | mean \|pairwise corr\| (↓ cleaner) | participation ratio (eff. dim) |
|---|---|---|
| swiglu | **0.0585 ± 0.0009** | **92.7 ± 6.2** |
| par_dense | 0.0656 ± 0.0013 | 66.1 ± 8.3 |
| par_dend | 0.0673 ± 0.0021 | 58.0 ± 8.9 |
| seq_dend | 0.0702 ± 0.0021 | 57.9 ± 6.3 |
| seq_dense | 0.0744 ± 0.0007 | 51.6 ± 5.2 |
| point | 0.1145 ± 0.0026 | 47.3 ± 2.0 |

Probe: the SwiGLU main's pre-`down` hidden (the representation projected toward the residual).
Plot: `results/analysis/phase2_proxies.png`.

---

## 5. Reading — the four pre-registered predictions

1. **No starvation — ✅ confirmed.** All augmentative variants within ~0.01 of `swiglu`
   (1.514–1.529), not phase 1's +0.12 collapse. The preserved-`x` design did its job.
2. **Attribution `*_dend < *_dense` — ✗ reversed.** `par_dend` (1.525) is *worse* than
   `par_dense` (1.514) by +0.011; `seq_dend` (1.529) worse than `seq_dense` by +0.015. Both gaps
   ≫ seed noise. Compartmentalization does slightly *negative* work on loss.
3. **Decisive bar (≤ `swiglu`) — ✗ not cleared.** Both dendritic variants are above `swiglu` by
   more than the noise band.
4. **Topology (sequential ≥ parallel) — ✗ not supported.** `seq_dend` is marginally worse than
   `par_dend`.

**Decision logic → Case B (clean null).** Not case A (no dendritic variant ≤ `swiglu` *and* <
control). Not case C (`*_dend` is not < `*_dense`). Test 3 concurs: `swiglu` is cleanest and
highest-dimensional; dendritic ≈ dense, both dirtier than plain SwiGLU (though all far cleaner
than the `point` baseline — the gate itself is the big tidiness win, not compartments).

One non-robust wrinkle worth flagging honestly: the **dense** add-ons (a two-way parallel/
sequential SwiGLU) edge `swiglu` by −0.005 (~1.4σ) — i.e. "an extra path helps a hair" — but
*compartmentalizing* that path erases the edge. That is the opposite of the hypothesis.

---

## 6. Test 2 — compositional generalization (the decisive test)

Two arms, run on the 5 survivors (`swiglu` + the 4 augmentative variants), 3 seeds, 2500 steps.

### 6.1 Toy PCFG — a length-matched recombination split

A small recursive bracket grammar (symbol-level tokens, vocab 19) with a **harmony rule**: when
bracket `b` opens inside parent `a`, the grammar emits a tag `H = parity(color(a), color(b))`.
Predicting `H` requires *combining* the parent's and child's features. A fixed subset of
`(parent, child)` pairs is **held out of training**; the test measures harmony-token accuracy at
held-out vs in-distribution positions on the *same* model. Both pools come from the same string
distribution, so the gap is not a sequence-length artifact (it directly answers the
char-level-tokenization concern). Every type and both parities are still seen in training, so the
rule `f` is fully learnable; only the specific combinations are novel.

**Harmony accuracy (mean ± std):**

| variant | in-dist acc | held-out acc | gap (in − held, ↓ better) |
|---|---|---|---|
| swiglu | 0.995 | 0.788 ± 0.035 | **0.207 ± 0.035** |
| par_dend | 0.994 | 0.827 ± 0.019 | 0.166 ± 0.019 |
| seq_dend | 0.996 | 0.859 ± 0.033 | 0.137 ± 0.034 |
| par_dense | 0.996 | 0.859 ± 0.015 | 0.136 ± 0.014 |
| seq_dense | 0.994 | 0.892 ± 0.022 | **0.102 ± 0.023** |

**In-distribution PCFG val loss (mean ± std):** swiglu 1.623 ± 0.008; par_dense 1.585 ± 0.011;
par_dend 1.578 ± 0.007; seq_dense 1.569 ± 0.009; **seq_dend 1.561 ± 0.004**.

Plots: `results/analysis/test2_pcfg.png`, `results/agg_pcfg.png`.

**Reading.**
- **Augmentation helps — robustly.** Every augmentative variant beats `swiglu` on both
  in-distribution loss (−0.04 to −0.06 nats, gaps ≫ std) *and* the compositional gap. The best
  augmentative variant roughly **halves** the generalization gap (`seq_dense` 0.102 vs `swiglu`
  0.207; ~2.5σ). All variants saturate in-distribution (~99.5%), so the gap differences are pure
  generalization.
- **Compartmentalization is not the source.** On the harmony gap the *dense* control is as good or
  better than its dendritic twin (`par_dense` 0.136 ≤ `par_dend` 0.166; `seq_dense` 0.102 ≤
  `seq_dend` 0.137) — the differences are only ~1σ, so the honest statement is *compartments do
  not improve compositional generalization beyond what the plain gated path already gives.*
- **Sequential ≥ parallel — supported** here (both loss and gap), unlike char-level.

### 6.2 Subword-BPE real text — transfer to an adjacent corpus

BPE (vocab 2048) trained on TinyShakespeare; transfer to the King James Bible (corpus B, unseen,
same era/register). Signal = corpus-B minus held-out-A loss (nats/token).

| variant | val loss A | corpus B loss | transfer gap (B − A, ↓ better) |
|---|---|---|---|
| swiglu | 5.143 ± 0.010 | 8.105 ± 0.060 | **2.963 ± 0.069** |
| par_dend | 4.878 ± 0.012 | 7.928 ± 0.071 | 3.050 ± 0.075 |
| par_dense | 4.980 ± 0.019 | 8.116 ± 0.124 | 3.136 ± 0.140 |
| seq_dend | **4.720 ± 0.016** | 7.953 ± 0.251 | 3.233 ± 0.265 |
| seq_dense | 4.763 ± 0.016 | 8.052 ± 0.312 | 3.289 ± 0.316 |

Plots: `results/analysis/test2_bpe.png`, `results/agg_bpe.png`.

**Reading.**
- **Augmentation strongly helps in-distribution fit** (all beat `swiglu` by 0.15–0.43 nats,
  ≫ std), and here **compartmentalization helps too**: `*_dend` < `*_dense` (par 4.878 vs 4.980,
  ~4.6σ; seq 4.720 vs 4.763, ~2σ). This is a **reversal of phase 1**, where compartmentalization
  hurt char-level fit — once the regime is gated + augmentative + subword, it earns its keep.
- **But it does not improve transfer.** `swiglu` has the *smallest* transfer gap; the augmentative
  variants fit Shakespeare better and carry that lead to corpus B, but their *gap* is the same or
  slightly larger (differences mostly within noise — the seq variants' gap std is ~0.3). Better
  fit, not better domain robustness. (Corpus B is a domain shift, not a compositional
  recombination, so this complements rather than duplicates the PCFG.)

### 6.3 The sweet-spot sweep — budget split (`main_frac`) and compartments (`branches`)

*Is there a "best of both worlds": most budget on raw main neurons, a small dendritic add-on for
richness?* We swept `main_frac ∈ {0.25, 0.5, 0.75, 0.9}` (fraction to the SwiGLU main) on
`seq_dend` and `seq_dense`, and `branches ∈ {2, 4, 8, 16}` on `seq_dend`, on both tasks (3 seeds).

**In-distribution fit vs `main_frac` (loss, ↓):**

| main_frac | PCFG seq_dend | PCFG seq_dense | BPE seq_dend | BPE seq_dense |
|---|---|---|---|---|
| 0.25 | **1.541** | 1.536 | **4.420** | 4.473 |
| 0.50 | 1.561 | 1.569 | 4.713 | 4.779 |
| 0.75 | 1.589 | 1.581 | 4.972 | 4.998 |
| 0.90 | 1.594 | 1.598 | 5.042 | 5.129 |

(swiglu: PCFG 1.623, BPE 5.145.)

**Generalization vs `main_frac`** — PCFG harmony gap (↓) / BPE transfer gap (↓):

| main_frac | PCFG dend gap | PCFG dense gap | BPE dend gap | BPE dense gap |
|---|---|---|---|---|
| 0.25 | 0.143 | **0.065** | 3.85 | 3.74 |
| 0.50 | 0.137 | 0.102 | 3.24 | 3.28 |
| 0.75 | 0.172 | 0.114 | 2.93 | 3.00 |
| 0.90 | 0.144 | 0.129 | 3.06 | 2.88 |

(swiglu: PCFG gap 0.207, BPE gap 2.97.) **Branches** (seq_dend, mf 0.5): PCFG gap 0.132 (b2) →
0.137 (b4) → 0.139 (b8) → 0.171 (b16) — more compartments, worse. Plots:
`results/sweep_{pcfg,bpe}_{loss,gap}.png`.

**Reading.**
1. **The optimum is add-on-heavy, not balanced.** Lower `main_frac` (more budget to the
   *sequential add-on*) monotonically improves in-distribution fit on both tasks (PCFG 1.62→1.54,
   BPE 5.15→4.42). "Preserve the raw main neurons" is the wrong instinct — budget is better spent
   on the add-on.
2. **The generalization winner is dense, not dendritic.** On the PCFG compositional gap the large
   *dense* sequential add-on (`seq_dense`, mf 0.25) is best by a wide margin — gap **0.065 vs
   SwiGLU 0.207 (~3.5σ)**, held-out accuracy 0.93. Compartmentalizing it (`seq_dend`) is robustly
   worse at mf 0.25 (0.143, ~2.2σ) and degrades further with more branches.
3. **Compartmentalization's one robust win is real-text fit.** On BPE, `seq_dend` < `seq_dense` at
   *every* split (mf 0.25: 4.420 vs 4.473, ~2.5σ) — but that edge does **not** transfer: the
   add-on-heavy configs have the *largest* BPE transfer gaps (more specialization), and the
   transfer gap is never beaten beyond noise.
4. **A fit ↔ generalization tension.** Compartmentalization buys in-distribution fit at the cost
   of compositional generalization; the two "worlds" are served by *opposite* choices (`*_dend`
   for fit, `*_dense` for generalization), both at high add-on budget. No swept setting lets
   compartments deliver the generalization win.

So the hoped-for "best of both worlds" does not exist *via compartmentalization*. The sweep's
sharper finding: a **large sequential gated add-on** is the lever — plain (dense) for compositional
generalization, compartmental for a small real-text fit bonus — and adding compartments or more
branches trades generalization away.

---

## 7. Conclusions

1. **The dendritic *augmentation* earns its parameters** on tasks that stress the FFN: a preserved
   SwiGLU main + a gated add-on (especially **sequential**) robustly beats a full SwiGLU at matched
   params on the PCFG and on subword real text, and roughly halves the PCFG compositional gap. It
   is **neutral only on flat char-level** — so phase 1's null was partly a low-power regime, which
   vindicates the tokenization concern.
2. **The dendritic *compartmentalization* helps fit, not generalization.** It improves
   in-distribution real-text (BPE) loss over its dense twin (robust) — a reversal of phase 1 — but
   does **not** beat the dense control on either generalization measure (PCFG harmony gap, BPE
   transfer gap). The plain gated extra path captures the generalization benefit; compartments add
   capacity-efficiency, not composition.
3. **The central hypothesis stays unsupported — and the sweep makes the negative firmer.** Across
   char, PCFG, BPE, *and* the full `main_frac`/`branches` sweep, the dense control matches or beats
   compartments on every *generalization* axis, and more compartments (branches) only make
   generalization worse. There is no "best of both worlds" via compartmentalization.
4. **What the sweep *does* surface** is a sharper, parameter-fair positive: a **large sequential
   gated add-on** is the real lever (budget is better spent on the add-on than on the main), with
   a genuine **fit ↔ generalization split** — plain/dense for compositional generalization
   (`seq_dense`, mf 0.25: PCFG gap 0.065 vs 0.207), compartmental for a small real-text fit bonus.
5. **Test 3 (char) cleanliness** still points the wrong way for compartments; not re-run on the
   structured tasks (label-based selectivity there is the natural follow-up).

This is a far more interesting outcome than two flat nulls: a clean, sweep-confirmed negative on the
headline claim, plus a genuine, parameter-fair positive for the augmentative gated topology that
only the structured / subword regimes revealed.

---

## 8. Threats to validity / what a skeptic should push on

- **Best augmentative variant ≠ compartmental.** The robust wins are carried by `seq_dense` /
  the augmentative path; the compartmental edge is fit-only and modest. Don't over-read it.
- **Transfer vs composition.** The PCFG measures rule recombination; the BPE arm measures *domain*
  transfer. They disagree on whether augmentation aids generalization (PCFG yes, BPE no), which is
  the honest, expected nuance — "compositional generalization" and "distribution-shift robustness"
  are different things.
- **Significance.** The compartments-vs-dense generalization differences are ~1σ; the *augmentation
  vs swiglu* and *compartment-helps-BPE-fit* effects are robust (≫ σ). Stated accordingly above.
- **`main_frac`/`branches` now swept; one backbone, one PCFG difficulty.** The budget-split and
  compartment-count axes are covered (§6.3); scale and grammar difficulty are not.
- **Seed variance on the BPE transfer gap** is large (~0.3 nats) — more seeds would tighten the
  transfer conclusion (the in-distribution fit and PCFG gap effects are already robust).

---

## 9. Next steps

1. **Scale sweep** (`d_model`, depth) on the augmentative winner (`seq_dense` at low `main_frac`) —
   does the large-sequential-add-on advantage hold or grow with scale? (`main_frac`/`branches`
   already swept in §6.3.)
2. **Label-based Test 3 on the PCFG** (selectivity / concepts-per-unit using the known token
   classes and grammatical roles) — does the augmentation's generalization edge show up as cleaner
   *labeled* features, even though char-level correlation did not?
3. **More seeds on the BPE transfer arm** to settle whether augmentation's larger transfer gap is
   real or noise.
4. **Scale / `branches` sweep** — does the compartments-help-fit effect grow with scale?

**Verdict.** Phase 2 closes the headline hypothesis as a **reportable negative** (compartmental
composition does not buy generalization, even gating-matched and non-starving), while surfacing a
**parameter-fair positive for the augmentative gated topology** on FFN-stressing tasks — the most
useful thing to carry into any write-up, and the honest answer to "does a richer neuron leave
visible fingerprints?": yes for an added gated path, no for compartmentalization-as-composition.

---

## 10. Reproduce

```bash
# Tier-1 matrix (6 variants x 3 seeds) — serial, one process per run
python experiments/run_seeds.py --seeds 0 1 2

# Test 1 aggregate + Test 3 proxies
python src/aggregate.py results/agg_phase2.png \
  v0_shakespeare swiglu_shakespeare par_dend_shakespeare par_dense_shakespeare \
  seq_dend_shakespeare seq_dense_shakespeare
python src/analyze_representations.py results/analysis/phase2_proxies.png \
  v0_shakespeare swiglu_shakespeare par_dend_shakespeare par_dense_shakespeare \
  seq_dend_shakespeare seq_dense_shakespeare

# Test 2 — PCFG (compositional gap) and BPE (transfer), 5 survivors x 3 seeds
python experiments/run_seeds.py --seeds 0 1 2 --configs \
  configs/swiglu_pcfg.yaml configs/par_dend_pcfg.yaml configs/seq_dend_pcfg.yaml \
  configs/par_dense_pcfg.yaml configs/seq_dense_pcfg.yaml
python src/eval_generalization.py results/analysis/test2_pcfg.png \
  swiglu_pcfg par_dend_pcfg seq_dend_pcfg par_dense_pcfg seq_dense_pcfg

python experiments/run_seeds.py --seeds 0 1 2 --configs \
  configs/swiglu_bpe.yaml configs/par_dend_bpe.yaml configs/seq_dend_bpe.yaml \
  configs/par_dense_bpe.yaml configs/seq_dense_bpe.yaml
python src/eval_generalization.py results/analysis/test2_bpe.png \
  swiglu_bpe par_dend_bpe seq_dend_bpe par_dense_bpe seq_dense_bpe
```

**Key files:** `src/ffn.py` (`GatedFFN`, `CompartmentalGatedFFN`, `DendriticBlock`, solvers,
`build_ffn`, parity); `src/data/toy_language.py` (PCFG + harmony split), `src/data/bpe*.py`
(subword arm + corpus B), `src/eval_generalization.py` (Test 2); `configs/*_{shakespeare,pcfg,
bpe}.yaml`. Dev: Python 3.13, torch cu124, RTX 4060 (8 GB); each run seconds–minutes, peak VRAM
≤0.6 GB.
