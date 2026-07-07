# Dendritic Transformer — Phase 3 Report (the reviewer pass: is it depth? and a truer dendrite)

*Round 3 of the project. An external LLM review read the repo (README, both reports, `ffn.py`,
`toy_language.py`, `eval_generalization.py`), accepted the methodology, and challenged the
**interpretation** of the one surviving Phase-2 positive. This report runs the controls that
challenge demanded. It supersedes the interpretive claims in [REPORT_PHASE2.md](REPORT_PHASE2.md)
§7 — the numbers there stand; the story we told about them does not.*

---

## 1. TL;DR

- **The Phase-2 "win" was FFN-internal depth, not dendrites.** Adding the missing control — a plain
  2-deep SwiGLU stack at matched params — **matches or edges** the best Phase-2 variant on the
  compositional-generalisation gap (XOR PCFG: `deep_swiglu` **0.091 ± .026** vs `seq_dense`
  0.102 ± .023 vs `swiglu` 0.207 ± .035). The winning variant `seq_dense` unfolds to
  `SwiGLU₂(x + s·SwiGLU₁(x))` — two full-width layers, nothing dendritic. "Sequential beats
  parallel" = "depth beats width at matched params."
- **The effect was largely an XOR artifact.** On an easier, separable OR rule the depth advantage
  over plain SwiGLU shrinks from **0.116 (XOR) to 0.023 (OR)** — within ~1–2σ. The Phase-2 headline
  0.065 gap was also a **winner's curse**: re-seeded from 3 to 8 seeds it regressed to
  **0.095 ± 0.047**.
- **A truer dendritic primitive (learned local routing + multiplicative coincidence) also loses.**
  On a new k-ary, non-decomposable **table** task (no XOR shortcut), every `arbor` corner loses to
  `deep_swiglu` (0.105); the "truest" corner (`arbor_lp`, learned+product) is **0.214** — tied with
  plain `swiglu` (0.210) and the **worst** on the board.
- **Enforced locality is causally inert.** Across cells whose effective input count spans
  **3.8 → 37 → 192** (verified in the logs, not just at init), the table gap is
  **0.214 / 0.205 / 0.186** — flat, with the *global* cell nominally best. Sparsemax genuinely
  pruned routing to a 45-of-192 support (frac_local 0.987); it changed nothing.
- **The one honest positive is narrow and correctly attributed:** a *frozen* local-window
  multiplicative detector (`arbor_fg` 0.158) **beats a plain same-size SwiGLU** (0.210) — the
  primitive is not inert — but it **loses to plain depth**, and the *specifically dendritic*
  ingredient (learned routing) is exactly what adds nothing.

**Net.** Phase 2's headline is reclassified from "a dendritic add-on earns its keep at the level of
a single neuron" to "FFN-internal **depth** helps, mostly on XOR-cored tasks; dendritic scaffolding
(compartments in Ph2, learned local multiplicative wiring here) is incidental." A clean, twice-
confirmed negative on the headline idea, on a task and a locality condition that can't be dismissed.

---

## 2. What the reviewer challenged, and why it was fair

Three load-bearing points, all uncontrolled in Phase 2:

1. **The winner is depth in a dendrite costume.** `seq_dense` (best generaliser) =
   `SwiGLU₂(x + s·SwiGLU₁(x))`: two full-width SwiGLU sub-layers, internal residual, learned scale.
   No compartments, full width — nothing per-neuron survives. `par_dense` =
   `SwiGLU₂(x) + s·SwiGLU₁(x)`. So "sequential > parallel" is "depth > width." The **missing control**
   is a plain 2-deep SwiGLU stack at matched params. The README claim that the effect earns its keep
   "at the level of a single *neuron*" is unsupported by the winning variant — that variant is
   layer-grained; the per-neuron grain lived only in the `*_dend` variants, which **lost**.
2. **The harmony rule is XOR (`parity`) — the textbook depth problem.** Grading on XOR over-determines
   a depth win. Need (a) a depth-independent control and (b) a non-XOR task.
3. **Locality was init-only.** Under softmax routing, each branch was *initialised* local but nothing
   held it there; a locality null could just mean locality decayed during training. Never measured.

---

## 3. Round 3a — the missing depth control (cheap, decisive)

**Build.** `DeepGatedFFN` in `src/ffn.py` (`ffn: deep_swiglu`): a plain 2-deep SwiGLU stack,
per-sublayer residual, **no** scale/compartments; each sublayer matched to `target//2` hidden via
`solve_glu_hidden`. `PCFG(harmony_rule=…)`: `parity` (XOR) / `or` (OR), the latter exposed as dataset
`toy_pcfg_or`. XOR/OR held-out sets are **byte-identical** to Phase 2 (verified), so existing runs
reproduce. Parity ≤0.065% on all cells; VRAM ~0.4 GB.

**Results (3 seeds, gap = in-dist − held-out accuracy, lower = better composition):**

| task | swiglu | **deep_swiglu** | seq_dense | par_dense |
|---|---|---|---|---|
| **XOR PCFG** gap | 0.207 ± .035 | **0.091 ± .026** | 0.102 ± .023 | 0.136 |
| **OR PCFG** gap  | 0.067 | **0.044** | 0.036 | 0.048 |

- **Depth matches/edges the Phase-2 "winner"** on XOR — confirms point #1. The scaffolding was
  incidental; the second layer did the work.
- **The advantage is XOR-specific** — the `deep_swiglu` − `swiglu` gap advantage collapses from
  **0.116 (XOR) → 0.023 (OR)**, within noise — confirms point #2.
- **BPE (real-text transfer):** `deep_swiglu` *fits* best (val 4.23 vs `swiglu` 5.14) but **overfits**
  (worst transfer gap 5.12 vs `swiglu` 2.96; `seq_dend` best corpus-B 7.95). Depth ≠ better transfer.
- **Winner's curse:** the Phase-2 headline `seq_dense_pcfg` @ mf0.25 gap **0.065** (3 seeds) →
  **0.095 ± 0.047** over 8 seeds. The smallest cell in a table regressed, as predicted.

---

## 4. Round 3b — a truer dendritic primitive + enforced locality + a non-XOR task

The reviewer's deepest point: every "dendrite" so far was a *soma* (gated SwiGLU) in a branch's role.
Phase 3b builds a faithful primitive and grades it on a task with no XOR shortcut.

### 4.1 The primitive — `ArborFFN` (`src/ffn.py`)

B branches; each branch:
- **(a) learns which input cluster it reads:** `r_j = norm(R_j) @ x`, with `R_j ∈ ℝ^{k×d}` on the
  simplex (`route_norm` ∈ {`softmax`, `sparsemax`}); `route_init` ∈ {`peaked`, `uniform`};
  `routing` ∈ {`frozen`, `learned`}. `taps` = k = d/B by default.
- **(b) detects multiplicative coincidence:** `branch_nonlin` = `product` ((Wa r)·(Wb r), pure
  bilinear) or `gate` (silu(Wa r)·(Wb r)).
- A **dense soma** returns to d, with a param-free **RMS-norm before the soma** as a blow-up guard
  (`act_var` ~1e-6, stable). Paired with `DendriticBlock(sequential)` so branches enrich `x`
  **before** the soma (connective tissue, not a second neuron layer).

Solver `solve_arbor_width`: learned-routing floor = B·k·d, so learned routing **buys fewer
detectors** at matched params — honest cost, printed per corner (`[arbor]` line): **frozen w = 128
vs learned w = 96**. Parity ≤0.087% on all corners.

**Enforced locality (the fix for point #3).** Under softmax, locality is init-only; `sparsemax`
(Martins 2016) gives differentiable *overproduce-then-prune* — routing starts broad and is pruned to
a sparse support. `arbor_stats.csv` logs routing entropy / eff-inputs / support / frac_local +
`act_var` every eval, with a final WARNING if frac_local < 0.5. Three locality cells (learned+product):

- `arbor_lp` **born-local-free** — peaked init (boost = ln(0.9(d−1)/0.1) ≈ 7.45), softmax
- `arbor_lp_globalfree` **born-global-free** — uniform init, softmax
- `arbor_lp_concentr` **born-global-concentrating** — uniform init, **sparsemax** → prunes toward local

### 4.2 The discriminator task — `toy_pcfg_table`

OR was too weak (binary → linearly separable). Replaced by a **k-ary non-decomposable**
`harmony_rule="table"`: `n_colors = 3`, a random balanced C×C lookup for the chord, every row/col
informative, base rate 0.33, held-out colour-combo coverage enforced. Nothing is parity; nothing
decomposes; composition without the depth-friendly shortcut. (OR demoted to a saturation check.)

### 4.3 Attribution — 2×2 corners + controls, on the TABLE (3 seeds, gap ↓)

| variant | in-dist acc | held-out acc | **gap (↓)** | what it is |
|---|---|---|---|---|
| `deep_swiglu` | 0.992 | 0.887 | **0.105 ± .039** | plain depth |
| `seq_dense` | 0.994 | 0.882 | 0.112 ± .018 | depth in disguise |
| `arbor_fg` | 0.991 | 0.834 | 0.158 ± .018 | frozen + gate |
| `arbor_fp` | 0.990 | 0.814 | 0.176 ± .011 | frozen + product |
| `arbor_lg` | 0.991 | 0.783 | 0.209 ± .016 | learned + gate |
| `swiglu` | 0.989 | 0.778 | 0.210 ± .018 | plain single layer |
| `arbor_lp` | 0.990 | 0.775 | 0.214 ± .017 | **learned + product (truest)** |

**Reads:** every `arbor` corner **loses to `deep_swiglu`** (~2.5–3σ for the learned corners). The
frozen corners (`fg` 0.158, `fp` 0.176) **beat plain `swiglu`** (0.210) — the multiplicative
local-window primitive is *not inert*. But the moment routing is *learned* (the dendrite-defining
feature), the edge vanishes: `arbor_lp` 0.214 ties `swiglu` and is worst overall.

**XOR anchor (PCFG, 3 seeds, gap ↓)** corroborates the ordering — depth wins, arbor trails:
`deep_swiglu` 0.091 ≤ `seq_dense` 0.102 < `arbor_lg` 0.117 < `arbor_fp` 0.144 ≈ `arbor_fg` 0.145 ≈
`arbor_lp` 0.148 ≪ `swiglu` 0.207.

### 4.4 Locality ablation — the knob genuinely moved, composition didn't (TABLE, 3 seeds)

| cell | routing | eff. inputs (seed0) | support | frac_local | **gap (↓)** |
|---|---|---|---|---|---|
| born-local-free (peaked+softmax) | local | **3.76** | 192 | 1.000 | 0.214 ± .017 |
| born-global-**concentrating** (sparsemax) | pruned | **37.15** | 45 / 192 | 0.987 | 0.205 ± .010 |
| born-global-free (uniform+softmax) | global | **191.96** | 192 | 0.000 | **0.186 ± .029** |

The manipulation is real: sparsemax pruned a uniform-init router from 192 inputs down to a 45-input
support (98.7% local); the peaked cell held at ~3.8 effective inputs; the free cell stayed fully
global (its WARNING fired, by design). Across that entire sweep the composition gap is **flat**, and
if anything the *global* cell is best. Enforced locality carries **no** benefit. (PCFG mirrors this:
0.148 / 0.205→n.a. — globalfree 0.174, concentr 0.138 — spread within ~1σ, no consistent locality
ordering across tasks → noise, and all worse than `deep_swiglu` 0.091.)

---

## 5. The checkpoint read (the two pre-registered questions)

1. **On the table, does `arbor_lp` / born-global-concentrating beat `deep_swiglu`?** — **No, decisively.**
   `arbor_lp` 0.214 and concentr 0.205 vs `deep_swiglu` 0.105 (~2.5–3σ worse). Every arbor corner
   loses; the best (frozen+gate 0.158) still loses clearly.
2. **Does the concentrating (local) cell beat the global cell?** — **No.** concentr 0.205 vs
   globalfree 0.186; the global cell is nominally better. Locality does not help.

Both reviewer escape hatches are closed: **not an XOR artifact** (depth wins on the k-ary table by
the same ~0.10 margin) and **not init-only locality** (sparsemax verifiably pruned and it changed
nothing).

---

## 6. Threats to validity / what a skeptic should push on

- **Scale.** One small backbone (d=192), one task family, ≤2500 steps. The claim is bounded to this
  regime; it does not rule out dendritic benefits at larger scale, longer training, or other tasks.
  But the **locality-null is causal** here (eff-inputs swept 3.8→192, flat gap) — that is stronger
  than a plain "didn't reach significance," and it argues against the simplest "needs better
  implementation" rescue.
- **Learned routing buys fewer detectors** (w 96 vs 128 at matched params). This is an honest cost of
  faithfulness, not a bug — but a skeptic could argue the learned corners are handicapped. Note the
  *frozen* local corners (same primitive, more detectors) still lose to depth, so the conclusion
  ("loses to depth") does not hinge on the detector-count cost.
- **Blocks C (connective-tissue B/k/main_frac sweep) and D (BPE headline) were deferred**, by design,
  pending this checkpoint. Given `arbor_lp` is already the weakest cell and locality is inert, the
  many-thin-detectors regime (C) is unlikely to reverse the sign; it is worth running only to harden
  the negative against a "you didn't try the thin-wire regime" objection, not as a search for signal.
- **"Beats a plain neuron" is against the easy baseline.** The frozen primitive's win is over
  single-layer `swiglu`, not over `deep_swiglu` (the fair same-compute comparison), which it loses to.
  Stated accordingly throughout.

---

## 7. Conclusions

1. **Headline idea — dendritic wiring / compartments → tidier, more composable concepts — is a firm
   NO**, now from two directions: compartments lost in Phase 2; a faithful learned-local-multiplicative
   dendrite loses here, with locality shown causally inert.
2. **The robust positive is depth**, not dendrites. "Enrich then fire" is a second layer; a plain deep
   SwiGLU stack does it at least as well as any dendritic dressing, and the Phase-2 win was largely an
   XOR effect (winner's-curse-inflated).
3. **The dendritic primitive adds something real but narrow:** a multiplicative, locally-scoped
   detector beats a plain same-size neuron — but loses to depth, and its dendrite-defining ingredient
   (learned local routing) is the part that adds nothing. Any rescue of full dendritic computation now
   carries burden-of-proof, not the benefit of an open door.

**Verdict.** Phase 3 turns Phase 2's "qualified yes" into a well-controlled negative on the headline,
plus a precisely-scoped, correctly-attributed minor positive (multiplicative enrichment < depth).

---

## 8. Reproduce

```bash
# 3a — the missing depth control + the OR foil (byte-identical XOR/OR held-out sets to Phase 2)
python experiments/run_round3a.py --seeds 0 1 2

# 3b CORE — ArborFFN 2×2 attribution (block A) + enforced-locality ablation (block B), 45 runs
#   (sparsemax cells run ~6× slower; peak VRAM ~0.58 GB)
python experiments/run_round3b.py --only core --seeds 0 1 2

# Evaluate (gap = in-dist − held-out; lower = better composition)
python src/eval_generalization.py results/analysis/round3b_table.png \
  swiglu_table deep_swiglu_table seq_dense_table \
  arbor_fg_table arbor_fp_table arbor_lg_table arbor_lp_table          # the decisive discriminator
python src/eval_generalization.py results/analysis/round3b_pcfg.png \
  swiglu_pcfg deep_swiglu_pcfg seq_dense_pcfg \
  arbor_fg_pcfg arbor_fp_pcfg arbor_lg_pcfg arbor_lp_pcfg              # XOR anchor
python src/eval_generalization.py results/analysis/round3b_loc_table.png \
  arbor_lp_table arbor_lp_globalfree_table arbor_lp_concentr_table     # locality 3-cell (table)
python src/eval_generalization.py results/analysis/round3b_loc_pcfg.png \
  arbor_lp_pcfg arbor_lp_globalfree_pcfg arbor_lp_concentr_pcfg        # locality 3-cell (XOR)

# Deferred (secondary): the connective-tissue sweep (block C) and BPE headline (block D)
# python experiments/run_round3b.py --only tissue --seeds 0 1 2
# python experiments/run_round3b.py --only bpe --seeds 0 1 2
```

Routing diagnostics (entropy / eff-inputs / support / frac_local / act_var) are written to
`results/<run>/seed<n>/arbor_stats.csv` every eval; a final WARNING flags any cell that stayed global.
