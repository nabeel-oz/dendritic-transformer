# Dendritic Transformer — Phase 1 Report (Real-Text Regime)

*Architectures of Mind, research idea #1. Status as of the first experimental campaign:
backbone + 5 parameter-matched FFN variants, Test 1 (learning speed) and Test 3 cheap
internal proxies run on character-level TinyShakespeare across 3 seeds. Test 2
(compositional generalization) not yet built.*

---

## 1. TL;DR

We replaced only the feed-forward (FFN) "neurons" inside a small transformer with
**dendrite-inspired** units — units that route inputs into isolated compartments, process
each on its own, then combine — holding *total trainable parameters equal* to a standard
baseline. We predicted three fingerprints: faster learning, better compositional
generalization, and cleaner (more disentangled) internal representations.

**On the real-text regime tested so far, the result is a robust negative.** At matched
parameters, the standard FFN is as good or better on every axis we can currently measure:

- **Learning speed:** standard FFN reaches lower loss; the dendritic *locality prior* costs
  ~0.12–0.14 nats/char (gaps 10–40× seed noise).
- **Internal cleanliness:** standard FFN has the *lowest* unit-correlation and *highest*
  effective dimensionality; dendritic variants are *more* redundant, not less.
- **Compute:** FLOPs are matched, but dendritic wall-clock is 1.5–1.9× higher.

A key methodological gain: a **depth-matched flat control** (`deep_flat`) showed that the
mild penalty of the "free-wiring" dendritic variant is explained entirely by its extra
depth, not by compartmentalization. Compartmentalization is loss-neutral; **input locality
is the specific thing that hurts** here.

**What remains genuinely open:** the locality prior was always hypothesized to pay off on
*compositional generalization* (recombining known pieces in unseen ways) — a regime not
exercised by next-character Shakespeare. That test (Test 2, a toy PCFG language) and the
label-based internal metrics it unlocks (Test 3 selectivity / concepts-per-unit) are the
fair next step.

---

## 2. Hypothesis

### 2.1 The idea
A standard artificial neuron is a *flat sum*: pour all inputs into one bucket, blend once,
emit a number. A transformer FFN is built from these. A **dendritic neuron** instead sorts
inputs into small groups, processes each group locally, and only then combines — so one unit
behaves like a tiny network. Biology works this way (a single cortical pyramidal neuron
needs a 5–8 layer ANN to imitate it; Beniaguev et al.).

**Claim (cause):** dendritic compartmentalization is a *structural prior* that pushes the
network toward disentangled internal representations — fewer polysemantic units, more units
each carrying one clean concept.

**Two downstream symptoms:** (1) *learns from less* — coherent output in fewer steps;
(2) *travels to the new* — better generalization to unseen recombinations.

**Falsifiable structure:** we want all three fingerprints together. Symptoms without internal
cleanliness ⇒ effect is something else. Cleanliness without symptoms ⇒ structure isn't doing
useful work.

### 2.2 The non-negotiable rule — fair comparison
A dendritic unit does more work internally than a point neuron. Comparing at equal *neuron
count* would let it win trivially by being quietly larger. **Every comparison holds total
trainable parameters equal (asserted in code), and reports compute (tokens, FLOPs,
wall-clock) alongside.** Any advantage must come from *organization*, not capacity.

---

## 3. Approach

### 3.1 Shared backbone (held fixed across all variants)
A small decoder-only GPT. Only the FFN sub-layer is swapped; attention, embeddings, norms,
and residual stream are identical. The backbone calls `build_ffn(cfg)` and never branches on
which variant it holds.

```
d_model=192, n_layer=4, n_head=6, context=128, ffn_mult=4 (hidden=768),
dropout=0.1, vocab=65 (char-level), ~1.8M params total (~1.18M in the FFNs)
AdamW, lr 6e-4 cosine w/ 200-step warmup, bf16 autocast, batch 64, 2500 steps
```

### 3.2 The five variants and what each isolates
All five are parameter-matched to V0 (within ≤0.38%; see §3.3).

| variant (`cfg.ffn`) | structure | role |
|---|---|---|
| **V0** `point` | `d→h→GELU→d` (2 layers, 1 nonlinearity) | standard baseline / parity anchor |
| **deep_flat** `deep_flat` | `d→w→w→d` dense (3 layers, 2 nonlinearities) | **depth-matched control** |
| **V2** `dendritic_free` | isolated branches, every branch reads **all** inputs | free-wiring control |
| **V1.strict** `dendritic_structured_strict` | V2 shape + frozen mask: each branch reads only its slice | locality prior; parity on *allocated* |
| **V1.equal** `dendritic_structured_equal` | genuinely sparse input, branches auto-widened | locality prior; parity on *active* |

This yields three clean single-variable contrasts:

| contrast | holds fixed | isolates |
|---|---|---|
| `deep_flat` vs `V0` | params | **depth alone** |
| `V2` vs `deep_flat` | params, depth | **compartmentalization** |
| `V1` vs `V2` | params, depth, compartments | **input locality** |

**Why two V1 variants.** A branch reading a *slice* has fewer real weights than one reading
*all* inputs, so you cannot simultaneously have (a) identical structure, (b) sparse-vs-dense
as the only difference, and (c) identical count of genuinely-working parameters. We built
both horns: **V1.strict** keeps identical shape to V2 and applies a frozen mask (parity on
*allocated* params; the mask leaves ~28% of input weights dead, so V1.strict is mildly
capacity-handicapped and can never win by being secretly bigger); **V1.equal** is genuinely
sparse with no dead weights and is auto-widened so its *active* params equal V0's.

### 3.3 Parameter parity (enforced before every run)
The metric checked depends on the variant — *allocated* for everything except V1.equal,
which is matched on *active* (working) params:

```python
def assert_param_parity(cfg, ffn, tol: float = 0.01) -> dict:
    ref = reference_point_params(cfg)            # V0 point-FFN param count
    name, metric = parity_metric(cfg, ffn)       # 'active' for equal, else 'allocated'
    rel = abs(metric - ref) / ref
    assert rel <= tol, f"PARAMETER PARITY FAILED for ffn={cfg.ffn}: ..."
    return {...}
```

Measured parity (per FFN, V0 target = 295,872):

| variant | allocated | active | matched on | rel diff |
|---|---|---|---|---|
| V0 point | 295,872 | 295,872 | allocated | 0.00% |
| deep_flat | 295,872 | 295,872 | allocated | 0.00% |
| V2 free | 294,752 | 294,752 | allocated | 0.38% |
| V1.strict | 294,752 | **214,112** | allocated | 0.38% |
| V1.equal | 296,844 | 296,844 | active | 0.33% |

### 3.4 The dendritic FFN, and a lesson that shaped it
A first, minimal design (`Linear→GELU→Linear` with branch structure only in the input
wiring) **collapsed**: with a single nonlinearity and a single soma, a "free-wiring"
dendritic block is *mathematically identical* to a point FFN — V0 and V2 produced bit-equal
losses. That defeats V2's job as a control. The fix: make each branch a **genuinely isolated
2-layer mini-MLP**, with a block-diagonal middle layer; the soma is the *only* cross-branch
mixing. This is what makes a branch "a tiny network" and keeps V2 ≠ V0.

```python
class DendriticFFN(nn.Module):
    # per branch j (isolated until the soma):
    #   z = act(W_in_j  x_j + b_in_j)      # x_j = all of x (dense/masked) or its slice (sparse)
    #   o = act(W_mid_j z   + b_mid_j)     # block-diagonal: branches never mix here
    # soma: y = W_soma concat_j(o_j) + b   # the ONLY cross-branch mixing
    def forward(self, x):
        lead = x.shape[:-1]
        if self.connectivity == "dense":
            a = self.in_proj(x).reshape(*lead, self.branches, self.branch_w)
        elif self.connectivity == "masked":
            a = F.linear(x, self.in_proj.weight * self.mask, self.in_proj.bias)
            a = a.reshape(*lead, self.branches, self.branch_w)
        else:  # sparse
            xr = x.reshape(*lead, self.branches, self.d_per)
            a = torch.einsum("...pk,pkw->...pw", xr, self.in_weight) + self.in_bias
        a = self.act(a)
        o = torch.einsum("...pw,pwc->...pc", a, self.mid_weight) + self.mid_bias  # block-diag
        o = self.act(o)
        o = o.reshape(*lead, self.branches * self.branch_out)
        return self.drop(self.soma(o))
```

The **deep_flat** control was added precisely because every `DendriticFFN` now has 3 weight
layers / 2 nonlinearities vs V0's 2/1 — so "dendritic vs V0" would otherwise confound depth
with compartmentalization. `deep_flat` is the depth-matched, non-compartmentalized twin:

```python
class FlatDeepFFN(nn.Module):                  # d -> w -> w -> d, fully dense
    def forward(self, x):
        x = self.act(self.fc1(x)); x = self.act(self.fc2(x)); return self.drop(self.fc3(x))
```

Branch widths are auto-solved to hit V0's budget (e.g. `deep_flat` w=384; dendritic branch
width wb=140 for dense/masked, 177 for the auto-widened sparse variant), so parity holds by
construction.

### 3.5 Mechanism axis: `branch_act`
A config flag swaps the branch nonlinearity (`gelu|tanh|sigmoid`); the soma stays linear.
Motivation: a biological dendritic branch fires a *self-limiting* spike — saturation
(tanh/sigmoid) might push branches toward all-or-nothing specialization (the disentanglement
fingerprint). Only the branch nonlinearity changes; V0/deep_flat stay GELU.

### 3.6 Measurement harness
- **Test 1** (`train.py` + `experiments/run_seeds.py` + `aggregate.py`): per-step CSV of
  `step, tokens_seen, wall_clock_s, flops, train_loss, val_loss, lr`; ≥3 seeds; mean±std with
  error bands. FLOPs measured once via `torch.utils.flop_counter.FlopCounterMode`.
- **Test 3 cheap proxies** (`analyze_representations.py`): rebuild each checkpoint, hook the
  activation fed to each FFN's final projection (a consistent "unit activation" across
  variants), pool over val batches, compute (1) **mean |pairwise correlation|** (lower =
  cleaner), (2) **participation ratio** `(Σλ)²/Σλ²` of the activation covariance (effective
  dimensionality). Aggregated over seeds.

---

## 4. Results

All runs: char-level TinyShakespeare, 3 seeds (0/1/2), 2500 steps, all parameter-matched.

### 4.1 Test 1 — race to coherence (final val loss, mean ± std)

| variant | final val loss | gap vs V0 | wall-clock | FLOPs |
|---|---|---|---|---|
| **V0 point** | **1.520 ± 0.004** | — | 42 s | 247 TFLOP |
| deep_flat | 1.555 ± 0.003 | +0.035 | 43 s | 247 TFLOP |
| V2 free | 1.550 ± 0.004 | +0.030 | 62 s | 246 TFLOP |
| V1.equal | 1.644 ± 0.011 | +0.124 | 78 s | 247 TFLOP |
| V1.strict | 1.658 ± 0.005 | +0.138 | 63 s | 246 TFLOP |

![Test 1 loss curves](results/agg_test1.png)

**Reading:**
- **The V2 penalty is depth, not compartments.** `deep_flat` (+0.035) ≈ V2 (+0.030). Adding a
  layer at fixed params is what costs fit; compartmentalization itself is loss-neutral (V2 is
  even a hair *better* than its flat depth-twin).
- **Locality is the real cost.** V1 (+0.12–0.14) is far worse, robustly (gaps ≫ std).
  V1.equal ≈ V1.strict, with V1.equal marginally better — consistent with it carrying its
  full active-param budget while V1.strict is mask-handicapped.
- **Compute parity ≠ wall-clock parity.** Matched FLOPs, but dendritic wall-clock is
  1.5–1.9× V0's: branched einsums are less hardware-efficient than one dense matmul.

### 4.2 The text de-garbles (V0 vs V1.strict, seed 0)

The story is legible char-by-char. Both start as noise; by step 2500 both produce
Shakespeare-shaped text, with V0 modestly more word-like (consistent with its lower loss).

**Step 0 (both, untrained):**
```
pe3TuLxFa NWQH I;?gIpQcQFKe?TAxAxtASLACRbRA3 G'zj3uppa33cKfdDm:s r!u,HI;g'!WBnvCCxd...
```

**V0 point — step 250 → 750 → 2500:**
```
[250]  Gine? / Fy you balls 'than the plor shing't Here sstheays lo. / AOvont INCLBPSTAY:
[750]  Lood Spering i' bok'd the part-soth, / To weethem is my hanced gon him be with
[2500] So I entrail, desire say thee: you are passed / apprinking in fair, perform of
       exercifielly / gentleman: and she not drespect for him! / CATESBY: / O sail, I am
       but for a conduction florship / Of the duke and dust provide
```

**V1.strict — step 250 → 750 → 2500:**
```
[250]  ichepe; hear bails thean the plor she' / Sowance sstheays ho. / AOU' wous w ie hen
[750]  Lod be eniguuing that thy padmose, / Theoug; gear if my lanced good thing. /
       Cizerval: Gevere, in the pinbed not. / LEORLAPURET:
[2500] Awainest fathous requsessted to high or Bohemia, / Pitizin if father that be my
       tentering to mak! / CATESH: / They is nothing so the having, / Why harts I king her
       favence fult, that new-suques!
```
Both learn speaker-tag/line structure and broadly English phonotactics; neither is fluent at
this budget, and the difference matches the small loss gap rather than a qualitative gulf.

### 4.3 Test 3 — internal disentanglement proxies (mean ± std)

| variant | mean \|pairwise corr\| (↓ cleaner) | participation ratio (eff. dim) |
|---|---|---|
| **V0 point** | **0.115 ± 0.003** | **46.8 ± 1.8** |
| deep_flat | 0.145 ± 0.001 | 26.6 ± 0.5 |
| V2 free | 0.141 ± 0.004 | 32.3 ± 2.5 |
| V1.equal | 0.179 ± 0.009 | 14.8 ± 2.5 |
| V1.strict | 0.191 ± 0.004 | 13.7 ± 1.3 |

![Test 3 proxies](results/analysis/test3_proxies.png)

**Reading (also negative).** V0 has the *lowest* unit correlation and *highest* effective
dimensionality. Dendritic variants — especially V1 — are *more* correlated and
*lower*-dimensional. The two proxies agree: high pairwise correlation drives low effective
dimensionality, i.e. the dendritic representations are **more redundant / more entangled**,
the opposite of the predicted cleanliness. Gaps ≫ seed noise.

*Caveat on interpretation:* lower participation ratio is only "good" if the few dimensions
are clean, meaningful factors — which would show as *low* correlation. Here it co-occurs with
*high* correlation, so it reflects redundancy, not tidy compression. The unambiguous signal
is the correlation column: V0 is cleanest.

### 4.4 Phase 3 — saturating branches (`branch_act=tanh`)

| variant | val loss (gelu → tanh) | mean\|corr\| (gelu → tanh) | participation (gelu → tanh) |
|---|---|---|---|
| V2 free | 1.550 → 1.661 | 0.141 → 0.154 | 32.2 → 19.7 |
| V1.strict | 1.658 → 1.697 | 0.191 → 0.180 | 13.7 → 13.9 |
| V1.equal | 1.644 → 1.696 | 0.179 → 0.161 | 14.8 → 16.4 |

![Test 1 tanh](results/agg_test1_tanh.png) ・ ![Test 3 tanh](results/analysis/test3_tanh.png)

**Reading.** tanh does **not** rescue the prior: it costs further loss (+0.04–0.11) and only
marginally/inconsistently moves the proxies (slightly lower correlation for the structured
V1 variants, slightly worse for V2). The saturating-dendrite hypothesis is not supported at
this scale.

---

## 5. Tentative conclusions

1. **On in-distribution real-text modeling, the dendritic prior is worse or neutral on every
   measured axis** (loss, internal cleanliness, wall-clock), and the effect is statistically
   robust across seeds.
2. **Compartmentalization per se is ~free; input locality is what costs.** The depth-matched
   control rules out the "you just added a layer" critique in both directions: depth alone
   accounts for V2's small penalty, and locality (V1) is the specific, larger cost.
3. **No evidence yet for the central claim.** The cheap internal proxies point the *wrong*
   way (more entangled, not less). This weakens — but does not yet falsify — the hypothesis,
   because the proxies are crude and the prior's intended payoff regime is untested.
4. **Saturation (tanh) is not the missing ingredient** at this scale.

These are real, reportable negatives. They do not depend on capacity differences (parity
asserted) and survive multi-seed noise.

---

## 6. Threats to validity / things a skeptic should push on

- **Wrong regime for the prior.** Next-char Shakespeare rewards in-distribution fit; the
  locality prior was hypothesized to help *compositional generalization*. The decisive test
  (Test 2) is not built. This is the biggest caveat.
- **Crude internal metrics.** Pairwise correlation + participation ratio are cheap proxies on
  *real-text* activations. The brief's stronger metrics — selectivity, concepts-per-unit —
  need *known feature labels*, which the toy language provides and Shakespeare does not.
- **Probe choice.** We probe the activation feeding each FFN's final projection. For
  dendritic variants these are branch outputs; within-branch correlation structure could
  partly reflect the block-diagonal design rather than "entanglement" per se.
- **Scale.** ~1.8M params, 2500 steps. The effect could change sign with scale (worth a
  sweep, since hardware is not the constraint — peak VRAM ≈ 0.3–0.5 GB of 8 GB).
- **One dataset, one tokenizer, one B=4, one branch depth.** Limited sweep of the design space.

---

## 7. Next steps (in priority order)

1. **Test 2 — toy language (the fair test for the prior).** Build `data/toy_language.py`: a
   small PCFG generator with a **compositional train/test split** (test set holds out
   grammatical recombinations never seen in training). Evaluate next-token accuracy /
   grammaticality on held-out structures (`eval_generalization.py`). This is the regime where
   a locality prior could trade in-distribution fit for out-of-distribution generalization.
   Also add a real-text transfer arm: val perplexity on held-out corpus A vs an adjacent
   Early-Modern Gutenberg corpus B (e.g. Marlowe).
2. **Test 3 full — selectivity + concepts-per-unit**, using the toy language's known feature
   labels (token classes, positions, grammatical roles). Extends `analyze_representations.py`.
   Depends on Test 2's labeled data.
3. **`branch_combine="gate"` (multiplicative interaction).** The one operation a point neuron
   cannot do in a single step: `content * sigmoid(mod)` across branch halves before the soma.
   Requires updating the parity solver (`combine_units = (B//2)*c`) so parity still holds.
   If "higher-dimensional representation" rather than rearrangement is the goal, this is the
   most principled lever.
4. **Scale / B / branch-depth sweep.** Cheap on this hardware; tests whether any effect is
   scale-dependent. Vary one axis at a time.
5. **Parked for phase 2 (do not build yet):** soft per-token routing inside the neuron (blurs
   toward MoE's "select" axis); two-compartment basal/apical neuron with a top-down signal (a
   genuine architecture change, not an FFN swap).

**Decision gate:** if Test 2 also shows no dendritic advantage, this is a clean, publishable
*null* — "fixed compartmental composition does not, at small scale, buy disentanglement or
generalization in a transformer FFN; MoE-style sparsity remains the demanding baseline." If
Test 2 shows a generalization gap in the prior's favor *despite* worse in-distribution loss,
the hypothesis lives and phase-2 (Mixture of Dendritic Experts) becomes worth scoping.

---

## 8. Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Test 1 matrix (5 variants x 3 seeds) + aggregate
python experiments/run_seeds.py --seeds 0 1 2
python src/aggregate.py results/agg_test1.png \
  v0_shakespeare deepflat_shakespeare v2free_shakespeare \
  v1strict_shakespeare v1equal_shakespeare

# Test 3 cheap proxies on the resulting checkpoints
python src/analyze_representations.py results/analysis/test3_proxies.png \
  v0_shakespeare deepflat_shakespeare v2free_shakespeare \
  v1strict_shakespeare v1equal_shakespeare
```

**Repo map (key files):** `src/ffn.py` (variants + parity + solvers), `src/model.py`
(backbone), `src/train.py` (loop + FLOPs logging), `src/aggregate.py` (Test 1),
`src/analyze_representations.py` (Test 3 proxies), `configs/*.yaml` (one per variant),
`experiments/run_seeds.py` (matrix launcher), `results/<run>/seed<N>/` (per-run CSV +
`samples.txt` + `ckpt.pt`). Dev: Python 3.13, torch 2.6 cu124, RTX 4060 (8 GB); each run is
seconds–minutes, peak VRAM ≈ 0.3–0.5 GB.

---

## 9. References (prior art to stay honest about)

- **Beniaguev, Segev & London** — a single cortical neuron ≈ a deep network (the biological
  motivation for "a neuron is a tiny network").
- **Chavlis & Poirazi (2025)** — dendrites → parameter-efficient learning (the efficiency
  case is largely already made by others; our contribution is the *representation* question
  inside a transformer).
- **Numenta, Active Dendrites** — dendrites for continual learning.
- **KANs** — the adjacent "structured unit" architecture to distinguish from.
- **Mixture of Experts (MoE)** — the demanding baseline for any *cleaner-representation* claim:
  MoE already reaches similar destinations via **sparsity** (units less polysemantic as routing
  gets sparser). The sharpened question this project owns: *can fixed compartmental
  composition produce disentanglement without sparsity — the way brains appear to — or is
  switching things off the only road to tidiness?* (See PROJECT_BRIEF_PHASE1.md §11.)
- **SpikingBrain (2025)** — stacks MoE with spiking neuron-level sparsity (precedent for
  combining network-level and neuron-level mechanisms).
