# Dendritic Transformer — Project Brief & Implementation Prompt

*A controlled comparison of a standard transformer against a dendrite-inspired variant.*

---

## Our Roles

I (the human) do not have a research background. My role is in providing the human creativity and intelligence. Your role is as the expert and technical intelligence, providing rigour in design and implementation, but also working as a partner to make the most of the potential of human thinking for discovery and insight. 

* Feel free to stop at logical breakpoints to ask questions. 
* When communicating make it intuitive and easy for me to  understand.
* I do have experience with ML but it helps to include brief descriptions when referring to a technical concept or term.

---

## 0. How to read this brief

This is both the **context** and the **plan**. The early sections explain *why* in plain language so the goal stays legible; the later sections are a precise technical spec for the agent doing the build. Every technical block opens with a one-line plain-language purpose so a non-specialist can sanity-check intent.

The governing spirit: **open curiosity, honest measurement, minimal first version.** We are not writing a paper for recognition. We are checking whether a richer neuron leaves visible fingerprints. Negative results are real results and must be reported as such.

---

## 1. The idea in one paragraph

A standard artificial neuron is a *flat sum*: it pours every input into one bucket, blends once, and emits a number. A transformer is built almost entirely from these. A **dendritic neuron** instead *sorts inputs into small groups, processes each group on its own, and only then combines them* — so a single unit behaves like a tiny network. Biology works this way (a single cortical pyramidal neuron needs a 5–8 layer network to imitate it). The question this project tests: if we replace only the feedforward neurons inside a transformer with dendritic ones, does the model organize what it learns into **cleaner, more separable concepts** — and does that cleaner organization show up as **faster learning** and **better generalization**?

---

## 2. The hypothesis (one cause, two symptoms)

**Cause (the real claim):** dendritic compartmentalization acts as a structural prior that pushes the network toward disentangled internal representations — fewer tangled, polysemantic units; more units that each carry one clean concept.

**Symptom 1 — learns from less.** On a matched budget, the dendritic model should reach coherent output in fewer training steps (less garbled text earlier).

**Symptom 2 — travels to the new.** The dendritic model should generalize better to text that recombines familiar pieces in unseen ways.

**Falsifiable structure:** symptoms without the internal cleanliness ⇒ the effect comes from something else. Cleanliness without the symptoms ⇒ the structure isn't doing useful work. We want all three fingerprints together.

---

## 3. The single most important rule: fair comparison

A dendritic neuron does more work inside than a point neuron. If we compare at equal *neuron count*, the dendritic model wins trivially because it is quietly larger. **Every comparison must hold total trainable parameters equal, and report compute (tokens seen and wall-clock) alongside.** Any advantage must come from *how the unit is organized*, never from extra capacity. This rule is non-negotiable and should be enforced in code (assert parameter parity within a small tolerance before any run).

---

## 4. The model variants

Purpose: isolate whether the *structural prior* matters, not merely whether adding nonlinearity helps. One shared transformer backbone; only the feedforward (FFN) sublayer changes. Attention, embeddings, norms, residual stream — all identical across variants.

- **V0 — Standard.** Conventional FFN: `Linear(d → h) → GELU → Linear(h → d)`. The baseline.
- **V1 — Dendritic (structured).** The FFN's hidden width `h` is partitioned into `B` branches. Each branch reads only a **fixed subset** of the input dimensions (a sparse, frozen input mask = the "sorting into groups" step), applies a local nonlinearity, then a soma step combines branches back to `d`. This carries the locality prior.
- **V2 — Dendritic (free wiring) — the control.** Identical branch/soma structure and identical parameter count as V1, but every branch reads **all** inputs (dense, no fixed mask). Same pieces, free wiring. This is the answer to "is it just N neurons firing?" — if V1 ≈ V2 inside, the structure did nothing; if V1 is cleaner than V2, the structural prior is the cause.

All three are parameter-matched to V0. Treat `B` (branch count) and mask sparsity as the primary knobs; default `B = 4`, each branch seeing ~`d/B` inputs.

> Implementation note for the agent: build the FFN as a single swappable module selected by a config string (`ffn: point | dendritic_structured | dendritic_free`). The backbone must not know which FFN it holds.

---

## 5. The three tests and what to measure

Purpose: turn "is it smarter" (unmeasurable) into three concrete fingerprints.

### Test 1 — Race to coherence (learns from less)
- Log train/val loss and perplexity against **tokens seen**, **steps**, and **wall-clock**.
- Save checkpoints at fixed token budgets; generate text samples at each so we can literally watch garbled → coherent, side by side.
- Run **≥ 3 seeds per variant**; report mean ± std. A gap inside the noise band is not a result.

### Test 2 — Generalization to adjacent input (travels to the new)
Two data regimes share this harness; only the dataset plugs in.
- **Real text:** train on corpus A, report val perplexity on held-out A *and* on an adjacent public-domain corpus B (same era/register, never seen). Transfer gap is the signal.
- **Toy language:** a small probabilistic grammar (PCFG) generates strings. Construct the split so the **test set contains grammatical recombinations never present in training** (compositional generalization). Measure next-token accuracy / grammaticality on held-out structures. The toy route is what lets us *construct* the exact unseen combinations; the real route is what makes it feel like language. Do both.

### Test 3 — Look inside (keeps concepts tidy)
Purpose: the crux. Compare V0 vs V1 vs V2 on internal organization, parameter-matched.
- **Selectivity:** for known features (trivial to define in the toy language — token classes, positions, grammatical roles), measure how sharply each branch/unit responds to one feature vs many. Report a selectivity index per unit.
- **Disentanglement proxies (cheap, robust, do these first):**
  - mean absolute pairwise correlation between unit activations (lower = cleaner);
  - effective dimensionality / participation ratio of FFN activations;
  - "concepts per unit" — for known toy features, count how many distinct features each unit fires for (fewer = more monosemantic).
- **Stretch (only after the above):** train a small sparse autoencoder on FFN activations and compare feature sparsity/count across variants. Flag as v2 — likely unnecessary for a first verdict and more compute.

---

## 6. Hardware budget (must fit comfortably)

Target machine: **laptop, RTX 4060 (8 GB VRAM), i7 12th-gen, 32 GB RAM.** This is plenty for the *small, controlled* models this project needs — scale is explicitly not the point.

- Tokenization: **character-level** by default (vocab ~65–100). It keeps the "watch text de-garble" story vivid and removes a confound. A small BPE (1–4k) is an optional later variation.
- Model size: `d_model 128–256`, `n_layer 4–6`, `n_head 4–8`, `context 128–256`, FFN hidden `4 × d_model`. Roughly 1–10M params — trivial for 8 GB.
- Precision: **bf16 autocast** (Ada-gen 4060 supports it). Batch 32–64; gradient accumulation only if ever needed.
- Data: TinyShakespeare (~1 MB) for real text; an adjacent public-domain Early Modern English text (e.g. another Project Gutenberg work of similar era) as corpus B; PCFG-generated strings for the toy language.
- Expected runtime: **minutes per run.** The full matrix (3 variants × 2 datasets × ≥3 seeds) should complete in a few hours, so it can run unattended.

Enforce a VRAM guardrail: if a config would exceed ~7 GB, shrink `context` or `d_model` automatically and log the change.

---

## 7. Repository shape

```
dendritic-transformer/
  README.md                  # the story + how to reproduce
  requirements.txt           # torch, numpy, matplotlib (keep deps minimal)
  configs/                   # one yaml/dataclass per run; declares variant + dataset + seed
  src/
    model.py                 # shared backbone; FFN selected by config (point/structured/free)
    ffn.py                   # the three swappable FFN modules + param-parity assert
    data/
      real_text.py           # loaders for corpus A and adjacent corpus B
      toy_language.py         # PCFG generator + compositional train/test split
    train.py                 # checkpointed training; multi-seed; logs to CSV
    eval_generalization.py    # Test 2
    analyze_representations.py# Test 3 (selectivity + disentanglement proxies)
    plotting.py              # loss curves, sample tables, internal-structure figures
  experiments/               # scripts that launch the full matrix
  results/                   # logs, checkpoints, figures (gitignore the heavy bits)
```

---

## 8. Build order (start minimal, prove each step)

1. Backbone + V0 point FFN. Train char-level TinyShakespeare. Confirm it learns and generates plausible text. (Sanity gate.)
2. Implement V1 and V2 FFNs as drop-ins. Assert parameter parity with V0. Confirm both train.
3. Test 1 rig: checkpointed loss + sample generations + multi-seed; produce the first side-by-side loss curve.
4. Toy-language generator + compositional split.
5. Test 2: generalization eval on both data regimes.
6. Test 3: selectivity + disentanglement proxies across V0/V1/V2.
7. Figures + README write-up of what we found (including nulls).

Stop and surface results to the human after step 3 — the first honest loss curve tells us whether to keep going or rethink the FFN design.

---

## 9. Honest-science guardrails (bake into the repo)

- State the prediction up front in the README before running (faster learning + better generalization + cleaner internals).
- Parameter parity asserted in code; compute reported every run.
- ≥3 seeds; report variance; never claim a within-noise difference.
- Keep the V2 free-wiring control in every internal comparison so effects attribute to the *structural prior*, not added nonlinearity.
- Commit to writing up negative results plainly.
- Everything config-driven and seeded for exact reproduction.

---

## 10. Project framing (for continuity)

This is research idea #1 of an open-ended project, *Architectures of Mind*, mapping neuroscience observations to computational analogues. Working dynamic: the human supplies conceptual direction and research taste; the agent supplies technical rigor, implementation, and candid assessment. Intended output: an accessible article paired with this working GitHub example. Resist publication framing; favor clarity and truth over impressiveness.

Key prior art the build should stay honest about (the efficiency case is largely already made by others; our contribution is the *representation* question inside a transformer): Beniaguev, Segev & London (single neuron ≈ deep net); Chavlis & Poirazi 2025 (dendrites → parameter-efficient learning); Numenta's Active Dendrites (continual learning); KANs as the adjacent "structured unit" architecture to distinguish ourselves from; and Mixture of Experts (MoE) — see section 11, the comparison that matters most for the representation claim.

---

## 11. Relationship to Mixture of Experts (context + a possible phase 2)

Plain-language purpose: anyone who sees this will ask "isn't this just Mixture of Experts?" This section answers that, and turns MoE from a rival into a possible future companion.

### The distinction: MoE *selects*, dendrites *compose*

Both break up the dense feedforward block, but along different axes.

- **MoE selects between whole sub-networks at the token level.** A learned router sends each token to a few of many expert FFNs; most experts are off for any given token; the goal is scale (more capacity at roughly constant compute per token). Inside each expert, the neurons are still ordinary point neurons. MoE doesn't change the neuron — it changes which block of neurons a token is routed to.
- **Dendritic units compose compartments within a single neuron at the feature level.** Every branch is active for every input, and the branches are integrated rather than selected; the goal is representation structure, not scale. It changes the unit itself.

One line: **MoE selects; dendrites compose.** Different axis, different mechanism, different motivation.

### Honest baseline note (this is why MoE matters here)

MoE already exhibits *both* fingerprints this project chases. It is reported to reach a target accuracy in roughly half the training epochs of a comparable dense model, and — importantly — sparse routing makes its units measurably *less polysemantic*, an effect attributable to the sparsity itself rather than to parameter count, and one that grows as routing gets sparser. So for any claim about *cleaner representations*, the demanding baseline is MoE, not the plain dense transformer.

v1 deliberately does **not** include an MoE arm — small-scale MoE is fiddly (routing collapse, load-balancing) and out of scope for the first minimal build — but the eventual write-up must acknowledge that MoE reaches similar destinations by a different road.

### The sharpened question this creates

MoE gets clean, specialized units by turning most of them *off* — cleanliness via **sparsity**. Biological dendrites keep every compartment *on* and integrate them — cleanliness, if any, via **composition**. The question worth owning, and the one that keeps this project distinct:

> *Can fixed compartmental composition produce disentangled representations without sparsity — the way brains appear to — or is switching things off the only road to tidiness?*

Either answer is worth knowing.

### Possible phase 2 — Mixture of Dendritic Experts (conditional, out of v1 scope)

Because the two mechanisms live on orthogonal axes, they can in principle stack: an MoE whose experts are built from dendritic units instead of point neurons — selective routing *and* internally richer units, i.e. "more complex neurons firing based on the expertise required." The combination of network-level sparsity with neuron-level complexity is not wholly unexplored (e.g. SpikingBrain, 2025, stacks MoE with *spiking* neuron-level sparsity for efficiency), but the specific pairing of MoE with dendritic *composition* units, studied for *representation structure* rather than efficiency, appears open.

Two cautions, stated plainly so phase 2 stays honest:

- **It is conditional on a phase-1 result.** Pursue it only if phase 1 shows dendritic units doing something the dense standard model does not — in particular, a different or sparsity-free form of cleanliness. If dendritic ≈ standard in phase 1, there is nothing to stack.
- **"Complementary" is a hypothesis, not a guarantee.** The two axes are architecturally orthogonal, but if dendritic cleanliness turns out to be its own implicit form of gating/sparsity, stacking may buy the same thing twice rather than compounding. Whether the benefits actually add up is itself the phase-2 experiment.

Keep phase 2 entirely out of the v1 build. Its presence here is to record the direction and protect against scope creep, not to expand the first version.
