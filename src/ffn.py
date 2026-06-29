"""Swappable feed-forward (FFN) sub-layer — the ONLY piece that differs across
variants.

Plain-language purpose: the FFN is the per-token "thinking" block inside a
transformer. The whole experiment swaps a standard point-neuron FFN (V0) for
dendrite-inspired ones while holding everything else fixed. The backbone asks
`build_ffn(cfg)` for a module and never learns which variant it got, so the
comparison stays clean (PROJECT_BRIEF_PHASE1.md section 4).

What makes a branch a "tiny network"
-------------------------------------
A dendritic branch must *process its inputs on its own and only then combine*
(section 1). So each branch here is a genuinely ISOLATED 2-layer mini-MLP
(local linear -> GELU -> per-branch linear -> GELU); the soma is the ONLY step
that mixes branches back to d. This isolation is what stops a "free-wiring"
dendritic block from silently collapsing into a plain FFN: with a single
nonlinearity and a single soma, `Linear -> GELU -> Linear` IS a point FFN, and
the branches would be nominal. The block-diagonal middle layer (+ its
nonlinearity) is what gives every variant real internal compartments.

Variants
--------
- point ........................ V0. Flat dense FFN. The baseline / parity anchor.
- dendritic_free ............... V2. Isolated branches + soma, but every branch's
                                 INPUT layer reads ALL inputs (dense, free wiring).
                                 The control: same internal richness as V1, no
                                 locality. So V1 - V2 isolates the locality prior
                                 from the mere extra-nonlinearity effect.
- dendritic_structured_strict .. V1.strict. SAME shape as V2, plus a fixed frozen
                                 mask so each branch's input layer sees only its
                                 slice. V1 vs V2 differ in ONE thing (the wiring).
                                 Parity on ALLOCATED params; the mask makes part of
                                 the input weights dead, so V1.strict has fewer
                                 *active* params than V2 and can never win by being
                                 secretly bigger.
- dendritic_structured_equal ... V1.equal. Genuinely sparse input wiring (no dead
                                 weights), branch width auto-sized WIDER so the
                                 count of *active* params equals V0/V2. Strict
                                 honesty on working capacity, at the cost of V1 and
                                 V2 differing in width as well as wiring.

The single most important rule (section 3): every variant is parameter-matched to
V0. `assert_param_parity` enforces it before any run, on the metric appropriate to
the variant (allocated for strict, active for equal).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_params(module: nn.Module) -> int:
    """Allocated trainable parameter count (everything with requires_grad)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def init_ffn_weights(module: nn.Module) -> None:
    """One shared init policy for EVERY FFN variant, so no variant gets an init
    edge (Test 1 is literally 'who learns faster', so init must not be a
    confound). All Linear weights and the raw branch parameters ~ N(0, 0.02^2);
    biases zero. (The backbone's `apply(_init_weights)` later re-sets the same
    0.02 on Linears — identical values, harmless.)"""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    for name, p in module.named_parameters():
        if name.endswith(("in_weight", "mid_weight", "gate_weight", "up_weight")):
            nn.init.normal_(p, mean=0.0, std=0.02)
        elif name.endswith(("in_bias", "mid_bias")):
            nn.init.zeros_(p)


class PointFFN(nn.Module):
    """V0 — Standard FFN: Linear(d -> hidden) -> GELU -> Linear(hidden -> d).

    A 'flat sum' neuron block: every input is poured into one bucket, blended
    once, and emitted. The baseline against which dendritic richness is judged.
    """

    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

    def allocated_params(self) -> int:
        return count_params(self)

    def active_params(self) -> int:
        return count_params(self)


class FlatDeepFFN(nn.Module):
    """Depth- and parameter-matched flat control: d -> w -> w -> d, fully dense,
    no compartments. Same 3 weight layers / 2 nonlinearities as every
    DendriticFFN, parameter-matched to V0. The 'just add a layer' baseline:
    'dendritic vs FlatDeep' isolates *compartmental structure* from *mere depth*.
    """

    def __init__(self, d_model: int, width: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.drop(self.fc3(x))

    def allocated_params(self) -> int:
        return count_params(self)

    def active_params(self) -> int:
        return count_params(self)


class DendriticFFN(nn.Module):
    """B isolated branch sub-networks + a soma that combines them back to d.

    Per branch j (all isolated from other branches until the soma):
        z = GELU(W_in_j  x_j + b_in_j)     # x_j = all of x (dense/masked) or its slice (sparse)
        o = GELU(W_mid_j z   + b_mid_j)     # block-diagonal: branches never mix here
    soma:
        y = W_soma concat_j(o_j) + b_soma   # the ONLY cross-branch mixing

    Connectivity of the INPUT layer is the swappable axis:
      - 'dense'  : every branch reads all d inputs (V2, free wiring)
      - 'masked' : dense weights + frozen block mask, each branch reads d/B (V1.strict)
      - 'sparse' : genuinely block-structured input, no dead weights (V1.equal)
    The middle layer is block-diagonal in every variant — that is what makes the
    branches real compartments rather than one wide layer.
    """

    _ACTS = {"gelu": nn.GELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}

    def __init__(self, d_model: int, branches: int, branch_w: int,
                 branch_out: int, connectivity: str, dropout: float = 0.0,
                 branch_act: str = "gelu"):
        super().__init__()
        assert d_model % branches == 0, (
            f"d_model={d_model} must be divisible by branches={branches}")
        self.d_model = d_model
        self.branches = branches
        self.branch_w = branch_w          # hidden units per branch (after input layer)
        self.branch_out = branch_out      # output units per branch (after middle layer)
        self.connectivity = connectivity
        self.d_per = d_model // branches  # input dims a structured branch reads

        H_in = branches * branch_w        # total branch-hidden width

        # --- input layer (the swappable connectivity) ---
        if connectivity == "dense":
            self.in_proj = nn.Linear(d_model, H_in)
        elif connectivity == "masked":
            self.in_proj = nn.Linear(d_model, H_in)
            mask = torch.zeros(H_in, d_model)
            for j in range(branches):
                mask[j * branch_w:(j + 1) * branch_w,
                     j * self.d_per:(j + 1) * self.d_per] = 1.0
            self.register_buffer("mask", mask)
        elif connectivity == "sparse":
            self.in_weight = nn.Parameter(torch.empty(branches, self.d_per, branch_w))
            self.in_bias = nn.Parameter(torch.zeros(branches, branch_w))
        else:
            raise ValueError(f"unknown connectivity: {connectivity!r}")

        # --- middle layer: block-diagonal, isolates branches (all variants) ---
        # (weights initialised by init_ffn_weights via build_ffn)
        self.mid_weight = nn.Parameter(torch.empty(branches, branch_w, branch_out))
        self.mid_bias = nn.Parameter(torch.zeros(branches, branch_out))

        # --- branch nonlinearity (the saturating-dendrite axis) ---
        if branch_act not in self._ACTS:
            raise ValueError(f"unknown branch_act: {branch_act!r}")
        self.branch_act = branch_act
        self.act = self._ACTS[branch_act]()

        # --- soma: the only cross-branch mixing (stays linear) ---
        self.soma = nn.Linear(branches * branch_out, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        lead = x.shape[:-1]
        # input layer -> (..., B, branch_w)
        if self.connectivity == "dense":
            a = self.in_proj(x).reshape(*lead, self.branches, self.branch_w)
        elif self.connectivity == "masked":
            a = F.linear(x, self.in_proj.weight * self.mask, self.in_proj.bias)
            a = a.reshape(*lead, self.branches, self.branch_w)
        else:  # sparse
            xr = x.reshape(*lead, self.branches, self.d_per)
            a = torch.einsum("...pk,pkw->...pw", xr, self.in_weight) + self.in_bias
        a = self.act(a)
        # block-diagonal middle layer -> (..., B, branch_out)
        o = torch.einsum("...pw,pwc->...pc", a, self.mid_weight) + self.mid_bias
        o = self.act(o)
        # soma combines branches
        o = o.reshape(*lead, self.branches * self.branch_out)
        return self.drop(self.soma(o))

    def allocated_params(self) -> int:
        return count_params(self)

    def active_params(self) -> int:
        """Working parameters that can actually affect the output."""
        if self.connectivity == "masked":
            dead = self.in_proj.weight.numel() - int(self.mask.sum().item())
            return count_params(self) - dead
        return count_params(self)


# --- Phase-2: augmentative dendrites on a preserved SwiGLU main path ----------
#
# Phase 1 was a clean negative for *substitutive* dendrites (branches that
# REPLACE the direct path: locality starves the unit). Phase 2 asks the narrower
# question: do dendrites, added as an AUGMENTATION on top of a preserved SwiGLU
# main path, beat a standard SwiGLU at matched parameters -- and is any gain due
# to compartmental structure rather than just an extra path? (PROJECT_BRIEF_PHASE2.md)
#
# Design note (gating held constant): the add-on is a *gated* (SwiGLU)
# compartmental FFN, not the phase-1 non-gated DendriticFFN. If the add-on were
# non-gated, the "beat SwiGLU" contrast would confound compartmentalization with
# gated-vs-non-gated (half the budget in a known-weaker primitive). With a gated
# add-on, gating / depth / the integrating projection are identical across arms
# and only compartmentalization varies. Gating also removes the phase-1 collapse
# trap: silu(gate)*up is already non-collapsible, so no two-nonlinearity branch
# trick is needed -- this module is simpler than DendriticFFN.


class GatedFFN(nn.Module):
    """SwiGLU -- the production-standard FFN (LLaMA/PaLM/Mistral). in = out = d.

    The phase-2 baseline to beat, and also the augmentative block's `main` soma
    and the `*_dense` control add-on. Bias-free (the GLU convention).
    """

    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))

    def allocated_params(self) -> int:
        return count_params(self)

    def active_params(self) -> int:
        return count_params(self)


class CompartmentalGatedFFN(nn.Module):
    """SwiGLU with compartmentalized branches -- the dendritic add-on.

    Each branch j computes a gated (SwiGLU) transform of ONLY its input
    compartment, then a dense soma integrates all branches back to d:
        z_j = silu(Wg_j x_j) * (Wu_j x_j)   # x_j = branch slice (sparse) / all of x (masked)
        y   = W_soma concat_j(z_j)          # dense: the ONLY cross-branch mixing

    The single gated stage is already non-collapsible (unlike a single GELU
    layer), so no extra block-diagonal middle layer is needed: this is the
    gating- and depth-matched twin of a plain SwiGLU (`GatedFFN`). The ONLY
    difference from plain SwiGLU is that gate/up are block-diagonal (each branch
    reads its compartment) rather than dense.

    Connectivity of the gate/up stage:
      - 'sparse' : each branch owns a (d/B -> branch_w) gate & up, no dead
                   weights; parity on ACTIVE params (auto-widen). Tier-1 default.
      - 'masked' : dense (d -> B*branch_w) gate & up + frozen block mask; parity
                   on ALLOCATED params. Tier-2 capacity-artifact confirmation.
    """

    def __init__(self, d_model: int, branches: int, branch_w: int,
                 connectivity: str = "sparse", dropout: float = 0.0):
        super().__init__()
        assert d_model % branches == 0, (
            f"d_model={d_model} must be divisible by branches={branches}")
        self.d_model = d_model
        self.branches = branches
        self.branch_w = branch_w
        self.connectivity = connectivity
        self.d_per = d_model // branches
        H = branches * branch_w

        if connectivity == "sparse":
            self.gate_weight = nn.Parameter(torch.empty(branches, self.d_per, branch_w))
            self.up_weight = nn.Parameter(torch.empty(branches, self.d_per, branch_w))
        elif connectivity == "masked":
            self.gate = nn.Linear(d_model, H, bias=False)
            self.up = nn.Linear(d_model, H, bias=False)
            mask = torch.zeros(H, d_model)
            for j in range(branches):
                mask[j * branch_w:(j + 1) * branch_w,
                     j * self.d_per:(j + 1) * self.d_per] = 1.0
            self.register_buffer("mask", mask)
        else:
            raise ValueError(f"unknown connectivity: {connectivity!r}")

        self.soma = nn.Linear(H, d_model, bias=False)  # dense integration
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        lead = x.shape[:-1]
        if self.connectivity == "sparse":
            xr = x.reshape(*lead, self.branches, self.d_per)
            g = torch.einsum("...pk,pkw->...pw", xr, self.gate_weight)
            u = torch.einsum("...pk,pkw->...pw", xr, self.up_weight)
            z = (F.silu(g) * u).reshape(*lead, self.branches * self.branch_w)
        else:  # masked
            g = F.linear(x, self.gate.weight * self.mask)
            u = F.linear(x, self.up.weight * self.mask)
            z = F.silu(g) * u
        return self.drop(self.soma(z))

    def allocated_params(self) -> int:
        return count_params(self)

    def active_params(self) -> int:
        if self.connectivity == "masked":
            live = int(self.mask.sum().item())
            dead = (self.gate.weight.numel() - live) + (self.up.weight.numel() - live)
            return count_params(self) - dead
        return count_params(self)


class DendriticBlock(nn.Module):
    """Augmentative dendritic FFN. The soma (`main`) is PRESERVED; dendrites only ADD.

        parallel   : out = main(x) + scale * addon(x)
        sequential : out = main(x + scale * addon(x))   # the soma computes ON the dendrites

    main  : the soma -- GatedFFN (SwiGLU).
    addon : CompartmentalGatedFFN (dendritic) or GatedFFN (dense control).
    scale : small NONZERO init (0.1) so at init the block ~= main and the add-on
            earns its way in. Do NOT init at 0 (that freezes the add-on gradient).
    """

    def __init__(self, addon: nn.Module, main: nn.Module, topology: str):
        super().__init__()
        assert topology in ("parallel", "sequential")
        self.addon, self.main, self.topology = addon, main, topology
        self.scale = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, x):
        a = self.scale * self.addon(x)
        return self.main(x) + a if self.topology == "parallel" else self.main(x + a)

    def allocated_params(self) -> int:
        return count_params(self)

    def active_params(self) -> int:
        return self.addon.active_params() + self.main.active_params() + self.scale.numel()


# --- branch-width sizing so each variant matches V0's parameter budget --------

def _dendritic_count(d: int, B: int, wb: int, c: int, in_per: int) -> int:
    """Total params of a DendriticFFN (incl. biases) for given dims.
    `in_per` = input dims each branch's input layer actually owns (d for dense/
    masked-allocated, d/B for genuinely sparse)."""
    in_layer = B * in_per * wb + B * wb
    mid_layer = B * wb * c + B * c
    soma = (B * c) * d + d
    return in_layer + mid_layer + soma


def solve_branch_width(d: int, B: int, target: int, sparse: bool) -> int:
    """Pick branch width wb (with branch_out = wb) so the FFN's parameter count
    best matches `target`. For 'masked'/'dense' use sparse=False (count the full
    dense input layer); for genuinely sparse input use sparse=True."""
    in_per = (d // B) if sparse else d
    best_wb, best_diff = 1, None
    for wb in range(1, 4 * d):
        n = _dendritic_count(d, B, wb, wb, in_per)
        diff = abs(n - target)
        if best_diff is None or diff < best_diff:
            best_wb, best_diff = wb, diff
        elif n - target > best_diff:
            break  # counts grow monotonically; no point continuing
    return best_wb


def solve_flat_deep_width(d: int, target: int) -> int:
    """Pick width w for a flat d->w->w->d FFN so total params best match `target`.
    params(w) = 2*d*w + w*w + 2*w + d  (monotonic in w)."""
    best_w, best_diff = 1, None
    for w in range(1, 8 * d):
        n = 2 * d * w + w * w + 2 * w + d
        diff = abs(n - target)
        if best_diff is None or diff < best_diff:
            best_w, best_diff = w, diff
        elif n - target > best_diff:
            break
    return best_w


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


def solve_compartmental_glu_width(d: int, B: int, target: int) -> int:
    """Per-branch width w for the SPARSE CompartmentalGatedFFN (bias-free):
    params = gate + up + soma = 2*(B * d/B * w) + (B*w)*d = d*w*(2 + B).
    Pick w nearest `target` (monotonic in w)."""
    best_w, best_diff = 1, None
    for w in range(1, 8 * d):
        n = d * w * (2 + B)
        diff = abs(n - target)
        if best_diff is None or diff < best_diff:
            best_w, best_diff = w, diff
        elif n - target > best_diff:
            break
    return best_w


def build_ffn(cfg) -> nn.Module:
    """Factory: select the FFN module from the config string. Applies the one
    shared init policy so no variant gets an init edge."""
    d, B, p = cfg.d_model, cfg.branches, cfg.dropout
    target = reference_point_params(cfg)
    kind = cfg.ffn
    if kind == "point":
        ffn = PointFFN(d, cfg.ffn_hidden, p)
    elif kind == "deep_flat":
        ffn = FlatDeepFFN(d, solve_flat_deep_width(d, target), p)
    elif kind in ("dendritic_free", "dendritic_structured_strict"):
        # dense and masked share the same dense shape -> identical allocated count
        wb = solve_branch_width(d, B, target, sparse=False)
        conn = "dense" if kind == "dendritic_free" else "masked"
        ffn = DendriticFFN(d, B, wb, wb, conn, p, cfg.branch_act)
    elif kind == "dendritic_structured_equal":
        wb = solve_branch_width(d, B, target, sparse=True)
        ffn = DendriticFFN(d, B, wb, wb, "sparse", p, cfg.branch_act)
    elif kind == "swiglu":
        ffn = GatedFFN(d, solve_glu_hidden(d, target), p)
    elif kind in ("par_dend", "par_dense", "seq_dend", "seq_dense"):
        # Augmentative block: a smaller SwiGLU `main` (soma) PRESERVES x, plus an
        # add-on that only adds. `*_dend` = compartmental SwiGLU; `*_dense` = plain
        # SwiGLU (the gating-/depth-matched control). main_frac splits the budget.
        topology = "parallel" if kind.startswith("par_") else "sequential"
        is_dend = kind.endswith("_dend")
        main_frac = getattr(cfg, "main_frac", 0.5)
        conn = getattr(cfg, "connectivity", "sparse")
        addon_budget = target - round(target * main_frac)
        if is_dend:
            if conn == "sparse":
                w = solve_compartmental_glu_width(d, B, addon_budget)
            else:  # masked: dense gate/up shape -> match allocated like plain SwiGLU
                w = max(1, solve_glu_hidden(d, addon_budget) // B)
            addon = CompartmentalGatedFFN(d, B, w, conn, p)
        else:
            addon = GatedFFN(d, solve_glu_hidden(d, addon_budget), p)
        main_budget = target - addon.allocated_params() - 1  # -1 for the scale scalar
        main = GatedFFN(d, solve_glu_hidden(d, main_budget), p)
        ffn = DendriticBlock(addon, main, topology)
    else:
        raise ValueError(f"unknown ffn variant: {kind!r}")
    init_ffn_weights(ffn)
    return ffn


# --- parameter parity (the non-negotiable fairness rule) ---------------------

def reference_point_params(cfg) -> int:
    """Allocated params of the V0 point FFN for this backbone — the parity target."""
    return PointFFN(cfg.d_model, cfg.ffn_hidden).allocated_params()


def parity_metric(cfg, ffn) -> tuple[str, int]:
    """Which count must match V0: 'active' for the capacity-honest sparse
    variants (no dead weights, auto-widened), 'allocated' otherwise."""
    sparse = getattr(cfg, "connectivity", "sparse") == "sparse"
    if cfg.ffn == "dendritic_structured_equal":
        return "active", ffn.active_params()
    if cfg.ffn in ("par_dend", "seq_dend") and sparse:
        return "active", ffn.active_params()
    return "allocated", ffn.allocated_params()


def assert_param_parity(cfg, ffn, tol: float = 0.01) -> dict:
    """Assert the FFN is parameter-matched to V0 within `tol`. Returns a report."""
    ref = reference_point_params(cfg)
    name, metric = parity_metric(cfg, ffn)
    rel = abs(metric - ref) / ref
    assert rel <= tol, (
        f"PARAMETER PARITY FAILED for ffn={cfg.ffn}: {name} params={metric} "
        f"vs V0={ref} (relative diff {rel:.3%} > tol {tol:.1%})")
    return {
        "variant": cfg.ffn,
        "v0_params": ref,
        "allocated": ffn.allocated_params(),
        "active": ffn.active_params(),
        "parity_on": name,
        "rel_diff": rel,
    }
