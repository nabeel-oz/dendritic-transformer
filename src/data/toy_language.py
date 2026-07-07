"""Toy compositional language (Test 2) — a recursive PCFG with a *harmony rule*
and a length-matched recombination split.

Plain-language purpose: phase 1 and the phase-2 in-distribution run were both
negative. The dendritic *composition* prior was always hypothesised to pay off
on **compositional generalization** — recombining known pieces in unseen ways —
not on next-token fit. This module builds the fair test for that claim.

Why this grammar (the design that makes the test meaningful)
-----------------------------------------------------------
A plain typed-bracket (Dyck) language is a trap: the correct closing bracket
depends only on the top of the stack, never on *which* nestings were seen in
training, so "held-out nesting pairs" would generalize for free (attention just
copies the nearest unmatched open) and the test could not discriminate FFN
topologies. So each bracket type `t` carries a binary feature `color(t)`, and
when a child bracket `b` opens directly inside a parent bracket `a`, the grammar
emits a HARMONY token `H = f(color(a), color(b))` right after `O_b` (here
`f = parity`). Predicting that token requires *combining* the parent's feature
(retrieved across the nested span) with the child's — a genuine composition the
FFN must compute.

The split (length-matched recombination)
----------------------------------------
We hold out a fixed subset of `(parent, child)` type pairs. Training/validation
strings NEVER contain a held-out direct nesting; the generalization set DOES.
Every individual type still appears (as parent and as child) in training, and
both parity outcomes are present — so the rule `f` is fully learnable from
training; only the *specific combinations* are novel. Train and test share the
same length/depth distribution (we only filter which pairs may co-occur), so a
generalization gap cannot be an artifact of sequence length — directly answering
the char-level tokenization concern.

Signal: accuracy of predicting the harmony token at held-out-pair positions vs
non-held-out positions, on the SAME trained model. Gap = composition that did
NOT generalize. A topology that composes better should show a smaller gap.

The split (held-out pairs, vocab) is fixed by `split_seed` independent of the
training run seed, so every variant trains and is evaluated on an identical task.
"""
from __future__ import annotations

import random

import torch


class PCFG:
    """Grammar definition + harmony rule + held-out pair set. Deterministic in
    `split_seed` so the task is identical across all model variants/seeds."""

    def __init__(self, n_types: int = 6, n_words: int = 4, max_depth: int = 4,
                 p_branch: float = 0.55, seq_stop: float = 0.55,
                 n_heldout: int = 10, max_len: int = 96, split_seed: int = 0,
                 harmony_rule: str = "parity", n_colors: int = 2):
        self.T = n_types
        self.W = n_words
        self.max_depth = max_depth
        self.p_branch = p_branch       # P(open a bracket | not at max depth)
        self.seq_stop = seq_stop       # P(end the current element sequence)
        self.max_len = max_len
        # harmony rule f(color(parent), color(child)) -> token:
        #   'parity' : XOR over 2 colors -- NOT linearly separable, wants a hidden
        #              layer (the known-hard anchor; over-determines a depth win).
        #   'or'     : OR over 2 colors -- linearly separable. DEMOTED to a saturation
        #              sanity check: with 2 colors the only non-separable binary rules
        #              are XOR/XNOR, so OR can't discriminate FFN topologies.
        #   'table'  : a fixed RANDOM, non-decomposable C x C lookup over n_colors>=3.
        #              Requires genuinely combining both features to generalise to
        #              held-out type pairs (gap > 0, architecture matters), but is NOT
        #              the specific XOR that depth trivially handles -- the round-3b
        #              DISCRIMINATOR of "XOR-specific" vs "general composition".
        #  Base rates differ across rules, so absolute accuracies are NOT comparable
        #  across rules -- only the within-rule in-dist->held-out gap.
        if harmony_rule not in ("parity", "or", "table"):
            raise ValueError(f"unknown harmony_rule: {harmony_rule!r}")
        self.harmony_rule = harmony_rule
        self.C = 2 if harmony_rule in ("parity", "or") else n_colors
        # colour per type: contiguous balanced blocks. For C=2 this is exactly the
        # legacy [0,0,0,1,1,1] assignment, so 'parity'/'or' tasks are byte-identical.
        per = max(1, self.T // self.C)
        self.col = [min(t // per, self.C - 1) for t in range(self.T)]
        self.F = 2 if harmony_rule in ("parity", "or") else self.C  # harmony range
        # random non-decomposable table (independent stream so it can't perturb the
        # held-out shuffle stream; fixed by split_seed -> identical across variants).
        self._table = self._build_table(split_seed) if harmony_rule == "table" else None

        # fixed token vocab (stable order -> reproducible stoi)
        toks = ["<sep>"]
        toks += [f"O{t}" for t in range(self.T)]
        toks += [f"C{t}" for t in range(self.T)]
        toks += [f"H{h}" for h in range(self.F)]
        toks += [f"w{w}" for w in range(self.W)]
        self.itos = {i: s for i, s in enumerate(toks)}
        self.stoi = {s: i for i, s in enumerate(toks)}
        self.vocab_size = len(toks)
        self.sep = self.stoi["<sep>"]

        self.heldout = self._choose_heldout(n_heldout, split_seed)

    def _build_table(self, seed: int):
        """A random C x C -> {0..C-1} lookup that genuinely depends on BOTH colours.
        Constraints so it is a strong discriminator (not a near-constant a model can
        shortcut): EVERY row and EVERY column is non-constant (so the child always
        matters given the parent and vice-versa), all C outcomes appear, the outcome
        distribution is near-balanced (max cell-count <= ceil(C*C/C)+1 -> base rate
        stays low), and it is not the additive `(a+b) mod C`."""
        import collections
        rng = random.Random(seed + 9999)
        C = self.C
        add = [[(a + b) % C for b in range(C)] for a in range(C)]
        max_count = (C * C) // C + 1                          # C=3 -> 4 (base rate <=0.44)
        for _ in range(100000):
            H = [[rng.randrange(C) for _ in range(C)] for _ in range(C)]
            rows_ok = all(len(set(H[a])) > 1 for a in range(C))
            cols_ok = all(len({H[a][b] for a in range(C)}) > 1 for b in range(C))
            counts = collections.Counter(H[a][b] for a in range(C) for b in range(C))
            balanced = len(counts) == C and max(counts.values()) <= max_count
            if rows_ok and cols_ok and balanced and H != add:
                return H
        raise RuntimeError("could not build a non-decomposable harmony table")

    # --- split construction ---------------------------------------------------
    def _choose_heldout(self, n_heldout: int, seed: int) -> set:
        """Pick `n_heldout` (parent, child) TYPE pairs to hold out so the rule stays
        learnable and the held-out test is genuine RECOMBINATION: every type still
        appears as a parent and as a child in training, and every harmony outcome
        survives. For the k-ary table we additionally require each held-out pair's
        COLOUR combination to be covered by some non-held-out pair (so the model has
        seen the colours + rule, only the specific type pairing is novel)."""
        rng = random.Random(seed)
        all_pairs = [(a, b) for a in range(self.T) for b in range(self.T)]
        for _ in range(10000):
            rng.shuffle(all_pairs)
            held = set(all_pairs[:n_heldout])
            train_pairs = [p for p in all_pairs if p not in held]
            parents = {a for a, _ in train_pairs}
            children = {b for _, b in train_pairs}
            outcomes = {self.harmony(a, b) for a, b in train_pairs}
            if not (len(parents) == self.T and len(children) == self.T
                    and len(outcomes) == self.F):
                continue
            if self.harmony_rule == "table":
                train_cols = {(self.col[a], self.col[b]) for a, b in train_pairs}
                if not all((self.col[a], self.col[b]) in train_cols
                           for a, b in held):
                    continue
            return held
        raise RuntimeError("could not build a valid held-out split; lower n_heldout")

    def harmony(self, parent: int, child: int) -> int:
        a, b = self.col[parent], self.col[child]
        if self.harmony_rule == "table":
            return self._table[a][b]
        if self.harmony_rule == "or":
            return a | b
        return (a + b) % 2   # parity (XOR)

    # --- generation -----------------------------------------------------------
    def _gen_seq(self, rng, depth, parent, out, hpos):
        """Append one element-sequence of the grammar to `out`; record harmony
        token positions (index in `out`, (parent, child) pair) in `hpos`."""
        while True:
            if depth < self.max_depth and rng.random() < self.p_branch:
                b = rng.randrange(self.T)
                out.append(self.stoi[f"O{b}"])
                if parent is not None:                  # nested -> harmony token
                    hpos.append((len(out), (parent, b)))
                    out.append(self.stoi[f"H{self.harmony(parent, b)}"])
                self._gen_seq(rng, depth + 1, b, out, hpos)
                out.append(self.stoi[f"C{b}"])
            else:
                out.append(self.stoi[f"w{rng.randrange(self.W)}"])
            if rng.random() < self.seq_stop:
                break

    def _raw_string(self, rng):
        out, hpos = [], []
        self._gen_seq(rng, 0, None, out, hpos)
        return out, hpos

    def sample(self, rng, mode: str):
        """Return (tokens, hpos) for one string.
          mode='train'    : no held-out pair anywhere (in-distribution)
          mode='heldout'  : at least one held-out pair present (generalization)
          mode='any'      : unconstrained (mixed; used for labeled eval)
        Rejection-samples to satisfy length and the mode constraint."""
        for _ in range(1000):
            out, hpos = self._raw_string(rng)
            if not out or len(out) > self.max_len:
                continue
            has_held = any(pair in self.heldout for _, pair in hpos)
            if mode == "train" and has_held:
                continue
            if mode == "heldout" and not has_held:
                continue
            return out, hpos
        # fall back to whatever we have (rare)
        return out, hpos

    def decode(self, ids) -> str:
        return " ".join(self.itos[int(i)] for i in ids)


class PCFGDataset:
    """Streams in-distribution PCFG strings as a contiguous token tensor with the
    same interface as CharDataset, so train.py is unchanged. The data stream is
    fixed by `data_seed` so every model variant trains on identical data; the
    held-out generalization set is produced on demand by `gen_eval_batch`."""

    def __init__(self, pcfg: PCFG | None = None, n_train_strings: int = 40000,
                 train_frac: float = 0.9, data_seed: int = 0):
        self.pcfg = pcfg or PCFG()
        self.vocab_size = self.pcfg.vocab_size
        self.stoi, self.itos = self.pcfg.stoi, self.pcfg.itos

        rng = random.Random(data_seed)
        ids = []
        for _ in range(n_train_strings):
            toks, _ = self.pcfg.sample(rng, "train")
            ids.extend(toks)
            ids.append(self.pcfg.sep)
        data = torch.tensor(ids, dtype=torch.long)
        n = int(train_frac * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s):  # symbols are space-separated tokens
        return [self.stoi[t] for t in s.split()]

    def decode(self, ids) -> str:
        return self.pcfg.decode(ids)

    def get_batch(self, split: str, batch_size: int, context: int, device: str):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - context - 1, (batch_size,))
        x = torch.stack([data[i:i + context] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + context] for i in ix])
        if device.startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # --- labeled generalization eval -----------------------------------------
    def gen_eval_batch(self, n_strings: int, context: int, eval_seed: int = 1234):
        """Generate labeled eval strings (allowing held-out nestings) padded to a
        rectangular batch. Returns (x[int], harmony_targets, mask_indist,
        mask_heldout) where the masks select harmony-token positions whose pair
        is / isn't held out. Position p's harmony token is predicted from the
        logits at p-1, so we expose the predict-from index."""
        rng = random.Random(eval_seed)
        seqs, labels = [], []   # labels: list of (pred_idx, target_id, is_heldout)
        maxlen = 0
        for _ in range(n_strings):
            toks, hpos = self.pcfg.sample(rng, "any")
            if len(toks) > context:
                continue
            lab = [(p - 1, toks[p], (pair in self.pcfg.heldout))
                   for p, pair in hpos if p >= 1]
            if not lab:
                continue
            seqs.append(toks)
            labels.append(lab)
            maxlen = max(maxlen, len(toks))
        x = torch.full((len(seqs), maxlen), self.pcfg.sep, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return x, labels
