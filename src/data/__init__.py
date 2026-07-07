"""Dataset registry: `load_dataset(name)` routes a config string to the right
loader. Every dataset exposes the same interface (vocab_size, stoi/itos,
encode/decode, get_batch(split, batch, context, device)) so the training loop
never branches on which data it holds.

  tinyshakespeare      char-level TinyShakespeare (phase 1 + phase-2 Test 1)
  toy_pcfg             recursive PCFG with the XOR (parity) harmony rule (Test 2)
  toy_pcfg_or          same PCFG with the OR-like (linearly-separable) rule -- a
                       saturation sanity check (2 colours can't discriminate topologies)
  toy_pcfg_table       same PCFG with a random NON-decomposable 3-colour harmony table
                       -- the round-3b discriminator of "XOR-specific" vs "general
                       composition" (composition needed, but not the depth-friendly XOR)
  tinyshakespeare_bpe  subword-BPE TinyShakespeare + corpus-B transfer (Test 2)
"""
from __future__ import annotations


def load_dataset(name: str = "tinyshakespeare"):
    if name == "tinyshakespeare":
        from .real_text import load_dataset as _ld
        return _ld(name)
    if name == "toy_pcfg":
        from .toy_language import PCFG, PCFGDataset
        return PCFGDataset(PCFG(harmony_rule="parity"))
    if name == "toy_pcfg_or":
        from .toy_language import PCFG, PCFGDataset
        return PCFGDataset(PCFG(harmony_rule="or"))
    if name == "toy_pcfg_table":
        from .toy_language import PCFG, PCFGDataset
        return PCFGDataset(PCFG(harmony_rule="table", n_colors=3))
    if name == "tinyshakespeare_bpe":
        from .bpe_text import load_bpe_dataset
        return load_bpe_dataset()
    raise ValueError(f"unknown dataset: {name!r}")
