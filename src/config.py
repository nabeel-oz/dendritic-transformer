"""Run configuration: a single dataclass loaded from a YAML file.

Keeping every knob in one typed place makes runs reproducible and makes it
trivial later to assert that V0/V1/V2 share an identical backbone while only the
`ffn` field differs.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    run_name: str = "v0_shakespeare"

    # which FFN to plug into the shared backbone. One of:
    #   phase 1: point | deep_flat | dendritic_free | dendritic_structured_strict |
    #            dendritic_structured_equal
    #   phase 2: swiglu | par_dend | par_dense | seq_dend | seq_dense
    ffn: str = "point"

    # backbone (identical across all variants)
    d_model: int = 192
    n_layer: int = 4
    n_head: int = 6
    context: int = 128
    ffn_mult: int = 4
    dropout: float = 0.1

    # dendritic-only knobs
    branches: int = 4
    branch_act: str = "gelu"   # gelu | tanh | sigmoid (phase-1 branch nonlinearity)

    # phase-2 augmentative-block knobs
    main_frac: float = 0.5     # budget fraction spent on the SwiGLU main (rest -> add-on)
    connectivity: str = "sparse"  # add-on gate/up wiring: sparse (active-matched) | masked

    # data
    dataset: str = "tinyshakespeare"

    # optimisation
    seed: int = 0
    batch_size: int = 64
    lr: float = 6e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 200
    max_steps: int = 5000
    eval_interval: int = 250
    eval_iters: int = 100
    sample_tokens: int = 300

    @property
    def ffn_hidden(self) -> int:
        return self.ffn_mult * self.d_model


def load_config(path: str | Path) -> RunConfig:
    """Parse a YAML file into a RunConfig, ignoring unknown keys with a warning."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    known = {f.name for f in fields(RunConfig)}
    unknown = set(raw) - known
    if unknown:
        print(f"[config] warning: ignoring unknown keys {sorted(unknown)}")
    clean = {k: v for k, v in raw.items() if k in known}
    return RunConfig(**clean)
