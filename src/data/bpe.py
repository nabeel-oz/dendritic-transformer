"""Minimal byte-level BPE (no external deps) — for the Test-2 subword arm.

Plain-language purpose: character-level forces the model to learn spelling from
scratch and lean on long-range attention, which can bury an FFN-topology effect.
A subword vocabulary lets semantics emerge sooner and makes the output head
non-trivial. This is a tiny, standard byte-level BPE (à la Karpathy's minbpe):
start from raw bytes, repeatedly merge the most frequent adjacent pair until the
target vocab size is reached. Trained once and cached to disk.
"""
from __future__ import annotations

import json
from pathlib import Path


def _get_pairs(ids):
    stats = {}
    for a, b in zip(ids, ids[1:]):
        stats[(a, b)] = stats.get((a, b), 0) + 1
    return stats


def _merge(ids, pair, idx):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class SimpleBPE:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int, train_limit: int | None = None):
        """Learn merges until `vocab_size`. Merges are learned on the first
        `train_limit` bytes (for speed) but apply to any text."""
        assert vocab_size >= 256
        learn = text[:train_limit] if train_limit else text
        ids = list(learn.encode("utf-8"))
        for k in range(vocab_size - 256):
            stats = _get_pairs(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + k
            ids = _merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = _get_pairs(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids) -> str:
        b = b"".join(self.vocab[int(i)] for i in ids)
        return b.decode("utf-8", errors="replace")

    # --- persistence (merges are enough to reconstruct vocab) ----------------
    def save(self, path: Path):
        data = {"merges": [[a, b, idx] for (a, b), idx in self.merges.items()]}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SimpleBPE":
        bpe = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for a, b, idx in data["merges"]:
            bpe.merges[(a, b)] = idx
            bpe.vocab[idx] = bpe.vocab[a] + bpe.vocab[b]
        return bpe
