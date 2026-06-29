"""Character-level real-text data (TinyShakespeare for the Step-1 sanity gate).

Plain-language purpose: download a ~1 MB block of Shakespeare, turn every
character into an integer, and serve random fixed-length next-character training
batches. Character-level keeps the "watch the text de-garble" story vivid and
removes the tokenizer as a confound (PROJECT_BRIEF_PHASE1.md section 6).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import torch

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

# results/data/ is gitignored — it is a reproducible cache, not source.
_CACHE_DIR = Path(__file__).resolve().parents[2] / "results" / "data"


class CharDataset:
    """Holds the corpus, the char<->int vocab, and the train/val token tensors."""

    def __init__(self, text: str, train_frac: float = 0.9):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(train_frac * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def get_batch(self, split: str, batch_size: int, context: int, device: str):
        """Random contiguous (x, y) next-char pairs from the chosen split."""
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - context - 1, (batch_size,))
        x = torch.stack([data[i : i + context] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + context] for i in ix])
        if device.startswith("cuda"):
            # pin + non-blocking copy for a small throughput win
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


def _download_tinyshakespeare() -> str:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / "tinyshakespeare.txt"
    if not path.exists():
        print(f"[data] downloading TinyShakespeare -> {path}")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    return path.read_text(encoding="utf-8")


def load_dataset(name: str = "tinyshakespeare") -> CharDataset:
    if name == "tinyshakespeare":
        return CharDataset(_download_tinyshakespeare())
    raise ValueError(f"unknown dataset: {name!r}")
