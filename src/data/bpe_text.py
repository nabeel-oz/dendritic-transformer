"""Subword-BPE real-text data for the Test-2 transfer arm.

Plain-language purpose: train on TinyShakespeare in a ~2k-token subword
vocabulary (non-trivial head, semantics emerge faster than char-level), then
measure perplexity transfer to an *adjacent, never-seen* Early-Modern English
corpus B (the King James Bible — same era/register). The held-out-A vs corpus-B
perplexity gap is the real-text generalization signal complementing the PCFG.

Everything is cached under results/data/ (gitignored): BPE merges + the encoded
token streams, so the 15 training runs just load tensors.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import torch

from .bpe import SimpleBPE
from .real_text import _download_tinyshakespeare

_CACHE = Path(__file__).resolve().parents[2] / "results" / "data"
_VOCAB = 2048
_TRAIN_LIMIT = 400_000          # learn merges on this many chars (speed)

# Corpus B: King James Bible (Gutenberg #10) — Early-Modern English, unseen.
_CORPUS_B_URLS = [
    "https://www.gutenberg.org/cache/epub/10/pg10.txt",
    "https://www.gutenberg.org/files/10/10-0.txt",
]


def _download_text(urls, dest: Path) -> str:
    if dest.exists():
        return dest.read_text(encoding="utf-8", errors="replace")
    last = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode("utf-8", errors="replace")
            dest.write_text(raw, encoding="utf-8")
            return raw
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[data] corpus-B fetch failed for {url}: {e}")
    raise RuntimeError(f"could not download corpus B: {last}")


def _strip_gutenberg(text: str) -> str:
    """Remove the Project Gutenberg header/footer boilerplate if present."""
    start = text.find("*** START OF THE PROJECT GUTENBERG")
    if start != -1:
        start = text.find("\n", start) + 1
        text = text[start:]
    end = text.find("*** END OF THE PROJECT GUTENBERG")
    if end != -1:
        text = text[:end]
    return text.strip()


def _encode_by_line(bpe: SimpleBPE, text: str) -> list[int]:
    """Encode line-by-line (keepends) so each BPE.encode call is short/fast."""
    ids: list[int] = []
    for piece in text.splitlines(keepends=True):
        ids.extend(bpe.encode(piece))
    return ids


class BPEDataset:
    """Same interface as CharDataset, plus a 'corpus_b' split for transfer eval."""

    def __init__(self, bpe: SimpleBPE, train_ids, val_ids, corpus_b_ids):
        self.bpe = bpe
        self.vocab_size = bpe.vocab_size
        self.itos = bpe.vocab                       # int -> bytes
        self.stoi = {v: k for k, v in bpe.vocab.items()}
        self.train_data = train_ids
        self.val_data = val_ids
        self.corpus_b_data = corpus_b_ids

    def decode(self, ids) -> str:
        return self.bpe.decode(ids)

    def _data(self, split):
        return {"train": self.train_data, "val": self.val_data,
                "corpus_b": self.corpus_b_data}[split]

    def get_batch(self, split: str, batch_size: int, context: int, device: str):
        data = self._data(split)
        ix = torch.randint(len(data) - context - 1, (batch_size,))
        x = torch.stack([data[i:i + context] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + context] for i in ix])
        if device.startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


def load_bpe_dataset(train_frac: float = 0.9) -> BPEDataset:
    _CACHE.mkdir(parents=True, exist_ok=True)
    merges_path = _CACHE / f"bpe_v{_VOCAB}.json"
    sh_path = _CACHE / f"bpe_v{_VOCAB}_shakespeare.pt"
    kjv_path = _CACHE / f"bpe_v{_VOCAB}_kjv.pt"

    shakespeare = _download_tinyshakespeare()
    corpus_b = _strip_gutenberg(
        _download_text(_CORPUS_B_URLS, _CACHE / "kjv_bible.txt"))

    if merges_path.exists():
        bpe = SimpleBPE.load(merges_path)
    else:
        print(f"[data] training BPE (vocab {_VOCAB}) on TinyShakespeare ...")
        bpe = SimpleBPE()
        bpe.train(shakespeare, _VOCAB, train_limit=_TRAIN_LIMIT)
        bpe.save(merges_path)

    if sh_path.exists():
        sh_ids = torch.load(sh_path)
    else:
        print("[data] BPE-encoding TinyShakespeare ...")
        sh_ids = torch.tensor(_encode_by_line(bpe, shakespeare), dtype=torch.long)
        torch.save(sh_ids, sh_path)
    if kjv_path.exists():
        kjv_ids = torch.load(kjv_path)
    else:
        print("[data] BPE-encoding corpus B (KJV) ...")
        kjv_ids = torch.tensor(_encode_by_line(bpe, corpus_b), dtype=torch.long)
        torch.save(kjv_ids, kjv_path)

    n = int(train_frac * len(sh_ids))
    return BPEDataset(bpe, sh_ids[:n], sh_ids[n:], kjv_ids)
