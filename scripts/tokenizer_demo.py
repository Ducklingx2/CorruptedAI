"""Demo: train and compare the three tokenizers. Not part of the library.

Usage (from the repo root, PowerShell):
    python scripts/tokenizer_demo.py
    python scripts/tokenizer_demo.py --bpe-corpus path\\to\\some_text.txt --vocab-size 1000
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from LLM.tokenizer import (  # noqa: E402
    BPETokenizer, CharacterTokenizer, WordTokenizer, load_tokenizer,
)

FALLBACK = "Hello AI! The quick brown fox jumps over the lazy dog. " * 20


def read(path: str | None) -> str:
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8")
    if path:
        print(f"(!) {path} not found, using built-in sample text")
    return FALLBACK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default="data/words.txt", help="corpus for word/char tokenizers")
    ap.add_argument("--bpe-corpus", default=None, help="corpus for BPE (default: --words)")
    ap.add_argument("--vocab-size", type=int, default=600)
    args = ap.parse_args()

    words_text = read(args.words)
    bpe_text = read(args.bpe_corpus) if args.bpe_corpus else words_text
    sample = "Hello AI!"

    word = WordTokenizer(); word.train(words_text)
    char = CharacterTokenizer(); char.train(words_text)
    bpe = BPETokenizer(); bpe.train(bpe_text, vocab_size=args.vocab_size)

    for name, tok in (("word", word), ("character", char), ("bpe", bpe)):
        ids = tok.encode(sample)
        print(f"{name:10s} vocab={tok.vocab_size:6d} ids={ids} -> {tok.decode(ids)!r}")

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bpe.json"
        bpe.save(path)
        again = load_tokenizer(path)
        print("BPE save/load identical:", again.encode(sample) == bpe.encode(sample))


if __name__ == "__main__":
    main()
