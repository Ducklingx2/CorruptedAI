"""Byte-level Byte Pair Encoding (BPE) tokenizer, implemented from scratch.

Algorithm
---------
1. Pre-tokenize text with a regex into chunks (words with their leading space,
   punctuation runs, whitespace runs). Merges never cross chunk boundaries,
   which stops the tokenizer learning junk like "end of" as one token.
2. Every chunk is converted to its UTF-8 bytes. Base vocabulary is the 256
   possible byte values, so ANY string can be encoded: there are no unknown
   tokens and ``decode(encode(x)) == x`` exactly for all valid ``str`` input.
3. Training: count how often each chunk occurs. Repeatedly find the most
   frequent adjacent pair of tokens (across all chunks, weighted by chunk
   frequency), assign it a new ID, and replace every occurrence of the pair.
   Ties are broken by the numerically smallest pair, so training is fully
   deterministic. Stop at ``vocab_size`` or when the best pair occurs fewer
   than ``min_frequency`` times.
4. Encoding: split into chunks, convert to bytes, then repeatedly apply the
   learned merge with the LOWEST rank (earliest learned) present in the chunk
   until none apply. This reproduces training order.
5. Decoding: concatenate each token's bytes, then decode as UTF-8.

ID layout: 0-3 special tokens, 4-259 bytes 0x00-0xFF, 260+ merges in learned order.

Trade-offs: no normalization is applied (exactness over convenience), and
non-Latin scripts need more tokens until merges for them are learned.
Training keeps incremental pair counts (updating only the words a merge touches)
and picks the best pair with a heap. It is pure Python: fine for corpora of a few
MB, not tuned for gigabytes of text.

Special tokens: ordinary text is never parsed for special tokens.
"""

from __future__ import annotations

import heapq
import re
from collections import Counter
from typing import Any, Sequence

from .base import Tokenizer
from .errors import TokenizerFormatError
from .special_tokens import ID_TO_SPECIAL_TOKEN, NUM_SPECIAL_TOKENS

_PRETOKEN_RE = re.compile(r" ?\w+| ?[^\w\s]+|\s+(?!\S)|\s+", re.UNICODE)

BYTE_OFFSET = NUM_SPECIAL_TOKENS          # ID of byte 0x00
BASE_VOCAB_SIZE = NUM_SPECIAL_TOKENS + 256  # first merge ID

Pair = tuple[int, int]


def _merge_pair(seq: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` (left to right)."""
    out: list[int] = []
    i = 0
    n = len(seq)
    a, b = pair
    while i < n:
        if i < n - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


class BPETokenizer(Tokenizer):
    TYPE = "bpe"

    def __init__(self) -> None:
        self._set_merges([])

    # ---- internals ------------------------------------------------------

    def _set_merges(self, merges: list[Pair]) -> None:
        self._merges: list[Pair] = list(merges)
        self._ranks: dict[Pair, int] = {p: r for r, p in enumerate(self._merges)}
        # Bytes for every non-special id; specials map to None.
        pieces: list[bytes | None] = [None] * NUM_SPECIAL_TOKENS
        pieces.extend(bytes([b]) for b in range(256))
        for a, b in self._merges:
            pieces.append(pieces[a] + pieces[b])  # type: ignore[operator]
        self._pieces = pieces
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return BASE_VOCAB_SIZE + len(self._merges)

    @property
    def merges(self) -> list[Pair]:
        return list(self._merges)

    # ---- training -------------------------------------------------------

    def train(
        self,
        text: str,
        vocab_size: int = 1000,
        min_frequency: int = 2,
    ) -> None:
        if vocab_size < BASE_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be >= {BASE_VOCAB_SIZE} "
                f"({NUM_SPECIAL_TOKENS} special tokens + 256 bytes)"
            )
        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")

        chunk_counts = Counter(_PRETOKEN_RE.findall(text))
        # Sorted for reproducibility (tie-breaking below makes order irrelevant
        # anyway, but this keeps behavior obviously deterministic).
        words: list[tuple[list[int], int]] = [
            ([BYTE_OFFSET + b for b in chunk.encode("utf-8")], freq)
            for chunk, freq in sorted(chunk_counts.items())
        ]

        # Incremental pair statistics: pair -> total weighted count, and
        # pair -> indices of words that (may) contain it. A max-heap with lazy
        # invalidation finds the best pair; entries are (-count, pair) so ties
        # resolve to the numerically smallest pair, exactly like a full rescan.
        pair_counts: dict[Pair, int] = {}
        where: dict[Pair, set[int]] = {}
        for idx, (seq, freq) in enumerate(words):
            for pair in zip(seq, seq[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
                where.setdefault(pair, set()).add(idx)
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[Pair] = []
        while BASE_VOCAB_SIZE + len(merges) < vocab_size:
            best: Pair | None = None
            count = 0
            while heap:
                neg_count, pair = heapq.heappop(heap)
                if pair_counts.get(pair, 0) == -neg_count and neg_count < 0:
                    best, count = pair, -neg_count
                    break  # entry is current; stale ones are skipped
            if best is None or count < min_frequency:
                break
            new_id = BASE_VOCAB_SIZE + len(merges)
            merges.append(best)

            changed: set[Pair] = set()
            for idx in where.pop(best):
                seq, freq = words[idx]
                new_seq = _merge_pair(seq, best, new_id)
                if new_seq == seq:
                    continue  # stale index entry
                for pair in zip(seq, seq[1:]):
                    pair_counts[pair] -= freq
                    changed.add(pair)
                for pair in zip(new_seq, new_seq[1:]):
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq
                    where.setdefault(pair, set()).add(idx)
                    changed.add(pair)
                words[idx] = (new_seq, freq)
            pair_counts.pop(best, None)
            for pair in changed:
                c = pair_counts.get(pair, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, pair))

        self._set_merges(merges)

    # ---- encode / decode ------------------------------------------------

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        seq = [BYTE_OFFSET + b for b in chunk.encode("utf-8")]
        while len(seq) > 1:
            best_rank = None
            best_pair: Pair | None = None
            for pair in zip(seq, seq[1:]):
                rank = self._ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None or best_rank is None:
                break
            seq = _merge_pair(seq, best_pair, BASE_VOCAB_SIZE + best_rank)
        self._cache[chunk] = seq
        return seq

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        for chunk in _PRETOKEN_RE.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return self._wrap(ids, add_special_tokens)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        parts: list[bytes] = []
        for raw in ids:
            i = self._check_id(raw)
            if i < NUM_SPECIAL_TOKENS:
                if not skip_special_tokens:
                    parts.append(ID_TO_SPECIAL_TOKEN[i].encode("utf-8"))
            else:
                parts.append(self._pieces[i])  # type: ignore[arg-type]
        # Token boundaries can split a multi-byte character, so decode only after
        # joining. Invalid byte sequences (e.g. truncated generation output)
        # become U+FFFD instead of raising.
        return b"".join(parts).decode("utf-8", errors="replace")

    # ---- serialization --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.TYPE, "merges": [list(p) for p in self._merges]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BPETokenizer":
        from .serialization import require_field

        raw = require_field(data, "merges", list)
        merges: list[Pair] = []
        seen: set[Pair] = set()
        for idx, item in enumerate(raw):
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in item)
            ):
                raise TokenizerFormatError(f"merges[{idx}] must be a list of two integers")
            a, b = item
            limit = BASE_VOCAB_SIZE + idx  # may only reference earlier IDs
            if not (NUM_SPECIAL_TOKENS <= a < limit and NUM_SPECIAL_TOKENS <= b < limit):
                raise TokenizerFormatError(
                    f"merges[{idx}] = {item} references an ID that does not exist yet "
                    f"(valid range {NUM_SPECIAL_TOKENS}..{limit - 1})"
                )
            pair = (a, b)
            if pair in seen:
                raise TokenizerFormatError(f"merges[{idx}] duplicates an earlier merge")
            seen.add(pair)
            merges.append(pair)
        tok = cls()
        tok._set_merges(merges)
        return tok
