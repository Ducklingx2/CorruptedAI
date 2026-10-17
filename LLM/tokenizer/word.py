"""Word-level tokenizer.

Tokenization rules
------------------
* Text is Unicode-normalized to NFC (and lowercased if ``lowercase=True``).
* Tokens are runs of word characters (letters, digits, underscore), optionally
  with internal apostrophes (``don't`` is one token), or single punctuation
  characters.
* Whitespace is discarded. ``decode`` re-joins tokens with single spaces and
  attaches closing punctuation to the previous word (``hello !`` -> ``hello!``).

Round-trip guarantee: ``decode(encode(x))`` equals ``x`` only when ``x`` is
already single-space separated, uses simple punctuation, and contains only
known words. Whitespace and unknown words are NOT preserved.

Vocabulary order: special tokens (0-3), then tokens by descending frequency,
ties broken alphabetically. Out-of-vocabulary words map to ``<UNK>``.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Sequence

from .base import Tokenizer
from .errors import NotTrainedError, TokenizerFormatError
from .special_tokens import (
    ID_TO_SPECIAL_TOKEN,
    NUM_SPECIAL_TOKENS,
    SPECIAL_TOKENS,
    UNK_ID,
)

_WORD_RE = re.compile(r"\w+(?:['\u2019]\w+)*|[^\w\s]", re.UNICODE)
_NO_SPACE_BEFORE = frozenset(".,!?;:%)]}")
_NO_SPACE_AFTER = frozenset("([{")


class WordTokenizer(Tokenizer):
    TYPE = "word"

    def __init__(
        self,
        lowercase: bool = False,
        max_vocab_size: int | None = None,
        min_frequency: int = 1,
    ) -> None:
        if max_vocab_size is not None and max_vocab_size <= NUM_SPECIAL_TOKENS:
            raise ValueError(f"max_vocab_size must exceed {NUM_SPECIAL_TOKENS}")
        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")
        self.lowercase = lowercase
        self.max_vocab_size = max_vocab_size
        self.min_frequency = min_frequency
        self._trained = False
        self._set_vocab([])

    # ---- internals ------------------------------------------------------

    def _set_vocab(self, words: list[str]) -> None:
        self._id_to_token: list[str] = [*SPECIAL_TOKENS, *words]
        self._token_to_id: dict[str, int] = {t: i for i, t in enumerate(self._id_to_token)}
        # Literal "<EOS>" in text must stay ordinary text, so only learned words
        # are looked up (special names can never be in `words`; see from_dict).
        self._word_to_id: dict[str, int] = {
            w: i + NUM_SPECIAL_TOKENS for i, w in enumerate(words)
        }

    def _tokenize(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFC", text)
        if self.lowercase:
            text = text.lower()
        return _WORD_RE.findall(text)

    # ---- public API -----------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    def train(self, text: str) -> None:
        counts = Counter(self._tokenize(text))
        # A literal "<PAD>" etc. in training text tokenizes to "<", "PAD", ">",
        # so it can never collide with a special token name.
        items = [(w, c) for w, c in counts.items() if c >= self.min_frequency]
        items.sort(key=lambda wc: (-wc[1], wc[0]))
        if self.max_vocab_size is not None:
            items = items[: self.max_vocab_size - NUM_SPECIAL_TOKENS]
        self._set_vocab([w for w, _ in items])
        self._trained = True

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not self._trained:
            raise NotTrainedError("WordTokenizer must be trained or loaded before encode()")
        ids = [self._word_to_id.get(tok, UNK_ID) for tok in self._tokenize(text)]
        return self._wrap(ids, add_special_tokens)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        if not self._trained:
            raise NotTrainedError("WordTokenizer must be trained or loaded before decode()")
        out: list[str] = []
        for raw in ids:
            i = self._check_id(raw)
            if i in ID_TO_SPECIAL_TOKEN and skip_special_tokens:
                continue
            out.append(self._id_to_token[i])
        return _join_tokens(out)

    # ---- serialization --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        if not self._trained:
            raise NotTrainedError("Cannot serialize an untrained WordTokenizer")
        return {
            "type": self.TYPE,
            "lowercase": self.lowercase,
            "vocab": self._id_to_token[NUM_SPECIAL_TOKENS:],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WordTokenizer":
        from .serialization import require_field

        lowercase = require_field(data, "lowercase", bool)
        vocab = require_field(data, "vocab", list)
        if not all(isinstance(w, str) and w for w in vocab):
            raise TokenizerFormatError("'vocab' must be a list of non-empty strings")
        if len(set(vocab)) != len(vocab):
            raise TokenizerFormatError("'vocab' contains duplicate entries")
        if set(vocab) & set(SPECIAL_TOKENS):
            raise TokenizerFormatError("'vocab' must not contain special token names")
        tok = cls(lowercase=lowercase)
        tok._set_vocab(list(vocab))
        tok._trained = True
        return tok


def _join_tokens(pieces: list[str]) -> str:
    result = ""
    prev = ""
    for piece in pieces:
        if not result:
            result = piece
        elif piece in _NO_SPACE_BEFORE or prev in _NO_SPACE_AFTER:
            result += piece
        else:
            result += " " + piece
        prev = piece
    return result
