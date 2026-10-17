"""Character-level tokenizer.

* Input is Unicode-normalized to NFC before training and encoding, so
  ``"e\\u0301"`` and ``"\\u00e9"`` are the same text. This is the only
  normalization applied; whitespace and punctuation are kept as characters.
* Vocabulary: special tokens (0-3), then every distinct character seen in
  training, sorted by code point (deterministic).
* Characters not seen in training become ``<UNK>``.

Round-trip guarantee: ``decode(encode(x)) == NFC(x)`` whenever every character
of ``x`` was seen in training. Unknown characters decode to ``<UNK>``
(or are dropped with ``skip_special_tokens=True``).
"""

from __future__ import annotations

import unicodedata
from typing import Any, Sequence

from .base import Tokenizer
from .errors import NotTrainedError, TokenizerFormatError
from .special_tokens import (
    ID_TO_SPECIAL_TOKEN,
    NUM_SPECIAL_TOKENS,
    SPECIAL_TOKENS,
    UNK_ID,
)


class CharacterTokenizer(Tokenizer):
    TYPE = "character"

    def __init__(self) -> None:
        self._trained = False
        self._set_chars([])

    def _set_chars(self, chars: list[str]) -> None:
        self._id_to_token: list[str] = [*SPECIAL_TOKENS, *chars]
        self._char_to_id: dict[str, int] = {c: i + NUM_SPECIAL_TOKENS for i, c in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    def train(self, text: str) -> None:
        text = unicodedata.normalize("NFC", text)
        self._set_chars(sorted(set(text)))
        self._trained = True

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not self._trained:
            raise NotTrainedError("CharacterTokenizer must be trained or loaded before encode()")
        text = unicodedata.normalize("NFC", text)
        ids = [self._char_to_id.get(c, UNK_ID) for c in text]
        return self._wrap(ids, add_special_tokens)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        if not self._trained:
            raise NotTrainedError("CharacterTokenizer must be trained or loaded before decode()")
        out: list[str] = []
        for raw in ids:
            i = self._check_id(raw)
            if i in ID_TO_SPECIAL_TOKEN and skip_special_tokens:
                continue
            out.append(self._id_to_token[i])
        return "".join(out)

    def to_dict(self) -> dict[str, Any]:
        if not self._trained:
            raise NotTrainedError("Cannot serialize an untrained CharacterTokenizer")
        return {"type": self.TYPE, "chars": self._id_to_token[NUM_SPECIAL_TOKENS:]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CharacterTokenizer":
        from .serialization import require_field

        chars = require_field(data, "chars", list)
        if not all(isinstance(c, str) and len(c) == 1 for c in chars):
            raise TokenizerFormatError("'chars' must be a list of single-character strings")
        if len(set(chars)) != len(chars):
            raise TokenizerFormatError("'chars' contains duplicate entries")
        tok = cls()
        tok._set_chars(list(chars))
        tok._trained = True
        return tok
