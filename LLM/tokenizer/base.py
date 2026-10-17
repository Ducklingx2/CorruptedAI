"""Abstract tokenizer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from .errors import InvalidTokenIdError
from .special_tokens import BOS_ID, EOS_ID


class Tokenizer(ABC):
    """Common interface: ``train``, ``encode``, ``decode`` plus save/load.

    ``TYPE`` is the identifier written to the serialized file.
    """

    TYPE: str = ""

    @abstractmethod
    def train(self, text: str) -> None:
        """Learn vocabulary (and merges, if any) from ``text``."""

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Convert text to token IDs. If requested, wrap with BOS/EOS."""

    @abstractmethod
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """Convert token IDs back to text. Raises InvalidTokenIdError."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Total number of IDs, including special tokens."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return the tokenizer-specific JSON-serializable state (incl. ``type``)."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tokenizer":
        """Rebuild a tokenizer from validated envelope + ``to_dict`` state."""

    # ---- shared helpers -------------------------------------------------

    @staticmethod
    def _wrap(ids: list[int], add_special_tokens: bool) -> list[int]:
        return [BOS_ID, *ids, EOS_ID] if add_special_tokens else ids

    def _check_id(self, token_id: object) -> int:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise InvalidTokenIdError(f"Token ID must be an int, got {token_id!r}")
        if not 0 <= token_id < self.vocab_size:
            raise InvalidTokenIdError(
                f"Token ID {token_id} is outside the vocabulary [0, {self.vocab_size})"
            )
        return token_id

    # ---- persistence ----------------------------------------------------

    def save(self, path: str | Path) -> None:
        from .serialization import save_tokenizer

        save_tokenizer(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        from .serialization import load_tokenizer

        tok = load_tokenizer(path)
        if not isinstance(tok, cls):
            raise TypeError(f"File contains a {type(tok).__name__}, not a {cls.__name__}")
        return tok
