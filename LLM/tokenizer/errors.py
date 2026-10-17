"""Exception types raised by the tokenizer subsystem."""

from __future__ import annotations


class TokenizerError(Exception):
    """Base class for all tokenizer errors."""


class NotTrainedError(TokenizerError):
    """Raised when a tokenizer that needs training is used before ``train``."""


class InvalidTokenIdError(TokenizerError, ValueError):
    """Raised when ``decode`` receives an ID that is not in the vocabulary."""


class TokenizerFormatError(TokenizerError, ValueError):
    """Raised when a serialized tokenizer file is corrupted or incompatible."""
