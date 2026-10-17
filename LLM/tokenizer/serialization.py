"""Versioned JSON serialization for tokenizers.

File layout (format_version 1)::

    {
      "format_version": 1,
      "type": "word" | "character" | "bpe",
      "special_tokens": {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3},
      ...tokenizer-specific fields...
    }

Files are plain JSON and are only ever parsed as data, never executed.
Any structural problem raises ``TokenizerFormatError`` with a specific message.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .base import Tokenizer
from .errors import TokenizerFormatError
from .special_tokens import SPECIAL_TOKEN_IDS

FORMAT_VERSION = 1


def require_field(data: dict[str, Any], key: str, expected: type) -> Any:
    """Return ``data[key]`` after checking it exists and has the right type."""
    if key not in data:
        raise TokenizerFormatError(f"Missing required field {key!r}")
    value = data[key]
    # bool is a subclass of int; never accept it where an int is expected.
    if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
        raise TokenizerFormatError(
            f"Field {key!r} must be {expected.__name__}, got {type(value).__name__}"
        )
    return value


def save_tokenizer(tokenizer: Tokenizer, path: str | Path) -> None:
    payload = {
        "format_version": FORMAT_VERSION,
        "special_tokens": dict(SPECIAL_TOKEN_IDS),
        **tokenizer.to_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file first so a crash cannot leave a half-written tokenizer.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def load_tokenizer(path: str | Path) -> Tokenizer:
    from .bpe import BPETokenizer
    from .character import CharacterTokenizer
    from .word import WordTokenizer

    registry: dict[str, type[Tokenizer]] = {
        WordTokenizer.TYPE: WordTokenizer,
        CharacterTokenizer.TYPE: CharacterTokenizer,
        BPETokenizer.TYPE: BPETokenizer,
    }

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise TokenizerFormatError(f"{path} is not valid JSON: {e}") from e
    except UnicodeDecodeError as e:
        raise TokenizerFormatError(f"{path} is not valid UTF-8 text: {e}") from e

    if not isinstance(data, dict):
        raise TokenizerFormatError("Top-level JSON value must be an object")

    version = require_field(data, "format_version", int)
    if version != FORMAT_VERSION:
        raise TokenizerFormatError(
            f"Unsupported format_version {version}; this code reads version {FORMAT_VERSION}"
        )

    special = require_field(data, "special_tokens", dict)
    if special != SPECIAL_TOKEN_IDS:
        raise TokenizerFormatError(
            f"special_tokens {special} do not match expected {SPECIAL_TOKEN_IDS}"
        )

    kind = require_field(data, "type", str)
    cls = registry.get(kind)
    if cls is None:
        raise TokenizerFormatError(
            f"Unknown tokenizer type {kind!r}; expected one of {sorted(registry)}"
        )
    return cls.from_dict(data)
