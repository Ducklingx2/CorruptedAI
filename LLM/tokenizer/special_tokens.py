"""Special tokens shared by every tokenizer.

The IDs are fixed (0-3) for every tokenizer so that models, datasets and
checkpoints can rely on them regardless of which tokenizer produced them.

Policy: ordinary text is NEVER interpreted as a special token. The literal
string ``"<EOS>"`` in user text is encoded as normal characters/words/bytes.
Special tokens are only produced via ``encode(..., add_special_tokens=True)``
(which adds BOS/EOS) or by constructing ID sequences directly.
"""

from __future__ import annotations

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)
SPECIAL_TOKEN_IDS: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
ID_TO_SPECIAL_TOKEN: dict[int, str] = {i: tok for tok, i in SPECIAL_TOKEN_IDS.items()}

PAD_ID = SPECIAL_TOKEN_IDS[PAD_TOKEN]
UNK_ID = SPECIAL_TOKEN_IDS[UNK_TOKEN]
BOS_ID = SPECIAL_TOKEN_IDS[BOS_TOKEN]
EOS_ID = SPECIAL_TOKEN_IDS[EOS_TOKEN]

NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)
