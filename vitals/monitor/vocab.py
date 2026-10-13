"""Token text from a GGUF's tokenizer metadata, without loading the model.

The monitor reads stored token ids and needs their exact text and character offsets (to find where a tool
argument's value was generated). ``Vocab.from_gguf`` reads ``tokenizer.ggml.tokens`` and decodes each token to its
bytes: byte-level BPE vocabularies (``gpt2``: Qwen, GLM) through the GPT-2 byte-to-unicode table, SentencePiece
ones (``llama``) through ``▁`` and ``<0xNN>`` byte tokens. Control and user-defined tokens are their own text.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_CONTROL, _USER_DEFINED, _BYTE = 3, 4, 6


def _byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


class Vocab:
    """Each token id's bytes, from GGUF tokenizer metadata (``from_gguf``); ``text_and_offsets`` turns stored
    token ids into text with each token's character span."""

    def __init__(self, pieces: list[bytes]):
        self.pieces = pieces

    @classmethod
    def from_tokens(cls, tokens: list[str], model: str = "gpt2", types: list[int] | None = None) -> Vocab:
        dec = _byte_decoder()
        out = []
        for i, t in enumerate(tokens):
            ty = types[i] if types else 1
            if ty in (_CONTROL, _USER_DEFINED):
                out.append(t.encode())
            elif model == "llama":
                if ty == _BYTE or (t.startswith("<0x") and t.endswith(">") and len(t) == 6):
                    out.append(bytes([int(t[3:5], 16)]))
                else:
                    out.append(t.replace("▁", " ").encode())
            else:
                try:
                    out.append(bytes(dec[c] for c in t))
                except KeyError:                       # a token outside the byte alphabet: keep its text
                    out.append(t.encode())
        return cls(out)

    @classmethod
    def from_gguf(cls, path: str | Path) -> Vocab:
        return _load(str(Path(path).resolve()))

    def piece(self, token: int) -> bytes:
        return self.pieces[int(token)]

    def text_and_offsets(self, ids) -> tuple[str, list[tuple[int, int]]]:
        """Decoded text of ``ids`` and every token's [start, end) character span in it. A token that ends inside a
        multi-byte character owns the characters that begin within its bytes."""
        buf = bytearray()
        bounds = []
        for t in ids:
            start = len(buf)
            buf += self.pieces[int(t)]
            bounds.append((start, len(buf)))
        text = bytes(buf).decode("utf-8", errors="replace")
        # character index of each byte offset: count UTF-8 lead bytes before it
        char_at = [0] * (len(buf) + 1)
        c = 0
        for i, b in enumerate(buf):
            char_at[i] = c
            if (b & 0xC0) != 0x80:
                c += 1
        char_at[len(buf)] = c
        if c != len(text):                              # invalid UTF-8 somewhere: offsets by decoded prefixes
            char_at = [len(bytes(buf[:i]).decode("utf-8", errors="replace")) for i in range(len(buf) + 1)]
        # a token starting on a continuation byte begins at the next character; one ending inside a character
        # owns it (its lead byte is counted before the token's end)
        return text, [(char_at[s], char_at[e]) for s, e in bounds]


@lru_cache(maxsize=4)
def _load(path: str) -> Vocab:
    from gguf import GGUFReader

    f = GGUFReader(path).fields
    model = bytes(f["tokenizer.ggml.model"].parts[-1]).decode()
    toks = f["tokenizer.ggml.tokens"]
    tokens = [bytes(toks.parts[i]).decode("utf-8", errors="replace") for i in toks.data]
    types = None
    if "tokenizer.ggml.token_type" in f:
        tt = f["tokenizer.ggml.token_type"]
        types = [int(tt.parts[i][0]) for i in tt.data]
    return Vocab.from_tokens(tokens, model, types)
