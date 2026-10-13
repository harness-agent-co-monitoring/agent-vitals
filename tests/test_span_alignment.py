"""Regression checks for named spans following split UTF-8 characters."""

import pytest

from vitals.backends.llamacpp import LlamaCppBackend


class PieceEngine:
    def __init__(self, pieces):
        self.pieces = pieces

    def tokenize(self, text):
        return list(range(len(self.pieces)))

    def token_bytes(self, token):
        return self.pieces[token]


def render(pieces, text, spans):
    backend = object.__new__(LlamaCppBackend)
    backend.engine = PieceEngine(pieces)
    return backend.render_turn([{"role": "raw", "content": text}], spans=spans)[0]


def test_split_character_and_later_section_have_exact_token_ranges():
    text = "A龘B rule"
    pieces = [b"A", b"\xe9", b"\xbe\x98", b"B", b" rule"]
    ctx = render(pieces, text, [{"name": "character", "text": "龘"},
                                {"name": "rule", "text": "rule", "msg_index": 0}])
    regions = {r.id: r for r in ctx.regions}
    char, rule = regions["span:character"], regions["span:rule"]
    assert (char.token_start, char.token_end) == (1, 3)
    assert (rule.token_start, rule.token_end) == (4, 5)
    assert (rule.char_start, rule.char_end) == (4, 8)
    assert rule.meta["token_alignment"] == "utf8_bytes"


def test_occurrence_anchor_and_shared_boundary_token():
    text = "龘 one rule two rule"
    pieces = [b"\xe9", b"\xbe\x98", b" one", b" rule", b" two", b" rule"]
    ctx = render(pieces, text, [{"name": "second", "text": "rule", "occurrence": 2},
                                {"name": "within", "text": "rule", "within": "two rule"}])
    for r in ctx.regions:
        if r.id.startswith("span:"):
            assert (r.token_start, r.token_end) == (5, 6)
            assert text[r.char_start:r.char_end] == "rule"


def test_non_roundtripping_tokens_fail_instead_of_assigning_wrong_span():
    with pytest.raises(ValueError, match="exactly reconstruct"):
        render([b"different"], "rule", [{"name": "rule", "text": "rule"}])
