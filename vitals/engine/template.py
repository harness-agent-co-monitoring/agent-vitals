"""Jinja rendering of the GGUF's embedded chat template.

The llama.cpp C API (`llama_chat_apply_template`) implements a lossy builtin
approximation: no `tools`, no template kwargs (e.g. Qwen's `enable_thinking`),
and for Qwen3.5 it omits the template's think-prefill. Rendering the actual
jinja source (GGUF metadata `tokenizer.chat_template`) is what llama-server
--jinja and HF transformers do — and what tool-calling harnesses need.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2


def _raise_exception(message: str):  # jinja templates call this on invalid input
    raise jinja2.TemplateError(message)


def _strftime_now(fmt: str) -> str:
    return datetime.now().strftime(fmt)


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """OpenAI protocol -> template conventions: tool_call `arguments` arrive as
    JSON strings on the wire but templates `tojson` them as objects."""
    out = deepcopy(messages)
    for m in out:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    pass  # leave as string; template will tojson the string
    return out


def _hf_tojson(x: Any, ensure_ascii: bool = False, indent: int | None = None,
               separators: tuple[str, str] | None = None, sort_keys: bool = False) -> str:
    """transformers' tojson filter: no HTML escaping, key order kept, HF keyword arguments."""
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


_HF_TOJSON_CALL = re.compile(r"tojson\s*\(\s*(ensure_ascii|separators|sort_keys)")


class ChatTemplate:
    def __init__(self, source: str, hf_tojson: bool | None = None):
        """hf_tojson: use transformers' `tojson` (key order kept, no HTML escaping) instead of jinja's
        (sorted keys, <>&' escaped). None = only when the template passes transformers-only keyword
        arguments to tojson, which jinja's filter rejects (GLM-5.3); other templates keep jinja's filter."""
        self.source = source
        # loopcontrols: {% break %} / {% continue %}, which HF's renderer also enables (GLM-5.3 uses them)
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.globals["raise_exception"] = _raise_exception
        env.globals["strftime_now"] = _strftime_now
        self.hf_tojson = bool(_HF_TOJSON_CALL.search(source)) if hf_tojson is None else hf_tojson
        if self.hf_tojson:
            env.filters["tojson"] = _hf_tojson
        self._tpl = env.from_string(source)

    @classmethod
    def from_engine(cls, engine) -> ChatTemplate | None:
        src = engine.meta("tokenizer.chat_template")
        return cls(src) if src else None

    def render(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        add_generation_prompt: bool = True,
        **template_kwargs: Any,
    ) -> str:
        """Render exactly as the model's authors specified. `template_kwargs`
        pass through as jinja variables (e.g. enable_thinking=True forces the
        Qwen3.5 prompt to end inside an open <think> block)."""
        return self._tpl.render(
            messages=_normalize_messages(messages),
            tools=tools or None,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )


def template_from_gguf(gguf_path: str | Path) -> str:
    """The chat template embedded in a GGUF file, read from its metadata without loading the weights."""
    import gguf

    field = gguf.GGUFReader(str(gguf_path)).fields["tokenizer.chat_template"]
    return bytes(field.parts[field.data[0]]).decode()
