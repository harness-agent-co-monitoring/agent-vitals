"""Writing a logged model call as a replayable trace-store turn."""

from __future__ import annotations

from vitals.core.schema import TokenMetrics


def forced_text(template, messages: list[dict], assistant: dict, *, tools: list[dict] | None = None,
                chat_kwargs: dict | None = None) -> str:
    """What the model emitted after the generation prompt, in the template's own serialization.

    Renders the conversation with the generation prompt, renders it again with
    the assistant message appended, and returns the difference. Reasoning,
    tool calls and the end-of-turn token come out exactly as the template
    writes them (the end-of-turn token is removed afterwards on token level).
    """
    if template is None:
        raise ValueError("the reference model has no chat template")
    kw = dict(chat_kwargs or {})
    prompt = template.render(messages, tools=tools, add_generation_prompt=True, **kw)
    full = template.render([*messages, assistant], tools=tools, add_generation_prompt=False, **kw)
    if not full.startswith(prompt):
        hint = ""
        if assistant.get("reasoning_content"):
            hint = (" The output carries reasoning, but this generation prompt does not open a reasoning "
                    "block: record the chat template options the request used, e.g. "
                    "chat_kwargs={'enable_thinking': True}.")
        raise ValueError("the recorded output is not a continuation of the generation prompt rendered with "
                         f"chat_kwargs={kw or None}.{hint}")
    return full[len(prompt):]


def strip_end_of_turn(engine, tokens: list[int]) -> list[int]:
    """Drop the trailing end-of-turn token and the formatting the template writes after it."""
    for i in range(len(tokens) - 1, -1, -1):
        if engine.is_eog(tokens[i]):
            return tokens[:i]
        if engine.detokenize([tokens[i]]).strip():
            break
    return tokens


def append_forced(store, episode_id: str, turn: int, tokens) -> None:
    """Queue one turn's forced tokens as generated-token rows. Their statistics stay zero until replay."""
    for step, token in enumerate(tokens):
        store.append(TokenMetrics(episode_id=episode_id, turn=turn, step=step, token_id=int(token),
                                  ent=0.0, m2=0.0, js=0.0, fork=False, region_id="gen"))


def record_turn(store, backend, episode_id: str, turn: int, messages: list[dict], assistant: dict, *,
                tools: list[dict] | None = None, chat_kwargs: dict | None = None) -> int:
    """Store one logged call: the request messages and the reply as forced tokens.

    ``assistant`` is the reply as an OpenAI-style message with ``content`` and, when present,
    ``reasoning_content`` and ``tool_calls``. Returns the number of forced tokens.
    """
    text = forced_text(backend.template, messages, assistant, tools=tools, chat_kwargs=chat_kwargs)
    tokens = strip_end_of_turn(backend.engine, backend.engine.tokenize(text, add_special=False, parse_special=True))
    store.write_messages(episode_id, turn, messages, tools=tools, chat_kwargs=chat_kwargs)
    append_forced(store, episode_id, turn, tokens)
    store.flush(episode_id)
    return len(tokens)
