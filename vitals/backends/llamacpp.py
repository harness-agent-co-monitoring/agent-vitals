"""The llama.cpp backend.

Implements the Backend protocol over vitals.engine.binding.LlamaEngine and
relies on four verified engine facts: non-perturbing cb_eval capture, full
hybrid-state seq_cp, fa-off attention materialization, and native-sampler
equivalence.

Episode lifecycle: every prefill() starts a fresh context (empty KV/recurrent
state, fresh sampler chain — reproducible per (prompt, sampler, seed)) and
re-registers capture per the negotiated CaptureSpec. Capture-off episodes run
with no callback registered at all (zero-overhead path).
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from vitals.backends.base import Backend, Capabilities, StepOutput, negotiate
from vitals.core.schema import AttentionMode, CaptureSpec, Region, RegionKind
from vitals.core.segmentation import segment_messages
from vitals.engine.binding import FLASH_ATTN_AUTO, FLASH_ATTN_DISABLED, LlamaEngine
from vitals.engine.capture import CaptureSession, plan_capture, pool_attention
from vitals.engine.nodes import ArchNodeMap, for_model, graph_layers
from vitals.engine.sampler import NativeSampler, SamplerConfig
from vitals.engine.template import ChatTemplate


def _rendered_groups(messages: list[dict]) -> list[int]:
    """Sizes of rendered message blocks: qwen-family chat templates merge
    consecutive `tool` messages into ONE user block of <tool_response>
    sections, so the rendered block count can be below len(messages)."""
    sizes: list[int] = []
    prev_tool = False
    for m in messages:
        is_tool = str(m.get("role")) == "tool"
        if is_tool and prev_tool:
            sizes[-1] += 1
        else:
            sizes.append(1)
        prev_tool = is_tool
    return sizes


def _byte_offsets(pieces: list[bytes], start: int = 0) -> list[int]:
    """Start offset of each piece, then the end offset, for pieces laid end to end from ``start``."""
    offs, cur = [], start
    for pc in pieces:
        offs.append(cur)
        cur += len(pc)
    offs.append(cur)
    return offs


def _locate_named_span(ctx: TurnCtx, req: dict, text: str, prompt_bytes: bytes, offs: list[int]) -> int:
    """Character offset of a named span's ``text`` in the rendered prompt, or -1 to skip the span.

    Most precise anchor first:
    1. "msg_index": the span asserts it lives in the i-th message, so only that
       message's rendered range is searched. Exact and collision-free (a short
       observation's text can recur verbatim inside an earlier message). An
       index beyond this turn's messages means the span belongs to a later
       turn: skip it, never fall back to a global match.
    2. "within" + "occurrence": resolve inside the n-th copy of an enclosing
       text (recurring feedback templates, byte-identical observations).
    3. bare text: global first occurrence.
    """
    prompt = ctx.prompt_text
    mi = req.get("msg_index")
    if mi is not None:
        mregs = [r for r in ctx.regions
                 if not (r.meta or {}).get("synthesized")
                 and not (r.meta or {}).get("generation_cue")
                 and not (r.meta or {}).get("named_span")]
        mi = int(mi)
        if mi < 0 or mi >= len(mregs):
            return -1
        mr = mregs[mi]
        if mr.token_start is not None and mr.token_end is not None:
            return prompt.find(
                text,
                len(prompt_bytes[:offs[mr.token_start]].decode("utf-8", errors="ignore")),
                len(prompt_bytes[:offs[mr.token_end]].decode("utf-8", errors="ignore")))
    within = req.get("within")
    anchor = str(within) if within else text
    pos, scan = -1, 0
    for _ in range(max(1, int(req.get("occurrence", 1)))):
        pos = prompt.find(anchor, scan)
        if pos < 0:
            return -1  # anchor absent: the missing region id is loud downstream
        scan = pos + 1
    if within:
        return prompt.find(text, pos, pos + len(anchor))
    return pos


@dataclass
class TurnCtx:
    """One turn's prompt as rendered: its regions with token spans, prompt length and generated tokens."""

    regions: list[Region] = field(default_factory=list)
    n_prompt: int = 0
    tokens: list[int] = field(default_factory=list)
    done: bool = False
    truncated: bool = False  # hit context capacity (finish_reason: length)
    prompt_text: str = ""
    # populated when CaptureSpec.prompt_scores: per-position [ent, m2,
    # surprisal] for prompt positions 1..n-1
    prompt_stats: object = None
    prompt_token_ids: list[int] | None = None   # the rendered prompt's tokens, every mode


class LlamaCppBackend(Backend):
    """A GGUF model on the pinned llama.cpp engine (``vitals.engine.binding``) with chat-template rendering,
    section resolution and the capture modes vitals uses. ``diagnostic_attention`` turns flash attention off so
    attention can be captured; ``n_seq_max=2`` gives replay and ablation a second sequence (the context is split
    between them)."""

    name = "llamacpp"

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 8192,
        n_gpu_layers: int = 999,
        diagnostic_attention: bool = False,
        sampler: SamplerConfig | None = None,
        node_map: ArchNodeMap | None = None,
        lib_path: str | Path | None = None,
        reuse_kv: bool = False,
        n_seq_max: int = 1,
    ):
        self.diagnostic_attention = diagnostic_attention
        self.sampler_config = sampler or SamplerConfig(temperature=0.0)
        self.engine = LlamaEngine(
            model_path,
            n_ctx=n_ctx,
            # single sequence: with the non-unified KV cache, n_ctx divides
            # across n_seq_max — one seq gives episodes the full context.
            # n_seq_max=2 lets replay keep a turn's prefix in seq 0 and decode
            # the forced output on a scratch seq (per-seq capacity = n_ctx / 2).
            n_seq_max=n_seq_max,
            n_gpu_layers=n_gpu_layers,
            flash_attn=FLASH_ATTN_DISABLED if diagnostic_attention else FLASH_ATTN_AUTO,
            lib_path=lib_path,
        )
        self.node_map = node_map or for_model(self.engine.arch, graph_layers(self.engine))
        self.template = ChatTemplate.from_engine(self.engine)  # None -> C-API fallback
        self._session: CaptureSession | None = None
        self._sampler: NativeSampler | None = None
        self.last_ctx: TurnCtx | None = None  # most recent prefill's ctx (regions w/ spans)
        # automatic sections (vitals.core.auto_sections) for render_turn calls that do not choose
        self.auto_sections = False
        # KV reuse across turns of one episode (opt-in; see prefill()).
        self.reuse_kv = reuse_kv
        self._kv_tokens: list[int] = []       # tokens currently resident in KV (seq 0)
        self._kv_plan_sig: tuple | None = None
        self.reuse_stats = {"reused_tokens": 0, "decoded_tokens": 0, "hits": 0, "misses": 0}
        # token id -> lossless piece bytes (the vocabulary is fixed), and the last
        # rendered prompt's (bytes, tokens, byte offsets) for prefix-reused tokenization
        self._piece_cache: dict[int, bytes] = {}
        self._tok_cache: tuple[bytes, list[int], list[int]] | None = None
        self.tokenize_stats = {"full": 0, "reused_tokens": 0, "tokenized_tokens": 0}

    # -- protocol ---------------------------------------------------------------

    def capabilities(self) -> Capabilities:
        modes = (
            (AttentionMode.NONE, AttentionMode.PER_HEAD, AttentionMode.REGION_POOLED)
            if self.diagnostic_attention
            else (AttentionMode.NONE,)
        )
        return Capabilities(
            full_logits=True,
            top_logprobs=True,
            activations=True,
            attention_modes=modes,
            branch_probes=True,
            interventions=False,
            note=(
                "attention capture requires diagnostic mode "
                "(LlamaCppBackend(diagnostic_attention=True), flash-attention off) "
                "and exists only on full-attention layers of hybrid architectures"
            ),
        )

    def _render(self, messages: list[dict], *, add_gen: bool, tools=None, chat_kwargs=None) -> str:
        if self.template is not None:
            return self.template.render(
                messages, tools=tools, add_generation_prompt=add_gen, **(chat_kwargs or {})
            )
        if tools or chat_kwargs:
            raise RuntimeError(
                "tools/template kwargs need the GGUF's jinja chat template; this model has none "
                "and the llama.cpp C-API fallback cannot render them"
            )
        return self.engine.apply_chat_template(messages, add_assistant=add_gen)

    # message-block start markers by template family; must tokenize to ONE token
    _BLOCK_MARKERS = ("<|im_start|>", "<|start_header_id|>")
    # Harmony-style templates (Muse Glimmer): one message can render as SEVERAL
    # `<|start|>{header}<|message|>...` blocks (reasoning to=self, one per tool
    # call, the answer to=user), so blocks are assigned to messages by header
    _HARMONY_MARKER, _HARMONY_BODY = "<|start|>", "<|message|>"

    def _harmony_spans(
        self, toks: list[int], messages: list[dict]
    ) -> tuple[list[tuple[int, int]], bool, list[int]] | None:
        """Message spans for Harmony-style templates. Each block's role is the
        first word of its header (system / user / assistant / tool / developer).
        Messages consume blocks in order; an assistant message takes every
        consecutive `assistant ...` block. The last block is the generation cue.
        Any disagreement between roles and headers -> None (spans degrade loudly)."""
        st = self.engine.tokenize(self._HARMONY_MARKER, add_special=False)
        bt = self.engine.tokenize(self._HARMONY_BODY, add_special=False)
        if len(st) != 1 or len(bt) != 1:
            return None
        pos = [i for i, t in enumerate(toks) if t == st[0]]
        if len(pos) < 2:
            return None
        heads = []
        for k, i in enumerate(pos):
            end = pos[k + 1] if k + 1 < len(pos) else len(toks)
            j = next((x for x in range(i + 1, end) if toks[x] == bt[0]), end)
            heads.append(self.engine.detokenize(toks[i + 1:j]).strip().split(" ")[0].lower())
        blocks = list(pairwise([0, *pos[1:], len(toks)]))  # attribute any preamble (BOS) to block 0
        if heads[-1] != "assistant":
            return None
        body, cue = list(zip(heads[:-1], blocks[:-1], strict=True)), blocks[-1]
        synth = bool(body) and body[0][0] == "system" and str(messages[0].get("role")) not in ("system", "developer")
        k = 1 if synth else 0
        spans = [body[0][1]] if synth else []
        for m in messages:
            role = str(m.get("role"))
            want = {"developer": "system"}.get(role, role)
            if k >= len(body) or body[k][0] != want:
                return None
            s, e = body[k][1]
            k += 1
            while role == "assistant" and k < len(body) and body[k][0] == "assistant":
                e = body[k][1][1]
                k += 1
            spans.append((s, e))
        if k != len(body):
            return None
        spans.append(cue)
        sizes = [1] * len(messages)
        return spans, synth, sizes

    # GLM templates: one role token opens each block and nothing closes it
    # (`[gMASK]<sop><|system|>...<|user|>...<|assistant|><think>...<|observation|>...`)
    _GLM_ROLES = {"<|system|>": "system", "<|user|>": "user", "<|assistant|>": "assistant",
                  "<|observation|>": "tool"}

    def _glm_spans(
        self, toks: list[int], messages: list[dict]
    ) -> tuple[list[tuple[int, int]], bool, list[int]] | None:
        """Message spans for GLM-style templates. Leading system blocks the conversation has no
        message for (GLM-5.3 always writes `Reasoning Effort: ...`, plus a tools block when tools are
        given) merge into one synthesized block. Consecutive tool messages share one observation
        block (subdivided at <tool_response> by the caller). The last block is the generation cue.
        Any disagreement between roles and blocks -> None (spans degrade loudly)."""
        ids = {}
        for marker, role in self._GLM_ROLES.items():
            t = self.engine.tokenize(marker, add_special=False)
            if len(t) != 1:
                return None
            ids[t[0]] = role
        pos = [i for i, t in enumerate(toks) if t in ids]
        if len(pos) < 2 or ids[toks[pos[-1]]] != "assistant":
            return None
        heads = [ids[toks[i]] for i in pos]
        blocks = list(pairwise([0, *pos[1:], len(toks)]))  # attribute the [gMASK]<sop> preamble to block 0
        body, cue = list(zip(heads[:-1], blocks[:-1], strict=True)), blocks[-1]
        lead_sys = next((i for i, (h, _) in enumerate(body) if h != "system"), len(body))
        real_sys = 1 if messages and str(messages[0].get("role")) in ("system", "developer") else 0
        n_synth = lead_sys - real_sys
        if n_synth < 0:
            return None
        spans, sizes, k = [], [], n_synth
        if n_synth:
            spans.append((body[0][1][0], body[n_synth - 1][1][1]))
        i = 0
        while i < len(messages):
            role = str(messages[i].get("role"))
            want = {"developer": "system"}.get(role, role)
            if k >= len(body) or body[k][0] != want:
                return None
            n = 1
            while role == "tool" and i + n < len(messages) and str(messages[i + n].get("role")) == "tool":
                n += 1
            spans.append(body[k][1])
            sizes.append(n)
            k += 1
            i += n
        if k != len(body):
            return None
        spans.append(cue)
        return spans, bool(n_synth), sizes

    def _block_spans(
        self, toks: list[int], n_msgs: int, group_sizes: list[int], messages: list[dict] | None = None
    ) -> tuple[list[tuple[int, int]], bool, list[int]] | None:
        """Token spans of rendered message blocks, by locating the atomic
        block-start marker token. Avoids partial template renders entirely
        (templates may validate the whole conversation and raise on prefixes).

        Two block-count interpretations are tried: one block per message, and
        the qwen-family merge where consecutive `tool` messages render as ONE
        user block (group_sizes). Each admits +1 (generation cue) or +2
        (template synthesized a system block, e.g. tools without a system
        message). Returns (spans, has_synth_system, sizes) with one span per
        rendered block tiling [0, len(toks)); any other marker count -> None
        (spans degrade loudly)."""
        for marker in self._BLOCK_MARKERS:
            mt = self.engine.tokenize(marker, add_special=False)
            if len(mt) != 1:
                continue
            pos = [i for i, t in enumerate(toks) if t == mt[0]]
            for sizes in ([1] * n_msgs, group_sizes):
                if len(pos) in (len(sizes) + 1, len(sizes) + 2):
                    spans = list(pairwise([0, *pos[1:], len(toks)]))  # attribute any preamble (BOS) to block 0
                    return spans, len(pos) == len(sizes) + 2, sizes
        if messages is not None:
            return self._harmony_spans(toks, messages) or self._glm_spans(toks, messages)
        return None

    def _assign_tool_group(
        self, regions: list[Region], toks: list[int], s: int, e: int
    ) -> None:
        """Subdivide one merged tool block: qwen-family templates render each
        consecutive tool message as a <tool_response>...</tool_response>
        section inside a single user block. Tile the block at the openers; on
        any count mismatch (e.g. an observation quoting the literal marker)
        leave just this group offset-less (loud, group-local)."""
        mt = self.engine.tokenize("<tool_response>", add_special=False)
        if not mt:
            return
        win = toks[s:e]
        occ = [s + i for i in range(len(win) - len(mt) + 1) if win[i : i + len(mt)] == mt]
        if len(occ) != len(regions):
            return
        for region, (bs, be) in zip(regions, pairwise([s, *occ[1:], e]), strict=False):
            region.token_start, region.token_end = bs, be

    def _piece(self, tok: int) -> bytes:
        cache = self.__dict__.setdefault("_piece_cache", {})     # tolerate backends built without __init__
        b = cache.get(tok)
        if b is None:
            b = cache[tok] = self.engine.token_bytes(tok)
        return b

    # special tokens: the tokenizer partitions the text on them, so the tokens
    # after one never depend on the text before it. Only anchors that are ONE
    # token in this vocabulary are used (a non-special spelling would not split).
    _TOK_ANCHORS = (b"<|im_start|>", b"<|start|>", b"<|start_header_id|>",
                    b"<|system|>", b"<|user|>", b"<|assistant|>", b"<|observation|>")

    def _tok_anchors(self) -> tuple[bytes, ...]:
        got = self.__dict__.get("_valid_tok_anchors")
        if got is None:
            got = tuple(a for a in self._TOK_ANCHORS
                        if len(self.engine.tokenize(a.decode(), add_special=False, parse_special=True)) == 1)
            self.__dict__["_valid_tok_anchors"] = got
        return got

    def _bos_id(self) -> int | None:
        """The BOS id when the tokenizer prepends one (Muse Glimmer does; Qwen does not)."""
        if "_bos" not in self.__dict__:
            bos, meta = None, getattr(self.engine, "meta", None)   # test doubles may lack metadata
            if meta is not None and str(meta("tokenizer.ggml.add_bos_token") or "").lower() == "true":
                try:
                    bos = int(meta("tokenizer.ggml.bos_token_id"))
                except (TypeError, ValueError):
                    bos = None
            self.__dict__["_bos"] = bos
        return self.__dict__["_bos"]

    def _text_pieces(self, toks: list[int], b: bytes) -> list[bytes]:
        """Token pieces for byte alignment. A prepended BOS carries bytes the rendered
        text does not contain, so it gets zero width (the model still sees it)."""
        pieces = [self._piece(t) for t in toks]
        if pieces and toks[0] == self._bos_id() and not b.startswith(pieces[0]):
            pieces[0] = b""
        return pieces

    def _tokenize_prompt(self, text: str) -> tuple[list[int], list[int], list[bytes]]:
        """tokenize(text) with its per-token byte offsets and pieces. When the
        previous prompt shares a prefix with this one (the next turn of an
        append-only history), the tokens up to the last special-token boundary
        inside that shared prefix are reused and only the remainder is
        tokenized; identical to a full tokenization because the tokenizer
        splits on special tokens first. Every reuse is verified to reproduce
        the prompt bytes exactly."""
        b = text.encode("utf-8")
        stats = self.__dict__.setdefault("tokenize_stats", {"full": 0, "reused_tokens": 0, "tokenized_tokens": 0})
        prev = self.__dict__.get("_tok_cache")
        if prev is not None:
            pb, ptoks, poffs = prev
            lo, hi = 0, min(len(b), len(pb))          # longest common prefix, by bisection on slices
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if b[:mid] == pb[:mid]:
                    lo = mid
                else:
                    hi = mid - 1
            cut = max((b.rfind(a, 0, lo) for a in self._tok_anchors()), default=-1)
            if cut > 0:
                k = bisect_left(poffs, cut)
                if k < len(ptoks) and poffs[k] == cut:
                    suffix = self.engine.tokenize(b[cut:].decode("utf-8"), add_special=False, parse_special=True)
                    pieces = self._text_pieces(ptoks[:k], b) + [self._piece(t) for t in suffix]
                    tail = b"".join(pieces[k:])
                    if tail == b[cut:]:
                        toks = ptoks[:k] + suffix
                        offs = poffs[:k] + _byte_offsets(pieces[k:], cut)
                        self._tok_cache = (b, toks, offs)
                        stats["reused_tokens"] += k
                        stats["tokenized_tokens"] += len(suffix)
                        return toks, offs, pieces
        toks = self.engine.tokenize(text)
        pieces = self._text_pieces(toks, b)
        offs = _byte_offsets(pieces)
        self._tok_cache = (b, toks, offs)
        stats["full"] += 1
        stats["tokenized_tokens"] += len(toks)
        return toks, offs, pieces

    def render_turn(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        chat_kwargs: dict | None = None,
        spans: list[dict] | None = None,
        auto_sections: bool | None = None,
    ) -> tuple[TurnCtx, list[int]]:
        """Pure turn preparation: template render, tokenization, message-region
        token spans, and named sub-span resolution. Touches no engine state
        (tokenize/detokenize only), so offline consumers -- attention replay,
        span repair -- reuse the exact serving-path region logic. With
        ``auto_sections`` (default: the backend's ``auto_sections``), the parts
        of the request no declared span covers get automatic sections
        (``vitals.core.auto_sections``)."""
        ctx = TurnCtx(regions=segment_messages(messages))
        raw = len(messages) == 1 and messages[0].get("role") == "raw"
        if raw:
            ctx.prompt_text = str(messages[0].get("content", ""))
        else:
            ctx.prompt_text = self._render(messages, add_gen=True, tools=tools, chat_kwargs=chat_kwargs)

        toks, offs, pieces = self._tokenize_prompt(ctx.prompt_text)
        if (getattr(self, "auto_sections", False) if auto_sections is None else auto_sections) and not raw:
            from vitals.core.auto_sections import with_auto_spans

            spans = with_auto_spans(messages, tools, ctx.prompt_text, spans)
        if raw:
            ctx.regions[0].token_start, ctx.regions[0].token_end = 0, len(toks)
        else:
            found = self._block_spans(toks, len(messages), _rendered_groups(messages), messages=messages)
            if found is not None:
                blocks, synth_system, sizes = found
                if synth_system:
                    # template created a leading block we have no message for
                    # (tools without a system message) -> a real schema region
                    ctx.regions.insert(
                        0,
                        Region(id="schema_0", kind=RegionKind.TOOL_SCHEMA, role="system",
                               meta={"synthesized": True}),
                    )
                    sizes = [1, *sizes]
                ri = 0
                for gsize, (s, e) in zip(sizes, blocks[:-1], strict=False):
                    grp = ctx.regions[ri:ri + gsize]
                    ri += gsize
                    if gsize == 1 and grp:
                        grp[0].token_start, grp[0].token_end = s, e
                    elif grp:
                        self._assign_tool_group(grp, toks, s, e)
                s, e = blocks[-1]
                ctx.regions.append(
                    Region(id="cue_0", kind=RegionKind.REASONING, role="assistant",
                           token_start=s, token_end=e, meta={"generation_cue": True})
                )
                if tools:
                    ctx.regions[0].meta["includes_tool_schema"] = len(tools)
            # else: spans stay None; attn_share degrades to None (loud, not wrong)

        # Named sub-spans (e.g. scaffold sections): each {"name", "text"} is
        # located in the rendered prompt (see _locate_named_span) and appended
        # as an extra pooled region. Overlap with the enclosing role region is
        # intended -- pool_attention sums each region independently, and
        # regions.jsonl records the ids that key the attn_share vector.
        if spans:
            # Keep partial UTF-8 pieces as bytes. Decoding each token separately
            # inserts replacement characters and shifts later named sections.
            prompt_bytes = ctx.prompt_text.encode("utf-8")
            if offs[-1] != len(prompt_bytes) or b"".join(pieces) != prompt_bytes:
                raise ValueError("cannot align named spans: token bytes do not exactly reconstruct the rendered prompt")
            for req in spans:
                name, text = str(req.get("name", "")), str(req.get("text", ""))
                if not name or not text:
                    continue
                cs = _locate_named_span(ctx, req, text, prompt_bytes, offs)
                if cs < 0:
                    continue
                ce = cs + len(text)
                bs = len(ctx.prompt_text[:cs].encode("utf-8"))
                be = len(ctx.prompt_text[:ce].encode("utf-8"))
                ts = max(0, bisect_right(offs, bs) - 1)
                te = bisect_left(offs, be)
                if te > ts:
                    ctx.regions.append(
                        Region(id=f"span:{name}", kind=RegionKind.USER, role="span",
                               char_start=cs, char_end=ce,
                               token_start=ts, token_end=te,
                               meta={"named_span": True, "token_alignment": "utf8_bytes",
                                     **({"auto": True} if req.get("auto") else {})})
                    )
        return ctx, toks

    def prefill(
        self,
        messages: list[dict],
        capture: CaptureSpec,
        *,
        tools: list[dict] | None = None,
        chat_kwargs: dict | None = None,
        spans: list[dict] | None = None,
    ) -> TurnCtx:
        negotiate(capture, self.capabilities(), self.name)
        plan = plan_capture(capture, self.node_map)  # raises CaptureUnsupported on bad layers
        self._session = CaptureSession(plan, self.node_map)
        plan_sig = (tuple(sorted(plan.names)), tuple(sorted(plan.modes.items())))

        # KV reuse (opt-in): if the new prompt's tokens EXACTLY extend the tokens
        # already resident in KV (same capture plan), decode only the delta.
        # Append-only by design: the hybrid recurrent state (DeltaNet) cannot be
        # partially rolled back, so any divergence -> full reset. Note the
        # measurement consequence: with reuse, prefill capture covers only the
        # delta rows (the reused prefix was captured in earlier turns).
        if not (self.reuse_kv and self._kv_plan_sig == plan_sig
                and self._kv_tokens and self.engine.n_past == len(self._kv_tokens)):
            # fresh episode/context: fresh KV/recurrent state, capture filter per spec
            self.engine.reset_context(capture_names=plan.names, capture_modes=plan.modes)
            self._kv_tokens = []
        self._kv_plan_sig = plan_sig
        if self._sampler is not None:
            self._sampler.free()
        self._sampler = NativeSampler(self.engine, self.sampler_config)

        ctx, toks = self.render_turn(messages, tools=tools,
                                     chat_kwargs=chat_kwargs, spans=spans)
        ctx.prompt_token_ids = list(toks)

        if self.reuse_kv and self._kv_tokens:
            n = len(self._kv_tokens)
            if len(toks) > n and toks[:n] == self._kv_tokens:
                # exact prefix extension: decode only the delta
                delta = toks[n:]
                self.engine.decode(delta)
                self._kv_tokens = list(toks)
                self.reuse_stats["hits"] += 1
                self.reuse_stats["reused_tokens"] += n
                self.reuse_stats["decoded_tokens"] += len(delta)
            else:
                # divergence (retokenization drift, edited history, new episode):
                # full reset — no partial rollback on hybrid recurrent state
                self.reuse_stats["misses"] += 1
                self.engine.reset_context()
                self.engine.decode(toks)
                self._kv_tokens = list(toks)
                self.reuse_stats["decoded_tokens"] += len(toks)
        else:
            if capture.prompt_scores:
                # scored prefill: same single pass, per-position logit reads
                ctx.prompt_stats = self.engine.score_prompt(toks)
            else:
                self.engine.decode(toks)
            if self.reuse_kv:
                self._kv_tokens = list(toks)
                self.reuse_stats["decoded_tokens"] += len(toks)
        ctx.n_prompt = len(toks)
        self.last_ctx = ctx
        return ctx

    def step(self, ctx: TurnCtx) -> StepOutput:
        resid, attn = self._session.harvest(self.engine) if self._session else ({}, {})
        share = pool_attention(attn, ctx.regions, self.engine.n_past) if attn else None
        return StepOutput(logits=self.engine.logits(-1), resid=resid, attn=attn, attn_share=share)

    def sample(self, ctx: TurnCtx, step: StepOutput, route=None) -> int:
        return self._sampler.sample(-1)

    def advance(self, ctx: TurnCtx, token: int) -> TurnCtx:
        ctx.tokens.append(token)
        if self.engine.is_eog(token):
            ctx.done = True
        elif self.engine.n_past + 1 >= self.engine.n_ctx_per_seq:
            # context capacity reached: end the turn as length-limited rather
            # than letting llama_decode fail with rc=1 mid-episode
            ctx.done = True
            ctx.truncated = True
        else:
            self.engine.decode([token])
            if self.reuse_kv:
                # this token is now resident in KV; the turn-ending token (EOG or
                # truncation) never is, and must not enter the prefix record
                self._kv_tokens.append(token)
        return ctx

    def done(self, ctx: TurnCtx) -> bool:
        return ctx.done

    # -- extras -------------------------------------------------------------------

    def detokenize(self, tokens: list[int]) -> str:
        return b"".join(self._piece(int(t)) for t in tokens).decode(errors="replace")

    def close(self) -> None:
        """Free the sampler, the engine, and the probe state cached on this backend (output weights on the GPU)."""
        self.__dict__.pop("_section_writes", None)
        if self._sampler is not None:
            self._sampler.free()
            self._sampler = None
        self.engine.close()
