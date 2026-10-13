"""ctypes binding over libllama, pinned to one llama.cpp release (b9910).

Deliberately small API surface: model/context lifecycle, decode, logits
access, sampler chain, memory_seq_cp/rm, sequence state, cb_eval. Struct
layouts are transcribed from the PINNED headers (vendor/llama.cpp @ b9910);
`doctor` verifies symbol presence so version drift fails at setup, not
mid-episode.
"""

from __future__ import annotations

import os

# torch's CUDA libs must be resolved BEFORE ggml binds its own (older) libcudart,
# or the device-side head pooling's later `import torch` fails with an
# undefined-symbol ImportError (cudaGetDriverEntryPointByVersion). Any path that
# loads libllama goes through this module, so the pre-import lives here.
# VITALS_REPLAY_HEAD_MEAN=0 opts out (no torch needed for host-side pooling).
if os.environ.get("VITALS_REPLAY_HEAD_MEAN", "1") != "0":
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

import ctypes as C
from pathlib import Path

import numpy as np

from vitals.engine import PINNED_COMMIT, PINNED_TAG
from vitals.engine.head_weights import for_node as head_weights_for
from vitals.engine.head_weights import layer_dropped

# --------------------------------------------------------------------------
# symbol table (verified by `vitals engine doctor`)
# --------------------------------------------------------------------------

REQUIRED_SYMBOLS = [
    # lifecycle
    "llama_backend_init",
    "llama_model_load_from_file",
    "llama_model_free",
    "llama_init_from_model",
    "llama_free",
    # decode + logits
    "llama_decode",
    "llama_get_logits",
    "llama_get_logits_ith",
    # tokenizer
    "llama_tokenize",
    "llama_token_to_piece",
    # sampler chain (native sampling semantics)
    "llama_sampler_chain_init",
    "llama_sampler_chain_add",
    "llama_sampler_sample",
    # branch probes
    "llama_memory_seq_cp",
    "llama_memory_seq_rm",
    # sequence state blobs
    "llama_state_seq_get_size",
    "llama_state_seq_get_data",
    "llama_state_seq_set_data",
]

_DEFAULT_CANDIDATES = [
    "vendor/llama.cpp/build/bin/libllama.so",  # project-local pinned build
    "vendor/llama.cpp/build/bin/libllama.dylib",
]


def find_libllama() -> Path:
    """Locate libllama: $VITALS_LLAMA_LIB, then a pinned build under the working
    directory or its parents, then one in the checkout agent-vitals is installed from."""
    env = os.environ.get("VITALS_LLAMA_LIB")
    if env:
        p = Path(env)
        if not p.exists():
            raise FileNotFoundError(f"VITALS_LLAMA_LIB={env} does not exist")
        return p
    here = Path.cwd()
    for base in [here, *here.parents, Path(__file__).resolve().parents[2]]:
        for cand in _DEFAULT_CANDIDATES:
            p = base / cand
            if p.exists():
                return p
    raise FileNotFoundError(
        "libllama not found. Set VITALS_LLAMA_LIB or build the pinned engine with "
        f"scripts/build_engine.sh (tag {PINNED_TAG})."
    )


def load(path: str | Path | None = None) -> C.CDLL:
    """dlopen libllama. The pinned build links with $ORIGIN rpath, so sibling
    libggml*.so resolve from the same directory."""
    lib_path = Path(path) if path else find_libllama()
    return C.CDLL(str(lib_path), mode=C.RTLD_GLOBAL)


def doctor(path: str | Path | None = None) -> dict:
    """Health check for the engine binding: library found, symbols present."""
    report: dict = {"pinned_tag": PINNED_TAG, "pinned_commit": PINNED_COMMIT, "ok": False}
    try:
        lib_path = Path(path) if path else find_libllama()
        report["lib"] = str(lib_path)
        lib = load(lib_path)
    except (FileNotFoundError, OSError) as e:
        report["error"] = str(e)
        return report
    missing = [s for s in REQUIRED_SYMBOLS if not hasattr(lib, s)]
    report["symbols_checked"] = len(REQUIRED_SYMBOLS)
    report["symbols_missing"] = missing
    report["ok"] = not missing
    return report


# --------------------------------------------------------------------------
# structs — transcribed from vendor/llama.cpp @ b9910. Field ORDER is the
# contract; any header change requires re-transcription and a pin bump.
# --------------------------------------------------------------------------

llama_pos = C.c_int32
llama_token = C.c_int32
llama_seq_id = C.c_int32

# ggml/include/ggml.h @ b9910
GGML_MAX_DIMS = 4
GGML_MAX_SRC = 10
GGML_MAX_NAME = 64
GGML_MAX_OP_PARAMS = 64


class GGMLTensor(C.Structure):
    _fields_ = [
        ("type", C.c_int),
        ("buffer", C.c_void_p),
        ("ne", C.c_int64 * GGML_MAX_DIMS),
        ("nb", C.c_size_t * GGML_MAX_DIMS),
        ("op", C.c_int),
        ("op_params", C.c_int32 * (GGML_MAX_OP_PARAMS // 4)),
        ("flags", C.c_int32),
        ("src", C.c_void_p * GGML_MAX_SRC),
        ("view_src", C.c_void_p),
        ("view_offs", C.c_size_t),
        ("data", C.c_void_p),
        ("name", C.c_char * GGML_MAX_NAME),
        ("extra", C.c_void_p),
        ("padding", C.c_char * 8),
    ]


# ggml dtypes we expect for activations
_GGML_TYPE_TO_NP = {0: np.float32, 1: np.float16, 30: np.dtype("uint16")}  # 30 = BF16 (raw bits)

# bool (*)(struct ggml_tensor * t, bool ask, void * user_data)
EvalCallback = C.CFUNCTYPE(C.c_bool, C.POINTER(GGMLTensor), C.c_bool, C.c_void_p)


class LlamaBatch(C.Structure):
    _fields_ = [
        ("n_tokens", C.c_int32),
        ("token", C.POINTER(llama_token)),
        ("embd", C.POINTER(C.c_float)),
        ("pos", C.POINTER(llama_pos)),
        ("n_seq_id", C.POINTER(C.c_int32)),
        ("seq_id", C.POINTER(C.POINTER(llama_seq_id))),
        ("logits", C.POINTER(C.c_int8)),
    ]


class LlamaModelParams(C.Structure):
    """llama_model_params, passed through opaquely.

    Its layout changes between llama.cpp builds (b9910: use_mmap/use_direct_io/use_mlock; later masters:
    load_mode, then lazy_mode, load_mtp, growing it from 72 to 80 bytes). Only n_gpu_layers is ever set,
    and it sits at offset 16 in every layout. The struct is declared oversized: on x86-64 a struct this
    large is returned through a caller-allocated buffer and passed by value as a stack copy, so a larger
    buffer is safe in both directions (the library writes and reads only its own size) and the library's
    defaults travel through byte for byte."""
    _fields_ = [
        ("devices", C.c_void_p),
        ("tensor_buft_overrides", C.c_void_p),
        ("n_gpu_layers", C.c_int32),
        ("split_mode", C.c_int),
        ("_rest", C.c_uint8 * 232),
    ]


class LlamaContextParams(C.Structure):
    _fields_ = [
        ("n_ctx", C.c_uint32),
        ("n_batch", C.c_uint32),
        ("n_ubatch", C.c_uint32),
        ("n_seq_max", C.c_uint32),
        ("n_rs_seq", C.c_uint32),
        ("n_outputs_max", C.c_uint32),
        ("n_threads", C.c_int32),
        ("n_threads_batch", C.c_int32),
        ("ctx_type", C.c_int),
        ("rope_scaling_type", C.c_int),
        ("pooling_type", C.c_int),
        ("attention_type", C.c_int),
        ("flash_attn_type", C.c_int),
        ("rope_freq_base", C.c_float),
        ("rope_freq_scale", C.c_float),
        ("yarn_ext_factor", C.c_float),
        ("yarn_attn_factor", C.c_float),
        ("yarn_beta_fast", C.c_float),
        ("yarn_beta_slow", C.c_float),
        ("yarn_orig_ctx", C.c_uint32),
        ("defrag_thold", C.c_float),
        ("cb_eval", EvalCallback),
        ("cb_eval_user_data", C.c_void_p),
        ("type_k", C.c_int),
        ("type_v", C.c_int),
        ("abort_callback", C.c_void_p),
        ("abort_callback_data", C.c_void_p),
        ("embeddings", C.c_bool),
        ("offload_kqv", C.c_bool),
        ("no_perf", C.c_bool),
        ("op_offload", C.c_bool),
        ("swa_full", C.c_bool),
        ("kv_unified", C.c_bool),
        ("samplers", C.c_void_p),
        ("n_samplers", C.c_size_t),
        ("ctx_other", C.c_void_p),
    ]


class LlamaSamplerChainParams(C.Structure):
    _fields_ = [("no_perf", C.c_bool)]


class LlamaContextParamsV2(C.Structure):
    """llama.cpp builds after b9910 (e.g. the Muse Glimmer master build) insert
    `n_outputs_max_per_seq` after `n_outputs_max`. Padding before `cb_eval`
    absorbs it, so the size and every field from `cb_eval` on are unchanged; the
    fields in between (n_threads ... defrag_thold, incl. flash_attn_type) move."""

    _fields_ = [*LlamaContextParams._fields_[:6], ("n_outputs_max_per_seq", C.c_uint32),
                *LlamaContextParams._fields_[6:]]


assert C.sizeof(LlamaContextParamsV2) == C.sizeof(LlamaContextParams)


def context_params_layout(lib: C.CDLL) -> type:
    """Pick the context-params struct the loaded library was built with.

    Both layouts have the same size, so the defaults can be fetched with either
    and reinterpreted. GGML_DEFAULT_N_THREADS initializes BOTH thread counts, so
    the right layout is the one where n_threads == n_threads_batch (> 0).
    $VITALS_LLAMA_CTX_ABI=v1|v2 overrides the detection."""
    forced = os.environ.get("VITALS_LLAMA_CTX_ABI", "").lower()
    if forced in ("v1", "v2"):
        return LlamaContextParams if forced == "v1" else LlamaContextParamsV2
    lib.llama_context_default_params.restype = LlamaContextParams
    raw = bytes(lib.llama_context_default_params())

    def plausible(cls: type) -> bool:
        p = cls.from_buffer_copy(raw)
        return 0 < p.n_threads == p.n_threads_batch and 0 < p.n_batch <= 1 << 20 and p.flash_attn_type in (-1, 0, 1)

    if plausible(LlamaContextParamsV2) and not plausible(LlamaContextParams):
        return LlamaContextParamsV2
    return LlamaContextParams


class LlamaChatMessage(C.Structure):
    _fields_ = [("role", C.c_char_p), ("content", C.c_char_p)]


FLASH_ATTN_AUTO = -1
FLASH_ATTN_DISABLED = 0
FLASH_ATTN_ENABLED = 1


def _bind(lib: C.CDLL) -> None:
    """Set restype/argtypes on the pinned API surface (idempotent)."""
    lib.llama_backend_init.restype = None
    lib.llama_model_default_params.restype = LlamaModelParams
    lib.llama_model_load_from_file.restype = C.c_void_p
    lib.llama_model_load_from_file.argtypes = [C.c_char_p, LlamaModelParams]
    lib.llama_model_free.restype = None
    lib.llama_model_free.argtypes = [C.c_void_p]
    ctx_params = context_params_layout(lib)
    lib.vitals_ctx_params = ctx_params
    if ctx_params is LlamaContextParamsV2 and os.environ.get("VITALS_ALLOW_CUDA_GRAPHS") != "1":
        # Newer builds (e.g. the Muse Glimmer master build) fault with "CUDA error: an illegal
        # instruction" when cb_eval splits the graph at kq_soft_max past ~5k positions;
        # without CUDA graphs the same decode completes.
        # ggml-cuda reads this once per process, before the first graph compute.
        os.environ.setdefault("GGML_CUDA_DISABLE_GRAPHS", "1")
    lib.llama_context_default_params.restype = ctx_params
    lib.llama_init_from_model.restype = C.c_void_p
    lib.llama_init_from_model.argtypes = [C.c_void_p, ctx_params]
    lib.llama_free.restype = None
    lib.llama_free.argtypes = [C.c_void_p]

    lib.llama_model_get_vocab.restype = C.c_void_p
    lib.llama_model_get_vocab.argtypes = [C.c_void_p]
    lib.llama_vocab_n_tokens.restype = C.c_int32
    lib.llama_vocab_n_tokens.argtypes = [C.c_void_p]
    lib.llama_vocab_is_eog.restype = C.c_bool
    lib.llama_vocab_is_eog.argtypes = [C.c_void_p, llama_token]
    lib.llama_model_n_layer.restype = C.c_int32
    lib.llama_model_n_layer.argtypes = [C.c_void_p]
    lib.llama_model_n_embd.restype = C.c_int32
    lib.llama_model_n_embd.argtypes = [C.c_void_p]

    lib.llama_tokenize.restype = C.c_int32
    lib.llama_tokenize.argtypes = [
        C.c_void_p, C.c_char_p, C.c_int32, C.POINTER(llama_token), C.c_int32, C.c_bool, C.c_bool,
    ]
    lib.llama_token_to_piece.restype = C.c_int32
    lib.llama_token_to_piece.argtypes = [
        C.c_void_p, llama_token, C.c_char_p, C.c_int32, C.c_int32, C.c_bool,
    ]

    lib.llama_decode.restype = C.c_int32
    lib.llama_decode.argtypes = [C.c_void_p, LlamaBatch]
    lib.llama_get_logits_ith.restype = C.POINTER(C.c_float)
    lib.llama_get_logits_ith.argtypes = [C.c_void_p, C.c_int32]
    lib.llama_get_logits.restype = C.POINTER(C.c_float)
    lib.llama_get_logits.argtypes = [C.c_void_p]
    lib.llama_synchronize.restype = None
    lib.llama_synchronize.argtypes = [C.c_void_p]

    lib.llama_get_memory.restype = C.c_void_p
    lib.llama_get_memory.argtypes = [C.c_void_p]
    lib.llama_memory_seq_cp.restype = None
    lib.llama_memory_seq_cp.argtypes = [C.c_void_p, llama_seq_id, llama_seq_id, llama_pos, llama_pos]
    lib.llama_memory_seq_rm.restype = C.c_bool
    lib.llama_memory_seq_rm.argtypes = [C.c_void_p, llama_seq_id, llama_pos, llama_pos]

    # per-sequence state (KV cache + recurrent state) as an opaque byte blob
    lib.llama_state_seq_get_size.restype = C.c_size_t
    lib.llama_state_seq_get_size.argtypes = [C.c_void_p, llama_seq_id]
    lib.llama_state_seq_get_data.restype = C.c_size_t
    lib.llama_state_seq_get_data.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, llama_seq_id]
    lib.llama_state_seq_set_data.restype = C.c_size_t
    lib.llama_state_seq_set_data.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, llama_seq_id]

    # sampler chain (native sampling semantics)
    lib.llama_sampler_chain_default_params.restype = LlamaSamplerChainParams
    lib.llama_sampler_chain_init.restype = C.c_void_p
    lib.llama_sampler_chain_init.argtypes = [LlamaSamplerChainParams]
    lib.llama_sampler_chain_add.restype = None
    lib.llama_sampler_chain_add.argtypes = [C.c_void_p, C.c_void_p]
    for name, argtypes in [
        ("llama_sampler_init_greedy", []),
        ("llama_sampler_init_dist", [C.c_uint32]),
        ("llama_sampler_init_top_k", [C.c_int32]),
        ("llama_sampler_init_top_p", [C.c_float, C.c_size_t]),
        ("llama_sampler_init_min_p", [C.c_float, C.c_size_t]),
        ("llama_sampler_init_temp", [C.c_float]),
    ]:
        fn = getattr(lib, name)
        fn.restype = C.c_void_p
        fn.argtypes = argtypes
    lib.llama_sampler_sample.restype = llama_token
    lib.llama_sampler_sample.argtypes = [C.c_void_p, C.c_void_p, C.c_int32]
    lib.llama_sampler_reset.restype = None
    lib.llama_sampler_reset.argtypes = [C.c_void_p]
    lib.llama_sampler_free.restype = None
    lib.llama_sampler_free.argtypes = [C.c_void_p]

    # chat template + metadata
    lib.llama_model_chat_template.restype = C.c_char_p
    lib.llama_model_chat_template.argtypes = [C.c_void_p, C.c_char_p]
    lib.llama_chat_apply_template.restype = C.c_int32
    lib.llama_chat_apply_template.argtypes = [
        C.c_char_p, C.POINTER(LlamaChatMessage), C.c_size_t, C.c_bool, C.c_char_p, C.c_int32,
    ]
    lib.llama_model_meta_val_str.restype = C.c_int32
    lib.llama_model_meta_val_str.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p, C.c_size_t]


def _load_ggml(lib_dir: Path) -> C.CDLL:
    """libggml-base from libllama's directory (shares its RTLD_GLOBAL namespace)."""
    for name in ("libggml-base.so", "libggml-base.dylib"):
        p = lib_dir / name
        if p.exists():
            g = C.CDLL(str(p), mode=C.RTLD_GLOBAL)
            g.ggml_backend_tensor_get.restype = None
            g.ggml_backend_tensor_get.argtypes = [C.POINTER(GGMLTensor), C.c_void_p, C.c_size_t, C.c_size_t]
            return g
    raise FileNotFoundError(f"libggml-base not found next to libllama in {lib_dir}")


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------

_CUDA_MEMCPY_D2D = 3  # cudaMemcpyKind: cudaMemcpyDeviceToDevice


class LlamaEngine:
    """Minimal decode engine over libllama: tokenize, prefill, step, logits,
    greedy/native sampling, seq branching, cb_eval capture.

    One instance = one model + one context. Not thread-safe. Sequence 0 is the
    main rollout; higher seq ids are probe branches (never written back).
    """

    _backend_initialized = False

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 8192,
        n_batch: int = 2048,
        n_seq_max: int = 4,
        n_gpu_layers: int = 999,
        flash_attn: int = FLASH_ATTN_AUTO,
        capture_names: set[str] | None = None,
        lib_path: str | Path | None = None,
        model_from: LlamaEngine | None = None,
    ):
        """``model_from`` shares that engine's loaded weights (a second context
        on the same model, e.g. a flash-attention prefill context beside a
        diagnostic one); the sharing engine must outlive this one, which then
        never frees the model."""
        if model_from is not None:
            self.lib, self.ggml, self.model = model_from.lib, model_from.ggml, model_from.model
        else:
            self.lib = load(lib_path)
            _bind(self.lib)
            self.ggml = _load_ggml((Path(lib_path) if lib_path else find_libllama()).parent)

        if not LlamaEngine._backend_initialized:
            self.lib.llama_backend_init()
            LlamaEngine._backend_initialized = True

        self._owns_model = model_from is None
        if self._owns_model:
            mp = self.lib.llama_model_default_params()
            mp.n_gpu_layers = n_gpu_layers
            self.model = self.lib.llama_model_load_from_file(str(model_path).encode(), mp)
        if not self.model:
            raise RuntimeError(f"failed to load model: {model_path}")
        self.vocab = self.lib.llama_model_get_vocab(self.model)
        self.n_vocab = self.lib.llama_vocab_n_tokens(self.vocab)
        self.n_layer = self.lib.llama_model_n_layer(self.model)
        self.n_embd = self.lib.llama_model_n_embd(self.model)

        # capture state (must outlive the context: ctypes callback + buffers)
        self._captured_parts: dict[str, list[np.ndarray]] = {}
        self._capture_names: set[str] = set(capture_names or ())
        self._capture_modes: dict[str, str] = {}  # name -> "full" | "last_row" | "head_mean" | "layer_mean" | "device"
        # "device" mode: name -> torch tensors (one per ubatch) copied on the GPU, see take_device_parts
        self._device_parts: dict[str, list] = {}
        self._device_errors: list[str] = []
        # "layer_mean" mode: (q, kv) -> {"acc": device (q, kv) sum of the pooled
        # rows over the captured layers, "n": layers added}; one entry per ubatch
        self._layer_acc: dict = {}
        self._cb = EvalCallback(self._on_eval)  # keep a reference or the trampoline is GC'd

        self._ctx_opts = dict(n_ctx=n_ctx, n_batch=n_batch, n_seq_max=n_seq_max, flash_attn=flash_attn)
        self.ctx = None
        self._create_context()

    def _create_context(self) -> None:
        cp = self.lib.llama_context_default_params()
        cp.n_ctx = self._ctx_opts["n_ctx"]
        cp.n_batch = self._ctx_opts["n_batch"]
        cp.n_ubatch = min(self._ctx_opts["n_batch"], 512)
        cp.n_seq_max = self._ctx_opts["n_seq_max"]
        cp.flash_attn_type = self._ctx_opts["flash_attn"]
        if self._capture_names:
            # capture-off contexts never register the callback: true zero-overhead path
            cp.cb_eval = self._cb
            cp.cb_eval_user_data = None
        self.ctx = self.lib.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise RuntimeError("failed to create llama context")
        self.mem = self.lib.llama_get_memory(self.ctx)
        self.n_past = 0

    @property
    def n_ctx_per_seq(self) -> int:
        """Effective token capacity per sequence: with the default non-unified KV
        cache, llama.cpp divides n_ctx across n_seq_max (verified empirically —
        exceeding this fails llama_decode with rc=1)."""
        return int(self._ctx_opts["n_ctx"]) // max(1, int(self._ctx_opts["n_seq_max"]))

    def reset_context(
        self,
        capture_names: set[str] | None = None,
        flash_attn: int | None = None,
        capture_modes: dict[str, str] | None = None,
    ) -> None:
        """Fresh context (empty KV/recurrent state) without reloading weights.
        Changing capture_names or flash_attn requires this — callback registration
        and FA mode are fixed at context creation. ``capture_modes`` maps node
        names to a copy mode: "full" (default), "last_row" (only the final query
        row per head; a full T×T prefill matrix is ~GBs at agent context lengths,
        its last row ~KBs), "head_mean" / "layer_mean" (pooled on the device, see
        take_layer_mean) or "device" (kept on the GPU, see take_device_parts)."""
        if capture_names is not None:
            self._capture_names = set(capture_names)
        if capture_modes is not None:
            self._capture_modes = dict(capture_modes)
        if flash_attn is not None:
            self._ctx_opts["flash_attn"] = flash_attn
        if self.ctx:
            self.lib.llama_free(self.ctx)
        self._clear_captures()
        self._create_context()

    def _clear_captures(self) -> None:
        self._captured_parts = {}
        self._layer_acc = {}
        self._device_parts = {}
        self._device_errors = []

    # -- capture callback ---------------------------------------------------

    def _on_eval(self, t_ptr, ask: bool, _ud) -> bool:
        t = t_ptr.contents
        name = t.name.decode(errors="replace")
        if ask:
            return name in self._capture_names
        # observe phase: copy device tensor to host
        np_dtype = _GGML_TYPE_TO_NP.get(t.type)
        if np_dtype is None:
            return True  # unsupported dtype: skip silently rather than abort the graph
        ne = [int(t.ne[i]) for i in range(GGML_MAX_DIMS)]

        mode = self._capture_modes.get(name)
        if mode == "device" and self._device_copy(t, name, np_dtype, ne):
            return True   # else no torch/CUDA: fall through to the host copy (same layout)
        if mode in ("head_mean", "layer_mean"):
            # Reduce over heads ON DEVICE. "head_mean" copies the (q, kv) mean to
            # the host (~24x less PCIe traffic than the per-head copy); "layer_mean"
            # also sums it over the captured layers on the device (take_layer_mean).
            # Falls back to the full copy below when torch/cudart are unavailable,
            # the tensor is not contiguous, or the one-time self-check disagrees.
            if self._gpu_head_mean(t_ptr, t, name, np_dtype, ne, accumulate=(mode == "layer_mean")):
                return True

        if mode == "last_row":
            # Copy only the final query row per head (ggml ne is innermost-first:
            # ne0=kv, ne1=q, ne2=heads). Strided per-head reads via nb[] offsets.
            # OVERWRITE across ubatches: ubatches run in order, so the last
            # ubatch's final row is the decode call's true final row.
            kv, q, heads = max(1, ne[0]), max(1, ne[1]), max(1, ne[2])
            itemsize = np.dtype(np_dtype).itemsize
            rows = np.empty((heads, kv), dtype=np_dtype)
            for h in range(heads):
                off = h * int(t.nb[2]) + (q - 1) * int(t.nb[1])
                self.ggml.ggml_backend_tensor_get(
                    t_ptr, rows[h].ctypes.data_as(C.c_void_p), off, kv * itemsize
                )
            self._captured_parts[name] = [rows]
            return True

        n_elems = int(np.prod([max(1, d) for d in ne]))
        buf = np.empty(n_elems, dtype=np_dtype)
        self.ggml.ggml_backend_tensor_get(
            t_ptr, buf.ctypes.data_as(C.c_void_p), 0, buf.nbytes
        )
        shape = tuple(int(d) for d in ne if d > 1) or (1,)
        # ggml ne is innermost-first; reverse for numpy row-major convention.
        # APPEND, never overwrite: one logical decode may span several ubatches
        # and chunks, each with its own graph evaluation — overwriting here
        # silently drops all but the last ubatch's rows once a prompt exceeds
        # n_ubatch.
        self._captured_parts.setdefault(name, []).append(buf.reshape(tuple(reversed(shape))))
        return True

    # -- device-side head pooling ---------------------------------------------

    _gpu_pool_state = None   # None = untried, False = unavailable/failed, dict = ready
    _gpu_pool_reason = None

    def _gpu_state(self):
        """torch + cudart handles for device-side copies, initialized once per process;
        None when unavailable (callers fall back to host copies)."""
        if self._gpu_pool_state is None:
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA not available to torch")
                rt = None
                for so in ("libcudart.so.12", "libcudart.so"):
                    try:
                        rt = C.CDLL(so)
                        break
                    except OSError:
                        continue
                if rt is None:
                    raise OSError("libcudart not found")
                rt.cudaMemcpy.restype = C.c_int
                LlamaEngine._gpu_pool_state = {"torch": torch, "rt": rt, "stage": None, "checked": False}
            except Exception as exc:
                LlamaEngine._gpu_pool_state = False
                LlamaEngine._gpu_pool_reason = f"init: {exc!r}"
        return self._gpu_pool_state or None

    def _device_copy(self, t, name, np_dtype, ne) -> bool:
        """"device" mode: copy the node's tensor into a new torch tensor on the GPU
        (shape = reversed ggml ne, i.e. tokens first). True when handled. A layout
        that a flat copy would scramble (non-contiguous view) or a non-f32 node is
        recorded as an error that take_device_parts raises -- never a silent
        wrong copy (exceptions cannot cross the ctypes callback)."""
        if np_dtype is not np.float32:
            self._device_errors.append(f"{name}: dtype {np_dtype}, device capture needs f32")
            return True
        nb = [int(t.nb[i]) for i in range(4)]
        stride = 4
        for i in range(4):
            if ne[i] > 1 and nb[i] != stride:
                self._device_errors.append(f"{name}: not contiguous (ne={ne}, nb={nb})")
                return True
            stride *= max(1, ne[i])
        if not t.data:
            self._device_errors.append(f"{name}: no data pointer")
            return True
        st = self._gpu_state()
        if st is None:
            return False
        torch = st["torch"]
        n = int(np.prod([max(1, d) for d in ne]))
        out = torch.empty(n, dtype=torch.float32, device="cuda")
        # ggml computed the node on its own stream; a D2D cudaMemcpy does not
        # synchronize the host, so fence on both sides before ggml reuses the buffer
        torch.cuda.synchronize()
        rc = st["rt"].cudaMemcpy(C.c_void_p(out.data_ptr()), C.c_void_p(t.data), C.c_size_t(n * 4),
                                 C.c_int(_CUDA_MEMCPY_D2D))
        torch.cuda.synchronize()
        if rc != 0:
            self._device_errors.append(f"{name}: cudaMemcpy rc={rc}")
            return True
        self._device_parts.setdefault(name, []).append(out.view(*reversed([max(1, d) for d in ne])))
        return True

    def take_device_parts(self) -> dict:
        """The "device"-mode captures of the last decode() call: name -> list of
        torch tensors, one per ubatch graph, in evaluation order. Clears them.
        Raises if any capture failed (see _device_copy). Without torch/CUDA the
        nodes were copied to the host instead and are returned as CPU tensors."""
        if self._device_errors:
            errs, self._device_errors = self._device_errors, []
            raise RuntimeError("device capture failed: " + "; ".join(sorted(set(errs))[:5]))
        out, self._device_parts = self._device_parts, {}
        for name, mode in self._capture_modes.items():
            if mode == "device" and name in self._captured_parts:
                import torch

                out.setdefault(name, []).extend(
                    torch.from_numpy(np.ascontiguousarray(p)) for p in self._captured_parts.pop(name))
        return out

    def _gpu_head_mean(self, t_ptr, t, name, np_dtype, ne, accumulate: bool = False) -> bool:
        """head-mean the attention tensor on the GPU; True when handled. With
        ``accumulate`` the (q, kv) result is summed into ``_layer_acc`` on the
        device instead of being copied to the host."""
        if self._gpu_pool_state is False:
            return False
        if np_dtype is not np.float32:
            LlamaEngine._gpu_pool_reason = f"dtype {np_dtype}"
            return False
        kv, q, heads = max(1, ne[0]), max(1, ne[1]), max(1, ne[2])
        nb = [int(t.nb[i]) for i in range(4)]
        itemsize = 4
        contiguous = (nb[0] == itemsize and nb[1] == kv * itemsize
                      and nb[2] == kv * q * itemsize)
        if not contiguous or not t.data:
            LlamaEngine._gpu_pool_reason = f"contig={contiguous} data={bool(t.data)} ne={ne} nb={nb}"
            return False
        st = self._gpu_state()
        if st is None:
            return False
        torch = st["torch"]
        n = heads * q * kv
        if st["stage"] is None or st["stage"].numel() < n:
            st["stage"] = torch.empty(n, dtype=torch.float32, device="cuda")
        # ggml computed this node on its own stream; tensor_get would sync, so do
        # the same before a raw D2D copy
        torch.cuda.synchronize()
        rc = st["rt"].cudaMemcpy(C.c_void_p(st["stage"].data_ptr()), C.c_void_p(t.data),
                                 C.c_size_t(n * itemsize), C.c_int(_CUDA_MEMCPY_D2D))
        if rc != 0:
            LlamaEngine._gpu_pool_state = False
            LlamaEngine._gpu_pool_reason = f"cudaMemcpy rc={rc}"
            return False
        wv = head_weights_for(name, heads)
        if wv is None:
            hm = st["stage"][:n].view(heads, q, kv).mean(dim=0)
        else:
            # per-head weights: a convex combination of the head rows
            wt = st.setdefault("wt", {}).get(name)
            if wt is None:
                wt = torch.as_tensor(wv / wv.sum(), dtype=torch.float32, device="cuda").view(heads, 1, 1)
                st["wt"][name] = wt
            hm = (st["stage"][:n].view(heads, q, kv) * wt).sum(dim=0)
        if not st["checked"]:
            # one-time self-check against the host-copy path
            got = hm.cpu().numpy()
            buf = np.empty(n, dtype=np.float32)
            self.ggml.ggml_backend_tensor_get(t_ptr, buf.ctypes.data_as(C.c_void_p), 0, buf.nbytes)
            ref = (buf.reshape(heads, q, kv).mean(axis=0) if wv is None
                   else np.tensordot(wv / wv.sum(), buf.reshape(heads, q, kv), axes=(0, 0)))
            if not np.allclose(got, ref, atol=1e-5):
                LlamaEngine._gpu_pool_state = False
                LlamaEngine._gpu_pool_reason = f"self-check max|d|={float(np.abs(got-ref).max()):.3e}"
                return False
            st["checked"] = True
        if accumulate:
            # a layer listed all-zero in the head-weights file leaves the layer
            # mean as well (vitals.replay.pooling.layer_head_mean drops it the same way)
            if not layer_dropped(name):
                slot = self._layer_acc.get((q, kv))
                if slot is None:
                    self._layer_acc[(q, kv)] = {"acc": hm, "n": 1}
                else:
                    slot["acc"] += hm
                    slot["n"] += 1
            # ggml may reuse the softmax buffer for the next node as soon as this
            # callback returns: the D2D copy and the reductions must be done
            torch.cuda.synchronize()
            return True
        out = hm.unsqueeze(0).cpu().numpy()          # (1, q, kv): a 1-head tensor
        self._captured_parts.setdefault(name, []).append(out)
        return True

    def take_layer_mean(self, n_kv: int | None = None):
        """The attention rows pooled over heads AND layers that the "layer_mean"
        capture mode accumulated on the device during the last decode call: a
        torch (q, kv) tensor sliced to ``n_kv`` (the used KV width, as
        CaptureSession.harvest slices), or None when nothing was accumulated.
        Parts from several ubatches are kv-padded with zeros (masked-future
        cells) and concatenated on q, like collect_captured. Clears the
        accumulator. Raises if the same decode also left host copies of
        layer_mean nodes (a mixed capture would average fewer layers)."""
        parts = self._layer_acc
        self._layer_acc = {}
        if not parts:
            return None
        names = {nm for nm, m in self._capture_modes.items() if m == "layer_mean"}
        if any(nm in self._captured_parts for nm in names):
            raise RuntimeError("layer_mean capture split between device and host copies")
        torch = self._gpu_pool_state["torch"]
        rows = [p["acc"] / p["n"] for p in parts.values()]
        if len(rows) > 1:
            kv_max = max(r.shape[1] for r in rows)
            rows = [torch.nn.functional.pad(r, (0, kv_max - r.shape[1])) for r in rows]
        out = rows[0] if len(rows) == 1 else torch.cat(rows, 0)
        return out[:, :n_kv] if n_kv is not None else out

    def collect_captured(self) -> dict[str, np.ndarray]:
        """Assembled capture for the last decode() call, clearing the accumulator.

        Multi-part assembly (one part per ubatch/chunk graph):
        - 2-D parts with equal last dim (residuals: [tokens, d]) concat on axis 0
        - 3-D parts with equal first dim (attention: [heads, q, kv] with growing
          kv per ubatch) are zero-padded on kv to the widest part, then concat on
          the query axis — zeros land on masked-future cells, matching softmax
          semantics
        - anything else keeps the LAST part (and that choice is explicit here)
        """
        out: dict[str, np.ndarray] = {}
        for name, parts in self._captured_parts.items():
            top = max(p.ndim for p in parts)
            if top >= 2 and len(parts) > 1:
                # a one-token ubatch loses its size-1 query axis in the host copy: put it back
                parts = [np.expand_dims(p, -2) if p.ndim == top - 1 else p for p in parts]
            if len(parts) == 1:
                out[name] = parts[0]
            elif all(p.ndim == 2 for p in parts) and len({p.shape[-1] for p in parts}) == 1:
                out[name] = np.concatenate(parts, axis=0)
            elif all(p.ndim == 3 for p in parts) and len({p.shape[0] for p in parts}) == 1:
                kv_max = max(p.shape[-1] for p in parts)
                padded = [
                    np.pad(p, ((0, 0), (0, 0), (0, kv_max - p.shape[-1]))) if p.shape[-1] < kv_max else p
                    for p in parts
                ]
                out[name] = np.concatenate(padded, axis=1)
            else:
                out[name] = parts[-1]
        self._captured_parts = {}
        return out

    def set_capture(self, names: set[str]) -> None:
        """Change the capture filter. Note: cb_eval registration is fixed at
        context creation; an engine built without capture_names has no callback
        (true zero-overhead path) and cannot enable capture later."""
        self._capture_names = set(names)

    # -- tokenizer ------------------------------------------------------------

    def tokenize(self, text: str, add_special: bool = True, parse_special: bool = True) -> list[int]:
        data = text.encode()
        n_max = len(data) + 64
        arr = (llama_token * n_max)()
        n = self.lib.llama_tokenize(self.vocab, data, len(data), arr, n_max, add_special, parse_special)
        if n < 0:
            raise RuntimeError(f"tokenize failed: need {-n} slots")
        return list(arr[:n])

    def token_bytes(self, token: int) -> bytes:
        """Lossless piece, including partial UTF-8 characters and special tokens."""
        buf = C.create_string_buffer(256)
        n = self.lib.llama_token_to_piece(self.vocab, int(token), buf, len(buf), 0, True)
        if n < 0:
            buf = C.create_string_buffer(-n)
            n = self.lib.llama_token_to_piece(self.vocab, int(token), buf, len(buf), 0, True)
        if n < 0:
            raise RuntimeError("could not obtain lossless token bytes")
        return buf.raw[:n]

    def detokenize(self, tokens: list[int]) -> str:
        out = []
        buf = C.create_string_buffer(256)
        for tok in tokens:
            n = self.lib.llama_token_to_piece(self.vocab, tok, buf, 256, 0, True)
            if n > 0:
                out.append(buf.raw[:n])
        return b"".join(out).decode(errors="replace")

    def is_eog(self, token: int) -> bool:
        return bool(self.lib.llama_vocab_is_eog(self.vocab, token))

    # -- metadata + chat template ----------------------------------------------

    def meta(self, key: str) -> str | None:
        buf = C.create_string_buffer(4096)
        n = self.lib.llama_model_meta_val_str(self.model, key.encode(), buf, 4096)
        if n < 0:
            return None
        if n >= 4096:  # value longer than the probe buffer (e.g. chat templates)
            buf = C.create_string_buffer(n + 1)
            n = self.lib.llama_model_meta_val_str(self.model, key.encode(), buf, n + 1)
        return buf.raw[:n].decode(errors="replace")

    @property
    def arch(self) -> str:
        return self.meta("general.architecture") or "unknown"

    def apply_chat_template(self, messages: list[dict], add_assistant: bool = True) -> str:
        """Render messages with the model's embedded chat template (native
        llama_chat_apply_template — same code path as llama-server)."""
        tmpl = self.lib.llama_model_chat_template(self.model, None)
        if not tmpl:
            raise RuntimeError("model has no embedded chat template; pass a raw prompt instead")
        keep = [(str(m.get("role", "user")).encode(), str(m.get("content", "")).encode()) for m in messages]
        arr = (LlamaChatMessage * len(keep))(*[LlamaChatMessage(role=r, content=c) for r, c in keep])
        size = sum(len(r) + len(c) for r, c in keep) + 1024 * (len(keep) + 2)
        buf = C.create_string_buffer(size)
        n = self.lib.llama_chat_apply_template(tmpl, arr, len(keep), add_assistant, buf, size)
        if n < 0:
            raise RuntimeError(f"llama_chat_apply_template failed (rc={n}) for this template")
        if n > size:
            buf = C.create_string_buffer(n + 1)
            n = self.lib.llama_chat_apply_template(tmpl, arr, len(keep), add_assistant, buf, n + 1)
        return buf.raw[:n].decode(errors="replace")

    # -- decode ----------------------------------------------------------------

    def _make_batch(self, tokens: list[int], pos0: int, seq_id: int, logits_all: bool) -> tuple:
        n = len(tokens)
        tok_arr = (llama_token * n)(*tokens)
        pos_arr = (llama_pos * n)(*range(pos0, pos0 + n))
        nseq_arr = (C.c_int32 * n)(*([1] * n))
        seq_val = (llama_seq_id * 1)(seq_id)
        seq_arr = (C.POINTER(llama_seq_id) * n)(*([seq_val] * n))
        logits_arr = (C.c_int8 * n)(*([1] * n if logits_all else [0] * (n - 1) + [1]))
        batch = LlamaBatch(
            n_tokens=n, token=tok_arr, embd=None, pos=pos_arr,
            n_seq_id=nseq_arr, seq_id=seq_arr, logits=logits_arr,
        )
        keepalive = (tok_arr, pos_arr, nseq_arr, seq_val, seq_arr, logits_arr)
        return batch, keepalive

    def decode(
        self, tokens: list[int], *, pos0: int | None = None, seq_id: int = 0, logits_all: bool = False
    ) -> None:
        """Feed tokens; advances self.n_past only for seq 0 with default pos.

        Prompts longer than n_batch are decoded in chunks. With logits_all the
        call must fit one chunk (each chunk overwrites the logit buffer);
        callers that need every row chunk themselves (see score_tokens).
        """
        if pos0 is None:
            pos0 = self.n_past
        self._clear_captures()  # capture accumulates within one logical decode
        n_batch = int(self._ctx_opts["n_batch"])
        if logits_all and len(tokens) > n_batch:
            raise NotImplementedError(
                f"logits_all over {len(tokens)} tokens exceeds n_batch={n_batch}; decode in chunks"
            )
        off = 0
        while off < len(tokens):
            chunk = tokens[off : off + n_batch]
            batch, _keep = self._make_batch(chunk, pos0 + off, seq_id, logits_all)
            rc = self.lib.llama_decode(self.ctx, batch)
            if rc != 0:
                raise RuntimeError(f"llama_decode failed rc={rc} at chunk offset {off}")
            off += len(chunk)
        if seq_id == 0 and pos0 == self.n_past:
            self.n_past += len(tokens)

    def logits(self, i: int = -1) -> np.ndarray:
        ptr = self.lib.llama_get_logits_ith(self.ctx, i)
        if not ptr:
            raise RuntimeError(f"no logits at output index {i}")
        return np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).copy()

    def logits_rows(self, n: int) -> np.ndarray:
        """The first ``n`` output rows of the last decode call as one (n, n_vocab)
        array: a zero-copy view of the engine's logit buffer, valid until the
        next decode. Row i is the row for the i-th token of that call when it
        was decoded with logits_all (the only case replay uses)."""
        ptr = self.lib.llama_get_logits(self.ctx)
        if not ptr:
            raise RuntimeError("no logits from the last decode")
        return np.ctypeslib.as_array(ptr, shape=(n, self.n_vocab))

    # -- convenience -----------------------------------------------------------

    def prefill(self, text: str, logits_all: bool = False) -> list[int]:
        toks = self.tokenize(text)
        self.decode(toks, logits_all=logits_all)
        return toks

    def score_tokens(self, tokens: list[int]) -> np.ndarray:
        """Teacher-forced logprobs of a token sequence: returns an array of
        length len(tokens)-1 where entry i is log p(tokens[i+1] | tokens[:i+1]).

        Works for prompts far beyond one batch (see _per_position). Scores from
        the engine's CURRENT state (call reset_context() first for a fresh
        score); advances n_past like any decode.
        """
        if len(tokens) < 2:
            return np.empty(0, dtype=np.float64)

        def logprob(row: np.ndarray, tok: int) -> float:
            lg = row.astype(np.float64)
            lg -= lg.max()
            return float(lg[tok] - np.log(np.exp(lg).sum()))

        return np.asarray(self._per_position(tokens, logprob), dtype=np.float64)

    def score_prompt(self, tokens: list[int]) -> np.ndarray:
        """One-pass prompt scoring while prefilling: for positions 1..n-1,
        the predictive distribution's entropy and top-2 margin plus the actual
        token's surprisal. Returns shape (n-1, 3): [ent, m2, surprisal].

        Advances state exactly like decode(tokens) -- this IS the prefill,
        with per-position logit reads (chunked at n_batch). The per-token cost
        is one vocab-sized softmax; ~seconds per multi-k prompt.
        """
        if len(tokens) < 2:
            return np.empty((0, 3), dtype=np.float64)

        def stats(row: np.ndarray, tok: int) -> tuple[float, float, float]:
            lg = row.astype(np.float64)
            lg -= lg.max()
            p = np.exp(lg)
            p /= p.sum()
            ent = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
            top2 = np.partition(p, -2)[-2:]
            return ent, float(top2[1] - top2[0]), float(-np.log(max(p[tok], 1e-12)))

        return np.asarray(self._per_position(tokens, stats), dtype=np.float64)

    def _per_position(self, tokens: list[int], fn) -> list:
        """Decode ``tokens`` in n_batch chunks with logits_all and return
        fn(logits at position i, tokens[i + 1]) for i in 0..n-2. Each chunk's
        rows are read before the next decode overwrites them."""
        n_batch = int(self._ctx_opts["n_batch"])
        out: list = []
        prev_last: np.ndarray | None = None
        off = 0
        while off < len(tokens):
            chunk = tokens[off : off + n_batch]
            self.decode(chunk, logits_all=True)
            if prev_last is not None:
                out.append(fn(prev_last, chunk[0]))
            for i in range(len(chunk) - 1):
                out.append(fn(self.logits(i), chunk[i + 1]))
            prev_last = self.logits(len(chunk) - 1)
            off += len(chunk)
        return out

    def greedy(self, n: int, on_step=None) -> list[int]:
        """Greedy continuation of seq 0 from current state."""
        out: list[int] = []
        for _ in range(n):
            tok = int(np.argmax(self.logits(-1)))
            if on_step is not None:
                on_step(tok, self)
            out.append(tok)
            if self.is_eog(tok):
                break
            self.decode([tok])
        return out

    # -- branching (probe contract: never written back) -------------------------

    def branch_copy(self, dst_seq: int, src_seq: int = 0) -> None:
        self.lib.llama_memory_seq_cp(self.mem, src_seq, dst_seq, -1, -1)

    def branch_drop(self, seq: int) -> None:
        self.lib.llama_memory_seq_rm(self.mem, seq, -1, -1)

    # -- sequence state blobs (KV cache + recurrent state) ------------------------

    def seq_state_get(self, seq: int = 0) -> np.ndarray:
        """The full state of one sequence as an opaque byte blob (llama_state_seq_get_data)."""
        size = int(self.lib.llama_state_seq_get_size(self.ctx, seq))
        buf = np.empty(size, dtype=np.uint8)
        n = int(self.lib.llama_state_seq_get_data(self.ctx, buf.ctypes.data_as(C.c_void_p), size, seq))
        if n != size:
            raise RuntimeError(f"llama_state_seq_get_data wrote {n} of {size} bytes")
        return buf

    def seq_state_set(self, blob: np.ndarray, seq: int = 0) -> None:
        """Restore a sequence from a blob produced by seq_state_get (possibly by
        another context on the same model); the sequence is cleared first."""
        blob = np.ascontiguousarray(blob, dtype=np.uint8)
        self.branch_drop(seq)
        n = int(self.lib.llama_state_seq_set_data(self.ctx, blob.ctypes.data_as(C.c_void_p), blob.nbytes, seq))
        if n == 0:
            raise RuntimeError("llama_state_seq_set_data rejected the blob")

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.lib.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "model", None):
            if getattr(self, "_owns_model", True):
                self.lib.llama_model_free(self.model)
            self.model = None
