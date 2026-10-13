"""Reading individual weight tensors from a (possibly split) GGUF, as torch tensors.

The write-share probe needs a few projection matrices per layer (output projections, MLA value
up-projections, per-head norms). They are read straight from the file with the ``gguf`` package,
dequantized when quantized, and kept on the device in bf16
(projections) or fp32 (small vectors). numpy shape = reversed ggml ``ne``, so a projection is
[n_out, n_in] and ``y = W @ x``.
"""

from __future__ import annotations

import re
from pathlib import Path

import gguf
import numpy as np
import torch


def shard_paths(path: str | Path) -> list[Path]:
    """All shards of a split GGUF given any one of them (or the single file)."""
    p = Path(path)
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", p.name)
    if not m:
        return [p]
    n = int(m.group(3))
    return [p.with_name(f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]


def default_device(device: str | torch.device | None = None) -> str | torch.device:
    """``device`` when given, else the GPU when torch sees one, else the CPU."""
    return device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")


class GGUFWeights:
    """Lazy tensor lookup over the shards of one model file."""

    def __init__(self, path: str | Path, device: str | torch.device | None = None):
        self.device = default_device(device)
        self._index: dict[str, object] = {}
        self._readers = []
        for shard in shard_paths(path):
            r = gguf.GGUFReader(str(shard))
            self._readers.append(r)
            for t in r.tensors:
                self._index[t.name] = t

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def get(self, name: str, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """The tensor ``name`` dequantized to ``dtype`` on the device."""
        t = self._index.get(name)
        if t is None:
            raise KeyError(f"tensor {name!r} not in the GGUF")
        kind = t.tensor_type.name
        raw = np.asarray(t.data)
        shape = tuple(int(x) for x in reversed(t.shape.tolist()))
        if kind == "F32":
            a = raw.astype(np.float32, copy=False)
        elif kind == "F16":
            a = raw.astype(np.float32)
        elif kind == "BF16":
            a = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            a = gguf.quants.dequantize(raw, t.tensor_type)
        a = np.array(a.reshape(shape), dtype=np.float32, copy=True)    # memory-mapped data is read-only
        return torch.from_numpy(a).to(device=self.device, dtype=dtype)

    def rows(self, name: str, idx: list[int], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Rows ``idx`` of the matrix ``name`` (e.g. a few unembedding rows) without loading the rest."""
        t = self._index.get(name)
        if t is None:
            raise KeyError(f"tensor {name!r} not in the GGUF")
        kind = t.tensor_type.name
        if kind not in ("F32", "F16", "BF16"):
            return self.get(name, dtype)[idx]
        n_cols = int(t.shape.tolist()[0])
        flat = np.asarray(t.data).reshape(-1)              # raw bytes or typed values, depending on the reader
        elem = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}[kind]
        if flat.dtype != elem:
            flat = flat.view(elem)
        raw = flat.reshape(-1, n_cols)[idx]
        if kind == "BF16":
            a = (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            a = raw.astype(np.float32)
        return torch.from_numpy(np.array(a, copy=True)).to(device=self.device, dtype=dtype)
