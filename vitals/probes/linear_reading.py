"""Reading measures for linear-attention layers, alongside the attention replay.

Attention layers keep every token, so a section's reading is measured against its length (lift).
Linear-attention layers (Gated DeltaNet, Kimi Delta Attention) keep one fading memory instead, so a section
can be forgotten before it can be read. Per turn and section this module reports, over all linear layers:

  retained    how much of what the section wrote is still in memory when the turn starts generating:
              sqrt(|S_r| now / |S_r| right after its last token), energies summed over heads. 1 = all kept.
  mem_share   the section's share of the whole memory at that point: per layer <S_r, S> / |S|^2 (heads
              summed), averaged over the linear layers. Sums to 1.
  read        the section's share of the linear layers' output at the generated tokens: per layer
              sum_h <o_r, o> / sum_h |o|^2, averaged over the layers and the turn's steps. Sums to 1.
              (Equal weight per layer, like the attention-side measures.)
  retrieval   read / mem_share: whether the agent pulls out what is still there. It plays the role of
              attention lift, with the memory share as the baseline instead of the length share (for an
              attention layer, which forgets nothing, the two baselines coincide). Reported only when the
              memory share is at least ``floor``.

The split per section is exact (vitals.engine.linear_memory): the parts add back to the engine's output.

Sections partition the tokens: a declared span (``span:<name>`` region) takes precedence over the message
region it sits in (``region:<id>``); nested spans go to the innermost; the forced output is ``output``.
Each section needs one memory slot per linear layer. When the slots run out, the least recently written
message regions fold into ``region:older`` (exact, by linearity), then declared spans into ``older
sections``. ``group_sep`` maps span names to the part before the first separator, for declarations with
many small sections.

Lifecycle, mirroring the replay's prefix reuse: the resident memories follow the prefix kept on sequence 0;
each turn forks a scratch copy for its own tail and forced output, which is dropped afterwards.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from vitals.core.sections import OUTPUT, section_labels  # noqa: F401  (section_labels: re-exported)
from vitals.engine.linear_memory import (
    AdvanceResult,
    SectionMemory,
    advance,
    concat_inputs,
    fp32_exact,
    layer_inputs,
    segments,
)
from vitals.engine.nodes import LinearMixerSpec

OLDER_REGIONS = "region:older"
OLDER_SPANS = "older sections"


def _kind(name: str) -> str:
    if name == OUTPUT:
        return "output"
    if name in (OLDER_REGIONS, OLDER_SPANS):
        return "bucket"
    return "region" if name.startswith("region:") else "span"


class _Track:
    """Memories plus their slot registry and the energy each slot had right after its last write."""

    def __init__(self, spec: LinearMixerSpec, max_slots: int):
        self.mem = SectionMemory(spec, max_slots)
        self.names: list[str] = []
        self.ids: dict[str, int] = {}            # name (or folded alias) -> slot
        self.written = np.zeros(max_slots)
        self.last = np.zeros(max_slots)          # turn counter of the last write, for folding

    def copy_from(self, other: _Track) -> None:
        self.mem.copy_from(other.mem)
        self.names, self.ids = list(other.names), dict(other.ids)
        self.written, self.last = other.written.copy(), other.last.copy()

    def reset(self) -> None:
        self.mem.reset()
        self.names, self.ids = [], {}
        self.written[:] = 0.0
        self.last[:] = 0.0

    def slot(self, name: str, clock: float) -> int:
        if name in self.ids:
            return self.ids[name]
        if len(self.names) < self.mem.max_slots:
            r = len(self.names)
            self.names.append(name)
            self.mem.n_slots = len(self.names)
            self.ids[name] = r
            return r
        return self._fold_into(name, clock)

    def _fold_into(self, name: str, clock: float) -> int:
        """Free a slot for ``name`` by folding the least recently written message region (else declared
        span) into its bucket. The first fold of a kind turns the victim's slot into the bucket in place
        (its memory already is the bucket's content) and then folds again."""
        for kind, bucket in (("region", OLDER_REGIONS), ("span", OLDER_SPANS)):
            cand = [r for r, n in enumerate(self.names) if _kind(n) == kind]
            if cand:
                return self._fold(name, clock, min(cand, key=lambda r: self.last[r]), bucket)
        raise RuntimeError("linear reading: no slot left to fold; raise max_slots")

    def _fold(self, name: str, clock: float, victim: int, bucket: str) -> int:
        if bucket not in self.ids:
            self.names[victim] = bucket
            self.ids[bucket] = victim
            return self._fold_into(name, clock)
        b = self.ids[bucket]
        self.mem.merge(victim, b)
        self.written[b] += self.written[victim]
        self.last[b] = max(self.last[b], self.last[victim])
        for alias, r in list(self.ids.items()):
            if r == victim:
                self.ids[alias] = b
        self.names[victim] = name
        self.ids[name] = victim
        self.written[victim] = 0.0
        self.last[victim] = clock
        return victim


class LinearReading:
    """Per-section reading of a hybrid model's linear-attention layers during replay.

    Create one per backend (``capture_names``/``capture_modes`` go into the engine's capture plan), then
    per turn: ``begin_turn`` -> ``feed("resident", ...)`` for the prefix kept on seq 0 -> ``fork`` ->
    ``feed("scratch", ...)`` for the tail -> ``gen_start`` -> ``feed("scratch", ...)`` for the forced
    output -> ``end_turn``. ``reset`` whenever the engine context is reset."""

    def __init__(self, engine, spec: LinearMixerSpec, *, max_slots: int = 64, chunk: int = 64,
                 check: bool = False, group_sep: str | None = None, floor: float = 1e-3,
                 layers_per_group: int | None = None):
        self.engine, self.spec = engine, spec
        self.opts = {k: v for k, v in dict(max_slots=max_slots, chunk=chunk, check=check, group_sep=group_sep,
                                           floor=floor, layers_per_group=layers_per_group).items()
                     if v != LinearReading.__init__.__kwdefaults__[k]}
        self.max_slots, self.chunk, self.check, self.group_sep, self.floor = max_slots, chunk, check, group_sep, floor
        # per layer: the chunk's fade factors (+ their exponent) and ~3 per-section [R, H, c, d_v] buffers
        fade = 2 * spec.n_v_heads * chunk * chunk * (spec.d_k if spec.decay == "channel" else 1) * 4
        bufs = 3 * max_slots * spec.n_v_heads * chunk * spec.d_v * 4
        self.layers_per_group = layers_per_group or max(1, min(len(spec.layers), (2 << 30) // (fade + bufs)))
        self.resident = _Track(spec, max_slots)
        self.scratch: _Track | None = None
        self._clock = 0.0
        self._labels: list[str] = []
        self._shares: dict[int, np.ndarray] = {}
        self._snap: dict | None = None
        self._err = [0.0, 0.0]

    # -- engine plumbing
    @property
    def capture_names(self) -> set[str]:
        return self.spec.node_names(check=self.check)

    @property
    def capture_modes(self) -> dict[str, str]:
        return dict.fromkeys(self.capture_names, "device")

    # -- lifecycle
    def reset(self) -> None:
        self.resident.reset()
        self.scratch = None

    def section_labels(self, regions, n_prompt: int, n_out: int) -> list[str]:
        """Section name of every prompt and forced-output position (see ``section_labels``)."""
        return section_labels(regions, n_prompt, n_out, group_sep=self.group_sep)

    def begin_turn(self, regions, n_prompt: int, n_out: int) -> None:
        """Label this turn's positions and register every section on the resident registry (folding, if
        needed, happens here, before any of the turn's reading is accumulated)."""
        self._clock += 1.0
        self._labels = self.section_labels(regions, n_prompt, n_out)
        for name in dict.fromkeys(self._labels):
            self.resident.slot(name, self._clock)
        self._shares, self._snap, self._err = {}, None, [0.0, 0.0]

    def fork(self) -> None:
        if self.scratch is None:
            self.scratch = _Track(self.spec, self.max_slots)
        self.scratch.copy_from(self.resident)

    def feed(self, which: str, pos0: int, n: int, *, want: tuple[int, int] | None = None) -> None:
        """Consume the engine's linear-layer captures of the decode call that just ran over positions
        [pos0, pos0 + n) and advance the ``which`` ("resident" or "scratch") memories. ``want`` = absolute
        position range whose output shares are recorded."""
        trk = self.resident if which == "resident" else self.scratch
        parts = self.engine.take_device_parts()
        names = self._labels[pos0:pos0 + n]
        if len(names) != n:
            raise ValueError(f"linear reading: positions {pos0}..{pos0 + n} outside the turn's labels")
        ids = torch.tensor([trk.ids[x] for x in names], device=trk.mem.S.device)
        wmask = None
        if want is not None:
            pos = torch.arange(pos0, pos0 + n, device=ids.device)
            wmask = (pos >= want[0]) & (pos < want[1])
        segs = segments(ids)
        res = AdvanceResult()
        L, nv, R = self.spec.layers, self.spec.n_v_heads, trk.mem.n_slots
        num: dict[int, torch.Tensor] = {}      # position -> [n_layers, R] sum_h <o_r, o>
        den: dict[int, torch.Tensor] = {}      # position -> [n_layers]    sum_h |o|^2

        def per_layer(g0: int, lg: int):
            def reduce(o: torch.Tensor, f: torch.Tensor, positions: list[int]) -> None:
                o = o.reshape(R, lg, nv, *o.shape[2:])
                f = f.reshape(lg, nv, *f.shape[1:])
                nm = torch.einsum("rlhnd,lhnd->nlr", o, f)
                dn = (f * f).sum((1, 3)).T
                for i, p in enumerate(positions):
                    if p not in num:
                        num[p] = torch.zeros(len(L), R, device=o.device)
                        den[p] = torch.zeros(len(L), device=o.device)
                    num[p][g0:g0 + lg] += nm[i]
                    den[p][g0:g0 + lg] += dn[i]
            return reduce

        with fp32_exact(), torch.no_grad():
            for g0 in range(0, len(L), self.layers_per_group):
                group = L[g0:g0 + self.layers_per_group]
                x = concat_inputs([layer_inputs(self.spec, parts, i, check=self.check) for i in group])
                if x.q.shape[0] != n:
                    raise ValueError(f"linear reading: captured {x.q.shape[0]} tokens for a {n}-token decode")
                S = trk.mem.S[:, trk.mem.heads(g0, len(group))]
                advance(S, x, ids, R, self.spec.decay, want=wmask, chunk=self.chunk,
                        segs=segs, result=res, on_output=per_layer(g0, len(group)))
        del parts
        for (_, b), e in zip(segs, res.seg_energy, strict=True):
            r = int(ids[b - 1])
            trk.written[r] = e
            trk.last[r] = self._clock
        for p in num:      # per-layer shares, then the mean over layers (each layer's shares sum to 1)
            self._shares[pos0 + p] = (num[p] / den[p].clamp_min(1e-30)[:, None]).mean(0).cpu().numpy()
        self._err[0] += res.err
        self._err[1] += res.ref

    def gen_start(self) -> None:
        """Snapshot retained / memory share from the scratch memories, after the turn's prompt."""
        trk = self.scratch
        with torch.no_grad():
            energy = trk.mem.energy().cpu().numpy()
            dots, tot = trk.mem.cross()
            live = tot > 0
            share = (dots[:, live] / tot[live]).mean(1) if bool(live.any()) else torch.full((len(energy),), float("nan"))
        self._snap = {"names": list(trk.names), "energy": energy, "written": trk.written[:len(energy)].copy(),
                      "share": share.cpu().numpy()}

    def end_turn(self, turn: int) -> tuple[list[dict], list[tuple[int, list[float]]]]:
        """Rows (one per section) and per-step output shares for the turn; drops the scratch state."""
        snap = self._snap or {}
        names = list(self.scratch.names if self.scratch else self.resident.names)
        steps = sorted(self._shares)
        mat = np.zeros((len(steps), len(names)))
        for i, p in enumerate(steps):
            v = self._shares[p]
            mat[i, :len(v)] = v
        read = mat.mean(0) if len(steps) else np.full(len(names), np.nan)
        rel = math.sqrt(self._err[0] / self._err[1]) if self._err[1] > 0 else None
        rows = []
        for r, name in enumerate(names):
            if not name:
                continue
            energy = snap["energy"][r] if r < len(snap.get("energy", [])) else np.nan
            written = snap["written"][r] if r < len(snap.get("written", [])) else np.nan
            share = snap["share"][r] if r < len(snap.get("share", [])) else np.nan
            retained = math.sqrt(energy / written) if written and written > 0 and np.isfinite(energy) else np.nan
            retrieval = read[r] / share if np.isfinite(share) and share >= self.floor else np.nan
            rows.append({"turn": turn, "section": name, "kind": _kind(name), "slot": r,
                         "retained": float(retained), "mem_share": float(share), "read": float(read[r]),
                         "retrieval": float(retrieval), "energy": float(energy), "written_energy": float(written),
                         "n_steps": len(steps), "mixer": self.spec.kind, "n_layers": len(self.spec.layers),
                         "check_rel_err": rel})
        first = steps[0] if steps else 0
        step_rows = [(p - first, mat[i].astype(np.float32).tolist()) for i, p in enumerate(steps)]
        self.abort_turn()
        return rows, step_rows

    def abort_turn(self) -> None:
        self._shares, self._snap = {}, None


# --------------------------------------------------------------------------- store files
