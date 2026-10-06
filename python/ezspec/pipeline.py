"""Non-destructive processing pipeline with provenance.

A pipeline is an ordered list of steps; each step is a call of a registered,
versioned operation with JSON parameters. Running it on the raw spectrum
produces the output of every step; the raw data are never modified.
Results are cached by (input content, operation, version, parameters), so
changing a parameter only recomputes the steps from there on.

The stored node structure ``{id, op, op_version, params, inputs, created_at}``
is DAG-ready (``inputs`` names the predecessor); the current GUI uses linear
chains.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .ops.registry import OpCall, get_op, pop_warnings
from .spectrum import Spectrum


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


@dataclass
class Step:
    op: str
    params: dict
    op_version: int
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    enabled: bool = True
    created_at: str = field(default_factory=_now)
    inputs: list = field(default_factory=list)
    label: str = ""

    @classmethod
    def new(cls, op: str, params: dict | None = None, enabled: bool = True, label: str = "") -> "Step":
        spec = get_op(op)
        return cls(op=op, params=spec.validate(params or {}), op_version=spec.version,
                   enabled=enabled, label=label)

    @property
    def title(self) -> str:
        return self.label or get_op(self.op).title

    def call(self) -> OpCall:
        return OpCall(self.op, dict(self.params))

    def to_dict(self) -> dict:
        return {"id": self.id, "op": self.op, "op_version": self.op_version, "params": self.params,
                "enabled": self.enabled, "inputs": list(self.inputs), "created_at": self.created_at,
                "label": self.label}

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        return cls(op=d["op"], params=dict(d["params"]), op_version=int(d["op_version"]),
                   id=d["id"], enabled=bool(d.get("enabled", True)), created_at=d.get("created_at", ""),
                   inputs=list(d.get("inputs", [])), label=d.get("label", ""))


@dataclass
class StepResult:
    step: Step
    input: Spectrum | None
    output: Spectrum | None
    warnings: list = field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0
    cached: bool = False


@dataclass
class PipelineRun:
    raw: Spectrum
    results: list

    @property
    def final(self) -> Spectrum:
        """Output of the last step that ran successfully (raw if none)."""
        out = self.raw
        for r in self.results:
            if r.error is not None:
                break
            if r.output is not None:
                out = r.output
        return out

    @property
    def ok(self) -> bool:
        return all(r.error is None for r in self.results)

    @property
    def warnings(self) -> list:
        return [(r.step, w) for r in self.results for w in r.warnings]

    def result(self, step_id: str) -> StepResult:
        for r in self.results:
            if r.step.id == step_id:
                return r
        raise KeyError(step_id)


def _spectrum_key(s: Spectrum) -> str:
    h = hashlib.sha256(s.content_hash().encode())
    h.update(repr(sorted(s.flags)).encode())
    for k in sorted(s.aux):
        h.update(k.encode())
        h.update(np.ascontiguousarray(s.aux[k]).tobytes())
    h.update(repr((s.x_unit, s.y_unit)).encode())
    return h.hexdigest()


class Pipeline:
    CACHE_SIZE = 64

    def __init__(self, steps: list | None = None):
        self.steps: list[Step] = list(steps or [])
        self._cache: OrderedDict = OrderedDict()

    # ---------------------------------------------------------------- editing
    def add(self, op: str, params: dict | None = None, index: int | None = None, **kw) -> Step:
        step = Step.new(op, params, **kw)
        self.insert(step, index)
        return step

    def insert(self, step: Step, index: int | None = None) -> None:
        if index is None:
            self.steps.append(step)
        else:
            self.steps.insert(index, step)
        self._relink()

    def remove(self, step_id: str) -> Step:
        i = self.index(step_id)
        step = self.steps.pop(i)
        self._relink()
        return step

    def move(self, step_id: str, new_index: int) -> None:
        step = self.steps.pop(self.index(step_id))
        self.steps.insert(new_index, step)
        self._relink()

    def update(self, step_id: str, params: dict) -> Step:
        step = self.get(step_id)
        merged = dict(step.params)
        merged.update(params)
        spec = get_op(step.op)
        step.params = spec.validate(merged)
        step.op_version = spec.version
        return step

    def set_enabled(self, step_id: str, enabled: bool) -> None:
        self.get(step_id).enabled = bool(enabled)

    def index(self, step_id: str) -> int:
        for i, s in enumerate(self.steps):
            if s.id == step_id:
                return i
        raise KeyError(step_id)

    def get(self, step_id: str) -> Step:
        return self.steps[self.index(step_id)]

    def _relink(self):
        prev = "raw"
        for s in self.steps:
            s.inputs = [prev]
            prev = s.id

    def __len__(self):
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    # ---------------------------------------------------------------- running
    def run(self, raw: Spectrum, upto: str | None = None) -> PipelineRun:
        """Run all steps (or up to and including ``upto``)."""
        results = []
        current = raw
        failed = False
        for step in self.steps:
            if failed:
                results.append(StepResult(step, None, None, error="vorheriger Schritt fehlgeschlagen"))
            elif not step.enabled:
                results.append(StepResult(step, current, current))
            else:
                r = self._run_step(step, current)
                results.append(r)
                if r.error is not None:
                    failed = True
                else:
                    current = r.output
            if upto is not None and step.id == upto:
                break
        return PipelineRun(raw, results)

    def _run_step(self, step: Step, s: Spectrum) -> StepResult:
        spec = get_op(step.op)
        warnings = []
        if step.op_version > spec.version:
            return StepResult(step, s, None, error=f"Operation {step.op} v{step.op_version} ist neuer "
                                                   f"als diese Programmversion (v{spec.version})")
        if step.op_version < spec.version:
            warnings.append(f"{step.op} wurde mit Version {step.op_version} erstellt, "
                            f"ausgeführt wird Version {spec.version}")
        key = hashlib.sha256((_spectrum_key(s) + step.op + str(spec.version)
                              + json.dumps(step.params, sort_keys=True)).encode()).hexdigest()
        if key in self._cache:
            out, w = self._cache[key]
            self._cache.move_to_end(key)
            return StepResult(step, s, out, warnings + w, cached=True)
        t0 = time.perf_counter()
        try:
            p = spec.validate(step.params)
            out = spec.func(s, **p)
            if spec.flags:
                out = out.with_flags(*spec.flags)
        except Exception as exc:  # noqa: BLE001 - reported to the user
            return StepResult(step, s, None, warnings, error=f"{type(exc).__name__}: {exc}",
                              seconds=time.perf_counter() - t0)
        out, w = pop_warnings(out)
        self._cache[key] = (out, w)
        if len(self._cache) > self.CACHE_SIZE:
            self._cache.popitem(last=False)
        return StepResult(step, s, out, warnings + w, seconds=time.perf_counter() - t0)

    # ---------------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        return {"format": "ezspec.pipeline", "version": 1, "steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, d: dict) -> "Pipeline":
        p = cls([Step.from_dict(s) for s in d.get("steps", [])])
        p._relink()
        return p

    def script_lines(self, var: str = "s") -> list:
        lines = []
        for st in self.steps:
            prefix = "" if st.enabled else "# (deaktiviert) "
            lines.append(prefix + st.call().code(var))
        return lines

    def copy(self) -> "Pipeline":
        return Pipeline.from_dict(json.loads(json.dumps(self.to_dict())))
