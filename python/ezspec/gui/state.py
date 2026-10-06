"""Application state, undoable commands and background workers.

All changes to pipelines, models and fit options go through snapshot
commands on a single QUndoStack, so every GUI edit is undoable and the
project always corresponds to a sequence of plain-JSON operations.
"""

from __future__ import annotations

import traceback

from PySide6 import QtCore
from PySide6.QtGui import QUndoCommand, QUndoStack

from ..fit import FitOptions
from ..models import Model
from ..pipeline import Pipeline
from ..project import Dataset, Project

_MERGE_IDS: dict = {}


def _merge_id(key) -> int:
    if key not in _MERGE_IDS:
        _MERGE_IDS[key] = len(_MERGE_IDS) + 1
    return _MERGE_IDS[key]


class SnapshotCommand(QUndoCommand):
    """Replace the pipeline / model / fit options of a dataset by a JSON snapshot."""

    def __init__(self, state, ds_id, kind, before, after, text, merge_key=None):
        super().__init__(text)
        self.state, self.ds_id, self.kind = state, ds_id, kind
        self.before, self.after = before, after
        self.merge_key = merge_key

    def id(self):
        return _merge_id((self.kind, self.ds_id, self.merge_key)) if self.merge_key is not None else -1

    def mergeWith(self, other):
        if not isinstance(other, SnapshotCommand) or other.merge_key != self.merge_key or \
                other.ds_id != self.ds_id or other.kind != self.kind:
            return False
        self.after = other.after
        return True

    def redo(self):
        self.state._apply(self.ds_id, self.kind, self.after)

    def undo(self):
        self.state._apply(self.ds_id, self.kind, self.before)


class DatasetCommand(QUndoCommand):
    def __init__(self, state, ds, add: bool, index=None):
        super().__init__(("Datensatz hinzufügen: " if add else "Datensatz entfernen: ") + ds.name)
        self.state, self.ds, self.add, self.index = state, ds, add, index

    def _insert(self):
        proj = self.state.project
        idx = len(proj.datasets) if self.index is None else self.index
        proj.datasets.insert(idx, self.ds)
        self.state.set_current(self.ds.id)
        self.state.projectChanged.emit()

    def _remove(self):
        proj = self.state.project
        self.index = proj.datasets.index(self.ds)
        proj.datasets.remove(self.ds)
        if self.state.current_id == self.ds.id:
            self.state.set_current(proj.datasets[0].id if proj.datasets else None)
        self.state.projectChanged.emit()

    def redo(self):
        self._insert() if self.add else self._remove()

    def undo(self):
        self._remove() if self.add else self._insert()


class RenameCommand(QUndoCommand):
    def __init__(self, state, ds, name):
        super().__init__(f"Umbenennen: {ds.name} → {name}")
        self.state, self.ds, self.old, self.new = state, ds, ds.name, name

    def redo(self):
        self.ds.name = self.new
        self.state.projectChanged.emit()

    def undo(self):
        self.ds.name = self.old
        self.state.projectChanged.emit()


def _structure(model_dict):
    """Model description without start values (start values do not invalidate a fit)."""
    out = []
    for c in model_dict.get("components", []):
        settings = {k: {kk: vv for kk, vv in v.items() if kk != "value"} for k, v in c.get("settings", {}).items()}
        out.append((c["kind"], c.get("prefix"), repr(sorted(c.get("options", {}).items(), key=str)),
                    repr(sorted(settings.items()))))
    return out


class AppState(QtCore.QObject):
    projectChanged = QtCore.Signal()
    currentChanged = QtCore.Signal()
    pipelineChanged = QtCore.Signal(str)
    modelChanged = QtCore.Signal(str)
    fitOptionsChanged = QtCore.Signal(str)
    fitChanged = QtCore.Signal(str)
    stepSelected = QtCore.Signal(object)
    busyChanged = QtCore.Signal(bool, str)
    message = QtCore.Signal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = Project()
        self.current_id: str | None = None
        self.selected_step: str | None = None
        self.undo = QUndoStack(self)
        self.variants: dict = {}            # ds_id -> {name: FitResult}
        self._runs: dict = {}               # ds_id -> PipelineRun

    # ------------------------------------------------------------------ access
    def current(self) -> Dataset | None:
        if self.current_id is None:
            return None
        try:
            return self.project.get(self.current_id)
        except KeyError:
            return None

    def set_current(self, ds_id):
        if ds_id != self.current_id:
            self.current_id = ds_id
            self.selected_step = None
            self.currentChanged.emit()

    def select_step(self, step_id):
        self.selected_step = step_id
        self.stepSelected.emit(step_id)

    def run(self, ds: Dataset | None = None):
        ds = ds or self.current()
        if ds is None:
            return None
        r = self._runs.get(ds.id)
        if r is None:
            r = ds.run_pipeline()
            self._runs[ds.id] = r
        return r

    def fit_current(self, ds) -> bool:
        """True if the dataset's fit result still corresponds to its data, model structure and options."""
        r = ds.fit_result if ds is not None else None
        if r is None:
            return False
        if _structure(r.model_spec) != _structure(ds.model.to_dict()):
            return False
        if r.options != ds.fit_options.to_dict():
            return False
        s = self.run(ds).final
        return r._internals.get("spectrum") is not None and \
            r._internals["spectrum"].content_hash() == s.content_hash() and \
            r._internals["spectrum"].flags == s.flags

    # ------------------------------------------------------------------ edits
    def _snapshot(self, ds, kind):
        if kind == "pipeline":
            return ds.pipeline.to_dict()
        if kind == "model":
            return ds.model.to_dict()
        return ds.fit_options.to_dict()

    def _apply(self, ds_id, kind, snap):
        ds = self.project.get(ds_id)
        if kind == "pipeline":
            new = Pipeline.from_dict(snap)
            ds.pipeline.steps = new.steps
            ds.pipeline._relink()
            self._runs.pop(ds_id, None)
            if self.selected_step and self.selected_step not in {s.id for s in ds.pipeline}:
                self.selected_step = None
            self.pipelineChanged.emit(ds_id)
        elif kind == "model":
            ds.model.components = Model.from_dict(snap).components
            self.modelChanged.emit(ds_id)
        else:
            ds.fit_options = FitOptions.from_dict(snap)
            self.fitOptionsChanged.emit(ds_id)

    def edit(self, kind: str, mutator, text: str, merge_key=None, ds: Dataset | None = None):
        """Apply ``mutator(obj)`` to a copy of the pipeline/model/options and push an undoable command."""
        ds = ds or self.current()
        if ds is None:
            return None
        before = self._snapshot(ds, kind)
        if kind == "pipeline":
            work = Pipeline.from_dict(before)
        elif kind == "model":
            work = Model.from_dict(before)
        else:
            work = FitOptions.from_dict(before)
        result = mutator(work)
        if kind == "model":
            work.check()
        after = work.to_dict()
        if after == before:
            return result
        self.undo.push(SnapshotCommand(self, ds.id, kind, before, after, text, merge_key))
        return result

    def add_dataset(self, ds: Dataset):
        self.undo.push(DatasetCommand(self, ds, True))

    def remove_dataset(self, ds: Dataset):
        self.undo.push(DatasetCommand(self, ds, False))

    def rename_dataset(self, ds: Dataset, name: str):
        if name and name != ds.name:
            self.undo.push(RenameCommand(self, ds, name))

    def new_project(self, project: Project | None = None):
        self.project = project or Project()
        self.undo.clear()
        self._runs.clear()
        self.variants.clear()
        self.current_id = None
        self.projectChanged.emit()
        self.set_current(self.project.datasets[0].id if self.project.datasets else None)
        self.currentChanged.emit()


# ============================================================================ workers
class _Signals(QtCore.QObject):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)


class Task(QtCore.QRunnable):
    """Run ``fn(progress=..., cancel=...)`` in the global thread pool."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = _Signals()
        self.cancelled = False
        self.setAutoDelete(True)

    def cancel(self):
        self.cancelled = True

    def run(self):
        try:
            out = self.fn(*self.args, **self.kwargs)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            tb = traceback.format_exc(limit=4)
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}\n\n{tb}")
            return
        self.signals.done.emit(out)


def start_task(fn, on_done, on_failed, on_progress=None, *args, **kwargs) -> Task:
    """Run ``fn(*args, **kwargs)`` in the thread pool. With ``on_progress`` the
    function additionally receives ``progress=callable(i, n)`` (thread-safe)."""
    task = Task(fn, *args, **kwargs)
    task.signals.done.connect(on_done)
    task.signals.failed.connect(on_failed)
    if on_progress is not None:
        task.signals.progress.connect(on_progress)
        task.kwargs["progress"] = task.signals.progress.emit
    QtCore.QThreadPool.globalInstance().start(task)
    return task
