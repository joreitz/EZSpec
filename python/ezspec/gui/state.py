"""Application state, undoable commands and background workers.

All changes to pipelines, models and fit options go through snapshot
commands on a single QUndoStack, so every GUI edit is undoable and the
project always corresponds to a sequence of plain-JSON operations.
"""

from __future__ import annotations

import copy
import traceback

from PySide6 import QtCore
from PySide6.QtGui import QUndoCommand, QUndoStack

from ..fit import FitOptions
from ..models import Model
from ..pipeline import Pipeline
from ..project import Dataset, Project
from ..view import VIEW_KEY, ViewState, next_color

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
    def __init__(self, state, ds, add: bool, index=None, text=None):
        super().__init__(text or ("Add dataset: " if add else "Remove dataset: ") + ds.name)
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


class AddDatasetsCommand(QUndoCommand):
    """Add several datasets at once (one undo step, one refresh of the panels)."""

    def __init__(self, state, datasets, text):
        super().__init__(text)
        self.state, self.datasets = state, list(datasets)

    def redo(self):
        self.state.project.datasets.extend(self.datasets)
        self.state.projectChanged.emit()
        self.state.set_current(self.datasets[0].id)

    def undo(self):
        proj = self.state.project
        ids = {d.id for d in self.datasets}
        proj.datasets[:] = [d for d in proj.datasets if d.id not in ids]
        for i in ids:
            self.state._runs.pop(i, None)
        self.state.projectChanged.emit()
        if self.state.current_id in ids:
            self.state.set_current(proj.datasets[-1].id if proj.datasets else None)


class RenameCommand(QUndoCommand):
    def __init__(self, state, ds, name):
        super().__init__(f"Rename: {ds.name} → {name}")
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
    modifiedChanged = QtCore.Signal()
    viewChanged = QtCore.Signal()               # which spectra are overlaid, colours, stacking, zoom regions
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
        # A file that is only being looked at (single click in the Explorer). It is not part of the
        # project and is replaced by the next preview; the first edit keeps it ("pins" it).
        self.preview: Dataset | None = None
        self.workspace_root = None          # Path of the opened folder, if any
        # How the datasets are looked at (overlays, colours, stacking). Kept in the project file,
        # but not part of the undo history: looking at data differently is not an edit of it.
        self.view = ViewState()
        self._touched = False               # changed outside the undo history (pages of the project)

    # ------------------------------------------------------------------ access
    def get(self, ds_id) -> Dataset:
        """Dataset by id – one of the project or the preview. KeyError if there is none."""
        if self.preview is not None and self.preview.id == ds_id:
            return self.preview
        return self.project.get(ds_id)

    def current(self) -> Dataset | None:
        if self.current_id is None:
            return None
        try:
            return self.get(self.current_id)
        except KeyError:
            return None

    def is_preview(self, ds) -> bool:
        return ds is not None and ds is self.preview

    # ------------------------------------------------------------------ unsaved changes
    @property
    def modified(self) -> bool:
        """True if there is something to save: an edit on the undo stack or a changed page."""
        return self._touched or not self.undo.isClean()

    def touch(self):
        """Something that is saved with the project, but not undoable, was changed."""
        if not self._touched:
            self._touched = True
            self.modifiedChanged.emit()

    def mark_saved(self):
        self._touched = False
        self.undo.setClean()
        self.modifiedChanged.emit()

    # ------------------------------------------------------------------ view
    def _view_changed(self):
        self.project.meta[VIEW_KEY] = self.view.to_dict()
        self.viewChanged.emit()

    def is_shown(self, ds_id) -> bool:
        return ds_id in self.view.shown

    def set_shown(self, ds_ids, on: bool = True, only: bool = False):
        """Keep datasets in the plot besides the active one (``only``: exactly these)."""
        ds_ids = [i for i in ds_ids if i is not None]
        before = list(self.view.shown)
        if only:
            self.view.shown = list(dict.fromkeys(ds_ids))
        elif on:
            self.view.shown = list(dict.fromkeys(self.view.shown + ds_ids))
        else:
            self.view.shown = [i for i in self.view.shown if i not in set(ds_ids)]
        if self.view.shown != before:
            self._view_changed()

    def shown_datasets(self) -> list:
        """Datasets in the plot, in drawing order: the ticked ones and the active one."""
        cur = self.current()
        out = [d for d in self.project.datasets if d.id in self.view.shown or d is cur]
        if cur is not None and cur is self.preview:
            out.append(cur)
        return out

    def color_of(self, ds) -> str:
        """Colour of a dataset; assigned once and then kept (a preview gets the next free one)."""
        color = self.view.colors.get(ds.id)
        if color is None:
            ids = {d.id for d in self.project.datasets}
            color = next_color(c for i, c in self.view.colors.items() if i in ids)
            if not self.is_preview(ds):
                self.view.colors[ds.id] = color
                self.project.meta[VIEW_KEY] = self.view.to_dict()
        return color

    def set_color(self, ds_ids, color: str):
        for i in ds_ids:
            self.view.colors[i] = color
        self._view_changed()

    def set_view(self, **options):
        """Change display options (``stacked``, ``gap``, ``normalize``, ``regions``)."""
        changed = False
        for key, value in options.items():
            if getattr(self.view, key) != value:
                setattr(self.view, key, value)
                changed = True
        if changed:
            self._view_changed()

    # ------------------------------------------------------------------ preview
    def set_preview(self, ds: Dataset | None):
        """Show ``ds`` without adding it to the project; replaces the previous preview."""
        old = self.preview
        if old is not None:
            self._runs.pop(old.id, None)
        self.preview = ds
        if ds is None:
            if old is not None and self.current_id == old.id:
                self.current_id = self.project.datasets[-1].id if self.project.datasets else None
                self.selected_step = None
        else:
            self.current_id = ds.id
            self.selected_step = None
        self.projectChanged.emit()
        self.currentChanged.emit()

    def pin_preview(self) -> Dataset | None:
        """Keep the preview: it becomes a dataset of the project (undoable)."""
        ds = self.preview
        if ds is None:
            return None
        self.preview = None
        self.undo.push(DatasetCommand(self, ds, True, text="Open: " + ds.name))
        return ds

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
        ds = self.get(ds_id)
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
        if self.is_preview(ds):             # working on a previewed file keeps it
            self.undo.beginMacro(text)
            self.pin_preview()
            self.undo.push(SnapshotCommand(self, ds.id, kind, before, after, text))
            self.undo.endMacro()
            return result
        self.undo.push(SnapshotCommand(self, ds.id, kind, before, after, text, merge_key))
        return result

    def add_dataset(self, ds: Dataset):
        self.undo.push(DatasetCommand(self, ds, True))

    def add_datasets(self, datasets, text: str):
        datasets = list(datasets)
        if datasets:
            self.undo.push(AddDatasetsCommand(self, datasets, text))

    def remove_datasets(self, datasets):
        """Remove several datasets as one undo step."""
        datasets = list(datasets)
        if any(self.is_preview(d) for d in datasets):
            self.set_preview(None)
        datasets = [d for d in datasets if d in self.project.datasets]
        if len(datasets) == 1:
            self.undo.push(DatasetCommand(self, datasets[0], False))
        elif datasets:
            self.undo.beginMacro(f"Remove {len(datasets)} datasets")
            for d in datasets:
                self.undo.push(DatasetCommand(self, d, False))
            self.undo.endMacro()

    def apply_setup(self, setup: dict, targets, model: bool = False) -> int:
        """Give ``targets`` the processing of a template (``model``: also its model and fit options)
        as one undo step. Returns the number of datasets that changed."""
        snaps = {"pipeline": setup["pipeline"]}
        if model:
            snaps["model"] = setup["model"]
            snaps["options"] = setup["fit_options"]
        commands = []
        for t in targets:
            if self.is_preview(t):
                continue
            for kind, snap in snaps.items():
                before = self._snapshot(t, kind)
                if before != snap:
                    commands.append(SnapshotCommand(self, t.id, kind, before, copy.deepcopy(snap),
                                                    f"{kind} of {t.name}"))
        if not commands:
            return 0
        n = len({c.ds_id for c in commands})
        what = "processing and model" if model else "processing"
        self.undo.beginMacro(f"Paste {what} of {setup.get('name', 'template')} to {n} dataset(s)")
        for c in commands:
            self.undo.push(c)
        self.undo.endMacro()
        return n

    def remove_dataset(self, ds: Dataset):
        if self.is_preview(ds):
            self.set_preview(None)
            return
        self.undo.push(DatasetCommand(self, ds, False))

    def rename_dataset(self, ds: Dataset, name: str):
        if name and name != ds.name:
            if self.is_preview(ds):
                self.pin_preview()
            self.undo.push(RenameCommand(self, ds, name))

    def new_project(self, project: Project | None = None):
        self.project = project or Project()
        self.undo.clear()
        self._runs.clear()
        self.variants.clear()
        self.preview = None
        self.current_id = None
        self._touched = False
        self.view = ViewState.from_dict(self.project.meta.get(VIEW_KEY))
        self.view.prune(d.id for d in self.project.datasets)
        self.projectChanged.emit()
        self.set_current(self.project.datasets[0].id if self.project.datasets else None)
        self.currentChanged.emit()
        self.viewChanged.emit()


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
