"""Folder workspaces: a directory of spectra opened as a whole.

The folder is only browsed; a file that is opened becomes a normal
:class:`~ezspec.project.Dataset` (raw bytes kept, SHA-256 recorded). The work
done on a folder is stored in ``workspace.ezspec`` inside it – an ordinary
project file that is loaded again when the folder is reopened.

Files are read without a dialog: with the *import profile* remembered for
their extension (e.g. "column 3 is y, x is in nm" for every ``*.csv`` of the
folder) or, without one, with automatic detection. Profiles are stored in
the project (``Project.meta['import_profiles']``).
"""

from __future__ import annotations

import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from .io import find_reader, get_reader, supported_extensions
from .project import Dataset, Project

WORKSPACE_FILE = "workspace.ezspec"
PROFILES_KEY = "import_profiles"


def workspace_file(root) -> Path:
    return Path(root) / WORKSPACE_FILE


def natural_key(name: str) -> list:
    """Sort key that orders embedded numbers numerically: ``T2`` < ``T10``."""
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", str(name))]


def spectrum_files(root, recursive: bool = False) -> list:
    """Files of a folder that a registered reader claims by extension, in the order of the
    Explorer: subfolders first, names in natural order."""
    root = Path(root)
    exts = set(supported_extensions())
    it = root.rglob("*") if recursive else root.iterdir()
    files = [p for p in it if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")]

    def key(p):
        parts = p.relative_to(root).parts
        return [(0, natural_key(d)) for d in parts[:-1]] + [(1, natural_key(parts[-1]))]
    return sorted(files, key=key)


# ============================================================================ import profiles
def profile_key(path) -> str:
    """Profiles are kept per file extension (lower case, with dot; '' for none)."""
    return Path(path).suffix.lower()


def profiles_of(project: Project) -> dict:
    return project.meta.setdefault(PROFILES_KEY, {})


def make_profile(reader: str, options: dict) -> dict:
    """Import settings as a JSON-able profile; only options the reader understands are kept."""
    known = get_reader(reader).options
    return {"reader": reader, "options": {k: v for k, v in options.items() if k in known}}


def set_profile(project: Project, path, reader: str, options: dict) -> dict:
    profile = make_profile(reader, options)
    profiles_of(project)[profile_key(path)] = profile
    return profile


@dataclass
class Loaded:
    """Result of reading one file without a dialog."""
    path: Path
    datasets: list = field(default_factory=list)
    note: str = ""              # e.g. the stored import settings did not fit this file
    error: str = ""             # the file could not be read at all

    @property
    def ok(self) -> bool:
        return bool(self.datasets)


class SpectraCache:
    """Parsed files, keyed by path, modification time, size and import settings.

    Makes flipping through a folder instant on the second visit and notices
    files that changed on disk. Thread-safe; holds the last ``maxsize`` files."""

    def __init__(self, maxsize: int = 48):
        self.maxsize = maxsize
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(path: Path, profile: dict | None):
        st = path.stat()
        return (str(path), st.st_mtime_ns, st.st_size, json.dumps(profile, sort_keys=True, default=str))

    def get(self, key):
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        return None

    def put(self, key, value) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.maxsize:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


def _parse(path: Path, raw: bytes, profile: dict | None) -> tuple:
    """(spectra, note). A profile that does not fit the file falls back to automatic detection."""
    from .io import read_bytes
    if profile:
        reader = find_reader(path, raw)
        # a profile belongs to one format; a file of another format with the same extension is read as it is
        if reader.name == profile.get("reader"):
            try:
                return read_bytes(path, raw, **profile.get("options", {})), ""
            except Exception as exc:  # noqa: BLE001 - e.g. fewer columns than the profile expects
                specs = read_bytes(path, raw)
                return specs, (f"{path.name}: the import settings for *{profile_key(path)} do not fit "
                               f"({exc}); read with automatic detection instead")
    return read_bytes(path, raw), ""


def load_file(path, profiles: dict | None = None, cache: SpectraCache | None = None) -> Loaded:
    """Read a file into datasets without asking: stored profile, else automatic detection."""
    path = Path(path)
    profile = (profiles or {}).get(profile_key(path))
    try:
        key = SpectraCache.key(path, profile) if cache is not None else None
        hit = cache.get(key) if cache is not None else None
        if hit is None:
            raw = path.read_bytes()
            specs, note = _parse(path, raw, profile)
            hit = (specs, raw, note)
            if cache is not None:
                cache.put(key, hit)
        specs, raw, note = hit
    except Exception as exc:  # noqa: BLE001 - reported per file, the other files are still opened
        return Loaded(path, error=f"{path.name}: {exc}")
    # spectra are immutable and can be shared; datasets (pipeline, model) are created fresh every time
    datasets = [Dataset(name=s.meta.get("name", path.stem), raw=s, raw_bytes=raw, raw_ext=path.suffix.lower())
                for s in specs]
    return Loaded(path, datasets, note)


def load_files(paths, profiles: dict | None = None, cache: SpectraCache | None = None, progress=None) -> list:
    """:func:`load_file` for several files; ``progress(i, n)`` is called after each one."""
    paths = list(paths)
    out = []
    for i, p in enumerate(paths):
        out.append(load_file(p, profiles, cache))
        if progress is not None:
            progress(i + 1, len(paths))
    return out


def datasets_of_file(datasets, path) -> list:
    """The datasets among ``datasets`` that were read from the file ``path``."""
    target = str(Path(path).resolve())
    out = []
    for ds in datasets:
        src = ds.source.get("path")
        if src and str(Path(src).resolve()) == target:
            out.append(ds)
    return out
