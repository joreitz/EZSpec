"""Registry of file readers – one :class:`Reader` per file format.

Adding a format means writing one function and registering it; the file
dialogs, the Explorer, drag and drop and ``read_spectra`` pick it up from
here::

    from ezspec.io import Reader, register_reader, source_meta
    from ezspec.spectrum import Spectrum

    def read_myformat(path, raw, x_unit=None):
        x, y = decode(raw)
        meta = {"name": path.stem, "source": source_meta(path, raw, "myformat", {})}
        return [Spectrum(x, y, x_unit=x_unit or "nm", meta=meta)]

    register_reader(Reader("myformat", "My instrument", (".myf",), read_myformat,
                           detect=lambda path, raw: raw[:4] == b"MYF1", options=("x_unit",)))

Installed packages can ship readers without touching EZSpec: an entry point
in the group ``ezspec.readers`` that resolves to a :class:`Reader` (or to a
callable returning one or several) is registered on first use.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ENTRY_POINT_GROUP = "ezspec.readers"


@dataclass(frozen=True)
class Reader:
    """A file format.

    ``read(path, raw, **options)`` returns a list of spectra (``raw`` are the
    file's bytes, already read). ``detect(path, raw)`` recognises the format
    from the content, for files with a foreign or missing extension.
    ``options`` names the keyword arguments ``read`` understands; they are
    what an import profile stores. Among several matching readers the one
    with the highest ``priority`` wins."""
    name: str
    title: str
    extensions: tuple
    read: Callable
    detect: Callable | None = None
    options: tuple = ()
    priority: int = 0

    def matches(self, path: Path, raw: bytes | None = None) -> bool:
        if path.suffix.lower() in self.extensions:
            return True
        if self.detect is None or raw is None:
            return False
        try:
            return bool(self.detect(path, raw))
        except Exception:  # noqa: BLE001 - a detector must never break the import of other formats
            return False


_READERS: dict = {}
_FALLBACK = "text"
_plugins_loaded = False


def register_reader(reader: Reader, replace: bool = False) -> Reader:
    if reader.name in _READERS and not replace:
        raise ValueError(f"a reader named {reader.name!r} is already registered")
    exts = tuple(e.lower() if e.startswith(".") else "." + e.lower() for e in reader.extensions)
    reader = Reader(reader.name, reader.title, exts, reader.read, reader.detect, tuple(reader.options),
                    reader.priority)
    _READERS[reader.name] = reader
    return reader


def unregister_reader(name: str) -> None:
    _READERS.pop(name, None)


def load_plugins(force: bool = False) -> None:
    """Register the readers of installed packages (entry-point group ``ezspec.readers``)."""
    global _plugins_loaded
    if _plugins_loaded and not force:
        return
    _plugins_loaded = True
    from importlib.metadata import entry_points
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            obj = ep.load()
            if callable(obj) and not isinstance(obj, Reader):
                obj = obj()
            for reader in (obj if isinstance(obj, (list, tuple)) else [obj]):
                register_reader(reader, replace=True)
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not prevent starting
            warnings.warn(f"EZSpec reader plugin {ep.name!r} could not be loaded: {exc}", stacklevel=2)


def list_readers() -> list:
    """All readers, highest priority first."""
    load_plugins()
    return sorted(_READERS.values(), key=lambda r: -r.priority)


def get_reader(name: str) -> Reader:
    load_plugins()
    try:
        return _READERS[name]
    except KeyError:
        raise ValueError(f"unknown reader {name!r} (available: {', '.join(sorted(_READERS))})") from None


def find_reader(path, raw: bytes | None = None) -> Reader:
    """The reader for a file: by extension or content; plain text is the fallback."""
    path = Path(path)
    for reader in list_readers():
        if reader.name != _FALLBACK and reader.matches(path, raw):
            return reader
    return get_reader(_FALLBACK)


def supported_extensions() -> list:
    """Lower-case extensions (with dot) of all registered formats."""
    return sorted({e for r in list_readers() for e in r.extensions})


def is_supported(path) -> bool:
    """True if the file's extension belongs to a registered format (no file access)."""
    return Path(path).suffix.lower() in supported_extensions()


def file_dialog_filter() -> str:
    """Qt file-dialog filter: all spectra, then one entry per format."""
    readers = sorted(list_readers(), key=lambda r: (r.name != _FALLBACK, r.title.lower()))
    every = " ".join("*" + e for r in readers for e in r.extensions)
    parts = [f"Spectra ({every})"]
    parts += [f"{r.title} ({' '.join('*' + e for e in r.extensions)})" for r in readers]
    return ";;".join(parts + ["All files (*)"])


def source_meta(path, raw: bytes, reader: str, options: dict) -> dict:
    """Provenance record a reader stores as ``meta['source']`` of every spectrum."""
    path = Path(path)
    return {"filename": path.name, "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "reader": reader, "options": options}
