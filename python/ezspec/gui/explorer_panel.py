"""Explorer: the opened folder as a tree of spectrum files.

A single click (or the arrow keys) previews a file, a double click or Enter
opens it as a dataset. The tree follows the file system, so spectra written
by an instrument while the program is running appear by themselves.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..io import supported_extensions
from ..workspace import natural_key, spectrum_files

PROJECT_EXT = ".ezspec"


class _Proxy(QtCore.QSortFilterProxyModel):
    """Folders first, natural order (T2 before T10), text filter on file names, open files in bold."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pattern = ""
        self.open_paths: set = set()

    def set_pattern(self, text):
        if hasattr(self, "endFilterChange"):     # Qt ≥ 6.10; invalidateFilter() is deprecated there
            self.beginFilterChange()
            self.pattern = text.strip().casefold()
            self.endFilterChange()
        else:
            self.pattern = text.strip().casefold()
            self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        fs = self.sourceModel()
        idx = fs.index(row, 0, parent)
        return fs.isDir(idx) or self.matches(fs.fileName(idx))

    def matches(self, name: str) -> bool:
        """Substring match, or a glob if the pattern contains wildcards (case-insensitive)."""
        if not self.pattern:
            return True
        name = name.casefold()
        if any(ch in self.pattern for ch in "*?["):
            return fnmatch.fnmatchcase(name, self.pattern)
        return self.pattern in name

    def lessThan(self, left, right):
        fs = self.sourceModel()
        dl, dr = fs.isDir(left), fs.isDir(right)
        if dl != dr:
            return dl
        return natural_key(fs.fileName(left)) < natural_key(fs.fileName(right))

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.FontRole and index.column() == 0 and self.open_paths:
            if self.sourceModel().filePath(self.mapToSource(index)) in self.open_paths:
                font = QtGui.QFont()
                font.setBold(True)
                return font
        return super().data(index, role)


class _Tree(QtWidgets.QTreeView):
    returnPressed = QtCore.Signal()

    def keyPressEvent(self, ev):
        if ev.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self.returnPressed.emit()
            ev.accept()
            return
        super().keyPressEvent(ev)


class ExplorerPanel(QtWidgets.QWidget):
    openFolderRequested = QtCore.Signal()
    previewRequested = QtCore.Signal(str)        # look at a file without opening it
    openRequested = QtCore.Signal(list)          # open files as datasets (no dialog)
    overlayRequested = QtCore.Signal(list)       # open files and show them together in the plot
    importRequested = QtCore.Signal(str)         # open one file through the import dialog

    def __init__(self, parent=None):
        super().__init__(parent)
        self.root: Path | None = None
        self.stack = QtWidgets.QStackedLayout(self)

        # page 0: no folder
        empty = QtWidgets.QWidget()
        el = QtWidgets.QVBoxLayout(empty)
        el.setContentsMargins(10, 10, 10, 10)
        hint = QtWidgets.QLabel("No folder opened.\n\nOpen a folder to browse its spectra: a single click previews "
                                "a file, a double click (or Enter) opens it.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        el.addWidget(hint)
        btn = QtWidgets.QPushButton("Open Folder…")
        btn.setObjectName("primary")
        btn.clicked.connect(self.openFolderRequested.emit)
        el.addWidget(btn)
        el.addStretch()
        self.stack.addWidget(empty)

        # page 1: tree
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        head = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel()
        self.title.setStyleSheet("font-weight: 600;")
        self.title.setMinimumWidth(40)
        head.addWidget(self.title, 1)
        other = QtWidgets.QToolButton()
        other.setText("…")
        other.setToolTip("Open another folder")
        other.clicked.connect(self.openFolderRequested.emit)
        head.addWidget(other)
        lay.addLayout(head)
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("Filter files, e.g. 300K or *_up.csv")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._filter_changed)
        lay.addWidget(self.filter)

        self.fs = QtWidgets.QFileSystemModel(self)
        self.fs.setFilter(QtCore.QDir.AllDirs | QtCore.QDir.Files | QtCore.QDir.NoDotAndDotDot)
        self.fs.setNameFilterDisables(False)
        self.proxy = _Proxy(self)
        self.proxy.setSourceModel(self.fs)
        self.proxy.setDynamicSortFilter(True)
        self.proxy.sort(0)
        self.tree = _Tree()
        self.tree.setModel(self.proxy)
        self.tree.setHeaderHidden(True)
        for col in range(1, self.fs.columnCount()):
            self.tree.hideColumn(col)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tree.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tree.setUniformRowHeights(True)
        self.tree.setDragEnabled(True)               # onto the plot: open and overlay
        self.tree.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.tree.doubleClicked.connect(self._double_clicked)
        self.tree.returnPressed.connect(self._open_selection)
        self.tree.selectionModel().currentChanged.connect(lambda *_: self._preview_timer.start())
        self.tree.setToolTip("Click: preview · Double click / Enter: open · Ctrl/Shift+click: select several")
        lay.addWidget(self.tree, 1)
        self.stack.addWidget(page)

        # arrow-key browsing: only the file the selection rests on is read
        self._preview_timer = QtCore.QTimer(self, singleShot=True, interval=70)
        self._preview_timer.timeout.connect(self._emit_preview)

    # ------------------------------------------------------------------ root
    def set_root(self, root):
        """Show a folder (``None``: back to the empty page)."""
        self.root = Path(root) if root is not None else None
        if self.root is None:
            self.stack.setCurrentIndex(0)
            return
        self.fs.setNameFilters(["*" + e for e in supported_extensions()] + ["*" + PROJECT_EXT])
        self.title.setText(self.root.name or str(self.root))
        self.title.setToolTip(str(self.root))
        self.filter.clear()
        self._apply_root()
        self.stack.setCurrentIndex(1)

    def _apply_root(self):
        if self.root is not None:
            self.tree.setRootIndex(self.proxy.mapFromSource(self.fs.setRootPath(str(self.root))))

    def _filter_changed(self, text):
        self.proxy.set_pattern(text)
        self._apply_root()                  # re-filtering invalidates the root index

    def set_open_paths(self, paths):
        """Files that are open as datasets (drawn in bold)."""
        paths = {Path(p).as_posix() for p in paths}
        if paths != self.proxy.open_paths:
            self.proxy.open_paths = paths
            self.tree.viewport().update()

    # ------------------------------------------------------------------ selection
    def _path(self, index) -> Path | None:
        if not index.isValid():
            return None
        return Path(self.fs.filePath(self.proxy.mapToSource(index)))

    def selected_files(self) -> list:
        out = []
        for idx in self.tree.selectionModel().selectedRows(0):
            p = self._path(idx)
            if p is not None and p.is_file():
                out.append(p)
        return sorted(out, key=lambda p: natural_key(str(p)))

    def visible_files(self) -> list:
        """Files directly inside the root that pass the filter, in display order."""
        root = self.tree.rootIndex()
        paths = [self._path(self.proxy.index(r, 0, root)) for r in range(self.proxy.rowCount(root))]
        return [p for p in paths if p is not None and p.is_file()]

    def select_path(self, path) -> bool:
        """Make ``path`` the current item (as a click would); False if it is not shown."""
        idx = self.proxy.mapFromSource(self.fs.index(str(path)))
        if not idx.isValid():
            return False
        self.tree.setCurrentIndex(idx)
        self.tree.scrollTo(idx)
        return True

    def _emit_preview(self):
        files = self.selected_files()
        if len(files) == 1 and len(self.tree.selectionModel().selectedRows(0)) == 1 \
                and files[0].suffix.lower() != PROJECT_EXT:
            self.previewRequested.emit(str(files[0]))

    def _double_clicked(self, index):
        p = self._path(index)
        if p is not None and p.is_file():
            self._preview_timer.stop()
            self.openRequested.emit([str(p)])

    def _open_selection(self):
        files = self.selected_files()
        if files:
            self._preview_timer.stop()
            self.openRequested.emit([str(p) for p in files])
            return
        idx = self.tree.currentIndex()       # Enter on a folder: expand / collapse
        if idx.isValid():
            self.tree.setExpanded(idx, not self.tree.isExpanded(idx))

    # ------------------------------------------------------------------ context menu
    def _context_menu(self, pos):
        idx = self.tree.indexAt(pos)
        clicked = self._path(idx) if idx.isValid() else self.root
        if clicked is None:
            return
        files = self.selected_files()
        if clicked.is_file() and clicked not in files:
            files = [clicked]
        menu = QtWidgets.QMenu(self)
        if files:
            title = "Open" if len(files) == 1 else f"Open {len(files)} files"
            menu.addAction(title, lambda: self.openRequested.emit([str(p) for p in files]))
            spectra = [str(p) for p in files if p.suffix.lower() != PROJECT_EXT]
            if spectra:
                menu.addAction("Open and overlay in plot", lambda: self.overlayRequested.emit(spectra))
            if len(files) == 1 and files[0].suffix.lower() != PROJECT_EXT:
                menu.addAction("Open with import options…", lambda: self.importRequested.emit(str(files[0])))
            menu.addSeparator()
        folder = clicked if clicked.is_dir() else clicked.parent
        menu.addAction(f"Open all spectra in “{folder.name}”", lambda: self._open_folder_files(folder))
        menu.addSeparator()
        menu.addAction("Copy path", lambda: QtWidgets.QApplication.clipboard().setText(str(clicked)))
        menu.addAction("Show in file manager",
                       lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder))))
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _open_folder_files(self, folder):
        # respect the filter the user is looking through
        files = [p for p in spectrum_files(folder) if self.proxy.matches(p.name)]
        if files:
            self.openRequested.emit([str(p) for p in files])
