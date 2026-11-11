"""Quick pick: type a few letters, press Enter.

Used for the command palette (every menu command of the program) and for
"Go to file" (the spectra of the opened folder and the open datasets).
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

MAX_SHOWN = 60


def fuzzy_score(query: str, text: str) -> float | None:
    """How well ``query`` matches ``text`` (higher is better), ``None`` if it does not.

    Every word of the query has to occur – as a substring or, failing that, as
    letters in the right order ("opfo" finds "Open Folder"). Matches at the
    start of a word and runs of adjacent letters count more."""
    text_l = text.casefold()
    total = 0.0
    for word in query.casefold().split():
        pos = text_l.find(word)
        if pos >= 0:
            total += 10.0 * len(word) + (6.0 if pos == 0 or not text_l[pos - 1].isalnum() else 0.0) - 0.01 * pos
            continue
        score, i, last, first = 0.0, 0, -2, None
        for ch in word:
            i = text_l.find(ch, i)
            if i < 0:
                return None
            first = i if first is None else first
            score += 1.0
            if i == last + 1:
                score += 2.0
            if i == 0 or not text_l[i - 1].isalnum():
                score += 3.0
            last = i
            i += 1
        if last - first + 1 > max(10, 4 * len(word)):
            return None                                 # letters scattered over the whole text: no match
        total += score
    return total - 0.001 * len(text)


def rank(query: str, items, limit: int = MAX_SHOWN) -> list:
    """Items ``(title, detail, …)`` matching the query, best first. Without a query: the first ones."""
    if not query.strip():
        return list(items[:limit])
    scored = []
    for k, item in enumerate(items):
        s = fuzzy_score(query, item[0])
        if s is None and item[1]:
            s = fuzzy_score(query, f"{item[0]} {item[1]}")
            s = None if s is None else s - 5.0          # a match in the detail counts less
        if s is not None:
            scored.append((-s, k, item))
    scored.sort(key=lambda t: t[:2])
    return [item for _s, _k, item in scored[:limit]]


def menu_commands(menu_bar) -> list:
    """All commands of a menu bar as ``(title, shortcut, action)``; the title is the menu path."""
    out = []

    def walk(menu, path):
        for action in menu.actions():
            if action.isSeparator():
                continue
            text = action.text().replace("&", "").rstrip("…").strip()
            if not text:
                continue
            sub = action.menu()
            if sub is not None:
                sub.aboutToShow.emit()                  # menus that are filled on demand (recent files)
                walk(sub, path + [text])
            elif action.isEnabled():
                out.append((" › ".join(path + [text]),
                            action.shortcut().toString(QtGui.QKeySequence.NativeText), action))
    for top in menu_bar.actions():
        if top.menu() is not None:
            walk(top.menu(), [top.text().replace("&", "")])
    return out


class QuickPick(QtWidgets.QDialog):
    """Popup with a search field and a list; ``items`` are ``(title, detail, callback)``."""

    def __init__(self, parent, placeholder: str, items):
        super().__init__(parent, QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint)
        self.items = list(items)
        self._shown = []
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        self.edit = QtWidgets.QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.textChanged.connect(self._filter)
        self.edit.installEventFilter(self)
        lay.addWidget(self.edit)
        self.list = QtWidgets.QListWidget()
        self.list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list.setUniformItemSizes(True)
        self.list.itemClicked.connect(lambda _it: self.accept_current())
        lay.addWidget(self.list)
        self.setStyleSheet("QuickPick { background: #fbfbfc; border: 1px solid #9aa3b2; }"
                           "QListWidget { border: none; background: transparent; }")
        width = min(640, max(380, parent.width() - 200)) if parent is not None else 560
        self.resize(width, 380)
        if parent is not None:                          # like VS Code: centred at the top of the window
            top = parent.mapToGlobal(QtCore.QPoint((parent.width() - width) // 2, 60))
            self.move(top)
        self._filter("")
        self.edit.setFocus()

    def _filter(self, text):
        self._shown = rank(text, self.items)
        self.list.clear()
        for title, detail, _cb in self._shown:
            it = QtWidgets.QListWidgetItem(title if not detail else f"{title}    {detail}")
            if detail:
                it.setToolTip(detail)
            self.list.addItem(it)
        if self._shown:
            self.list.setCurrentRow(0)

    def eventFilter(self, obj, ev):
        if obj is self.edit and ev.type() == QtCore.QEvent.KeyPress:
            key = ev.key()
            if key in (QtCore.Qt.Key_Down, QtCore.Qt.Key_Up, QtCore.Qt.Key_PageDown, QtCore.Qt.Key_PageUp):
                n = self.list.count()
                if n:
                    step = {QtCore.Qt.Key_Down: 1, QtCore.Qt.Key_Up: -1, QtCore.Qt.Key_PageDown: 8,
                            QtCore.Qt.Key_PageUp: -8}[key]
                    self.list.setCurrentRow(min(max(self.list.currentRow() + step, 0), n - 1))
                return True
            if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                self.accept_current()
                return True
        return super().eventFilter(obj, ev)

    def accept_current(self):
        row = self.list.currentRow()
        if 0 <= row < len(self._shown):
            callback = self._shown[row][2]
            self.accept()
            QtCore.QTimer.singleShot(0, callback)      # after the popup is gone (the command may open a dialog)
