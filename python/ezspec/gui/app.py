"""Entry point: ``ezspec`` (or ``python -m ezspec``)."""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    try:
        from PySide6 import QtWidgets
    except ImportError:  # pragma: no cover
        print("Die GUI benötigt PySide6 und pyqtgraph:  pip install 'ezspec[gui]'", file=sys.stderr)
        return 1
    from .theme import apply_palette, configure_pyqtgraph
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setApplicationName("EZSpec")
    app.setOrganizationName("EZSpec")
    configure_pyqtgraph()
    apply_palette(app)
    from .main_window import MainWindow
    win = MainWindow()
    win.show()
    for arg in argv[1:]:
        if arg.endswith(".ezspec"):
            win.open_project(arg)
        else:
            win.import_data(arg)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
