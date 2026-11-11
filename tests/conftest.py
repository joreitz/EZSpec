"""Shared test configuration."""

import pytest


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path_factory):
    """Every test gets its own, empty settings file.

    The program stores window layout, recent folders and the folder of the last
    session with ``QSettings("EZSpec", "EZSpec")`` (native format). Without this
    redirection the tests would read and overwrite the settings of the person
    running them."""
    try:
        from PySide6 import QtCore
    except ImportError:
        yield
        return
    QtCore.QSettings.setPath(QtCore.QSettings.NativeFormat, QtCore.QSettings.UserScope,
                             str(tmp_path_factory.mktemp("settings")))
    yield
