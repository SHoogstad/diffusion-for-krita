from . import QT6

if QT6:
    # Moved from QtWidgets to QtGui in Qt6, still importable from here like in Qt5
    from PyQt6.QtGui import QAction, QActionGroup, QShortcut  # noqa: F401
    from PyQt6.QtWidgets import *
else:
    from PyQt5.QtWidgets import *  # type: ignore
