from . import QT6

if QT6:
    from PyQt6.QtGui import *
else:
    from PyQt5.QtGui import *  # type: ignore

    # Moved from QtWidgets to QtGui in Qt6
    from PyQt5.QtWidgets import QAction, QActionGroup, QShortcut  # type: ignore  # noqa: F401
