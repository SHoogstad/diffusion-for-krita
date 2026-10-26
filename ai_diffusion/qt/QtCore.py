from . import QT6

if QT6:
    from PyQt6.QtCore import *
else:
    from PyQt5.QtCore import *  # type: ignore
