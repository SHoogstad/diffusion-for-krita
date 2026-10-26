from . import QT6

if QT6:
    from PyQt6.QtNetwork import *
else:
    from PyQt5.QtNetwork import *  # type: ignore
