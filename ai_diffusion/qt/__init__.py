"""Qt binding used by the plugin: PyQt6 on Krita 6, PyQt5 on Krita 5.

The binding must match the one Krita itself was built with, which Krita has already imported by
the time plugins load. Outside of Krita (tests, scripts) PyQt6 is preferred if installed.
Set DIFFUSION_QT=5 or DIFFUSION_QT=6 to force a binding.
"""

import importlib
import os
import sys


def _choose_binding() -> str:
    forced = os.environ.get("DIFFUSION_QT", "")
    if forced in ("5", "6"):
        return f"PyQt{forced}"
    for name in ("PyQt6", "PyQt5"):
        if name in sys.modules:
            return name
    for name in ("PyQt6", "PyQt5"):
        try:
            importlib.import_module(f"{name}.QtCore")
            return name
        except ImportError:
            pass
    raise ImportError("Diffusion for Krita requires PyQt6 (Krita 6) or PyQt5 (Krita 5)")


binding = _choose_binding()
QT6 = binding == "PyQt6"

sip = importlib.import_module(f"{binding}.sip")
