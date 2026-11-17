"""Load a plugin update without restarting Krita.

The update replaced the files of the plugin. The running (old) version then drops all modules of
the package, imports the new ones and lets the new version take over. Krita keeps the objects it
was given at startup (the extension, the docker, the menu actions), so those are handed over:
the old extension forwards to the new one, the docker gets a new panel, the actions are
connected to the new functions.

`request()` runs in the old version, `take_over()` in the new one, so a version can always be
loaded by the one before it. Keep the hand-over dict compatible (only add keys).
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from .qt.QtCore import Qt, QTimer
from .qt.QtWidgets import QApplication
from .util import client_logger as log

_flag = "_diffusion_for_krita_reloading"
_package = __name__.rpartition(".")[0]
_dock_classes: list[type] = []  # docker classes of earlier versions (set by take_over)


class _Clock:
    """Logs how long each step of a reload takes (Krita does not redraw meanwhile)."""

    def __init__(self):
        import time

        self._time = time.perf_counter
        self._start = self._last = self._time()
        self._laps: list[str] = []

    def lap(self, step: str):
        now = self._time()
        self._laps.append(f"{step} {now - self._last:.2f}s")
        self._last = now

    def done(self):
        total = self._time() - self._start
        log.info(f"Plugin update: loaded in {total:.2f}s ({', '.join(self._laps)})")


def _busy():
    from .model.root import root

    return any(m.jobs.any_executing() for m in root.models)


def request(on_failure, wait_ms=2000):
    """Load the new version as soon as nothing is generating. `on_failure` is called (with the
    error) if it can't be done, then a restart of Krita is needed after all."""

    def attempt():
        if _busy():
            QTimer.singleShot(wait_ms, attempt)
            return
        # Krita can't redraw while the new version loads (a few seconds at most): show that
        # it is busy, and let "Loading the new version" appear first.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            reload_plugin()
        except Exception as e:
            log.exception(f"Could not load the plugin update without a restart: {e}")
            on_failure(e)
        finally:
            QApplication.restoreOverrideCursor()

    QTimer.singleShot(100, attempt)


def _plugin_modules():
    return {
        name: module
        for name, module in sys.modules.items()
        if name == _package or name.startswith(_package + ".")
    }


def reload_plugin():
    from .extension import AIToolsExtension
    from .ui.diffusion import ImageDiffusionWidget

    old_extension = AIToolsExtension.current
    # Krita makes dockers with the class it got at startup, so after earlier reloads there are
    # docks of several (older) classes.
    classes = [c for c in _dock_classes if c is not ImageDiffusionWidget] + [ImageDiffusionWidget]
    docks = [dock for cls in classes for dock in cls.alive()]
    hand_over: dict[str, Any] = {
        "extension": old_extension,
        "docks": docks,
        "dock_classes": classes,
        "actions": list(getattr(old_extension, "_actions", [])),
    }

    # Import the new version first: if that fails, the old one keeps running as it is.
    clock = _Clock()
    old_modules = _plugin_modules()
    for name in old_modules:
        del sys.modules[name]
    setattr(sys, _flag, True)
    try:
        new_reload = importlib.import_module(__name__)
        importlib.import_module(f"{_package}.extension")
        importlib.import_module(f"{_package}.ui.diffusion")
    except Exception:
        for name in _plugin_modules():
            del sys.modules[name]
        sys.modules.update(old_modules)
        raise
    finally:
        delattr(sys, _flag)
    clock.lap("import the new version")

    log.info("Plugin update: stopping the old version")
    for dock in docks:  # the panels of the old version must not outlive its event loop
        if shutdown := getattr(dock.widget(), "shutdown", None):
            shutdown()
    if old_extension is not None:
        old_extension.retire()
    clock.lap("stop the old version")
    new_reload.take_over(hand_over)
    clock.lap("start the new version")
    clock.done()


def take_over(hand_over: dict[str, Any]):
    """Runs in the new version: start it and give it what Krita holds."""
    from krita import Krita

    from . import __version__
    from .extension import AIToolsExtension
    from .ui.diffusion import DiffusionPanel

    clock = _Clock()
    extension = AIToolsExtension(Krita.instance())
    if old := hand_over.get("extension"):
        old._successor = extension
    extension.take_over_actions(hand_over.get("actions", []))
    clock.lap("settings and dialog")

    _dock_classes[:] = hand_over.get("dock_classes", [])
    for cls in _dock_classes:
        cls.panel_factory = DiffusionPanel
    for dock in hand_over.get("docks", []):
        dock.set_panel(DiffusionPanel())
    extension.setup()
    clock.lap("docker")
    for dock in hand_over.get("docks", []):
        if panel := dock.widget():
            panel.update_content()  # the open image, with its history
    clock.lap("image and history")
    log.info(f"Plugin update: version {__version__} is running")
    clock.done()
    return extension
