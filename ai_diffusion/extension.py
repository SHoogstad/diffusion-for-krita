import sys
from collections.abc import Callable
from pathlib import Path

from krita import DockWidgetFactory, DockWidgetFactoryBase, Extension, Krita, Window  # type: ignore

from . import __version__, eventloop
from .backend.server import ServerState
from .document import stop_polling
from .model.model import Workspace
from .model.root import root
from .qt.QtWidgets import QAction
from .settings import settings
from .ui import actions
from .ui.diffusion import ImageDiffusionWidget
from .ui.settings import SettingsDialog
from .util import client_logger as log

# Set while a plugin update is loaded without a restart (reload.py): Krita already has an
# extension and a docker factory then, the new extension is made by reload.take_over().
_reloading = getattr(sys, "_diffusion_for_krita_reloading", False)


class AIToolsExtension(Extension):
    """Krita keeps the first extension object for the whole session. After a plugin update was
    loaded without a restart, it hands everything to its `_successor` (the extension of the new
    version). Keep `current`, `_successor`, `_actions`, `retire()` and `take_over_actions()`
    stable."""

    current: "AIToolsExtension | None" = None

    def __init__(self, parent):
        super().__init__(parent)
        self._actions: list[tuple[str, QAction]] = []
        self._successor: Extension | None = None
        type(self).current = self

        log.info(f"Extension initialized, Version: {__version__}, Python: {sys.version}")

        extension_dir = Path(__file__).parent
        debugpy_path = extension_dir / "debugpy" / "src"
        if debugpy_path.exists() and not _reloading:  # it listens on a port, only once
            try:
                sys.path.insert(0, str(debugpy_path))
                import debugpy

                debugpy.listen(("127.0.0.1", 5678), in_process_debug_adapter=True)
                log.info("Developer mode: debugpy listening on port 5678")
            except ImportError:
                pass

        pykrita_dir = extension_dir.parent
        if pykrita_dir.name != "pykrita" and not (pykrita_dir / ".git").exists():
            log.warning(
                "Plugin is not installed in a 'pykrita' directory, this may break user files "
                f"and settings. Detected installation path is: {pykrita_dir}"
            )

        eventloop.setup()
        settings.load()
        root.init()
        self._settings_dialog = SettingsDialog(root.server)

        notifier = Krita.instance().notifier()
        notifier.setActive(True)
        notifier.applicationClosing.connect(self.shutdown)  # type: ignore

    def setup(self):
        eventloop.run(root.autostart(self._update_server_ui))

    def _update_server_ui(self):
        SettingsDialog.instance().connection.update_ui()  # the dialog is renewed on a look change

    def shutdown(self):
        root.server.terminate()
        root.connection.shutdown()
        eventloop.stop()

    def retire(self):
        """A new plugin version takes over: stop this one without ending anything the new
        version continues (a Runpod pod stays up, the new version connects to it again)."""
        import asyncio
        from contextlib import suppress

        with suppress(Exception):
            Krita.instance().notifier().applicationClosing.disconnect(self.shutdown)
        steps = [(root.connection.disconnect(stop_pod=False), "disconnect")]
        if root.server.state is ServerState.running:
            steps.append((root.server.stop(), "stop the server"))
        for task, what in steps:
            try:
                eventloop.run_until_complete(asyncio.wait_for(task, timeout=10))
            except Exception as e:
                log.warning(f"Plugin reload: could not {what}: {e}")
        root.server.terminate()
        root.connection._idle_timer.stop()
        stop_polling()
        for _name, action in self._actions:
            with suppress(Exception):  # nothing connected
                action.triggered.disconnect()
        if dialog := SettingsDialog._instance:
            dialog.hide()
            dialog.deleteLater()
        eventloop.stop(close=False)

    def _action_functions(self) -> dict[str, Callable]:
        return {
            "settings": lambda *_: SettingsDialog.instance().show(),
            "generate": actions.generate,
            "cancel": actions.cancel_active,
            "cancel_queued": actions.cancel_queued,
            "cancel_all": actions.cancel_all,
            "toggle_preview": actions.toggle_preview,
            "apply": actions.apply,
            "apply_alternative": actions.apply_alternative,
            "create_region": actions.create_region,
            "switch_workspace_generation": actions.set_workspace(Workspace.generation),
            "switch_workspace_upscaling": actions.set_workspace(Workspace.upscaling),
            "switch_workspace_live": actions.set_workspace(Workspace.live),
            "switch_workspace_graph": actions.set_workspace(Workspace.custom),
            "toggle_workspace": actions.toggle_workspace,
            "toggle_edit_mode": actions.toggle_edit_mode,
        }

    def take_over_actions(self, existing: list[tuple[str, QAction]]):
        """Connect the actions Krita already has (made by the previous version) to this one."""
        functions = self._action_functions()
        for name, action in existing:
            if func := functions.get(name):
                action.triggered.connect(func)
                self._actions.append((name, action))

    def createActions(self, window: Window):
        if self._successor is not None:  # a newer version was loaded, Krita still calls us
            return self._successor.createActions(window)
        for name, func in self._action_functions().items():
            action = window.createAction(f"ai_diffusion_{name}", "", "")
            action.triggered.connect(func)
            self._actions.append((name, action))


if not _reloading:
    Krita.instance().addExtension(AIToolsExtension(Krita.instance()))
    Krita.instance().addDockWidgetFactory(
        DockWidgetFactory(
            "imageDiffusion",
            DockWidgetFactoryBase.DockPosition.DockRight,  # type: ignore
            ImageDiffusionWidget,
        )
    )
