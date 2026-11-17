"""Loading a plugin update without restarting Krita (ai_diffusion/reload.py).

The reload drops and re-imports the whole package, so it runs in a subprocess on a copy of the
plugin, never in the test process.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent

script = r"""
import importlib, os, sys
from pathlib import Path
plugin = Path(sys.argv[1])
sys.path.insert(0, str(plugin.parent))
import ai_diffusion  # adds tests/mock (next to the plugin) for the krita module

from PyQt6.QtWidgets import QApplication  # noqa
app = QApplication([])
errors = []
import traceback
sys.excepthook = lambda *a: (errors.append(a), traceback.print_exception(*a))

import krita
import ai_diffusion.extension as ext_module
from ai_diffusion.ui.diffusion import ImageDiffusionWidget
from ai_diffusion.ui.settings import SettingsDialog
import ai_diffusion.eventloop as old_loop

def pump():
    for _ in range(30):
        app.processEvents()

first = ext_module.AIToolsExtension.current
window = krita.Window()
first.createActions(window)
actions = dict(first._actions)
dock = ImageDiffusionWidget()
old_panel = dock.widget()
pump()

def update(version):
    init = plugin / "__init__.py"
    text = init.read_text()
    old = text.split('__version__ = "')[1].split('"')[0]
    init.write_text(text.replace(f'__version__ = "{old}"', f'__version__ = "{version}"'))
    importlib.import_module("ai_diffusion.reload").reload_plugin()
    pump()

update("9.9.9")
import ai_diffusion
assert ai_diffusion.__version__ == "9.9.9", ai_diffusion.__version__
from ai_diffusion.extension import AIToolsExtension
from ai_diffusion.ui.diffusion import DiffusionPanel
from ai_diffusion.ui.settings import SettingsDialog as NewDialog
second = AIToolsExtension.current
assert second is not first and first._successor is second
assert type(dock.widget()) is DiffusionPanel, type(dock.widget())  # the new version's panel
assert dock.widget() is not old_panel
assert old_loop._timer is None  # the old event loop is stopped
assert sys.modules["ai_diffusion.eventloop"]._timer.isActive()

# The menu actions Krita holds now run the new version
actions["settings"].trigger()
pump()
assert NewDialog.instance().isVisible()
assert not SettingsDialog._instance or SettingsDialog._instance is not NewDialog.instance()

# A docker Krita makes later (new window) still uses the old class, with the new panel
late = ImageDiffusionWidget()
assert type(late.widget()) is DiffusionPanel
first.createActions(krita.Window())  # forwarded to the new extension
assert len(second._actions) == 2 * len(actions)

# And again: the docks made with the first class are found by the third version
update("9.9.10")
import ai_diffusion
assert ai_diffusion.__version__ == "9.9.10"
from ai_diffusion.ui.diffusion import DiffusionPanel as Third
from ai_diffusion.extension import AIToolsExtension as ThirdExt
assert type(dock.widget()) is Third and type(late.widget()) is Third
assert second._successor is ThirdExt.current
actions["generate"].trigger()  # no document open: must not raise
pump()

# A new version that can't be imported: the running one stays as it is, restart needed
panel = dock.widget()
(plugin / "ui" / "diffusion.py").write_text("this is not python(")
failures = []
importlib.import_module("ai_diffusion.reload").request(failures.append)
pump()
assert len(failures) == 1 and isinstance(failures[0], SyntaxError), failures
import ai_diffusion
assert ai_diffusion.__version__ == "9.9.10"
assert dock.widget() is panel and ThirdExt.current is second._successor
assert sys.modules["ai_diffusion.eventloop"]._timer.isActive()
assert not errors, errors
print("reload ok")
"""


def test_reload_plugin(tmp_path: Path):
    plugin = tmp_path / "ai_diffusion"
    shutil.copytree(
        root_dir / "ai_diffusion",
        plugin,
        ignore=shutil.ignore_patterns("debugpy", "__pycache__", "tests", "docs"),
    )
    shutil.copytree(root_dir / "tests" / "mock", tmp_path / "tests" / "mock")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", HOME=str(tmp_path))
    env.pop("PYTHONPATH", None)
    script_file = tmp_path / "run.py"
    script_file.write_text(script.replace("PyQt6", _binding()))
    result = subprocess.run(
        [sys.executable, str(script_file), str(plugin)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=tmp_path,
        check=False,
    )
    assert "reload ok" in result.stdout, result.stdout + result.stderr


def _binding():
    from ai_diffusion.qt import binding

    return binding
