"""Krita's DoubleSliderSpinBox is a QObject wrapper that Krita keeps alive after its widget is
deleted. When a look switch rebuilds the pages, the old StrengthWidgets must stop listening to the
settings and the model. Otherwise every settings or style change raises, and Krita's error dialog
(a nested event loop) lets the settings dialog be renewed in the middle of its own write()."""

import os
import subprocess
import sys
import tempfile

from tests.test_docker_rebuild import root_dir
from tests.test_docker_rebuild import script as docker_script

setup = docker_script.split("assert click()", 1)[0]
script = (
    setup.replace(
        "from ai_diffusion import eventloop",
        "import ai_diffusion.ui.slider_spinbox as ss\n"
        "ss.DoubleSliderSpinBox = ss.FallbackDoubleSliderSpinBox\n"
        "from ai_diffusion import eventloop",
        1,
    )
    + r"""
import gc
gc.disable()
from ai_diffusion.ui.widget import StrengthWidget
from ai_diffusion.ui.settings import SettingsDialog
kept = []  # like Krita's wrappers: never freed
_init = StrengthWidget.__init__
def init(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    kept.append(self)
StrengthWidget.__init__ = init

def hook(*args):  # like Krita's error dialog: show the error in a nested event loop
    errors.append(args)
    pump()
sys.excepthook = hook

panel2 = DiffusionPanel()
dialog = SettingsDialog(root.server)
dialog.show()
pump()
for color in ("#ff8800", "#9b59b6"):
    tab = SettingsDialog._instance.interface
    tab._widgets["interface_color"].value = color
    tab.write()
    pump()
settings.changed.emit("batch_size", 3)
model.style_changed.emit(model.style)
pump()
assert any(w._released for w in kept), "no StrengthWidget lost its spinbox"
assert not errors, [str(e[1]) for e in errors]
print("leftovers ok")
"""
)


def test_look_switch_leaves_no_listening_sliders():
    with tempfile.TemporaryDirectory() as home:
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", HOME=home)
        result = subprocess.run(
            [sys.executable, "-c", script, str(root_dir)],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=root_dir,
            check=False,
        )
    assert "leftovers ok" in result.stdout, result.stdout + result.stderr


improve_script = (
    setup
    + r"""
from ai_diffusion import eventloop as el
from ai_diffusion.ui import region as region_ui
from ai_diffusion.ui.region import ActiveRegionWidget

pending = []
async def slow_improve(*args, **kwargs):  # the LLM call, still running during the switch
    pending.append(asyncio.get_event_loop().create_future())
    return await pending[0]
region_ui.improve_prompt = slow_improve

def run_until(condition):
    for _ in range(2000):
        app.processEvents()
        el.process_python_events()
        if condition():
            return

widget = panel.findChildren(ActiveRegionWidget)[0]
region = widget.region
region.positive = "a cat"
task = el.run(widget._run_improve_prompt(region))
run_until(lambda: pending or task.done())
assert pending, "Improve Prompt did not start"
settings.interface_color = "#ff8800"  # the page is rebuilt while the LLM runs
pump()
pending[0].set_result("a fluffy cat")
run_until(task.done)
assert task.exception() is None, task.exception()
assert region.positive == "a fluffy cat", region.positive
print("improve ok")
"""
)


def test_look_switch_during_improve_prompt_keeps_the_result():
    with tempfile.TemporaryDirectory() as home:
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", HOME=home)
        result = subprocess.run(
            [sys.executable, "-c", improve_script, str(root_dir)],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=root_dir,
            check=False,
        )
    assert "improve ok" in result.stdout, result.stdout + result.stderr
