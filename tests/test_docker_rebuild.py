"""The docker pages work when they are made while a document is open: on a look switch, after a
plugin reload, or when the docker opens after the image. Widgets need a QApplication, which the
test process doesn't have, so this runs in a subprocess."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

root_dir = Path(__file__).parent.parent

script = r"""
import asyncio, sys, tempfile, traceback
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from ai_diffusion.qt.QtWidgets import QApplication
from ai_diffusion.qt.QtCore import QEvent
app = QApplication([])
errors = []
sys.excepthook = lambda *a: (errors.append(a), traceback.print_exception(*a))
from ai_diffusion import eventloop
eventloop.setup()
from ai_diffusion.model.root import root
root.init()
from krita import Krita
from ai_diffusion.document import KritaDocument
from ai_diffusion.model.connection import Connection, ConnectionState
from ai_diffusion.model.custom_workflow import WorkflowCollection
from ai_diffusion.model.model import DocumentModel
from ai_diffusion.settings import InterfaceLook, settings
from ai_diffusion.ui import theme
from ai_diffusion.ui.diffusion import DiffusionPanel, WelcomeWidget
from tests.mock.client import MockClient

def pump():
    for _ in range(30):
        app.processEvents()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)

async def stage():
    doc = Krita.instance().openDocument("test")
    Krita.instance().setActiveDocument(doc)
    conn = Connection()
    conn.connect(MockClient())
    for _ in range(200):
        await asyncio.sleep(0)
        if conn.state is ConnectionState.connected:
            break
    wf = WorkflowCollection(conn, folder=Path(tempfile.mkdtemp()))
    root._connection = conn
    return DocumentModel(KritaDocument.active(), conn, wf)

task = eventloop.run(stage())
while not task.done():
    app.processEvents()
model = task.result()
root.model_for_active_document = lambda: model  # the document is open before the docker
WelcomeWidget.requires_update = property(lambda s: False)
WelcomeWidget.has_news = property(lambda s: False)
calls = []
DocumentModel.generate = lambda self: calls.append(self)

panel = DiffusionPanel()
panel.update_content()
pump()

def click():
    calls.clear()
    panel._generation.generate_button.click()
    pump()
    return calls == [model]

assert click(), "Generate does nothing when the docker opens after the image"
for look in (InterfaceLook.country, InterfaceLook.legacy, InterfaceLook.plain):
    settings.interface_look = look
    pump()
    assert theme.legacy == (look is InterfaceLook.legacy)
    assert click(), f"Generate does nothing after switching to the {look.name} look"
    model.seed = model.seed + 1  # reaches no deleted widget (the queue popup's bindings)
    model.batch_count = 2
    pump()
assert not errors, errors
print("docker ok")
"""


def test_docker_pages_bind_when_made_with_open_document():
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
    assert "docker ok" in result.stdout, result.stdout + result.stderr
