import functools
import inspect
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ai_diffusion.qt.QtCore import QCoreApplication

sys.path.append(str(Path(__file__).parent.parent))
from ai_diffusion import eventloop, util
from ai_diffusion.backend import network
from ai_diffusion.document import KritaDocument

from .config import result_dir

# Create the Qt application instance at module level, before any test module is
# imported (collected).  Several modules create QTimers at import time
# (e.g. PoseLayers in document.py), and Qt requires QCoreApplication to exist
# before any timer can be started.
_qt_app = QCoreApplication.instance() or QCoreApplication([])
eventloop.setup()

root_dir = Path(__file__).parent.parent


def pytest_addoption(parser):
    parser.addoption("--test-install", action="store_true")
    parser.addoption("--ci", action="store_true")
    parser.addoption("--benchmark", action="store_true")


def pytest_collection_modifyitems(session, config, items: list[pytest.Item]):
    def order(item: pytest.Item):
        if not item.parent or "test_workflow" not in item.parent.name:
            return 0
        if "flux2" in item.name:
            return 4
        elif "flux" in item.name:
            return 3
        elif "sdxl" in item.name:
            return 2
        else:
            return 1

    items.sort(key=order)


class QtTestApp:
    def run(self, coro):
        task = eventloop.run(coro)
        while not task.done():
            _qt_app.processEvents()
        return task.result()

    def __call__(self, func):
        """Use as a decorator on async test functions to run them in the Qt event loop.

        Example::

            @qtapp
            async def test_foo():
                result = await bar()
                assert result == expected
        """
        if not inspect.iscoroutinefunction(func):
            raise TypeError(f"@qtapp can only decorate async functions, got: {func.__name__!r}")

        @functools.wraps(func)
        def wrapper(**kwargs):
            self.run(func(**kwargs))

        return wrapper


# Module-level singleton — import this in test files to use as a decorator:
qtapp = QtTestApp()


# Can also use qtapp as Pytest fixture and explicitly call qtapp.run()
@pytest.fixture(scope="session", name="qtapp")
def _qtapp_fixture():
    return qtapp


@pytest.fixture(autouse=True)
def reset_krita_state():
    from krita import Krita

    Krita._instance = None  # type: ignore[assignment]
    KritaDocument._instances.clear()


@pytest.fixture(scope="session", autouse=True)
def clear_appdata():
    user_dir = util.user_data_dir
    assert user_dir.name == ".appdata", "expected local test appdata dir"

    data_dir = user_dir / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)


@pytest.fixture(scope="session", autouse=True)
def clear_results():
    if result_dir.exists():
        for file in result_dir.iterdir():
            if file.is_dir():
                shutil.rmtree(file)
            else:
                file.unlink()
    result_dir.mkdir(exist_ok=True)


@pytest.fixture()
def local_download_server():
    script = root_dir / "scripts" / "file_server.py"
    port = 51222

    with subprocess.Popen([sys.executable, str(script), str(port)]) as proc:
        assert proc.poll() is None
        network.HOSTMAP = network.HOSTMAP_LOCAL
        yield f"http://localhost:{port}"
        network.HOSTMAP = {}

        proc.terminate()
        proc.wait()
