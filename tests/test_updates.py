import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import pytest
from aiohttp import ClientSession
from PyQt5.QtCore import pyqtBoundSignal

from ai_diffusion.model.updates import AutoUpdate, UpdateState, parse_release_tag
from ai_diffusion.platform_tools import ZipFile

from .conftest import CloudService, qtapp


class SignalObserver:
    def __init__(self, signal: pyqtBoundSignal):
        self.events = []
        signal.connect(self.on_changed)

    def on_changed(self, value):
        self.events.append(value)

    def reset(self):
        self.events = []


def http_session(service_url: str):
    service_token = os.environ["INTERSTICE_INFRA_TOKEN"]
    headers = {"Authorization": f"Bearer {service_token}"}
    return ClientSession(service_url, headers=headers)


class FakeGithub:
    """Serves /releases/latest and the release zip, like the GitHub API does."""

    def __init__(self):
        self.tag = ""
        self.archive = b""
        self.digest = ""
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                url = f"http://127.0.0.1:{fake.port}"
                if self.path == "/releases/latest":
                    asset = {"name": "plugin.zip", "browser_download_url": f"{url}/plugin.zip"}
                    if fake.digest:
                        asset["digest"] = fake.digest
                    body = json.dumps({"tag_name": fake.tag, "assets": [asset]}).encode()
                    ctype = "application/json; charset=utf-8"
                elif self.path == "/plugin.zip":
                    body = fake.archive
                    ctype = "application/zip"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        Thread(target=self.server.serve_forever, daemon=True).start()

    def publish(self, tag: str, archive: bytes, digest: str | None = None):
        self.tag = tag
        self.archive = archive
        self.digest = f"sha256:{hashlib.sha256(archive).hexdigest()}" if digest is None else digest


@pytest.fixture()
def github():
    fake = FakeGithub()
    yield fake
    fake.server.shutdown()


def test_parse_release_tag():
    assert parse_release_tag("v1.54.0-krita5") == (1, 54, 0, 0)
    assert parse_release_tag("v1.54.0-krita5.3") == (1, 54, 0, 3)
    assert parse_release_tag("1.55.1-krita5.1") == (1, 55, 1, 1)
    assert parse_release_tag("v1.54.0") is None
    assert parse_release_tag("v2.0.0-krita6") is None
    assert parse_release_tag("v1.54.0-krita5.1") < parse_release_tag("v1.55.0-krita5")  # type: ignore


@qtapp
async def test_auto_update(github: FakeGithub, tmp_path: Path):
    install_dir = tmp_path / "install"
    install_plugin_dir = install_dir / "test_plugin"
    install_plugin_dir.mkdir(parents=True)
    install_test_file = install_plugin_dir / "test_file.txt"
    install_test_file.write_text("local produce is the best")

    build_dir = tmp_path / "build"
    build_archive = build_dir / "test_plugin.zip"
    build_dir.mkdir()
    with ZipFile(build_archive, "w") as zip_file:
        zip_file.writestr(
            "test_plugin/test_file.txt", "if you're feeling orange, try flying a kite"
        )
    archive = build_archive.read_bytes()

    updater = AutoUpdate(
        current_version="v1.54.0-krita5.1", plugin_dir=install_dir, api_url=github.url
    )
    state_changes = SignalObserver(updater.state_changed)

    # Same and older releases are not offered
    for tag in ["v1.54.0-krita5.1", "v1.53.0-krita5.4"]:
        github.publish(tag, archive)
        state_changes.reset()
        await updater.check()
        assert state_changes.events == [UpdateState.checking, UpdateState.latest]

    # Release without a digest can't be verified
    github.publish("v1.54.0-krita5.2", archive, digest="")
    await updater.check()
    assert updater.state is UpdateState.failed_check

    # Newer release is offered, downloaded and installed
    github.publish("v1.54.0-krita5.2", archive)
    state_changes.reset()
    await updater.check()
    assert state_changes.events == [UpdateState.checking, UpdateState.available]
    assert updater.latest_version == "v1.54.0-krita5.2"

    state_changes.reset()
    await updater.run()
    assert state_changes.events == [
        UpdateState.downloading,
        UpdateState.installing,
        UpdateState.restart_required,
    ]
    assert install_test_file.read_text() == "if you're feeling orange, try flying a kite"


@qtapp
async def test_auto_update_bad_hash(github: FakeGithub, tmp_path: Path):
    archive = b"not the archive you are looking for"
    github.publish("v9.0.0-krita5", archive, digest="sha256:" + "0" * 64)
    updater = AutoUpdate(current_version="v1.0.0-krita5", plugin_dir=tmp_path, api_url=github.url)
    await updater.check()
    assert updater.state is UpdateState.available
    await updater.run()
    assert updater.state is UpdateState.failed_update
    assert "corrupted" in updater.error


async def test_authorization(cloud_service: CloudService):
    if not cloud_service.enabled:
        pytest.skip("Cloud service not running")
    async with ClientSession(cloud_service.url) as session:
        # Version check is public
        async with session.get("/plugin/latest?version=1.2.3") as response:
            assert response.status == 200

        # Upload requires authorization
        async with session.put("/plugin/upload/1.2.3") as response:
            assert response.status == 401
