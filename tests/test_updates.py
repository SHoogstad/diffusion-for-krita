import hashlib
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import pytest

from ai_diffusion.model.updates import AutoUpdate, UpdateState, parse_release_tag
from ai_diffusion.platform_tools import ZipFile
from ai_diffusion.qt.QtCore import pyqtBoundSignal

from .conftest import qtapp


class SignalObserver:
    def __init__(self, signal: pyqtBoundSignal):
        self.events = []
        signal.connect(self.on_changed)

    def on_changed(self, value):
        self.events.append(value)

    def reset(self):
        self.events = []


class FakeGithub:
    """Serves /releases, /releases/latest and the release zip, like the GitHub API does."""

    def __init__(self):
        self.releases: list[dict] = []
        self.archive = b""
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/releases/latest":
                    stable = [r for r in fake.releases if not r["prerelease"]]
                    body = json.dumps(stable[-1] if stable else {}).encode()
                    ctype = "application/json; charset=utf-8"
                elif self.path.startswith("/releases?"):
                    body = json.dumps(list(reversed(fake.releases))).encode()
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

            def log_message(self, format, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        Thread(target=self.server.serve_forever, daemon=True).start()

    def publish(self, tag: str, archive: bytes, digest: str | None = None, prerelease=False):
        self.archive = archive
        digest = f"sha256:{hashlib.sha256(archive).hexdigest()}" if digest is None else digest
        asset = {"name": "plugin.zip", "browser_download_url": f"{self.url}/plugin.zip"}
        if digest:
            asset["digest"] = digest
        self.releases.append({"tag_name": tag, "prerelease": prerelease, "assets": [asset]})


@pytest.fixture()
def github():
    fake = FakeGithub()
    yield fake
    fake.server.shutdown()


def test_parse_release_tag():
    assert parse_release_tag("v1.54.0-krita5") == (1, 54, 0)
    assert parse_release_tag("v1.54.1-krita5.3") == (1, 54, 1)
    assert parse_release_tag("1.55.1-krita5.2") == (1, 55, 1)
    assert parse_release_tag("v1.56.0") == (1, 56, 0)
    assert parse_release_tag("v2.0.0-krita6") is None
    assert parse_release_tag("v1.54.0-pre") is None
    assert parse_release_tag("v1.54.0-krita5") < parse_release_tag("v1.54.1-krita5.3")  # type: ignore
    assert parse_release_tag("v1.55.3-krita5.3") < parse_release_tag("v1.56.0")  # type: ignore
    assert parse_release_tag("v1.58.1.1") == (1, 58, 1, 1)
    assert parse_release_tag("v1.58.1") < parse_release_tag("v1.58.1.1")  # type: ignore
    assert parse_release_tag("v1.58.1.1") < parse_release_tag("v1.58.2")  # type: ignore


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
        current_version="v1.54.1-krita5.3", plugin_dir=install_dir, api_url=github.url
    )
    state_changes = SignalObserver(updater.state_changed)

    # Same and older releases are not offered
    for tag in ["v1.54.1-krita5.3", "v1.54.0-krita5.4"]:
        github.publish(tag, archive)
        state_changes.reset()
        await updater.check()
        assert state_changes.events == [UpdateState.checking, UpdateState.latest]

    # Release without a digest can't be verified
    github.publish("v1.54.2-krita5.3", archive, digest="")
    await updater.check()
    assert updater.state is UpdateState.failed_check

    # Newer release is offered, downloaded and installed
    github.publish("v1.54.2-krita5.3", archive)
    state_changes.reset()
    await updater.check()
    assert state_changes.events == [UpdateState.checking, UpdateState.available]
    assert updater.latest_version == "v1.54.2-krita5.3"

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


@qtapp
async def test_auto_update_from_krita5_tag(github: FakeGithub, tmp_path: Path):
    github.publish("v1.55.3-krita5.3", b"old")
    github.publish("v1.56.0", b"new")
    updater = AutoUpdate(
        current_version="v1.55.3-krita5.3", plugin_dir=tmp_path, api_url=github.url
    )
    await updater.check()
    assert updater.state is UpdateState.available
    assert updater.latest_version == "v1.56.0"


@qtapp
async def test_auto_update_prerelease(github: FakeGithub, tmp_path: Path):
    github.publish("v1.54.1-krita5.3", b"stable")
    github.publish("v1.55.0-krita5.3", b"preview", prerelease=True)

    stable = AutoUpdate(
        current_version="v1.54.1-krita5.3",
        plugin_dir=tmp_path,
        api_url=github.url,
        include_prereleases=False,
    )
    await stable.check()
    assert stable.state is UpdateState.latest

    preview = AutoUpdate(
        current_version="v1.54.1-krita5.3",
        plugin_dir=tmp_path,
        api_url=github.url,
        include_prereleases=True,
    )
    await preview.check()
    assert preview.state is UpdateState.available
    assert preview.latest_version == "v1.55.0-krita5.3"

    # A newer stable release wins over an older pre-release
    github.publish("v1.55.1-krita5.3", b"stable 2")
    await preview.check()
    assert preview.latest_version == "v1.55.1-krita5.3"

    # Running a pre-release with the option off doesn't downgrade to the latest stable release
    on_preview = AutoUpdate(
        current_version="v1.56.0-krita5.3",
        plugin_dir=tmp_path,
        api_url=github.url,
        include_prereleases=False,
    )
    await on_preview.check()
    assert on_preview.state is UpdateState.latest


@qtapp
async def test_auto_update_from_prerelease(github: FakeGithub, tmp_path: Path):
    github.publish("v1.54.1-krita5.3", b"stable")
    github.publish("v1.55.0-krita5.3", b"preview", prerelease=True)
    github.publish("v1.55.1-krita5.3", b"preview 2", prerelease=True)

    # Installed version is a pre-release: newer pre-releases are offered with the option off
    updater = AutoUpdate(
        current_version="v1.55.0-krita5.3",
        plugin_dir=tmp_path,
        api_url=github.url,
        include_prereleases=False,
    )
    await updater.check()
    assert updater.state is UpdateState.available
    assert updater.latest_version == "v1.55.1-krita5.3"

    # Stable installs still ignore them
    stable = AutoUpdate(
        current_version="v1.54.1-krita5.3",
        plugin_dir=tmp_path,
        api_url=github.url,
        include_prereleases=False,
    )
    await stable.check()
    assert stable.state is UpdateState.latest
