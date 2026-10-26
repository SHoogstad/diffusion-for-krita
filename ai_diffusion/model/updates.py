import hashlib
import os
import re
import shutil
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

from .. import __release__, eventloop
from ..backend.network import RequestManager
from ..platform_tools import ZipFile
from ..qt.QtCore import QObject, pyqtSignal
from ..settings import settings
from ..util import client_logger as log
from .properties import ObservableProperties, Property


class UpdateState(Enum):
    unknown = 1
    checking = 2
    available = 3
    latest = 4
    downloading = 5
    installing = 6
    restart_required = 7
    failed_check = 8
    failed_update = 9


class UpdatePackage(NamedTuple):
    version: str
    url: str
    sha256: str


def parse_release_tag(tag: str) -> tuple[int, ...] | None:
    """Parse "v1.56.0" into the plugin version (1, 56, 0). Builds run on Krita 5 and 6.
    Older Krita 5 only releases were tagged "v1.54.1-krita5.3" and are still recognized."""
    m = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)(?:-krita5(?:\.\d+)?)?", tag.strip())
    if m is None:
        return None
    return tuple(int(x) for x in m.groups())


class AutoUpdate(QObject, ObservableProperties):
    # Updates come from GitHub releases of Diffusion for Krita.
    default_api_url = os.getenv(
        "KRITA5_UPDATE_URL", "https://api.github.com/repos/SHoogstad/diffusion-for-krita"
    )

    state = Property(UpdateState.unknown)
    latest_version = Property("")
    error = Property("")

    state_changed = pyqtSignal(UpdateState)
    latest_version_changed = pyqtSignal(str)
    error_changed = pyqtSignal(str)

    def __init__(
        self,
        plugin_dir: Path | None = None,
        current_version: str | None = None,
        api_url: str | None = None,
        include_prereleases: bool | None = None,
    ):
        super().__init__()
        self.plugin_dir = plugin_dir or Path(__file__).parent.parent.parent
        self.current_version = current_version or __release__
        self.api_url = api_url or self.default_api_url
        self._include_prereleases = include_prereleases
        self._package: UpdatePackage | None = None
        self._temp_dir: TemporaryDirectory | None = None
        self._request_manager: RequestManager | None = None

    def check(self):
        return eventloop.run(
            self._handle_errors(
                self._check, UpdateState.failed_check, "Failed to check for new plugin version"
            )
        )

    async def _check(self):
        if self.state is UpdateState.restart_required:
            return

        self.state = UpdateState.checking
        log.info(f"Checking for latest plugin version at {self.api_url}")
        releases = await self._net.get(f"{self.api_url}/releases?per_page=30", timeout=10)
        releases = [
            r
            for r in releases
            if not r.get("draft") and parse_release_tag(r.get("tag_name", "")) is not None
        ]
        # Installs of a pre-release keep following pre-releases, even with the option off
        on_prerelease = any(
            r["tag_name"] == self.current_version and r.get("prerelease") for r in releases
        )
        if not (self.include_prereleases or on_prerelease):
            releases = [r for r in releases if not r.get("prerelease")]
        result = max(releases, key=lambda r: parse_release_tag(r["tag_name"]) or (), default={})
        tag = result.get("tag_name", "")
        latest = parse_release_tag(tag)
        current = parse_release_tag(self.current_version)
        asset = next(
            (a for a in result.get("assets", []) if a.get("name", "").endswith(".zip")), None
        )
        digest = (asset.get("digest") or "") if asset else ""
        if latest is None:
            log.error(f"Invalid plugin update information: {tag}")
            self.state = UpdateState.failed_check
            self.error = "Failed to retrieve plugin update information"
            return

        self.latest_version = tag
        if current is not None and latest <= current:
            log.info("Plugin is up to date!")
            self.state = UpdateState.latest
        elif asset is None or not digest.startswith("sha256:"):
            log.error(f"Invalid plugin update information: {result}")
            self.state = UpdateState.failed_check
            self.error = "Plugin update package is incomplete"
        else:
            log.info(f"New plugin version available: {tag}")
            self._package = UpdatePackage(
                version=tag,
                url=asset["browser_download_url"],
                sha256=digest.removeprefix("sha256:"),
            )
            self.state = UpdateState.available

    def run(self):
        return eventloop.run(
            self._handle_errors(self._run, UpdateState.failed_update, "Failed to update plugin")
        )

    async def _run(self):
        assert self.latest_version and self._package

        self._temp_dir = TemporaryDirectory()
        archive_path = Path(self._temp_dir.name) / f"krita_ai_diffusion-{self.latest_version}.zip"
        log.info(f"Downloading plugin update {self._package.url}")
        self.state = UpdateState.downloading
        archive_data = await self._net.download(self._package.url)

        sha256 = hashlib.sha256(archive_data).hexdigest()
        if sha256 != self._package.sha256:
            log.error(f"Update package hash mismatch: {sha256} != {self._package.sha256}")
            raise RuntimeError("Downloaded plugin package is corrupted or incomplete")

        archive_path.write_bytes(archive_data)
        source_dir = Path(self._temp_dir.name) / f"krita_ai_diffusion-{self.latest_version}"
        log.info(f"Extracting plugin archive into {source_dir}")
        self.state = UpdateState.installing
        with ZipFile(archive_path) as zip_file:
            zip_file.extractall(source_dir)

        log.info(f"Installing new plugin version to {self.plugin_dir}")
        shutil.copytree(source_dir, self.plugin_dir, dirs_exist_ok=True)
        self.current_version = self.latest_version
        self.state = UpdateState.restart_required

    @property
    def include_prereleases(self):
        if self._include_prereleases is None:
            return settings.auto_update_prerelease
        return self._include_prereleases

    @property
    def is_available(self):
        return self.latest_version is not None and self.latest_version != self.current_version

    @property
    def _net(self):
        if self._request_manager is None:
            self._request_manager = RequestManager()
        return self._request_manager

    async def _handle_errors(self, func, error_state: UpdateState, message: str):
        try:
            return await func()
        except Exception as e:
            log.exception(e)
            self.error = f"{message}: {e}"
            self.state = error_state
            return None
