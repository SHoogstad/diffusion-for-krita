import asyncio
import hashlib
import os
import re
import shutil
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

from .. import __prerelease__, __release__, eventloop
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
    reloading = 10  # loading the new version without a restart


def _precompile(package_dir: Path):
    import compileall
    import time

    start = time.perf_counter()
    try:
        skip = re.compile(r"[/\\](debugpy|tests|docs|__pycache__)[/\\]")
        compileall.compile_dir(str(package_dir), quiet=1, rx=skip)
        log.info(f"Plugin update: compiled in {time.perf_counter() - start:.1f}s")
    except Exception as e:  # only slower then
        log.warning(f"Plugin update: could not compile the new version ahead: {e}")


class UpdatePackage(NamedTuple):
    version: str
    url: str
    sha256: str
    prerelease: bool = False


def parse_release_tag(tag: str) -> tuple[int, ...] | None:
    """Parse "v1.56.0" into the plugin version (1, 56, 0). Builds run on Krita 5 and 6.
    Hotfixes have a fourth number, "v1.58.1.1" -> (1, 58, 1, 1), which sorts after (1, 58, 1).
    Older Krita 5 only releases were tagged "v1.54.1-krita5.3" and are still recognized."""
    m = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?(?:-krita5(?:\.\d+)?)?", tag.strip())
    if m is None:
        return None
    return tuple(int(x) for x in m.groups() if x is not None)


def _release_key(release: dict):
    """Order of releases: by version, and a full release after a pre-release of the same
    version (eg. a pre-release which was later made the full release with more fixes)."""
    version = parse_release_tag(release.get("tag_name", "")) or ()
    return (version, not release.get("prerelease"))


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
        current_prerelease: bool | None = None,
    ):
        super().__init__()
        self.plugin_dir = plugin_dir or Path(__file__).parent.parent.parent
        self.current_version = current_version or __release__
        # A pre-release build: the full release with the same version number is newer
        if current_prerelease is None:
            current_prerelease = __prerelease__ if current_version is None else False
        self.current_prerelease = current_prerelease
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
        if self.state in (UpdateState.restart_required, UpdateState.reloading):
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
        on_prerelease = self.current_prerelease or any(
            r["tag_name"] == self.current_version and r.get("prerelease") for r in releases
        )
        if not (self.include_prereleases or on_prerelease):
            releases = [r for r in releases if not r.get("prerelease")]
        result = max(releases, key=_release_key, default={})
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
        is_final = not result.get("prerelease")
        if current is not None and (latest, is_final) <= (current, not on_prerelease):
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
                prerelease=not is_final,
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
        self.current_prerelease = self._package.prerelease
        if self.reload_after_install:
            from .. import reload

            self.state = UpdateState.reloading
            # Compile the new files now, in the background: loading them is then quicker,
            # and that part blocks Krita.
            await asyncio.to_thread(_precompile, self.plugin_dir / "ai_diffusion")
            reload.request(self._reload_failed)
        else:
            self.state = UpdateState.restart_required

    reload_after_install = False  # load the new version without a restart (set by root)

    def _reload_failed(self, error: Exception):
        self.error = str(error)
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
