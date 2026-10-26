from __future__ import annotations

from typing import cast

from krita import Krita

from .. import __release__, eventloop, util
from ..backend import resources, runpod
from ..backend.client import Client, MissingResources
from ..backend.resources import Arch, ResourceId
from ..backend.server import Server, ServerState
from ..localization import Localization
from ..localization import translate as _
from ..model.connection import ConnectionState, apply_performance_preset
from ..model.root import collect_diagnostics, root
from ..model.updates import UpdateState
from ..qt.QtCore import QSize, Qt, QUrl, pyqtSignal
from ..qt.QtGui import (
    QColor,
    QCursor,
    QDesktopServices,
    QFontDatabase,
    QFontMetrics,
    QGuiApplication,
    QPainter,
)
from ..qt.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QStyleOption,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from ..settings import (
    ImageFileFormat,
    PerformancePreset,
    RunpodStorage,
    ServerMode,
    Settings,
    settings,
)
from ..style import Style
from .server import ServerWidget
from .settings_widgets import (
    ComboBoxSetting,
    FileListSetting,
    SettingsTab,
    SliderSetting,
    SpinBoxSetting,
    SwitchSetting,
)
from .style import StylePresets
from .theme import add_header, green, grey, logo, prompt_max_line_count, red, yellow


class InitialSetupWidget(QWidget):
    finished = pyqtSignal(ServerMode)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 0)
        self.setLayout(layout)

        label_title = QLabel("<b>" + _("Welcome to Image Generation in Krita") + "</b>", self)
        label_sub = QLabel(
            _(
                "To create images, the plugin needs to connect to a backend server. Please choose one of the options below (you can always switch later)."
            ),
            self,
        )
        label_sub.setWordWrap(True)
        layout.addWidget(label_title)
        layout.addWidget(label_sub)
        layout.addSpacing(20)

        def add_option(title: str, desc_text: str, button_text: str, mode: ServerMode):
            header = QLabel("<b>" + title + "</b>", self)
            desc = QLabel(desc_text, self)
            desc.setMaximumWidth(600)
            desc.setWordWrap(True)
            button = QPushButton(button_text, self)
            button.setMinimumHeight(int(1.3 * button.sizeHint().height()))
            button.setMaximumWidth(300)
            button.clicked.connect(self._choose(mode))
            layout.addWidget(header)
            layout.addWidget(desc)
            layout.addWidget(button)
            layout.addSpacing(16)

        add_option(
            _("Option {number}", number=1) + ": " + _("Runpod"),
            _(
                "Rent a GPU on {link} with your own account, billed per second while it runs. No local installation or powerful hardware needed.",
                link="<a href='https://www.runpod.io'>runpod.io</a>",
            ),
            _("Set up Runpod"),
            ServerMode.runpod,
        )
        add_option(
            _("Option {number}", number=2) + ": " + _("Local Managed Server"),
            _(
                "Install and run a local ComfyUI server on your machine. Installation and updates are performed automatically by the plugin. Requires a compatible GPU (NVIDIA with at least 6GB VRAM recommended)."
            ),
            _("Start Installation"),
            ServerMode.managed,
        )
        add_option(
            _("Option {number}", number=3) + ": " + _("Custom ComfyUI"),
            _(
                "Connect to an existing installation of ComfyUI. It can be on the same machine, or a remote machine over the network. You are responsible to setup ComfyUI and install required custom nodes and models."
            )
            + "<br><a href='https://github.com/SHoogstad/diffusion-for-krita#comfyui-setup'>ComfyUI Setup Guide</a>",
            _("Connect via URL"),
            ServerMode.external,
        )
        layout.addStretch()

    def _choose(self, mode: ServerMode):
        def handler():
            settings.server_mode = mode
            settings.save()
            self.finished.emit(mode)

        return handler


class RunpodWidget(QWidget):
    """Configure, start and stop a Runpod GPU pod running ComfyUI."""

    def __init__(self, parent):
        super().__init__(parent)
        self._gpus: list[runpod.GpuType] = []
        self._loading = False

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 12, 4, 4)
        self.setLayout(layout)

        header = QLabel(
            "<b>Runpod</b> - "
            + _(
                "GPU pods on {link}, billed to your own account per second while they run.",
                link="<a href='https://www.runpod.io'>runpod.io</a>",
            ),
            self,
        )
        header.setWordWrap(True)
        header.setOpenExternalLinks(True)
        layout.addWidget(header)
        layout.addSpacing(8)

        add_header(layout, Settings._runpod_api_key)
        key_layout = QHBoxLayout()
        self._api_key = QLineEdit(self)
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key.setPlaceholderText("rpa_...")
        self._api_key.editingFinished.connect(self._change_api_key)
        key_layout.addWidget(self._api_key)
        self._load_button = QPushButton(_("Load GPUs"), self)
        self._load_button.clicked.connect(self._load_gpus)
        key_layout.addWidget(self._load_button)
        layout.addLayout(key_layout)

        add_header(layout, Settings._runpod_gpu)
        self._gpu = QComboBox(self)
        self._gpu.currentIndexChanged.connect(self._change_gpu)
        layout.addWidget(self._gpu)

        add_header(layout, Settings._runpod_workloads)
        workload_layout = QHBoxLayout()
        self._workloads: dict[str, QCheckBox] = {}
        for id, name in runpod.workload_names.items():
            box = QCheckBox(name, self)
            box.toggled.connect(self._change_workloads)
            self._workloads[id] = box
            workload_layout.addWidget(box)
        workload_layout.addStretch()
        layout.addLayout(workload_layout)
        self._checkpoints = QCheckBox(Settings._runpod_checkpoints.name, self)
        self._checkpoints.setToolTip(Settings._runpod_checkpoints.desc)
        self._checkpoints.toggled.connect(self._change_workloads)
        layout.addWidget(self._checkpoints)

        add_header(layout, Settings._runpod_storage)
        self._storage_group = QButtonGroup(self)
        self._storage: dict[RunpodStorage, QRadioButton] = {}
        self._estimates: dict[RunpodStorage, QLabel] = {}
        for storage in RunpodStorage:
            radio = QRadioButton(storage.value, self)
            radio.toggled.connect(self._change_storage)
            self._storage_group.addButton(radio)
            estimate = QLabel(self)
            estimate.setWordWrap(True)
            estimate.setTextFormat(Qt.TextFormat.RichText)
            estimate.setContentsMargins(24, 0, 0, 6)
            estimate.setStyleSheet(f"color: {grey}")
            self._storage[storage] = radio
            self._estimates[storage] = estimate
            layout.addWidget(radio)
            layout.addWidget(estimate)
        self._delete_volume = QPushButton(_("Delete network volume"), self)
        self._delete_volume.clicked.connect(self._delete_volume_clicked)
        layout.addWidget(self._delete_volume, alignment=Qt.AlignmentFlag.AlignLeft)

        idle_layout = QHBoxLayout()
        idle_label = QLabel(Settings._runpod_idle_minutes.name, self)
        idle_label.setToolTip(Settings._runpod_idle_minutes.desc)
        self._idle = QSpinBox(self)
        self._idle.setRange(0, 240)
        self._idle.setSuffix(" " + _("min"))
        self._idle.setSpecialValueText(_("Never"))
        self._idle.valueChanged.connect(self._change_idle)
        idle_layout.addWidget(idle_label)
        idle_layout.addWidget(self._idle)
        idle_layout.addStretch()
        layout.addLayout(idle_layout)
        layout.addSpacing(8)

        self._status = QLabel(self)
        self._status.setWordWrap(True)
        self._status.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._status)

        self._start_button = QPushButton(_("Start pod"), self)
        self._start_button.setMinimumWidth(200)
        self._start_button.setMinimumHeight(int(1.3 * self._start_button.sizeHint().height()))
        self._start_button.clicked.connect(self._start_or_stop)
        layout.addWidget(self._start_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch()

        self.read()
        root.connection.pod_status_changed.connect(self.update_state)

    def read(self):
        self._api_key.setText(settings.runpod_api_key)
        for id, box in self._workloads.items():
            box.blockSignals(True)
            box.setChecked(id in settings.runpod_workloads)
            box.blockSignals(False)
        self._checkpoints.blockSignals(True)
        self._checkpoints.setChecked(settings.runpod_checkpoints)
        self._checkpoints.blockSignals(False)
        self._storage[settings.runpod_storage].setChecked(True)
        self._idle.setValue(settings.runpod_idle_minutes)
        self._fill_gpus()
        if settings.runpod_api_key and not self._gpus:
            self._load_gpus()
        self.update_state()

    def _fill_gpus(self):
        self._gpu.blockSignals(True)
        self._gpu.clear()
        gpus = self._gpus or []
        if not any(g.id == settings.runpod_gpu for g in gpus):
            self._gpu.addItem(settings.runpod_gpu, settings.runpod_gpu)
        for gpu in gpus:
            label = gpu.label
            if gpu.availability == "NONE":
                label += " - " + _("not available")
            self._gpu.addItem(label, gpu.id)
        index = self._gpu.findData(settings.runpod_gpu)
        self._gpu.setCurrentIndex(max(index, 0))
        self._gpu.blockSignals(False)
        self._update_estimates()

    def _load_gpus(self):
        if self._loading or not settings.runpod_api_key:
            return

        async def load():
            self._loading = True
            self._load_button.setEnabled(False)
            try:
                self._gpus = await runpod.RunpodApi(settings.runpod_api_key).gpus()
                self._fill_gpus()
            except Exception as e:
                self._status.setText(f"<b>{_('Error')}</b>: {e}")
                self._status.setStyleSheet(f"color: {red}")
            finally:
                self._loading = False
                self._load_button.setEnabled(True)

        eventloop.run(load())

    def _selected_gpu(self):
        return next((g for g in self._gpus if g.id == settings.runpod_gpu), None)

    def _update_estimates(self):
        gpu = self._selected_gpu()
        price = gpu.price if gpu else 0.0
        model_gb = runpod.model_size_gb(settings.runpod_workloads, settings.runpod_checkpoints)
        for storage, label in self._estimates.items():
            estimate = runpod.estimate(storage, model_gb, price)
            label.setText(estimate.describe(storage, model_gb))

    def _change_api_key(self):
        key = self._api_key.text().strip()
        if key != settings.runpod_api_key:
            settings.runpod_api_key = key
            settings.save()
            self._gpus = []
            self._load_gpus()

    def _change_gpu(self):
        settings.runpod_gpu = self._gpu.currentData() or settings.runpod_gpu
        self._update_estimates()

    def _change_workloads(self):
        if self._workloads["illu"].isChecked() and not self._workloads["sdxl"].isChecked():
            self._workloads["sdxl"].setChecked(True)  # Illustrious builds on the SDXL models
            return
        workloads = [id for id, box in self._workloads.items() if box.isChecked()]
        settings.runpod_workloads = workloads or ["sdxl"]
        settings.runpod_checkpoints = self._checkpoints.isChecked()
        self._update_estimates()

    def _change_storage(self):
        for storage, radio in self._storage.items():
            if radio.isChecked():
                settings.runpod_storage = storage
        self.update_state()

    def _change_idle(self):
        settings.runpod_idle_minutes = self._idle.value()

    def _start_or_stop(self):
        connection = root.connection
        if connection.pod_running:
            connection.stop_pod()
        else:
            settings.save()
            connection.start_pod()

    def _delete_volume_clicked(self):
        answer = QMessageBox.question(
            self,
            _("Delete network volume"),
            _(
                "Delete the Runpod network volume with all downloaded models? The next start downloads them again."
            ),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        async def delete():
            try:
                await runpod.RunpodApi(settings.runpod_api_key).delete_volume(
                    settings.runpod_volume_id
                )
                settings.runpod_volume_id = ""
                settings.save()
                self._status.setText(_("Network volume deleted."))
                self._status.setStyleSheet(f"color: {grey}")
            except Exception as e:
                self._status.setText(f"<b>{_('Error')}</b>: {e}")
                self._status.setStyleSheet(f"color: {red}")
            self.update_state()

        eventloop.run(delete())

    def update_state(self):
        connection = root.connection
        state = connection.state
        running = connection.pod_running
        busy = state in (ConnectionState.pod_starting, ConnectionState.connecting)
        editable = not running
        for widget in (self._gpu, self._checkpoints, *self._workloads.values()):
            widget.setEnabled(editable)
        for radio in self._storage.values():
            radio.setEnabled(editable)
        self._delete_volume.setVisible(bool(settings.runpod_volume_id))
        self._delete_volume.setEnabled(not running)
        self._start_button.setText(_("Stop pod") if running else _("Start pod"))
        self._start_button.setEnabled(bool(settings.runpod_api_key) and not busy or running)

        if state is ConnectionState.pod_starting:
            self._status.setText(connection.pod_status)
            self._status.setStyleSheet(f"color: {yellow}; font-weight:bold")
        elif state is ConnectionState.connected and running:
            gpu = self._selected_gpu()
            cost = f" - ${gpu.price:.2f}/h" if gpu else ""
            self._status.setText(_("Pod running, connected") + cost)
            self._status.setStyleSheet(f"color: {green}; font-weight:bold")
        elif state in (ConnectionState.connecting, ConnectionState.discover_models):
            self._status.setText(_("Connecting"))
            self._status.setStyleSheet(f"color: {yellow}; font-weight:bold")
        elif state is ConnectionState.error and connection.error:
            msg = connection.error.removeprefix("Error: ")
            self._status.setText(f"<b>{_('Error')}</b>: {msg}")
            self._status.setStyleSheet(f"color: {red}")
        elif connection.error:  # e.g. stopped after idle time
            self._status.setText(connection.error)
            self._status.setStyleSheet(f"color: {grey}")
        else:
            self._status.setText(_("No pod running"))
            self._status.setStyleSheet(f"color: {grey}; font-style:italic")


_server_mode_text = {
    ServerMode.undefined: "Undefined",
    ServerMode.runpod: _("Runpod"),
    ServerMode.managed: _("Local Managed Server"),
    ServerMode.external: _("Custom Server"),
}
_server_mode_status = {
    "not_started": (_("Not started"), grey),
    "not_installed": (_("Not installed"), grey),
    "not_running": (_("Not running"), grey),
    "not_connected": (_("Not connected"), grey),
    "connecting": (_("Connecting"), yellow),
    "connected": (_("Connected"), green),
    "error": (_("Error"), red),
}


class ServerModeButton(QPushButton):
    toggled = pyqtSignal(ServerMode)

    def __init__(self, mode: ServerMode, status: str, parent=None):
        self._text = _server_mode_text[mode]
        super().__init__(self._text, parent)
        self.mode = mode
        self._status = status
        self._is_checked = False

        font = QFontMetrics(self.font())
        self._text_width = font.horizontalAdvance(self._text)
        max_width = max(font.horizontalAdvance(s[0]) for s in _server_mode_status.values())
        self.setMinimumWidth(self._text_width + max_width + 32)
        self.setFixedHeight(int(1.3 * self.sizeHint().height()))

        self.clicked.connect(self._toggle)

    def _toggle(self):
        self.toggled.emit(self.mode)

    def setChecked(self, a0: bool):
        self._is_checked = a0
        self.update()

    def isChecked(self) -> bool:
        return self._is_checked

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, status: str):
        self._status = status
        self.update()

    def paintEvent(self, a0):
        status_text, color = _server_mode_status.get(self._status, (_("Unknown"), red))
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        style = util.ensure(self.style())
        if self.isChecked():
            opt.state |= QStyle.StateFlag.State_Sunken
        style.drawPrimitive(QStyle.PrimitiveElement.PE_PanelButtonCommand, opt, painter, self)

        rect = self.rect().adjusted(8, 0, -8, 0)
        bold = self.font()
        bold.setBold(True)
        painter.setFont(bold)
        painter.drawText(
            rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._text
        )
        painter.setPen(QColor(color))
        painter.setFont(self.font())
        painter.drawText(
            rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, status_text
        )
        painter.end()


class ServerModeSelect(QWidget):
    changed = pyqtSignal(ServerMode)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._runpod_button = ServerModeButton(ServerMode.runpod, "not_started", self)
        self._managed_button = ServerModeButton(ServerMode.managed, "not_installed", self)
        self._external_button = ServerModeButton(ServerMode.external, "not_connected", self)

        for button in (self._runpod_button, self._managed_button, self._external_button):
            button.toggled.connect(self._change_mode)

        layout.addWidget(self._runpod_button)
        layout.addWidget(self._managed_button)
        layout.addWidget(self._external_button)
        layout.addStretch()

    def _change_mode(self, mode: ServerMode):
        self.mode = mode
        self.changed.emit(mode)

    @property
    def mode(self):
        if self._runpod_button.isChecked():
            return ServerMode.runpod
        elif self._managed_button.isChecked():
            return ServerMode.managed
        elif self._external_button.isChecked():
            return ServerMode.external
        return ServerMode.undefined

    @mode.setter
    def mode(self, mode: ServerMode):
        self._runpod_button.setChecked(mode is ServerMode.runpod)
        self._managed_button.setChecked(mode is ServerMode.managed)
        self._external_button.setChecked(mode is ServerMode.external)

    def update_status(self, state: ConnectionState, server_state: ServerState):
        self._runpod_button.status = "not_started"
        self._external_button.status = "not_connected"
        match server_state:
            case ServerState.not_installed:
                self._managed_button.status = "not_installed"
            case ServerState.stopped:
                self._managed_button.status = "not_running"
            case _:
                self._managed_button.status = "not_connected"

        match self.mode, state, server_state:
            case ServerMode.runpod, ConnectionState.pod_starting | ConnectionState.connecting, _:
                self._runpod_button.status = "connecting"
            case ServerMode.runpod, ConnectionState.discover_models, _:
                self._runpod_button.status = "connecting"
            case ServerMode.runpod, ConnectionState.connected, _:
                self._runpod_button.status = "connected"
            case ServerMode.runpod, ConnectionState.error, _:
                self._runpod_button.status = "error"
            case ServerMode.managed, _, ServerState.starting:
                self._managed_button.status = "connecting"
            case ServerMode.managed, ConnectionState.connecting, _:
                self._managed_button.status = "connecting"
            case ServerMode.managed, ConnectionState.connected, ServerState.running:
                self._managed_button.status = "connected"
            case ServerMode.managed, ConnectionState.error, _:
                self._managed_button.status = "error"
            case ServerMode.external, ConnectionState.disconnected, _:
                self._external_button.status = "disconnected"
            case ServerMode.external, ConnectionState.connecting, _:
                self._external_button.status = "connecting"
            case ServerMode.external, ConnectionState.connected, _:
                self._external_button.status = "connected"
            case ServerMode.external, ConnectionState.error, _:
                self._external_button.status = "error"


class ConnectionSettings(SettingsTab):
    def __init__(self, server: Server):
        super().__init__(_("Server Configuration"))
        self._server = server

        self._server_mode = ServerModeSelect(self)
        self._server_mode.changed.connect(self._change_server_mode)

        self._setup_widget = InitialSetupWidget(self)
        self._runpod_widget = RunpodWidget(self)
        self._server_widget = ServerWidget(server, self)
        self._connection_widget = QWidget(self)
        self._server_stack = QStackedWidget(self)
        self._server_stack.addWidget(self._setup_widget)
        self._server_stack.addWidget(self._runpod_widget)
        self._server_stack.addWidget(self._server_widget)
        self._server_stack.addWidget(self._connection_widget)

        connection_layout = QVBoxLayout()
        connection_layout.setContentsMargins(0, 0, 0, 0)
        self._connection_widget.setLayout(connection_layout)

        add_header(connection_layout, Settings._server_url)
        server_layout = QHBoxLayout()
        self._server_url = QLineEdit(self._connection_widget)
        self._server_url.textChanged.connect(self.write)
        server_layout.addWidget(self._server_url)
        self._connect_button = QPushButton(_("Connect"), self._connection_widget)
        self._connect_button.clicked.connect(self._connect)
        server_layout.addWidget(self._connect_button)
        connection_layout.addLayout(server_layout)

        self._connection_status = QLabel(self._connection_widget)
        self._supported_workloads = QLabel(self._connection_widget)
        self._supported_workloads.setWordWrap(True)
        self._supported_workloads.setTextFormat(Qt.TextFormat.RichText)
        self._supported_workloads.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self._supported_workloads.setOpenExternalLinks(True)

        anchor = _("View log files")
        open_log_button = QLabel(f"<a href='file://{util.log_dir}'>{anchor}</a>", self)
        open_log_button.setToolTip(str(util.log_dir))
        open_log_button.linkActivated.connect(self._open_logs)

        status_layout = QHBoxLayout()
        status_layout.addWidget(self._connection_status)
        status_layout.addWidget(open_log_button, alignment=Qt.AlignmentFlag.AlignRight)

        connection_layout.addLayout(status_layout)
        connection_layout.addWidget(self._supported_workloads)
        connection_layout.addStretch()

        self._layout.addWidget(self._server_mode)
        self._layout.addWidget(self._server_stack)

        self.update_server_status()
        self._update_server_mode(settings.server_mode)

        root.connection.state_changed.connect(self.update_server_status)
        root.connection.error_changed.connect(self.update_server_status)
        root.connection.progress_changed.connect(self.update_server_status)
        self._setup_widget.finished.connect(self._setup_finished)
        self._server_widget.state_changed.connect(self.update_server_status)

    def _setup_finished(self, mode: ServerMode):
        self._server_mode.mode = mode
        self._update_server_mode(mode)

    def _update_server_mode(self, mode: ServerMode):
        self._server_mode.setVisible(mode is not ServerMode.undefined)
        widget = {
            ServerMode.runpod: self._runpod_widget,
            ServerMode.managed: self._server_widget,
            ServerMode.external: self._connection_widget,
            ServerMode.undefined: self._setup_widget,
        }[mode]
        self._server_stack.setCurrentWidget(widget)

    def update_ui(self):
        self._server_widget.update_ui()

    def _read(self):
        self._server_mode.mode = settings.server_mode
        self._server_mode.update_status(root.connection.state, self._server.state)
        self._update_server_mode(settings.server_mode)
        self._server_url.setText(settings.server_url)

    def _write(self):
        settings.server_mode = self._server_mode.mode
        settings.server_url = self._server_url.text()

    def _change_server_mode(self):
        self._update_server_mode(self._server_mode.mode)
        self.write()

    def _connect(self):
        if client := root.connection.create_client(settings):
            root.connection.connect(client)

    def update_server_status(self):
        connection = root.connection
        self._server_mode.update_status(connection.state, self._server.state)
        self._runpod_widget.update_state()
        self._connect_button.setEnabled(True)
        if connection.state == ConnectionState.connected:
            self._connection_status.setText(_("Connected"))
            self._connection_status.setStyleSheet(f"color: {green}; font-weight:bold")
        elif connection.state == ConnectionState.connecting:
            self._connection_status.setText(_("Connecting"))
            self._connection_status.setStyleSheet(f"color: {yellow}; font-weight:bold")
            self._connect_button.setEnabled(False)
        elif connection.state == ConnectionState.discover_models:
            progress = f" ({connection.progress[0]}/{connection.progress[1]})"
            self._connection_status.setText(_("Discovering models") + progress)
            self._connection_status.setStyleSheet(f"color: {yellow}; font-weight:bold")
            self._connect_button.setEnabled(False)
        elif connection.state == ConnectionState.disconnected:
            self._connection_status.setText(_("Disconnected"))
            self._connection_status.setStyleSheet(f"color: {grey}; font-style:italic")
        elif connection.state == ConnectionState.error:
            msg = connection.error.removeprefix("Error: ") if connection.error else "Unknown error"
            self._connection_status.setText("<b>" + _("Error") + f"</b>: {msg}")
            self._connection_status.setStyleSheet(f"color: {red};")

        self._supported_workloads.clear()
        if connection.state in [ConnectionState.connected, ConnectionState.error]:
            if connection.missing_resources is not None:
                self._show_missing_resources(connection.missing_resources, connection.state)

    def _show_missing_resources(self, res: MissingResources, state: ConnectionState):
        def model_name(id: ResourceId, with_file=False):
            if res := resources.find_resource(id):
                if with_file:
                    return f"{res.name} ({', '.join(f.name for f in res.files)})"
                return res.name
            if isinstance(id.identifier, str):
                return f"{id.kind.value} {id.identifier}"
            return f"{id.kind.value} {id.identifier.value}"

        text = ""
        if isinstance(res.missing, list):
            text = (
                _("The following ComfyUI custom nodes are missing or too old")
                + ":<ul>"
                + "\n".join(f"<li>{p.name} <a href='{p.url}'>{p.url}</a></li>" for p in res.missing)
                + "</ul>"
                + _(
                    "Please install or update the custom node package, then restart the server and try again."
                )
                + _("If nodes are still missing, check the ComfyUI output at startup for errors.")
                + "<br>"
            )
        else:
            basic = [m for lst in res.missing.values() for m in lst if m.arch is Arch.all]
            basic = util.unique(basic, key=lambda m: m.string)
            if len(basic) > 0:
                text = _("Missing common models (required)") + ":\n<ul>"
                text += "\n".join(f"<li>{model_name(m, True)}</li>" for m in basic)
                text += "</ul>"
            text += _("Detected workloads for the following base models:") + "\n<ul>"
            for arch, missing in res.missing.items():
                if arch in [Arch.all, Arch.illu_v]:
                    continue
                text += f"<li><b>{arch.value}</b>: "
                if len(missing) == 0:
                    text += _("supported")
                else:
                    names = [model_name(m) for m in missing if m.arch is arch]
                    if len(names) > 0:
                        text += _("missing") + " " + ", ".join(names)
                    else:
                        text += _("models found")
                text += "</li>"
            text += "</ul>"

        link = "<a href='https://github.com/SHoogstad/diffusion-for-krita#comfyui-setup'>Custom ComfyUI Setup</a>"
        text += _(
            "See {link} for required models.<br>Check the client.log file for more details.",
            link=link,
        )
        style = "" if state is ConnectionState.error else f"color: {grey};"
        self._supported_workloads.setStyleSheet(style)
        self._supported_workloads.setText(text)

    def _open_logs(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(util.log_dir)))


class DiffusionSettings(SettingsTab):
    def __init__(self):
        super().__init__(_("Diffusion Settings"))

        S = Settings
        self.add("selection_feather", SliderSetting(S._selection_feather, self, 0, 25, suffix="%"))
        self.add("selection_blend", SliderSetting(S._selection_blend, self, 0, 100, suffix=" px"))
        self.add("selection_padding", SliderSetting(S._selection_padding, self, 0, 25, suffix=" %"))
        self.add("color_match", SwitchSetting(S._color_match, parent=self))
        self.add("nsfw_filter", ComboBoxSetting(S._nsfw_filter, parent=self))

        nsfw_settings = [(_("Disabled"), 0.0), (_("Basic"), 0.65), (_("Strict"), 0.8)]
        self._widgets["nsfw_filter"].set_items(nsfw_settings)
        DiffusionSettings._warning_shown = self._warning_shown or settings.nsfw_filter > 0

        self._layout.addStretch()

    _warning_shown = False

    def _write(self):
        if self._widgets["nsfw_filter"].value > 0 and not self._warning_shown:
            DiffusionSettings._warning_shown = True
            QMessageBox.warning(
                self,
                _("NSFW Filter Warning"),
                _(
                    "The NSFW filter is a basic tool to exclude explicit content from generated images. It is NOT a guarantee and may not catch all inappropriate content. Please use responsibly and always review the generated images."
                ),
            )


class InterfaceSettings(SettingsTab):
    def __init__(self):
        super().__init__(_("Interface Settings"))

        S = Settings
        self.add("language", ComboBoxSetting(S._language, parent=self))
        self.add("prompt_translation", ComboBoxSetting(S._prompt_translation, parent=self))
        self.add(
            "prompt_line_count",
            SpinBoxSetting(S._prompt_line_count, self, 1, prompt_max_line_count),
        )
        self.add(
            "show_negative_prompt",
            SwitchSetting(S._show_negative_prompt, (_("Show"), _("Hide")), self),
        )
        self.add("show_steps", SwitchSetting(S._show_steps, parent=self))
        self.add("recent_styles_count", SpinBoxSetting(S._recent_styles_count, self, 0, 10))

        self.add("tag_files", FileListSetting(S._tag_files, files=self._tag_files(), parent=self))
        tag_files = cast(FileListSetting, self._widgets["tag_files"])
        self._layout.addWidget(tag_files.list_widget)
        tag_files.add_button(
            Krita.instance().icon("reload-preset"),
            _("Look for new tag files"),
            self._update_tag_files,
        )
        tag_files.add_button(
            Krita.instance().icon("document-open"),
            _("Open folder where custom tag files can be placed"),
            self._open_tag_folder,
        )

        self.add(
            "generation_finished_action",
            ComboBoxSetting(S._generation_finished_action, parent=self),
        )
        self.add("apply_behavior", ComboBoxSetting(S._apply_behavior, parent=self))
        self.add("apply_region_behavior", ComboBoxSetting(S._apply_region_behavior, parent=self))
        self.add("apply_behavior_live", ComboBoxSetting(S._apply_behavior_live, parent=self))
        self.add(
            "apply_region_behavior_live",
            ComboBoxSetting(S._apply_region_behavior_live, parent=self),
        )
        self.add("new_seed_after_apply", SwitchSetting(S._new_seed_after_apply, parent=self))
        self.add("save_image_format", ComboBoxSetting(S._save_image_format, parent=self))
        self.add("save_image_metadata", SwitchSetting(S._save_image_metadata, parent=self))
        self.add("debug_dump_workflow", SwitchSetting(S._debug_dump_workflow, parent=self))

        self._widgets["save_image_format"].value_changed.connect(self._update_image_format_widgets)

        languages = [(lang.name, lang.id) for lang in Localization.available]
        self._widgets["language"].set_items(languages)
        self.update_translation(root.connection.client_if_connected)

        for w in ["apply_region_behavior", "apply_region_behavior_live"]:
            self._widgets[w].show_label = False

        self._layout.addStretch()

    def read(self):
        super().read()
        self._update_image_format_widgets()

    def _tag_files(self) -> list[str]:
        plugin_tags_path = util.plugin_dir / "tags"
        user_tags_path = util.user_data_dir / "tags"
        files = set()
        files.update(path.stem for path in plugin_tags_path.glob("*.csv"))
        files.update(path.stem for path in user_tags_path.glob("*.csv"))

        return list(files)

    def _update_tag_files(self):
        tag_files = cast(FileListSetting, self._widgets["tag_files"])
        tag_files.reset_files(self._tag_files())

    def _open_tag_folder(self):
        user_tag_folder = util.user_data_dir / "tags"
        user_tag_folder.mkdir(exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(user_tag_folder)))

    def update_translation(self, client: Client | None):
        translation = cast(ComboBoxSetting, self._widgets["prompt_translation"])
        languages = [("Disabled", "")]
        if client:
            languages += [(lang.name, lang.code) for lang in client.features.languages]
        translation.enabled = client is not None
        translation.set_items(languages)
        self.read()

    def _update_image_format_widgets(self):
        fmt: ImageFileFormat = self._widgets["save_image_format"].value
        self._widgets["save_image_metadata"].enabled = fmt.extension == "png"


class HistorySizeWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, maximum: int, step: int, parent=None):
        super().__init__(parent)

        self._history_size = QSpinBox(self)
        self._history_size.setMinimum(5)
        self._history_size.setMaximum(maximum)
        self._history_size.setSingleStep(step)
        self._history_size.setSuffix(" MB")
        self._history_size.valueChanged.connect(self._change_value)

        self._history_usage = QLabel(self)
        self._history_usage.setStyleSheet(f"font-style:italic; color: {green};")

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._history_size)
        layout.addWidget(self._history_usage)
        self.setLayout(layout)

    def _change_value(self):
        self.value_changed.emit()

    @property
    def value(self):
        return self._history_size.value()

    @value.setter
    def value(self, v):
        self._history_size.setValue(v)

    def update_usage(self, usage: float):
        self._history_usage.setText(_("Currently using") + f" {usage:.1f} MB")


class PerformanceSettings(SettingsTab):
    def __init__(self):
        super().__init__(_("Performance Settings"))

        add_header(self._layout, Settings._history_size)
        self._history_size = HistorySizeWidget(maximum=20000, step=100, parent=self)
        self._history_size.value_changed.connect(self.write)
        self._layout.addWidget(self._history_size)

        add_header(self._layout, Settings._history_storage)
        self._history_storage = HistorySizeWidget(maximum=2000, step=5, parent=self)
        self._history_storage.value_changed.connect(self.write)
        self._layout.addWidget(self._history_storage)

        add_header(self._layout, Settings._performance_preset)
        self._device_info = QLabel(self)
        self._device_info.setStyleSheet("font-style:italic")
        self._layout.addWidget(self._device_info)

        self._performance_preset = QComboBox(self)
        for preset in PerformancePreset:
            self._performance_preset.addItem(preset.value)
        self._performance_preset.currentIndexChanged.connect(self._change_performance_preset)
        self._layout.addWidget(self._performance_preset, alignment=Qt.AlignmentFlag.AlignLeft)

        self._advanced = QWidget(self)
        self._advanced.setEnabled(settings.performance_preset is PerformancePreset.custom)
        self._layout.addWidget(self._advanced)
        advanced_layout = QVBoxLayout()
        advanced_layout.setContentsMargins(8, 0, 0, 4)
        self._advanced.setLayout(advanced_layout)

        self._batch_size = SliderSetting(Settings._batch_size, self._advanced, 1, 16)
        self._batch_size.value_changed.connect(self.write)
        advanced_layout.addWidget(self._batch_size)

        self._resolution_multiplier = SliderSetting(
            Settings._resolution_multiplier, self._advanced, 0.3, 1.5, suffix="x", decimals=1
        )
        self._resolution_multiplier.value_changed.connect(self.write)
        advanced_layout.addWidget(self._resolution_multiplier)

        self._max_pixel_count = SpinBoxSetting(
            Settings._max_pixel_count, self._advanced, 1, 99, 1, " MP"
        )
        self._max_pixel_count.value_changed.connect(self.write)
        advanced_layout.addWidget(self._max_pixel_count)

        self._tiled_vae = SwitchSetting(
            Settings._tiled_vae, text=(_("Always"), _("Automatic")), parent=self._advanced
        )
        self._tiled_vae.value_changed.connect(self.write)
        advanced_layout.addWidget(self._tiled_vae)

        self._dynamic_caching = SwitchSetting(Settings._dynamic_caching, parent=self)
        self._dynamic_caching.value_changed.connect(self.write)
        self._layout.addWidget(self._dynamic_caching)

        self._multi_threading = SwitchSetting(Settings._multi_threading, parent=self)
        self._multi_threading.value_changed.connect(self.write)
        self._layout.addWidget(self._multi_threading)

        self._layout.addStretch()

    def _change_performance_preset(self, index):
        self.write()
        is_custom = settings.performance_preset is PerformancePreset.custom
        self._advanced.setEnabled(is_custom)
        if (
            settings.performance_preset is PerformancePreset.auto
            and root.connection.state is ConnectionState.connected
        ):
            apply_performance_preset(settings, root.connection.client.device_info)
        if not is_custom:
            self.read()

    def update_client_info(self):
        if root.connection.state is ConnectionState.connected:
            client = root.connection.client
            self._device_info.setText(
                _("Device")
                + f": [{client.device_info.type.upper()}] {client.device_info.name} ({client.device_info.vram} GB)"
            )

    def _read(self):
        self._history_size.value = settings.history_size
        self._history_size.update_usage(root.active_model.jobs.memory_usage)
        self._history_storage.value = settings.history_storage
        self._history_storage.update_usage(root.get_active_model_used_storage() / (1024**2))
        self._multi_threading.value = settings.multi_threading
        self._batch_size.value = settings.batch_size
        self._performance_preset.setCurrentIndex(
            list(PerformancePreset).index(settings.performance_preset)
        )
        self._resolution_multiplier.value = settings.resolution_multiplier
        self._max_pixel_count.value = settings.max_pixel_count
        self._tiled_vae.value = settings.tiled_vae
        self._dynamic_caching.value = settings.dynamic_caching
        self.update_client_info()

    def _write(self):
        settings.history_size = self._history_size.value
        settings.history_storage = self._history_storage.value
        settings.multi_threading = self._multi_threading.value
        settings.batch_size = int(self._batch_size.value)
        settings.resolution_multiplier = self._resolution_multiplier.value
        settings.max_pixel_count = self._max_pixel_count.value
        settings.tiled_vae = self._tiled_vae.value
        settings.performance_preset = list(PerformancePreset)[
            self._performance_preset.currentIndex()
        ]
        settings.dynamic_caching = self._dynamic_caching.value


class AboutSettings(SettingsTab):
    def __init__(self):
        super().__init__(_("Plugin Information and Updates"))

        large = self.font()
        large.setPointSize(large.pointSize() + 2)

        extra_large = self.font()
        extra_large.setPointSize(extra_large.pointSize() + 4)

        bold = self.font()
        bold.setBold(True)

        italic = self.font()
        italic.setItalic(True)

        header_layout = QHBoxLayout()
        header_logo = QLabel(self)
        font_height = QFontMetrics(extra_large).height() + 4
        header_logo.setPixmap(logo().scaled(font_height * 2, font_height * 2))
        header_logo.setMaximumSize(font_height * 2, font_height * 2)
        header_text = QLabel("Generative AI\nfor Krita", self)
        header_text.setFont(extra_large)
        header_layout.addWidget(header_logo)
        header_layout.addWidget(header_text)

        current_version_name = QLabel(_("Current version") + ":", self)
        current_version_value = QLabel(__release__, self)

        latest_version_name = QLabel(_("Latest version") + ":", self)
        self._latest_version_value = QLabel(self)
        self._latest_version_value.setFont(bold)

        self._update_error = QLabel(self)
        self._update_error.setFont(italic)

        self._update_checkbox = QCheckBox(_("Check for updates on startup"), self)
        self._update_checkbox.setChecked(settings.auto_update)
        self._update_checkbox.stateChanged.connect(self.write)

        self._prerelease_checkbox = QCheckBox(_("Include pre-releases"), self)
        self._prerelease_checkbox.setToolTip(Settings._auto_update_prerelease.desc)
        self._prerelease_checkbox.setChecked(settings.auto_update_prerelease)
        self._prerelease_checkbox.stateChanged.connect(self.write)

        self._check_button = QPushButton(_("Check for Updates"), self)
        self._check_button.setMinimumWidth(font_height * 6)
        self._check_button.clicked.connect(self._check_updates)

        self._update_button = QPushButton(_("Download and Install"), self)
        self._update_button.setMinimumWidth(font_height * 6)
        self._update_button.clicked.connect(self._run_update)

        sys_header = QLabel(_("System Information"), self)
        sys_header.setFont(large)
        sys_desc = QLabel(_("Please attach this information when reporting issues!"), self)
        sys_desc.setFont(italic)
        sys_button = QPushButton(_("Collect Diagnostics"), self)
        sys_button.setMinimumWidth(font_height * 6)
        sys_button.clicked.connect(self._collect_diagnostics)
        anchor = _("View log files")
        open_log_button = QLabel(f"<a href='file://{util.log_dir}'>{anchor}</a>", self)
        open_log_button.setToolTip(str(util.log_dir))
        open_log_button.linkActivated.connect(self._open_logs)

        doc_header = QLabel(_("Documentation and Support"), self)
        doc_header.setFont(large)

        doc_links = QLabel(_links_text, self)
        doc_links.setOpenExternalLinks(True)
        doc_contact = QLabel(_contact_text, self)
        doc_contact.setOpenExternalLinks(True)

        self._layout.addLayout(header_layout)
        self._layout.addSpacing(10)
        current_version_layout = QHBoxLayout()
        current_version_layout.addWidget(current_version_name)
        current_version_layout.addWidget(current_version_value)
        current_version_layout.addStretch()
        self._layout.addLayout(current_version_layout)
        latest_version_layout = QHBoxLayout()
        latest_version_layout.addWidget(latest_version_name)
        latest_version_layout.addWidget(self._latest_version_value)
        latest_version_layout.addStretch()
        self._layout.addLayout(latest_version_layout)
        self._layout.addWidget(self._update_error)
        self._layout.addWidget(self._update_checkbox)
        self._layout.addWidget(self._prerelease_checkbox)
        update_layout = QHBoxLayout()
        update_layout.addWidget(self._check_button)
        update_layout.addWidget(self._update_button)
        update_layout.addStretch()
        self._layout.addLayout(update_layout)
        self._layout.addSpacing(20)
        self._layout.addWidget(sys_header)
        self._layout.addWidget(sys_desc)
        self._layout.addWidget(sys_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self._layout.addWidget(open_log_button)
        self._layout.addSpacing(20)
        self._layout.addWidget(doc_header)
        self._layout.addSpacing(5)
        doc_layout = QHBoxLayout()
        doc_layout.addWidget(doc_links)
        doc_layout.addSpacing(40)
        doc_layout.addWidget(doc_contact)
        doc_layout.addStretch()
        self._layout.addLayout(doc_layout)
        self._layout.addStretch()

        root.auto_update.state_changed.connect(self._update_content)
        self._update_content()

    def _update_content(self):
        self._check_button.setEnabled(False)
        self._update_button.setEnabled(False)
        self._update_error.clear()

        au = root.auto_update
        match au.state:
            case UpdateState.unknown:
                self._latest_version_value.setText(_("Not checked"))
                self._check_button.setEnabled(True)
            case UpdateState.checking:
                self._latest_version_value.setText(_("Checking for updates..."))
            case UpdateState.latest:
                self._latest_version_value.setText(au.latest_version)
                self._check_button.setEnabled(True)
            case UpdateState.available:
                self._latest_version_value.setText(au.latest_version)
                self._check_button.setEnabled(True)
                self._update_button.setEnabled(True)
            case UpdateState.downloading:
                self._latest_version_value.setText(_("Downloading package..."))
            case UpdateState.installing:
                self._latest_version_value.setText(_("Installing new version..."))
            case UpdateState.failed_check:
                self._latest_version_value.setText(_("Unknown"))
                self._update_error.setText(au.error)
                self._check_button.setEnabled(True)
            case UpdateState.failed_update:
                self._latest_version_value.setText(_("Update failed"))
                self._update_error.setText(au.error)
                self._check_button.setEnabled(True)
                self._update_button.setEnabled(True)
            case UpdateState.restart_required:
                self._latest_version_value.setText(
                    _("Please restart Krita to complete the update!")
                )

    def _check_updates(self):
        root.auto_update.check()

    def _run_update(self):
        root.auto_update.run()

    def _read(self):
        self._update_checkbox.setChecked(settings.auto_update)
        self._prerelease_checkbox.setChecked(settings.auto_update_prerelease)

    def _write(self):
        settings.auto_update = self._update_checkbox.isChecked()
        settings.auto_update_prerelease = self._prerelease_checkbox.isChecked()

    def _collect_diagnostics(self):
        diagnostics = collect_diagnostics()
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(diagnostics)

        window = QDialog(self)
        window.setWindowTitle(_("Diagnostics Information"))
        layout = QVBoxLayout()
        text = QTextEdit(window)
        text.setReadOnly(True)
        text.setText(diagnostics)
        text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.SmallestReadableFont))
        if clipboard is not None:
            msg = _("System information has been copied to the clipboard.")
            layout.addWidget(QLabel("✔️ " + msg))
        layout.addSpacing(6)
        layout.addWidget(text)
        window.setLayout(layout)
        window.resize(min(self.width(), 800), 640)
        window.exec()

    def _open_logs(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(util.log_dir)))


_links_text = """
<a href='https://github.com/SHoogstad/diffusion-for-krita'>GitHub</a><br><br>
<a href='https://github.com/SHoogstad/diffusion-for-krita#readme'>Documentation</a>
"""

_contact_text = """
<a href='https://github.com/SHoogstad/diffusion-for-krita/issues'>Issues</a>
"""


class SettingsDialog(QDialog):
    _instance = None

    @classmethod
    def instance(cls) -> SettingsDialog:
        assert cls._instance is not None
        return cls._instance

    def __init__(self, server: Server):
        super().__init__()
        type(self)._instance = self

        self.setWindowTitle(_("Configure Diffusion for Krita"))
        self.setMinimumSize(QSize(960, 480))
        if screen := QGuiApplication.screenAt(QCursor.pos()):
            size = screen.availableSize()
            min_w = min(size.width(), QFontMetrics(self.font()).horizontalAdvance("M") * 100)
            self.resize(QSize(min_w, int(size.height() * 0.8)))

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.connection = ConnectionSettings(server)
        self.styles = StylePresets(server)
        self.diffusion = DiffusionSettings()
        self.interface = InterfaceSettings()
        self.performance = PerformanceSettings()
        self.about = AboutSettings()

        self._stack = QStackedWidget(self)
        self._list = QListWidget(self)
        self._list.setFixedWidth(120)

        def create_list_item(text: str, widget: QWidget):
            item = QListWidgetItem(text, self._list)
            item.setSizeHint(QSize(112, 24))
            self._stack.addWidget(widget)

        create_list_item(_("Connection"), self.connection)
        create_list_item(_("Styles"), self.styles)
        create_list_item(_("Diffusion"), self.diffusion)
        create_list_item(_("Interface"), self.interface)
        create_list_item(_("Performance"), self.performance)
        create_list_item(_("Plugin"), self.about)

        self._list.setCurrentRow(0)
        self._list.currentRowChanged.connect(self._change_page)
        layout.addWidget(self._list)

        inner = QVBoxLayout()
        layout.addLayout(inner)
        inner.addWidget(self._stack)
        inner.addSpacing(6)

        self._restore_button = QPushButton(_("Restore Defaults"), self)
        self._restore_button.clicked.connect(self.restore_defaults)

        version_label = QLabel(_("Plugin version") + f": {__release__}", self)
        version_label.setStyleSheet(f"font-style:italic; color: {grey};")

        anchor = _("Open Settings folder")
        self._open_folder_link = QLabel(f"<a href='file://{util.user_data_dir}'>{anchor}</a>", self)
        self._open_folder_link.linkActivated.connect(self._open_settings_folder)
        self._open_folder_link.setToolTip(str(util.user_data_dir))

        self._close_button = QPushButton(_("Ok"), self)
        self._close_button.clicked.connect(self._close)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self._restore_button)
        button_layout.addStretch()
        button_layout.addWidget(version_label)
        button_layout.addStretch()
        button_layout.addWidget(self._open_folder_link)
        button_layout.addSpacing(8)
        button_layout.addWidget(self._close_button)
        inner.addLayout(button_layout)

        root.connection.state_changed.connect(self._update_connection)
        root.connection.models_changed.connect(self.styles.update_model_lists)

    def read(self):
        self.connection.read()
        self.styles.read()
        self.diffusion.read()
        self.interface.read()
        self.performance.read()
        self.about.read()

    def restore_defaults(self):
        settings.restore()
        settings.save()
        self.read()

    def show(self, style: Style | None = None):
        self.read()
        self.connection.update_ui()
        super().show()

        if style:
            self._list.setCurrentRow(1)
            self.styles.current_style = style
        self._close_button.setFocus()

    def _change_page(self, index):
        self._stack.setCurrentIndex(index)

    def _update_connection(self):
        self.connection.update_server_status()
        if root.connection.state is ConnectionState.connected:
            self.interface.update_translation(root.connection.client)
            self.performance.update_client_info()

    def _open_settings_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(util.user_data_dir)))

    def _close(self):
        _ = self.close()
