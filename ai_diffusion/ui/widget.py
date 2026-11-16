from __future__ import annotations

from collections.abc import Callable
from itertools import chain
from typing import Any, ClassVar, cast

from ..backend.client import filter_supported_styles, resolve_arch
from ..backend.workflow import apply_strength, snap_to_percent
from ..localization import translate as _
from ..model.connection import ConnectionState
from ..model.jobs import JobKind, JobState
from ..model.model import (
    DocumentModel,
    Error,
    ErrorKind,
    ProgressKind,
    QueueMode,
    SamplingQuality,
    Workspace,
    no_error,
)
from ..model.properties import Bind, Binding, bind, bind_combo
from ..model.root import root
from ..qt.QtCore import QEvent, QMetaObject, QPointF, QRectF, QSize, Qt, pyqtSignal
from ..qt.QtGui import (
    QCloseEvent,
    QColor,
    QFontMetrics,
    QGuiApplication,
    QIcon,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPaintDevice,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPalette,
    QPen,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
)
from ..qt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollBar,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QStyleOption,
    QToolButton,
    QWidget,
    QWidgetAction,
)
from ..settings import Settings, settings
from ..style import Style, Styles, sort_recent_styles
from ..text import (
    char16_index_to_str_index,
    char16_len,
    edit_attention,
    pattern_comment,
    pattern_layer,
    pattern_lora,
    pattern_weight_expr,
    pattern_wildcard,
    select_on_cursor_pos,
    str_index_to_char16_index,
)
from ..util import ensure
from . import actions, theme
from .autocomplete import PromptAutoComplete
from .slider_spinbox import DoubleSliderSpinBox
from .theme import SignalBlocker


class QueuePopup(QMenu):
    def __init__(self, supports_batch=True, parent: QWidget | None = None):
        super().__init__(parent)
        self._connections: list[QMetaObject.Connection | Binding] = []

        palette = self.palette()
        self.setObjectName("QueuePopup")
        self.setStyleSheet(
            theme.docker_stylesheet
            + f"""
            QWidget#QueuePopup {{
                background-color: {palette.window().color().name()};
                border: 1px solid {theme.accent_cool}; padding: 6px;
            }}"""
        )

        self._layout = QGridLayout()
        self.setLayout(self._layout)

        counts_label = QLabel(_("Jobs"), self)
        counts_layout = QHBoxLayout()
        counts_layout.setContentsMargins(0, 0, 0, 0)
        counts_layout.addWidget(QLabel(_("Document:"), self))
        self._counts_document = QLabel("0", self)
        self._counts_document.setStyleSheet(f"color: {theme.highlight}; font-weight: bold;")
        counts_layout.addWidget(self._counts_document)
        counts_layout.addWidget(QLabel(_("Total:"), self))
        counts_layout.addSpacing(4)
        self._counts_total = QLabel("0", self)
        self._counts_total.setStyleSheet(f"color: {theme.highlight}; font-weight: bold;")
        counts_layout.addWidget(self._counts_total)
        counts_layout.addStretch()
        self._layout.addWidget(counts_label, 0, 0)
        self._layout.addLayout(counts_layout, 0, 1)

        batch_label = QLabel(_("Batches"), self)
        batch_label.setVisible(supports_batch)
        self._layout.addWidget(batch_label, 1, 0)
        batch_layout = QHBoxLayout()
        self._batch_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._batch_slider.setMinimum(1)
        self._batch_slider.setMaximum(10)
        self._batch_slider.setSingleStep(1)
        self._batch_slider.setPageStep(1)
        self._batch_slider.setVisible(supports_batch)
        self._batch_slider.setToolTip(_("Number of jobs to enqueue at once"))
        self._batch_label = QLabel("1", self)
        self._batch_label.setVisible(supports_batch)
        batch_layout.addWidget(self._batch_slider)
        batch_layout.addWidget(self._batch_label)
        self._layout.addLayout(batch_layout, 1, 1)

        self._seed_label = QLabel(_("Seed"), self)
        self._layout.addWidget(self._seed_label, 2, 0)
        self._seed_input = QDoubleSpinBox(self)
        self._seed_check = QCheckBox(self)
        self._seed_check.setText(_("Fixed"))
        self._seed_input.setMinimum(0)
        self._seed_input.setMaximum(2**32 - 1)
        self._seed_input.setDecimals(0)
        self._seed_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._seed_input.setToolTip(
            _(
                "The seed controls the random part of the output. A fixed seed value will always produce the same result for the same inputs."
            )
        )
        self._randomize_seed = QToolButton(self)
        self._randomize_seed.setIcon(theme.icon("random"))
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(self._seed_check)
        seed_layout.addWidget(self._seed_input)
        seed_layout.addWidget(self._randomize_seed)
        self._layout.addLayout(seed_layout, 2, 1)

        resolution_multiplier_label = QLabel(_("Resolution"), self)
        self._resolution_multiplier_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._resolution_multiplier_slider.setRange(3, 15)
        self._resolution_multiplier_slider.setValue(10)
        self._resolution_multiplier_slider.setSingleStep(1)
        self._resolution_multiplier_slider.setPageStep(1)
        self._resolution_multiplier_slider.setToolTip(Settings._resolution_multiplier.desc)
        self._resolution_multiplier_slider.valueChanged.connect(self._set_resolution_multiplier)
        self._resolution_multiplier_display = QLabel("1.0 x", self)
        self._resolution_multiplier_display.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._resolution_multiplier_display.setMinimumWidth(20)
        resolution_multiplier_layout = QHBoxLayout()
        resolution_multiplier_layout.addWidget(self._resolution_multiplier_slider)
        resolution_multiplier_layout.addWidget(self._resolution_multiplier_display)
        self._layout.addWidget(resolution_multiplier_label, 3, 0)
        self._layout.addLayout(resolution_multiplier_layout, 3, 1)

        enqueue_label = QLabel(_("Enqueue"), self)
        self._queue_mode_combo = QComboBox(self)
        self._queue_mode_combo.addItem(_("at the Back"), QueueMode.back)
        self._queue_mode_combo.addItem(_("in Front (new jobs first)"), QueueMode.front)
        self._queue_mode_combo.addItem(_("Replace Queue"), QueueMode.replace)
        self._layout.addWidget(enqueue_label, 4, 0)
        self._layout.addWidget(self._queue_mode_combo, 4, 1)

        cancel_label = QLabel(_("Cancel"), self)
        self._layout.addWidget(cancel_label, 5, 0)
        self._cancel_active = self._create_cancel_button(_("Active"), actions.cancel_active)
        self._cancel_queued = self._create_cancel_button(_("Queued"), actions.cancel_queued)
        self._cancel_all = self._create_cancel_button(_("All"), actions.cancel_all)
        cancel_layout = QHBoxLayout()
        cancel_layout.addWidget(self._cancel_active)
        cancel_layout.addWidget(self._cancel_queued)
        cancel_layout.addWidget(self._cancel_all)
        self._layout.addLayout(cancel_layout, 5, 1)

        self._model = root.active_model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        Binding.disconnect_all(self._connections)
        self._model = model
        self._randomize_seed.setEnabled(model.fixed_seed)
        self._seed_input.setValue(model.seed)
        self._seed_input.setEnabled(model.fixed_seed)
        self._batch_label.setText(str(model.batch_count))
        self._connections = [
            bind(model, "batch_count", self._batch_slider, "value"),
            model.batch_count_changed.connect(lambda v: self._batch_label.setText(str(v))),
            model.seed_changed.connect(lambda: self._seed_input.setValue(self._model.seed)),
            self._seed_input.valueChanged.connect(lambda v: setattr(self._model, "seed", int(v))),
            bind(model, "fixed_seed", self._seed_check, "checked", Bind.one_way),
            self._seed_check.toggled.connect(lambda v: setattr(self._model, "fixed_seed", v)),
            model.fixed_seed_changed.connect(self._seed_input.setEnabled),
            model.fixed_seed_changed.connect(self._randomize_seed.setEnabled),
            self._randomize_seed.clicked.connect(model.generate_seed),
            model.resolution_multiplier_changed.connect(self._update_resolution_multiplier),
            bind_combo(model, "queue_mode", self._queue_mode_combo),
            model.jobs.count_changed.connect(self._update_job_count),
        ]
        self._update_job_count()

    def _create_cancel_button(self, name: str, action: Callable[[], None]):
        button = QToolButton(self)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setText(name)
        button.setIcon(theme.icon("cancel"))
        button.setEnabled(False)
        button.clicked.connect(action)
        return button

    def _update_job_count(self):
        has_active = self._model.jobs.any_executing()
        n_queued = self._model.jobs.count(JobState.queued)
        n_total = sum(m.jobs.count(JobState.queued) for m in root.models)
        self._counts_document.setText(str(n_queued + (1 if has_active else 0)))
        self._counts_total.setText(str(n_total + (1 if has_active else 0)))
        self._cancel_active.setEnabled(has_active)
        self._cancel_queued.setEnabled(n_queued > 0)
        self._cancel_all.setEnabled(has_active or n_queued > 0)

    def _update_resolution_multiplier(self):
        slider_value = round(self.model.resolution_multiplier * 10)
        if self._resolution_multiplier_slider.value() != slider_value:
            self._resolution_multiplier_slider.setValue(slider_value)

    def _set_resolution_multiplier(self, value: int):
        self.model.resolution_multiplier = value / 10
        self._resolution_multiplier_display.setText(f"{(value / 10):.1f} x")

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        if parent := cast(QWidget, self.parent()):
            parent.close()
        return super().closeEvent(a0)


class QueueButton(QToolButton):
    def __init__(self, supports_batch=True, parent: QWidget | None = None):
        super().__init__(parent)
        self._model = root.active_model

        self._popup = QueuePopup(supports_batch)
        popup_action = QWidgetAction(self)
        popup_action.setDefaultWidget(self._popup)
        self.addAction(popup_action)

        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._connect_model()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        if self._model != model:
            Binding.disconnect_all(self._connections)
            self._model = model
            self._popup.model = model
            self._connect_model()

    def _connect_model(self):
        self._connections = [
            self._model.jobs.count_changed.connect(self._update),
            self._model.progress_kind_changed.connect(self._update),
        ]
        self._update()

    def _update(self):
        count = self._model.jobs.count(JobState.queued)
        queued_msg = _("{count} jobs queued.", count=count)
        cancel_msg = _("Click to cancel.")

        if self._model.progress_kind is ProgressKind.upload:
            self.setIcon(theme.icon("queue-upload"))
            self.setToolTip(_("Uploading models.") + f" {queued_msg} {cancel_msg}")
            count += 1
        elif self._model.jobs.any_executing():
            self.setIcon(theme.icon("queue-active"))
            if count > 0:
                self.setToolTip(_("Generating image.") + f" {queued_msg} {cancel_msg}")
            else:
                self.setToolTip(_("Generating image.") + f" {cancel_msg}")
            count += 1
        elif count > 0:
            self.setIcon(theme.icon("queue-waiting"))
            self.setToolTip(f"{queued_msg} {cancel_msg}")
        else:
            self.setIcon(theme.icon("queue-inactive"))
            self.setToolTip(_("Idle."))

        self.setText(f"{count} ")

    def sizeHint(self) -> QSize:
        original = super().sizeHint()
        width = original.height() * 0.75 + self.fontMetrics().horizontalAdvance(" 99 ") + 30
        return QSize(int(width), original.height())

    def paintEvent(self, a0):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        fm = self.fontMetrics()
        rect = QRectF(self.rect()).adjusted(0.5, 1.5, -0.5, -1.5)
        radius = rect.height() / 2
        is_busy = self.text().strip() not in ("", "0")
        border = QColor(theme.grey)
        border.setAlpha(110)
        if self.underMouse() and self.isEnabled():
            border = QColor(theme.accent_cool)
        elif is_busy:
            border = QColor(theme.accent)
        painter.setPen(QPen(border, 1))
        filled = is_busy or not theme.is_dark
        painter.setBrush(QColor(theme.base) if filled else Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, radius, radius)

        size = float(fm.height())
        ratio = self.devicePixelRatioF() or 1.0
        pixmap = self.icon().pixmap(int(size * ratio))
        x = rect.left() + radius * 0.7
        painter.drawPixmap(
            QRectF(x, (self.height() - size) / 2, size, size), pixmap, QRectF(pixmap.rect())
        )
        x += size + 4
        font = self.font()
        font.setBold(is_busy)
        painter.setFont(font)
        text_color = self.palette().color(QPalette.ColorRole.WindowText)
        painter.setPen(QColor(theme.accent) if is_busy else text_color)
        align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        painter.drawText(QRectF(x, 0, self.width() - x, self.height()), align, self.text().strip())
        # chevron: the pill opens the queue menu
        cx, cy = rect.right() - radius * 0.7 - 3, self.height() / 2
        chevron = QPen(text_color, 1.4)
        chevron.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(chevron)
        painter.setOpacity(0.7)
        painter.drawLine(QPointF(cx - 3, cy - 1.5), QPointF(cx, cy + 1.5))
        painter.drawLine(QPointF(cx, cy + 1.5), QPointF(cx + 3, cy - 1.5))


class StyleSelectWidget(QWidget):
    value_changed = pyqtSignal(Style)
    quality_changed = pyqtSignal(SamplingQuality)

    def __init__(self, parent: QWidget | None, show_quality=False):
        super().__init__(parent)
        self._styles: list[Style] = []
        self._value = Styles.list().default

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._combo = QComboBox(self)
        theme.style_select(self._combo)
        self._combo.setIconSize(QSize(22, 22))
        self._combo.setToolTip(_("Style: the model and its settings used to generate"))
        self.update_styles()
        self._combo.currentIndexChanged.connect(self.change_style)
        layout.addWidget(self._combo, 3)

        if show_quality:
            self._quality_combo = QComboBox(self)
            theme.style_select(self._quality_combo)
            self._quality_combo.addItem(_("Fast"), SamplingQuality.fast.value)
            self._quality_combo.addItem(_("Quality"), SamplingQuality.quality.value)
            self._quality_combo.currentIndexChanged.connect(self.change_quality)
            layout.addWidget(self._quality_combo, 1)

        settings_btn = QToolButton(self)
        settings_btn.setIcon(theme.icon("settings"))
        settings_btn.setStyleSheet(theme.ghost_button_stylesheet)
        settings_btn.setToolTip(_("Edit this style"))
        settings_btn.setFixedSize(self._combo.sizeHint().height(), self._combo.sizeHint().height())
        settings_btn.clicked.connect(self.show_settings)
        layout.addWidget(settings_btn)

        Styles.list().changed.connect(self.update_styles)
        Styles.list().name_changed.connect(self.update_styles)
        root.connection.state_changed.connect(self.update_styles)
        settings.changed.connect(self._on_settings_changed)

    def update_styles(self):
        if root.connection.state is not ConnectionState.connected:
            return
        client = root.connection.client_if_connected
        filtered = filter_supported_styles(Styles.list().filtered(), client)
        recent, remaining = sort_recent_styles(
            filtered, settings.recent_styles, settings.recent_styles_count
        )
        if self._value not in chain(recent, remaining):
            recent.insert(0, self._value)
        self._styles = recent + remaining
        with SignalBlocker(self._combo):
            self._combo.clear()
            for style in recent:
                icon = theme.checkpoint_icon(resolve_arch(style, client))
                self._combo.addItem(icon, f"{style.name} ★", style.filename)
            if recent and remaining:
                self._combo.insertSeparator(len(recent))
            for style in remaining:
                icon = theme.checkpoint_icon(resolve_arch(style, client))
                self._combo.addItem(icon, style.name, style.filename)
            self._combo.setCurrentText(self._value.name)

    def change_style(self):
        filename = self._combo.currentData()
        if filename is None:
            return  # separator item selected
        style = next((s for s in self._styles if s.filename == filename), None)
        if style is None or style == self._value:
            return
        self._value = style
        self.value_changed.emit(style)

    def change_quality(self):
        quality = SamplingQuality(self._quality_combo.currentData())
        self.quality_changed.emit(quality)

    def _on_settings_changed(self, name: str, value: object):
        if "recent_styles" in name:
            self.update_styles()

    def show_settings(self):
        from .settings import SettingsDialog

        SettingsDialog.instance().show(self._value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, style: Style):
        if style != self._value:
            self._value = style
            if style not in self._styles:
                self.update_styles()
            else:
                idx = self._combo.findData(style.filename)
                self._combo.setCurrentIndex(idx)


class PromptHighlighter(QSyntaxHighlighter):
    def __init__(self, parent):
        super().__init__(parent)

        self._comment_fmt = QTextCharFormat()
        self._comment_fmt.setForeground(QColor(theme.grey))
        self._comment_fmt.setFontItalic(True)

        self._weight_fmt = QTextCharFormat()
        self._weight_fmt.setForeground(QColor(theme.red))

        self._keyword_fmt = QTextCharFormat()
        self._keyword_fmt.setForeground(QColor(theme.strong_highlight))

        self._grey_fmt = QTextCharFormat()
        self._grey_fmt.setForeground(QColor(theme.grey))

    def highlightBlock(self, text: str | None):
        if text is None:
            return
        comment_start = len(text)
        m = pattern_comment.search(text)
        if m:
            comment_start = m.start()
            self.setFormat(m.start(), m.end() - m.start(), self._comment_fmt)

        for m in pattern_weight_expr.finditer(text):
            if m.start(1) < comment_start:
                self.setFormat(m.start(), 1, self._grey_fmt)
                self.setFormat(m.start(1), m.end(1) - m.start(1), self._weight_fmt)
                self.setFormat(m.end(1), 1, self._grey_fmt)

        for m in pattern_lora.finditer(text):
            if m.start() < comment_start:
                # keyword "lora" starts at m.start()+1 (after '<'), length 4
                self.setFormat(m.start(), 1, self._grey_fmt)
                self.setFormat(m.start() + 1, 4, self._keyword_fmt)
                self.setFormat(m.end(1), 1, self._grey_fmt)
                if m.group(2):
                    self.setFormat(m.start(2), m.end(2) - m.start(2), self._weight_fmt)
                    self.setFormat(m.end(2), 1, self._grey_fmt)

        for m in pattern_layer.finditer(text):
            if m.start() < comment_start:
                # keyword "layer" starts at m.start()+1 (after '<'), length 5
                self.setFormat(m.start(), 1, self._grey_fmt)
                self.setFormat(m.start() + 1, 5, self._keyword_fmt)
                self.setFormat(m.end(1), 1, self._grey_fmt)

        for m in pattern_wildcard.finditer(text):
            if m.start() < comment_start:
                wildcard_text = m.group(0)
                wildcard_start = m.start()
                self.setFormat(wildcard_start, 1, self._keyword_fmt)
                for i, char in enumerate(wildcard_text):
                    if char == "|":
                        self.setFormat(wildcard_start + i, 1, self._keyword_fmt)
                self.setFormat(wildcard_start + len(wildcard_text) - 1, 1, self._keyword_fmt)


class ResizeHandle(QWidget):
    """A small resize handle that appears at the bottom of the prompt widget."""

    handle_dragged = pyqtSignal(int)

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setFixedSize(22, 8)
        self._dragging = False

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        if ensure(a0).button() == Qt.MouseButton.LeftButton:
            self._dragging = True

    def mouseReleaseEvent(self, a0: QMouseEvent | None) -> None:
        self._dragging = False

    def mouseMoveEvent(self, a0: QMouseEvent | None) -> None:
        if not self._dragging:
            return
        y_pos = self.mapToParent(ensure(a0).pos()).y()
        self.handle_dragged.emit(y_pos)

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        if not self.isVisible():
            return
        painter = QPainter(self)
        painter.setPen(self.palette().color(QPalette.ColorRole.PlaceholderText).lighter(100))
        painter.setBrush(painter.pen().color())
        w, h = self.width(), self.height()
        for i, x in enumerate(range(2, w - 1, 3)):
            y = 2 * h // 3 if i % 2 == 0 else h // 3
            painter.drawEllipse(x - 1, y - 1, 2, 2)


class TextPromptWidget(QPlainTextEdit):
    activated = pyqtSignal()
    text_changed = pyqtSignal(str)
    handle_dragged = pyqtSignal(int)

    def __init__(self, line_count=2, is_negative=False, parent=None):
        super().__init__(parent)
        self._line_count = line_count
        self._is_negative = is_negative

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTabChangesFocus(True)
        self.setFrameStyle(QFrame.Shape.NoFrame)

        self._completer = PromptAutoComplete(self)
        self.textChanged.connect(self.notify_text_changed)

        self._highlighter = PromptHighlighter(self.document())
        self._resize_handle: ResizeHandle | None = None

        palette: QPalette = self.palette()
        self._base_color = palette.color(QPalette.ColorRole.Base)
        self.is_negative = is_negative
        self.line_count = line_count

    def event(self, e: QEvent | None):
        assert e is not None
        # Ctrl+Backspace should be handled by QPlainTextEdit, not Krita.
        if e.type() == QEvent.Type.ShortcutOverride:
            assert isinstance(e, QKeyEvent)
            if e.matches(QKeySequence.StandardKey.DeleteStartOfWord):
                e.accept()
        return super().event(e)

    def keyPressEvent(self, e: QKeyEvent | None):
        assert e is not None
        if self._completer.is_active and e.key() in PromptAutoComplete.action_keys:
            e.ignore()
            return

        self.handle_weight_adjustment(e)

        if e.key() == Qt.Key.Key_Return and e.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            self.activated.emit()
        else:
            super().keyPressEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._resize_handle:
            self._place_resize_handle()

    def focusOutEvent(self, e):
        super().focusOutEvent(e)
        if scroll := self.verticalScrollBar():
            scroll.triggerAction(QScrollBar.SliderAction.SliderToMinimum)

    def focusNextPrevChild(self, next):
        if self._completer.is_active:
            return False
        return super().focusNextPrevChild(next)

    def notify_text_changed(self):
        self._completer.check_completion()
        self.text_changed.emit(self.text)

    @property
    def text(self):
        return self.toPlainText()

    @text.setter
    def text(self, value: str):
        if value == self.text:
            return
        with SignalBlocker(self):  # avoid auto-completion on non-user input
            self.setPlainText(value)

    @property
    def is_resizable(self):
        return self._resize_handle is not None

    @is_resizable.setter
    def is_resizable(self, value: bool):
        if value and self._resize_handle is None:
            self._resize_handle = ResizeHandle(self)
            self._resize_handle.handle_dragged.connect(self.handle_dragged)
            self._place_resize_handle()
            self._resize_handle.show()
        if not value and self._resize_handle is not None:
            self._resize_handle.handle_dragged.disconnect(self.handle_dragged)
            self._resize_handle.deleteLater()
            self._resize_handle = None

    def _place_resize_handle(self):
        if self._resize_handle:
            rect = self.geometry()
            self._resize_handle.move(
                (rect.width() - self._resize_handle.width()) // 2,
                rect.height() - self._resize_handle.height(),
            )

    @property
    def line_count(self):
        return self._line_count

    @line_count.setter
    def line_count(self, value: int):
        self._line_count = value
        fm = QFontMetrics(ensure(self.document()).defaultFont())
        self.setFixedHeight(fm.lineSpacing() * value + 10)

    @property
    def is_negative(self):
        return self._is_negative

    @is_negative.setter
    def is_negative(self, value: bool):
        self._is_negative = value
        if not value:
            self.setPlaceholderText(_("Describe the content you want to see, or leave empty."))
        else:
            self.setPlaceholderText(_("Describe content you want to avoid."))

        if value:
            self.setContentsMargins(0, 2, 0, 2)
            self.setFrameStyle(QFrame.Shape.StyledPanel)
            self.setStyleSheet("QFrame { background: rgba(255, 0, 0, 15); }")
        else:
            self.setFrameStyle(QFrame.Shape.NoFrame)

    @property
    def has_focus(self):
        return self.hasFocus()

    @has_focus.setter
    def has_focus(self, value: bool):
        if value:
            self.setFocus()

    def move_cursor_to_end(self):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)

    def handle_weight_adjustment(self, event: QKeyEvent):
        """Handles Ctrl + (arrow key up / arrow key down) attention weight adjustment."""
        if event.key() in [Qt.Key.Key_Up, Qt.Key.Key_Down] and (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            cursor = self.textCursor()
            text = self.toPlainText()

            if cursor.hasSelection():
                start = char16_index_to_str_index(text, cursor.selectionStart())
                end = char16_index_to_str_index(text, cursor.selectionEnd())
            else:
                pos = char16_index_to_str_index(text, cursor.position())
                start, end = select_on_cursor_pos(text, pos)

            target_text = text[start:end]
            text_after_edit = edit_attention(target_text, event.key() == Qt.Key.Key_Up)
            text = text[:start] + text_after_edit + text[end:]
            self.setPlainText(text)
            start_c16 = str_index_to_char16_index(text, start)
            cursor = self.textCursor()
            cursor.setPosition(min(start_c16 + char16_len(text_after_edit), char16_len(text)))
            cursor.setPosition(min(start_c16, char16_len(text)), QTextCursor.MoveMode.KeepAnchor)
            self.setTextCursor(cursor)


class StrengthWidget(DoubleSliderSpinBox):
    value_changed = pyqtSignal(float)

    def __init__(self, range: tuple[float, float] = (0.01, 1.0), prefix=True):
        super().__init__()
        self._model: DocumentModel | None = None
        self._value = 100
        self._range = (range[0] * 100, range[1] * 100)

        self.setSoftMinimum(self._range[0])
        self.setSoftMaximum(self._range[1])
        self.setRange(min(self._range[0], 1), 100, 0)
        super().setValue(self._value)

        w = self.widget()  # the internal QDoubleSpinBox
        w.setSingleStep(5)
        if prefix:
            w.setPrefix(_("Strength") + ": ")
        w.setSuffix("%")
        if range[0] == 0:
            w.setSpecialValueText(_("Off"))

        w.valueChanged.connect(self.notify_changed)
        self.draggingFinished.connect(self._complete_drag)
        settings.changed.connect(self._update_suffix)

    def notify_changed(self, value: float):
        value = int(value)
        if self.isDragging():
            self._change_suffix(value)
            return

        step = 1 if value - self._value > 0 else -1
        if not self.isDragging() and self._model is not None:
            # keep going until we hit a new snap point
            current_point = self._nearest_snap_point(self._value)
            while self._nearest_snap_point(value) == current_point and value > 1:
                value += step
            value = self._nearest_snap_point(value)
        if self._update_value(value):
            self.value_changed.emit(self.value)

    def _complete_drag(self):
        value = int(self.widget().value())
        if settings.show_steps:
            value = self._nearest_snap_point(value)
        if self._update_value(value):
            self.value_changed.emit(self.value)

    def _nearest_snap_point(self, value: int) -> int:
        if self._model and (snap := _nearest_percent(self._model, value)):
            return snap
        return max(round(value / 5) * 5, int(self._range[0]))

    def _update_value(self, value: int):
        if value != self._value:
            self._value = value
            with SignalBlocker(self.widget()):
                super().setValue(value)
            self._update_suffix()
            return True
        return False

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        if self._model:
            self._model.style_changed.disconnect(self._update_suffix)
            self._model.edit_mode_changed.disconnect(self._update_suffix)
            self._model.animation.sampling_quality_changed.disconnect(self._update_suffix)
        self._model = model
        self._model.style_changed.connect(self._update_suffix)
        self._model.edit_mode_changed.connect(self._update_suffix)
        self._model.animation.sampling_quality_changed.connect(self._update_suffix)
        self._update_suffix()

    @property
    def value(self):
        return self._value / 100

    @value.setter
    def value(self, value: float):
        self.setValue(value)

    def _change_suffix(self, value: int):
        if not self._model or not settings.show_steps:
            self.widget().setSuffix("%")
            return

        steps, start_at_step = _apply_strength(self._model, value)
        self.widget().setSuffix(f"% - {steps - start_at_step}/{steps} steps")

    def _update_suffix(self):
        self._change_suffix(self._value)

    def setValue(self, value: float):
        self._update_value(round(value * 100))

    def setVisible(self, visible: bool):
        self.widget().setVisible(visible)

    def setEnabled(self, enabled: bool):
        self.widget().setEnabled(enabled)

    def setToolTip(self, tooltip: str):
        self.widget().setToolTip(tooltip)


def _get_steps(model: DocumentModel) -> tuple[int, int]:
    is_live = model.workspace is Workspace.live
    if model.workspace is Workspace.animation:
        is_live = model.animation.sampling_quality is SamplingQuality.fast
    return model.active_style.get_steps(is_live=is_live)


def _apply_strength(model: DocumentModel, strength_percent: int) -> tuple[int, int]:
    if strength_percent <= 0:
        return 0, 0
    min_steps, max_steps = _get_steps(model)
    strength = strength_percent / 100
    return apply_strength(strength, steps=max_steps, min_steps=min_steps)


def _nearest_percent(model: DocumentModel, strength_percent: int) -> int | None:
    _, max_steps = _get_steps(model)
    steps, start_at_step = _apply_strength(model, strength_percent)
    return snap_to_percent(steps, start_at_step, max_steps=max_steps)


class LayerCountWidget(QWidget):
    value_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 4

        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setMinimum(1)
        self._slider.setMaximum(10)
        self._slider.setValue(self._value)
        self._slider.setSingleStep(1)
        self._slider.valueChanged.connect(self._notify_changed)

        self._input = QSpinBox(self)
        self._input.setPrefix(_("Layers") + ": ")
        self._input.setMinimum(1)
        self._input.setMaximum(10)
        self._input.setValue(self._value)
        self._input.valueChanged.connect(self._notify_changed)

        self._layout.addWidget(self._slider)
        self._layout.addWidget(self._input)

    def _notify_changed(self, value: int):
        if value != self._value:
            self._value = value
            self._update()
            self.value_changed.emit(self._value)

    def _update(self):
        with SignalBlocker(self._slider), SignalBlocker(self._input):
            self._slider.setValue(self._value)
            self._input.setValue(self._value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: int):
        if value == self._value:
            return
        self._value = value
        self._update()


class WorkspaceTab(QToolButton):
    """One workspace in the tab strip. The active tab shows its name, the others only an icon."""

    def __init__(self, workspace: Workspace, text: str, parent: QWidget):
        super().__init__(parent)
        self.workspace = workspace
        self._text = text
        font = self.font()
        font.setBold(True)
        self.setFont(font)
        self.setCheckable(True)
        self.setToolTip(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def _icon_size(self):
        return self.fontMetrics().height() + 2

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        height = fm.height() + 14
        width = self._icon_size() + 14
        if self.isChecked():
            width += fm.horizontalAdvance(self._text) + 6
        return QSize(width, height)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, a0):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        is_hover = self.underMouse() and self.isEnabled()
        painter.setPen(Qt.PenStyle.NoPen)
        if self.isChecked():
            painter.setBrush(QColor(theme.base))
            if not theme.is_dark:  # a white tab on a light window needs an edge
                painter.setPen(QPen(QColor(theme.line_base), 1))
            painter.drawRoundedRect(rect, theme.radius, theme.radius)
            painter.setPen(Qt.PenStyle.NoPen)
            bar = QRectF(rect.left() + 7, rect.bottom() - 5, rect.width() - 14, 4.5)
            theme.paint_flag_bar(painter, bar)
            painter.setPen(Qt.PenStyle.NoPen)
        elif is_hover:
            painter.setBrush(QColor(theme.line))
            painter.drawRoundedRect(rect, theme.radius, theme.radius)

        size = self._icon_size()
        icon_rect = QRectF(7, (self.height() - size) / 2 - 1, size, size)
        text_color = self.palette().color(QPalette.ColorRole.WindowText)
        rest = 0.55 if theme.is_dark else 0.72  # thin lines need more weight on light
        painter.setOpacity(1.0 if self.isChecked() else (0.95 if is_hover else rest))
        glyph_color = None if self.isChecked() else text_color  # active = brand gradient
        theme.paint_workspace_glyph(painter, icon_rect, self.workspace.name, glyph_color)
        if self.isChecked():
            painter.setOpacity(1.0)
            painter.setPen(self.palette().color(QPalette.ColorRole.WindowText))
            text_rect = QRectF(self.rect()).adjusted(7 + size + 6, -1, -4, -1)
            align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            painter.drawText(text_rect, align, self._text)


class WorkspaceSelectWidget(QWidget):
    """Tab strip to switch between workspaces. All of them are visible at once."""

    _icons: ClassVar[dict[Workspace, QIcon]] = {
        Workspace.generation: theme.icon("workspace-generation"),
        Workspace.upscaling: theme.icon("workspace-upscaling"),
        Workspace.live: theme.icon("workspace-live"),
        Workspace.animation: theme.icon("workspace-animation"),
        Workspace.custom: theme.icon("workspace-custom"),
    }

    _value = Workspace.generation

    def __init__(self, parent):
        super().__init__(parent)
        names = {
            Workspace.generation: _("Generate"),
            Workspace.upscaling: _("Upscale"),
            Workspace.live: _("Live"),
            Workspace.animation: _("Animation"),
            Workspace.custom: _("Graph"),
        }
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._tabs: list[WorkspaceTab] = []
        for workspace, name in names.items():
            tab = WorkspaceTab(workspace, name, self)
            tab.clicked.connect(self._activate(workspace))
            layout.addWidget(tab)
            self._tabs.append(tab)
        layout.addStretch()
        self.value = Workspace.generation

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, workspace: Workspace):
        self._value = workspace
        for tab in self._tabs:
            tab.setChecked(tab.workspace is workspace)
            tab.updateGeometry()

    def paintEvent(self, a0):
        # the brand stroke dissolves into the empty space right of the tabs
        left = self._tabs[-1].geometry().right() + 14
        if self.width() - left > 50:
            rect = QRectF(left, 4, self.width() - left - 4, self.height() - 8)
            theme.paint_dissolve(QPainter(self), rect, step=6, seed=3, opacity=0.85)

    def _activate(self, workspace: Workspace):
        switch = actions.set_workspace(workspace)

        def handler():
            self.value = self._value  # the tab follows the model, not the click
            switch()

        return handler


def _get_width_dip(s: QPaintDevice):
    # get device-indpendent width for things like QPixmap, which report their size in physical pixels
    if ratio := s.devicePixelRatioF():
        return int(s.width() / ratio)
    return s.width()


class GenerateButton(QPushButton):
    ctrl_clicked = pyqtSignal()

    def __init__(self, kind: JobKind, parent: QWidget):
        super().__init__(parent)
        self.model = root.active_model
        self._operation = _("Generate")
        self._kind = kind
        self._seed_icon = theme.icon("seed")
        self._resolution_icon = theme.icon("resolution-multiplier")
        font = self.font()
        font.setBold(True)
        self.setFont(font)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)

    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value: str):
        self._operation = value
        self.update()

    def minimumSizeHint(self):
        fm = self.fontMetrics()
        return QSize(fm.horizontalAdvance(self._operation) + 40, 12 + int(1.3 * fm.height()))

    def mouseReleaseEvent(self, e: QMouseEvent | None):
        if (
            e
            and e.button() == Qt.MouseButton.LeftButton
            and bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
        ):
            self.ctrl_clicked.emit()
            e.accept()
            self.setDown(False)
        else:
            super().mouseReleaseEvent(e)

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        fm = self.fontMetrics()
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        is_enabled = self.isEnabled()
        is_hover = self.underMouse() and is_enabled

        # the one filled, warm button of the docker
        painter.setPen(Qt.PenStyle.NoPen)
        if is_enabled:
            painter.setBrush(theme.accent_gradient(rect))
        else:
            painter.setBrush(QColor(theme.line))
        painter.drawRoundedRect(rect, theme.radius, theme.radius)
        if is_enabled and rect.width() > 120:
            # the stroke of the logo: solid at the start, dissolving into mint at the end
            clip = QPainterPath()
            clip.addRoundedRect(rect, theme.radius, theme.radius)
            painter.save()
            painter.setClipPath(clip)
            tail = QRectF(rect.right() - rect.width() * 0.2, 0, rect.width() * 0.22, rect.height())
            theme.paint_dissolve(
                painter,
                tail,
                step=5,
                seed=5,
                opacity=0.95,
                color=QColor(theme.accent_text),
                reverse=True,
            )
            painter.restore()
            painter.setPen(Qt.PenStyle.NoPen)
        if self.isDown():
            painter.setBrush(QColor(0, 0, 0, 40))
            painter.drawRoundedRect(rect, theme.radius, theme.radius)
        elif is_hover:
            painter.setBrush(QColor(255, 255, 255, 50))
            painter.drawRoundedRect(rect, theme.radius, theme.radius)

        color = QColor(theme.accent_text if is_enabled else theme.grey)
        ratio = self.devicePixelRatioF() or 1.0

        def draw_icon(icon: QIcon, size: float, x: float):
            pixmap = theme.tinted(icon.pixmap(int(size * ratio)), color)
            target = QRectF(x, (self.height() - size) / 2, size, size)
            painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))

        icon_size = 0.0 if self.icon().isNull() else fm.height() * 1.2
        gap = 6 if icon_size else 0
        content_width = fm.horizontalAdvance(self._operation) + gap + icon_size
        x = max(6.0, 0.5 * (self.width() - content_width))
        if icon_size:
            draw_icon(self.icon(), icon_size, x)
        painter.setPen(color)
        text_rect = QRectF(self.rect()).adjusted(x + icon_size + gap, 0, 0, 0)
        align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        painter.drawText(text_rect, align, self._operation)

        # hints on hover: fixed seed and resolution multiplier are active
        hint_size = float(fm.height())
        hint_x = self.width() - hint_size - 8
        if is_hover and self.model.fixed_seed:
            draw_icon(self._seed_icon, hint_size, hint_x)
            hint_x -= hint_size + 4
        if is_hover and self.model.resolution_multiplier != 1:
            draw_icon(self._resolution_icon, hint_size, hint_x)


class ChipToggle(QCheckBox):
    """A checkbox drawn as a pill: outlined when off, filled blue when on."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        return QSize(fm.horizontalAdvance(self.text()) + 20, fm.height() + 8)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def hitButton(self, pos):
        return self.rect().contains(pos)

    def paintEvent(self, a0):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2
        text_color = self.palette().color(QPalette.ColorRole.WindowText)
        painter.setOpacity(1.0 if self.isEnabled() else 0.45)
        if self.isChecked():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.blue_fill))
            painter.drawRoundedRect(rect, radius, radius)
            text_color = QColor(theme.fill_text)
        else:
            border = QColor(theme.accent_cool if self.underMouse() else theme.grey)
            border.setAlpha(255 if self.underMouse() else (120 if theme.is_dark else 170))
            painter.setPen(QPen(border, 1))
            painter.setBrush(Qt.BrushStyle.NoBrush if theme.is_dark else QColor(theme.base))
            painter.drawRoundedRect(rect, radius, radius)
        painter.setPen(text_color)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())
        if self.hasFocus() and not self.isChecked():
            painter.setPen(QPen(QColor(theme.accent_cool), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, radius, radius)


class AccentProgressBar(QProgressBar):
    """A slim, rounded progress line filled with the brand gradient."""

    alt_color: str | None = None  # plain color instead of the gradient (eg. while uploading)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setTextVisible(False)
        self.setFixedHeight(5)

    def paintEvent(self, a0):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        rect = QRectF(self.rect())
        radius = rect.height() / 2
        painter.setBrush(QColor(theme.line_base))
        painter.drawRoundedRect(rect, radius, radius)
        span = self.maximum() - self.minimum()
        if span <= 0 or self.value() <= self.minimum():
            return
        ratio = min(1.0, (self.value() - self.minimum()) / span)
        fill = QRectF(rect)
        fill.setWidth(max(rect.height(), rect.width() * ratio))
        if self.alt_color:
            painter.setBrush(QColor(self.alt_color))
        else:
            painter.setBrush(theme.accent_gradient(rect))
        painter.drawRoundedRect(fill, radius, radius)


class ErrorBox(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._error = no_error
        self._original_error = ""

        self.setObjectName("errorBox")
        self.setFrameStyle(QFrame.Shape.StyledPanel)

        self._label = QLabel(self)
        self._label.setWordWrap(True)
        self._label.setOpenExternalLinks(True)
        self._label.setTextFormat(Qt.TextFormat.RichText)

        self._copy_button = QToolButton(self)
        self._copy_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self._copy_button.setIcon(theme.icon("copy"))
        self._copy_button.setToolTip(_("Copy error message to clipboard"))
        self._copy_button.setAutoRaise(True)
        self._copy_button.clicked.connect(self._copy_error)

        self._icon = QLabel(self)
        self._icon.setFixedSize(18, 18)
        self._icon.setScaledContents(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 4, 6)
        layout.setSpacing(8)
        layout.addWidget(self._icon, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._label, 1)
        layout.addWidget(self._copy_button, 0, Qt.AlignmentFlag.AlignTop)

        self.reset()

    def reset(self, color: str = theme.red):
        self._copy_button.setVisible(False)
        self._label.setStyleSheet(f"color: {color};")
        is_error = color == theme.red
        tint = QColor(theme.accent if is_error else color)
        ground = f"rgba({tint.red()}, {tint.green()}, {tint.blue()}, {34 if is_error else 22})"
        self._icon.setPixmap(theme.icon("alert" if is_error else "warning").pixmap(36, 36))
        self.setStyleSheet(
            f"QFrame#errorBox {{ background: {ground}; border: 1px solid {tint.name()};"
            f" border-radius: {theme.radius}px; }}"
        )
        self.setVisible(False)

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, error: Error):
        self.reset()
        self._error = error
        self._original_error = error.message if error else ""
        if error.kind is ErrorKind.insufficient_funds:
            self._show_payment_error(error.data)
        elif error.kind.is_warning:
            self._show_warning(error.kind, error.message)
        elif error:
            self._show_error(error.message)

    def _show_error(self, text: str):
        server_error_prefix = _("Server error")
        if text.startswith(f"{server_error_prefix}: Validation error:"):
            text = text[len(server_error_prefix) + 1 :]
        if text.count("\n") > 3:
            lines = text.split("\n")
            n = 1
            text = lines[-n]
            while n < len(lines) and text.strip() == "":
                n += 1
                text = lines[-n]
        if len(text) > 60 * 3:
            text = text[: 60 * 2] + " [...] " + text[-60:]
        self._label.setText(text)
        if text != self._original_error:
            self._label.setToolTip(self._original_error)
        self._copy_button.setVisible(True)
        self.setVisible(True)

    def _show_warning(self, kind: ErrorKind, text: str):
        self.reset(theme.yellow)
        if kind is ErrorKind.incompatible_lora:
            text = (
                _(
                    "Selected LoRA model could not be applied. Please make sure it is compatible with the checkpoint base model you are using."
                )
                + " <a href='https://github.com/SHoogstad/diffusion-for-krita#base-models'>"
                + _("Learn more")
                + "</a>"
            )
        self._label.setText(text)
        self.setVisible(True)

    def _show_payment_error(self, data: dict[str, Any] | None):
        self.reset(theme.yellow)
        message = "Insufficient funds"
        if data:
            message = _(
                "Insufficient funds - generation would cost {cost} tokens. Remaining tokens: {tokens}",
                cost=data["cost"],
                tokens=data["credits"],
            )
        self._label.setText(message)
        self.setVisible(True)

    def _copy_error(self):
        if clipboard := QGuiApplication.clipboard():
            clipboard.setText(self._original_error)


def create_wide_tool_button(icon_name: str, text: str, parent=None):
    button = QToolButton(parent)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
    button.setIcon(theme.icon(icon_name))
    button.setToolTip(text)
    button.setAutoRaise(True)
    icon_height = button.iconSize().height()
    button.setIconSize(QSize(int(icon_height * 1.25), icon_height))
    return button


def create_framed_label(text: str, parent=None):
    frame = QFrame(parent)
    frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Plain)
    label = QLabel(parent=frame)
    label.setText(text)
    frame_layout = QHBoxLayout()
    frame_layout.setContentsMargins(4, 2, 4, 2)
    frame_layout.addWidget(label)
    frame.setLayout(frame_layout)
    return frame, label


def _paint_tool_drop_down(widget: QToolButton, text: str | None = None):
    opt = QStyleOption()
    opt.initFrom(widget)
    painter = QPainter(widget)
    style = ensure(widget.style())
    align = (
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignAbsolute
    )
    rect = widget.rect()
    pixmap = widget.icon().pixmap(int(rect.height() * 0.75))
    element = QStyle.PrimitiveElement.PE_Widget
    if opt.state & QStyle.StateFlag.State_MouseOver:
        element = QStyle.PrimitiveElement.PE_PanelButtonCommand
    style.drawPrimitive(element, opt, painter, widget)
    style.drawItemPixmap(painter, rect.adjusted(4, 0, 0, 0), align, pixmap)
    if text:
        text_rect = rect.adjusted(_get_width_dip(pixmap) + 4, 0, 0, 0)
        style.drawItemText(painter, text_rect, align, widget.palette(), True, text)
    painter.translate(int(0.5 * rect.width() - 10), 0)
    style.drawPrimitive(QStyle.PrimitiveElement.PE_IndicatorArrowDown, opt, painter)
