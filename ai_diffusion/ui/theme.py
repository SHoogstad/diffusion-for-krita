from __future__ import annotations

from pathlib import Path

from ..backend.client import Client
from ..files import FileFormat
from ..localization import translate as _
from ..platform_tools import is_windows
from ..qt.QtCore import QObject, QPointF, QRectF, QSize, Qt
from ..qt.QtGui import (
    QColor,
    QFontMetrics,
    QGuiApplication,
    QIcon,
    QLinearGradient,
    QPainter,
    QPalette,
    QPixmap,
)
from ..qt.QtWidgets import QLabel, QVBoxLayout, QWidget
from ..settings import Setting
from ..style import Arch
from ..util import client_logger as log

_app = QGuiApplication.instance()
_palette = _app.palette() if isinstance(_app, QGuiApplication) else QPalette()
is_dark = _palette.color(QPalette.ColorRole.Window).lightness() < 128

base = _palette.color(QPalette.ColorRole.Base).name()
green = "#30b030" if is_dark else "#209020"
yellow = "#c0c030" if is_dark else "#706020"
red = "#d07a40" if is_dark else "#c07630"
grey = "#888888" if is_dark else "#666666"
highlight = "#7ee0d0" if is_dark else "#1c7a6d"
strong_highlight = "#ffc857" if is_dark else "#b4540f"
progress_alt = "#a16207" if is_dark else "#ca8a04"
active = _palette.color(QPalette.ColorRole.Highlight).name()
line = _palette.color(QPalette.ColorRole.Window).darker(120).name()
line_base = _palette.color(QPalette.ColorRole.Base).darker(120).name()

# Brand colors, taken from the logo: a warm stroke (coral to amber) that dissolves into mint.
accent = "#ff8a5c" if is_dark else "#ee6f3c"
accent_warm = "#ffc857" if is_dark else "#f2a52c"
accent_cool = "#7ee0d0" if is_dark else "#1f9e8c"
accent_text = "#2a1f45"  # text and icons on top of the warm accent
radius = 5

primary_button_stylesheet = f"""
    QPushButton {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {accent}, stop:1 {accent_warm});
        color: {accent_text}; font-weight: bold; border: none; border-radius: {radius}px;
        padding: 6px 14px;
    }}
    QPushButton:hover {{ background: {accent_warm}; }}
    QPushButton:pressed {{ background: {accent}; }}
    QPushButton:disabled {{ background: {line}; color: {grey}; }}
"""

ghost_button_stylesheet = f"""
    QToolButton {{
        border: 1px solid {line_base}; border-radius: {radius}px; background: transparent;
        padding: 2px;
    }}
    QToolButton:hover {{ border-color: {accent_cool}; }}
    QToolButton:checked {{ border-color: {accent}; background: {base}; }}
    QToolButton::menu-indicator {{ image: none; }}
"""

flat_combo_stylesheet = f"""
    QComboBox {{ border: none; background-color: transparent; padding: 1px 12px 1px 2px; }}
    QComboBox QAbstractItemView {{ selection-color: {highlight}; }}
"""

prompt_max_line_count = 40

copy_to_clipboard_string = _("Copy to clipboard")  # keeping translations for future use

icon_path = Path(__file__).parent.parent / "icons"


def icon(name: str):
    theme = "dark" if is_dark else "light"
    path = icon_path / f"{name}-{theme}.svg"
    if not path.exists():
        path = path.with_suffix(".png")
    if not path.exists():
        log.error(f"Icon {name} not found for theme {theme}")
        return QIcon()
    return QIcon(str(path))


def checkpoint_icon(arch: Arch, format: FileFormat | None = None, client: Client | None = None):
    if client:
        if not client.supports_arch(arch):
            if arch is Arch.sdxl and client.supports_arch(Arch.illu):
                arch = Arch.illu  # assume SDXL models are Illustrious if SDXL isn't supported
            else:
                return icon("warning")
        if format is FileFormat.diffusion and not client.models.for_arch(arch).has_te_vae:
            return icon("warning")
    if arch is Arch.sd15:
        return icon("sd-version-15")
    elif arch is Arch.sdxl:
        return icon("sd-version-xl")
    elif arch is Arch.sd3:
        return icon("sd-version-3")
    elif arch is Arch.flux:
        return icon("sd-version-flux")
    elif arch is Arch.flux_k:
        return icon("sd-version-flux-k")
    elif arch.is_flux2:
        return icon("sd-version-flux-2")
    elif arch is Arch.illu:
        return icon("sd-version-illu")
    elif arch is Arch.illu_v:
        return icon("sd-version-illu-v")
    elif arch is Arch.chroma:
        return icon("sd-version-chroma")
    elif arch.is_qwen_like or arch is Arch.qwen2:
        return icon("sd-version-qwen")
    elif arch is Arch.zimage:
        return icon("sd-version-z-image")
    elif arch is Arch.anima:
        return icon("sd-version-anima")
    elif arch is Arch.ernie:
        return icon("sd-version-ernie")
    elif arch is Arch.krea2:
        return icon("sd-version-krea2")
    elif arch is Arch.moss:
        return icon("sd-version-moss")
    elif arch.is_fibo2:
        return icon("sd-version-fibo2")
    elif arch in (Arch.h3, Arch.ph3):
        return icon("sd-version-h3")
    else:
        log.warning(f"Unresolved SD version {arch}, cannot fetch icon")
        return icon("warning")


def accent_gradient(rect: QRectF, cool_end=False):
    """The brand gradient across `rect`: coral to amber, optionally running on into mint."""
    gradient = QLinearGradient(QPointF(rect.left(), 0), QPointF(rect.right(), 0))
    gradient.setColorAt(0, QColor(accent))
    if cool_end:
        gradient.setColorAt(0.6, QColor(accent_warm))
        gradient.setColorAt(1, QColor(accent_cool))
    else:
        gradient.setColorAt(1, QColor(accent_warm))
    return gradient


def tinted(pixmap: QPixmap, color: str | QColor):
    """A copy of `pixmap` with every visible pixel painted in `color` (alpha is kept)."""
    result = QPixmap(pixmap)
    if result.isNull():
        return result
    painter = QPainter(result)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(result.rect(), QColor(color))
    painter.end()
    return result


class AccentLine(QWidget):
    """A thin horizontal rule in the brand gradient."""

    def __init__(self, parent: QWidget | None = None, height=2):
        super().__init__(parent)
        self.setFixedHeight(height)

    def paintEvent(self, a0):
        rect = QRectF(self.rect())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent_gradient(rect, cool_end=True))
        painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)


def logo():
    return QPixmap(str(icon_path / "logo-128.png"))


def add_header(layout: QVBoxLayout, setting: Setting):
    title_label = QLabel(setting.name)
    title_label.setStyleSheet("font-weight:bold")
    desc_label = QLabel(setting.desc)
    desc_label.setWordWrap(True)
    layout.addSpacing(6)
    layout.addWidget(title_label)
    layout.addWidget(desc_label)


def set_text_clipped(label: QLabel, text: str, padding=4):
    metrics = QFontMetrics(label.font())
    elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight, label.width() - padding)
    label.setText(elided)


def screen_scale(widget: QWidget, size: QSize):
    if is_windows:  # Not sure about other OS
        scale = widget.logicalDpiX() / 96.0
        return QSize(int(size.width() * scale), int(size.height() * scale))
    return size


class SignalBlocker:
    def __init__(self, obj: QObject):
        self._obj = obj

    def __enter__(self):
        self._obj.blockSignals(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._obj.blockSignals(False)
