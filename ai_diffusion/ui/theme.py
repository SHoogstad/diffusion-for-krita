from __future__ import annotations

import random
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
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QPolygonF,
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
highlight = "#8db4f7" if is_dark else "#21468b"
strong_highlight = "#f4626c" if is_dark else "#ae1c28"
progress_alt = "#a16207" if is_dark else "#ca8a04"
active = _palette.color(QPalette.ColorRole.Highlight).name()
line = _palette.color(QPalette.ColorRole.Window).darker(120).name()
line_base = _palette.color(QPalette.ColorRole.Base).darker(120).name()

# Brand colors: red, white and blue, after the Dutch flag (#ae1c28, #ffffff, #21468b).
# Dark themes get brighter red and blue so they keep their contrast.
accent = "#e5343f" if is_dark else "#ae1c28"  # red: the primary action
accent_warm = "#f25560" if is_dark else "#c7303b"  # red, lit (hover, end of a red fill)
accent_cool = "#6c9bf0" if is_dark else "#21468b"  # blue: state that is on, focus, links
accent_white = "#ffffff"
accent_text = "#ffffff"  # text and icons on top of red or blue
flag_blue = "#21468b"
blue_fill = "#2f5fc4" if is_dark else flag_blue  # blue ground under white text
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
    """The red fill across `rect`, or with `cool_end` the whole flag: red, white, blue."""
    gradient = QLinearGradient(QPointF(rect.left(), 0), QPointF(rect.right(), 0))
    gradient.setColorAt(0, QColor(accent))
    if cool_end:
        gradient.setColorAt(0.5, QColor(accent_white))
        gradient.setColorAt(1, QColor(accent_cool))
    else:
        gradient.setColorAt(1, QColor(accent_warm))
    return gradient


def accent_at(t: float):
    """Color of the brand stroke at position t: red (0), white (0.5), blue (1)."""
    a, b, c = QColor(accent), QColor(accent_white), QColor(accent_cool)
    lo, hi, f = (a, b, t * 2) if t < 0.5 else (b, c, t * 2 - 1)
    f = min(max(f, 0.0), 1.0)
    return QColor(
        int(lo.red() + (hi.red() - lo.red()) * f),
        int(lo.green() + (hi.green() - lo.green()) * f),
        int(lo.blue() + (hi.blue() - lo.blue()) * f),
    )


def paint_dissolve(
    painter: QPainter,
    rect: QRectF,
    step=6.0,
    seed=7,
    opacity=1.0,
    color: QColor | None = None,
    reverse=False,
):
    """The signature of the plugin: a stroke that dissolves into noise, like in the logo.
    A halftone of dots on a staggered grid, full at the left of `rect`, shrinking and
    dropping out towards the right (`reverse` flips the direction)."""
    rng = random.Random(seed)
    rows = max(1, int(rect.height() / step))
    cols = max(2, int(rect.width() / step))
    top = rect.top() + (rect.height() - rows * step) / 2 + step / 2
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    for col in range(cols):
        t = col / (cols - 1)
        for row in range(rows):
            if rng.random() < t * 0.6:
                continue  # noise takes over
            radius = step * 0.42 * (1 - t) ** 1.2 + 0.35
            pos = 1 - t if reverse else t
            x = rect.left() + (rect.width() - step) * pos + step / 2
            y = top + row * step + (step / 2 if col % 2 else 0) - step / 4
            dot = QColor(color) if color else accent_at(t)
            dot.setAlphaF(opacity * (0.35 + 0.65 * (1 - t)))
            painter.setBrush(dot)
            painter.drawEllipse(QPointF(x, y), radius, radius)
    painter.restore()


def paint_workspace_glyph(painter: QPainter, rect: QRectF, name: str, color: QColor | None):
    """Own workspace symbols, drawn on a 24 unit grid. No color = the brand gradient."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.translate(rect.topLeft())
    painter.scale(rect.width() / 24, rect.height() / 24)
    brush = accent_gradient(QRectF(0, 0, 24, 24)) if color is None else color
    pen = QPen(brush, 1.9)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

    def stroke(path: QPainterPath):
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def fill(path: QPainterPath):
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(brush)
        painter.drawPath(path)

    def star(cx: float, cy: float, r: float):
        k = r * 0.14
        path = QPainterPath(QPointF(cx, cy - r))
        path.quadTo(cx + k, cy - k, cx + r, cy)
        path.quadTo(cx + k, cy + k, cx, cy + r)
        path.quadTo(cx - k, cy + k, cx - r, cy)
        path.quadTo(cx - k, cy - k, cx, cy - r)
        return path

    if name == "generation":
        fill(star(10, 13.5, 9))
        fill(star(19.5, 5, 4))
    elif name == "upscaling":
        small = QPainterPath()
        small.addRoundedRect(QRectF(3, 13, 8, 8), 1.6, 1.6)
        fill(small)
        frame = QPainterPath(QPointF(3, 8.5))
        frame.lineTo(3, 5)
        frame.quadTo(3, 3, 5, 3)
        frame.lineTo(19, 3)
        frame.quadTo(21, 3, 21, 5)
        frame.lineTo(21, 19)
        frame.quadTo(21, 21, 19, 21)
        frame.lineTo(15.5, 21)
        stroke(frame)
        arrow = QPainterPath(QPointF(12.5, 11.5))
        arrow.lineTo(17, 7)
        arrow.moveTo(13, 7)
        arrow.lineTo(17, 7)
        arrow.lineTo(17, 11)
        stroke(arrow)
    elif name == "live":
        bolt = QPainterPath()
        points = [(13.5, 2), (5, 13.6), (10.8, 13.6), (9.5, 22), (19, 9.8), (12.8, 9.8)]
        bolt.addPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
        bolt.closeSubpath()
        fill(bolt)
    elif name == "animation":
        back = QPainterPath(QPointF(8, 4))
        back.lineTo(19, 4)
        back.quadTo(21, 4, 21, 6)
        back.lineTo(21, 14.5)
        stroke(back)
        front = QPainterPath()
        front.addRoundedRect(QRectF(3, 8, 14, 12), 2, 2)
        stroke(front)
        play = QPainterPath()
        play.addPolygon(QPolygonF([QPointF(8.3, 11.2), QPointF(8.3, 16.8), QPointF(13, 14)]))
        play.closeSubpath()
        fill(play)
    else:  # custom graph
        links = QPainterPath(QPointF(6, 12))
        links.cubicTo(12, 12, 11, 5.5, 17, 5.5)
        links.moveTo(6, 12)
        links.cubicTo(12, 12, 11, 18.5, 17, 18.5)
        stroke(links)
        nodes = QPainterPath()
        for x, y in [(5, 12), (19, 5.5), (19, 18.5)]:
            nodes.addEllipse(QPointF(x, y), 2.8, 2.8)
        fill(nodes)
    painter.restore()


def paint_flag_bar(painter: QPainter, rect: QRectF):
    """A small tricolor: red, white and blue side by side."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    clip = QPainterPath()
    clip.addRoundedRect(rect, rect.height() / 2, rect.height() / 2)
    painter.setClipPath(clip)
    third = rect.width() / 3
    for i, color in enumerate((accent, accent_white, accent_cool)):
        part = QRectF(rect.left() + i * third, rect.top(), third + 0.5, rect.height())
        painter.fillRect(part, QColor(color))
    painter.restore()


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
        painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        paint_flag_bar(painter, rect)


class BrandBanner(QWidget):
    """Header of the welcome screen: the logo on its night-blue ground, the name, and the
    stroke dissolving into noise across the rest of the card."""

    def __init__(self, subtitle: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._subtitle = subtitle
        self._logo = logo()
        self.setFixedHeight(84)

    def paintEvent(self, a0):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        rect = QRectF(self.rect())
        ground = QLinearGradient(rect.topLeft(), rect.bottomRight())
        ground.setColorAt(0, QColor(flag_blue))
        ground.setColorAt(1, QColor("#14294f"))
        card = QPainterPath()
        card.addRoundedRect(rect, 10, 10)
        painter.fillPath(card, ground)
        painter.setClipPath(card)

        logo_size = 60.0
        logo_rect = QRectF(10, (rect.height() - logo_size) / 2, logo_size, logo_size)
        painter.drawPixmap(logo_rect, self._logo, QRectF(self._logo.rect()))

        text_x = logo_rect.right() + 10
        title_font = self.font()
        title_font.setBold(True)
        title_font.setPointSizeF(title_font.pointSizeF() * 1.6)
        painter.setFont(title_font)
        painter.setPen(QColor("#ffffff"))
        title_rect = QRectF(text_x, 14, rect.width() - text_x, 34)
        align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        painter.drawText(title_rect, align, "Diffusion")
        title_width = QFontMetrics(title_font).horizontalAdvance("Diffusion")
        painter.setFont(self.font())
        painter.setPen(QColor("#c9d6ee"))
        painter.drawText(QRectF(text_x, 46, rect.width() - text_x, 22), align, self._subtitle)

        subtitle_width = QFontMetrics(self.font()).horizontalAdvance(self._subtitle)
        dots_left = text_x + max(title_width, subtitle_width) + 18
        if rect.width() - dots_left > 40:
            dots = QRectF(dots_left, 0, min(rect.width() - dots_left, 300), rect.height())
            paint_dissolve(painter, dots, step=8, seed=11, opacity=0.9)


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
