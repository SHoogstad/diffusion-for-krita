from __future__ import annotations

import random
import tempfile
from functools import lru_cache
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
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QPolygonF,
)
from ..qt.QtWidgets import QComboBox, QLabel, QStyle, QStyledItemDelegate, QVBoxLayout, QWidget
from ..settings import Setting
from ..style import Arch
from ..util import client_logger as log
from ..util import read_json_with_comments
from . import flags
from .flags import Country

_app = QGuiApplication.instance()
_palette = _app.palette() if isinstance(_app, QGuiApplication) else QPalette()
is_dark = _palette.color(QPalette.ColorRole.Window).lightness() < 128

base = _palette.color(QPalette.ColorRole.Base).name()
text = _palette.color(QPalette.ColorRole.Text).name()
text_color = _palette.color(QPalette.ColorRole.WindowText).name()
green = "#30b030" if is_dark else "#209020"
yellow = "#c0c030" if is_dark else "#706020"
red = "#f4626c" if is_dark else "#ae1c28"
grey = "#888888" if is_dark else "#666666"
progress_alt = "#a16207" if is_dark else "#ca8a04"
active = _palette.color(QPalette.ColorRole.Highlight).name()
line = _palette.color(QPalette.ColorRole.Window).darker(120).name()
line_base = _palette.color(QPalette.ColorRole.Base).darker(120).name()

# Brand colors. The country look takes them from the flag of the user's country (see flags.py),
# the plain look from one chosen color. apply_look() sets everything below, at import from the
# saved settings and again after the settings are loaded; a change needs a restart of Krita.
radius = 5
_white, _ink = "#ffffff", "#141414"
_window = _palette.color(QPalette.ColorRole.Window)

# Colors of the generated icon files (scripts/make_icons.py) and the logo, replaced on load
_icon_red = {"dark": "#f0505a", "light": "#ae1c28"}
_icon_blue = {"dark": "#7ea8f5", "light": "#21468b"}
_logo_colors = ("#e5343f", "#ffffff", "#5b8def")
_logo_ground = ("#1b2b4f", "#0c1428")


def _luminance(color: QColor):
    def channel(v: float):
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(color.redF())
        + 0.7152 * channel(color.greenF())
        + 0.0722 * channel(color.blueF())
    )


def contrast(a: str | QColor, b: str | QColor):
    la, lb = _luminance(QColor(a)), _luminance(QColor(b))
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


@lru_cache(maxsize=512)
def _toward(color: str, background: str, ratio: float):
    """`color` made lighter or darker (away from `background`) until it has `ratio` contrast."""
    c = QColor(color)
    bg = QColor(background)
    lighten = _luminance(bg) < 0.18
    h, s, l, _ = c.getHslF()
    h = max(h, 0.0)
    for _ in range(60):
        if contrast(c, bg) >= ratio:
            break
        l = min(1.0, l + 0.02) if lighten else max(0.0, l - 0.02)
        c = QColor.fromHslF(h, s, l)
        if l in (0.0, 1.0):
            break
    return c.name()


def _text_on(color: str):
    """White or near black, whichever reads better on `color` (white wins ties)."""
    return _white if contrast(color, _white) >= 2.6 else _ink


def _ground(color: str, lightness: float):
    """A deep shade of `color` for the banner and the logo tile."""
    h, s, _, _ = QColor(color).getHslF()
    if s < 0.08:  # black, white or grey: a neutral ground
        return QColor.fromHslF(0.62, 0.15, lightness).name()
    return QColor.fromHslF(max(h, 0.0), min(s, 0.62), lightness).name()


def _is_yellowish(color: str):
    h, s, _, _ = QColor(color).getHslF()
    return s > 0.3 and 0.09 < h < 0.2


def _peek_settings():
    """The look settings, read from the file: the theme is imported before settings.load()."""
    from ..settings import InterfaceLook, settings

    values = {
        "interface_look": settings.interface_look,
        "interface_country": settings.interface_country,
        "interface_color": settings.interface_color,
    }
    try:
        path = settings.default_path
        if path.exists():
            data = read_json_with_comments(path)
            look = data.get("interface_look", values["interface_look"].name)
            values["interface_look"] = InterfaceLook[look] if isinstance(look, str) else look
            values["interface_country"] = str(data.get("interface_country", ""))
            values["interface_color"] = str(data.get("interface_color", values["interface_color"]))
    except Exception as e:
        log.warning(f"Could not read the look settings, using defaults: {e}")
    return values


country: Country | None = None  # the country of the look, None for the plain look


def apply_look(look=None, country_code: str | None = None, color: str | None = None):
    """Set all brand colors and stylesheets. Arguments left out come from the settings."""
    global country, accent, accent_warm, accent_cool, accent_text, accent_white, white_edge
    global fill_text, blue_fill, flag_blue, banner_end, bands, banner_bands, highlight
    global strong_highlight, primary_button_stylesheet, ghost_button_stylesheet
    global select_stylesheet, docker_stylesheet, flat_combo_stylesheet, _icon_map, _badge_colors
    global _icon_cache_dir, _chevron, _icons, _mode, _edge, ui_bands, _band_outline, _logo
    from ..settings import InterfaceLook

    values = _peek_settings() if look is None else {}
    look = look if look is not None else values["interface_look"]
    if country_code is None:
        country_code = values.get("interface_country", "") if values else settings_country()
    if color is None:
        color = values.get("interface_color", "") if values else settings_color()
    if not QColor(color).isValid():
        color = "#3d7be0"

    country = flags.resolve(country_code) if look is InterfaceLook.country else None
    if country is not None:
        raw_accent = country.accent
        raw_cool = country.cool or QColor(country.accent).darker(135).name()
        bands = country.bands
    else:  # plain: one color does everything
        raw_accent = raw_cool = QColor(color).name()
        bands = (raw_accent,)

    window = _window.name()
    accent = _toward(raw_accent, window, 2.2)  # the primary action
    accent_warm = QColor(accent).lighter(114).name()  # lit (hover, end of the fill)
    accent_text = _text_on(accent)  # text and icons on top of the accent
    accent_cool = _toward(raw_cool, window, 3.0)  # state that is on, focus, links
    highlight = accent_cool
    strong_highlight = accent
    blue_fill = _toward(raw_cool, window, 1.9) if is_dark else raw_cool  # ground under text
    fill_text = _text_on(blue_fill)
    accent_white = _white
    white_edge = "#8d97ad"  # outline that keeps white visible on light themes
    ground_hue = raw_cool if not _is_yellowish(raw_cool) else raw_accent  # no muddy olive
    flag_blue = _ground(ground_hue, 0.24)  # banner and logo ground
    banner_end = _ground(ground_hue, 0.11)
    banner_bands = tuple(_toward(b, flag_blue, 2.0) for b in bands)  # strokes on that ground
    ui_bands = tuple(_toward(b, window, 2.0) if is_dark else b for b in bands)  # on the window
    _band_outline = tuple(contrast(b, window) < 1.3 for b in ui_bands)

    mode = "dark" if is_dark else "light"
    icon_red = _toward(raw_accent, window, 3.0)
    icon_blue = accent_cool
    _icon_map = {_icon_red[mode]: icon_red, _icon_blue[mode]: icon_blue}
    _badge_colors = {"r": _toward(raw_accent, _white, 3.0), "b": _toward(raw_cool, _white, 3.0)}
    _badge_cache.clear()
    key = "-".join([mode, icon_red, icon_blue, fill_text, flag_blue, *banner_bands])
    _icon_cache_dir = (
        Path(tempfile.gettempdir()) / "krita-ai-diffusion-icons" / key.replace("#", "")
    )

    _icons = icon_path.as_posix()
    _mode = mode
    _edge = grey if is_dark else "#9aa1ad"  # outline of unchecked boxes
    _chevron = f"{_icons}/chevron-{mode}.svg"
    check = _recolored_file(icon_path / f"check-{mode}.svg", {"#ffffff": fill_text}).as_posix()
    strokes = [banner_bands[i % len(banner_bands)] for i in range(3)]
    if len(banner_bands) == 2:  # two colors: outer strokes in the first, middle in the second
        strokes = [banner_bands[0], banner_bands[1], banner_bands[0]]
    logo_map = dict(zip(_logo_colors, strokes))
    logo_map.update(zip(_logo_ground, (flag_blue, banner_end)))
    _logo = _recolored_logo(logo_map)

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

    # The selectors at the top of the docker (style, model, workflow): a card with the name in
    # bold, a blue edge on hover and focus, and a list with a blue selection.
    select_stylesheet = f"""
        QComboBox {{
            background: {base}; border: 1px solid {line_base}; border-radius: {radius}px;
            padding: 4px 6px 4px 8px; font-weight: bold;
        }}
        QComboBox:hover, QComboBox:focus, QComboBox:on {{ border-color: {accent_cool}; }}
        QComboBox:disabled {{ color: {grey}; }}
        QComboBox::drop-down {{ border: none; width: 24px; }}
        QComboBox::down-arrow {{ image: url("{_chevron}"); width: 12px; height: 12px; }}
        QComboBox QAbstractItemView {{
            background: {base}; border: 1px solid {accent_cool}; padding: 3px; outline: none;
        }}
        QComboBox QAbstractItemView::item {{
            min-height: 22px; padding: 2px 4px; border-radius: 3px; color: {text};
        }}
        QComboBox QAbstractItemView::item:selected {{
            background: {blue_fill}; color: {fill_text};
        }}
    """

    # One look for the standard widgets of the docker and the settings dialog: fields are cards
    # with a blue edge on hover and focus, whatever is checked or selected is blue.
    docker_stylesheet = f"""
        QPushButton {{
            background: {base}; border: 1px solid {line_base}; border-radius: {radius}px;
            padding: 5px 12px;
        }}
        QPushButton:hover {{ border-color: {accent_cool}; }}
        QPushButton:pressed {{ background: {line}; }}
        QPushButton:disabled {{ color: {grey}; }}

        QToolButton {{
            background: transparent; border: 1px solid transparent; border-radius: {radius}px;
            padding: 3px;
        }}
        QToolButton:hover {{ background: {line}; }}
        QToolButton:pressed {{ background: {line_base}; }}
        QToolButton:checked {{ background: {base}; border-color: {accent_cool}; }}
        QToolButton[popupMode="1"] {{ padding-right: 14px; }}
        QToolButton::menu-button {{
            border: none; background: transparent; width: 14px; border-radius: {radius}px;
        }}
        QToolButton::menu-button:hover {{ background: {line_base}; }}
        QToolButton::menu-arrow {{ image: url("{_icons}/chevron-{_mode}.svg"); width: 8px; height: 8px; }}

        QLineEdit, QAbstractSpinBox, QComboBox {{
            background: {base}; border: 1px solid {line_base}; border-radius: {radius}px;
            padding: 3px 6px; selection-background-color: {blue_fill}; selection-color: {fill_text};
        }}
        QLineEdit:hover, QAbstractSpinBox:hover, QComboBox:hover,
        QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus, QComboBox:on {{
            border-color: {accent_cool};
        }}
        QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {{ color: {grey}; }}
        QAbstractSpinBox QLineEdit, QComboBox QLineEdit {{
            border: none; background: transparent; padding: 0;
        }}
        QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
            subcontrol-origin: border; width: 16px; border: none; background: transparent;
        }}
        QAbstractSpinBox::up-button {{ subcontrol-position: top right; }}
        QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; }}
        QAbstractSpinBox::up-arrow {{ image: url("{_icons}/chevron-up-{_mode}.svg"); width: 8px; height: 8px; }}
        QAbstractSpinBox::down-arrow {{ image: url("{_icons}/chevron-{_mode}.svg"); width: 8px; height: 8px; }}
        QComboBox::drop-down {{ border: none; width: 22px; }}
        QComboBox::down-arrow {{ image: url("{_icons}/chevron-{_mode}.svg"); width: 10px; height: 10px; }}
        QComboBox QAbstractItemView {{
            background: {base}; border: 1px solid {accent_cool}; outline: none;
            selection-background-color: {blue_fill}; selection-color: {fill_text};
        }}

        QCheckBox, QRadioButton {{ spacing: 7px; }}
        QCheckBox::indicator, QGroupBox::indicator, QRadioButton::indicator {{
            width: 14px; height: 14px; background: {base}; border: 1px solid {_edge};
        }}
        QCheckBox::indicator, QGroupBox::indicator {{ border-radius: 4px; }}
        QRadioButton::indicator {{ border-radius: 8px; }}
        QCheckBox::indicator:hover, QGroupBox::indicator:hover, QRadioButton::indicator:hover {{
            border-color: {accent_cool};
        }}
        QCheckBox::indicator:checked, QGroupBox::indicator:checked {{
            background: {blue_fill}; border-color: {blue_fill};
            image: url("{check}");
        }}
        QRadioButton::indicator:checked {{
            background: {accent_white}; border: 4px solid {blue_fill}; width: 8px; height: 8px;
        }}

        QGroupBox {{
            border: 1px solid {line_base}; border-radius: {radius + 1}px; margin-top: 12px;
            padding-top: 8px; font-weight: bold;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}

        QSlider::groove:horizontal {{ height: 4px; background: {line_base}; border-radius: 2px; }}
        QSlider::sub-page:horizontal {{ background: {blue_fill}; border-radius: 2px; }}
        QSlider::handle:horizontal {{
            width: 12px; height: 12px; margin: -5px 0; border-radius: 7px;
            background: {accent_white}; border: 2px solid {blue_fill};
        }}
        QSlider::handle:horizontal:hover {{ border-color: {accent}; }}

        QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
        QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
        QScrollBar::handle {{ background: {_edge}; border-radius: 4px; margin: 1px; }}
        QScrollBar::handle:vertical {{ min-height: 28px; }}
        QScrollBar::handle:horizontal {{ min-width: 28px; }}
        QScrollBar::handle:hover {{ background: {accent_cool}; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

        QMenu {{ background: {base}; border: 1px solid {line_base}; padding: 4px; }}
        QMenu::item {{ padding: 5px 22px 5px 10px; border-radius: 3px; }}
        QMenu::icon {{ padding-left: 8px; }}
        QMenu::item:selected {{ background: {blue_fill}; color: {fill_text}; }}
        QMenu::item:disabled {{ color: {grey}; }}
        QMenu::separator {{ height: 1px; background: {line_base}; margin: 4px 6px; }}

        QTabWidget::pane {{ border: 1px solid {line_base}; border-radius: {radius + 1}px; top: -1px; }}
        QTabBar::tab {{
            background: transparent; border: none; border-bottom: 2px solid transparent;
            padding: 6px 12px; margin-right: 2px; color: {grey};
        }}
        QTabBar::tab:hover {{ color: {text_color}; }}
        QTabBar::tab:selected {{
            color: {text_color}; font-weight: bold; border-bottom: 2px solid {accent};
        }}

        QFrame[frameShape="4"], QFrame[frameShape="5"] {{ color: {line_base}; }}

        QProgressBar {{
            border: none; background: {line_base}; border-radius: 3px; text-align: center;
        }}
        QProgressBar::chunk {{ background: {accent}; border-radius: 3px; }}
    """

    flat_combo_stylesheet = f"""
        QComboBox {{ border: 1px solid transparent; background-color: transparent; padding: 1px 12px 1px 2px; }}
        QComboBox:hover {{ border-color: {accent_cool}; }}
        QComboBox QAbstractItemView {{ selection-color: {highlight}; }}
    """


def settings_country():
    from ..settings import settings

    return settings.interface_country


def settings_color():
    from ..settings import settings

    return settings.interface_color


def _recolored_logo(colors: dict[str, str]):
    """The logo SVG with all colors swapped at once (a stroke may take a color another stroke
    gives up, so replace through placeholders)."""
    placeholders = {old: f"#@{i}@" for i, old in enumerate(colors)}
    try:
        text = (icon_path / "logo.svg").read_text(encoding="utf-8")
        for old, mark in placeholders.items():
            text = text.replace(old, mark)
        for old, mark in placeholders.items():
            text = text.replace(mark, colors[old])
        target = _icon_cache_dir / "logo.svg"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target
    except Exception as e:
        log.warning(f"Could not recolor the logo: {e}")
        return icon_path / "logo-128.png"


def _recolored_file(path: Path, colors: dict[str, str]):
    """A copy of the SVG at `path` with `colors` replaced, cached per look. Falls back to the
    original file if nothing changes or the copy can't be written."""
    try:
        text = path.read_text(encoding="utf-8")
        changed = text
        marks = {old: f"#@{i}@" for i, old in enumerate(colors)}  # swaps stay correct
        for old, mark in marks.items():
            changed = changed.replace(old, mark).replace(old.upper(), mark)
        for old, mark in marks.items():
            changed = changed.replace(mark, colors[old])
        if changed == text:
            return path
        target = _icon_cache_dir / path.name
        if not target.exists() or target.read_text(encoding="utf-8") != changed:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(changed, encoding="utf-8")
        return target
    except Exception as e:
        log.warning(f"Could not recolor {path.name}: {e}")
        return path


prompt_max_line_count = 40

copy_to_clipboard_string = _("Copy to clipboard")  # keeping translations for future use

icon_path = Path(__file__).parent.parent / "icons"


# Model families: a badge in red or blue with white letters
_badges = {
    "15": ("1.5", "b"), "xl": ("XL", "b"), "3": ("SD3", "b"), "illu": ("IL", "b"),
    "illu-v": ("IV", "b"), "anima": ("AN", "b"), "flux": ("FX", "r"), "flux-k": ("FK", "r"),
    "flux-2": ("F2", "r"), "chroma": ("CH", "r"), "krea2": ("K2", "r"), "qwen": ("QW", "b"),
    "z-image": ("ZI", "r"), "ernie": ("ER", "b"), "moss": ("MS", "r"), "fibo2": ("FB", "b"),
    "h3": ("H3", "r"),
}  # fmt: skip
_badge_cache: dict[str, QIcon] = {}


def model_badge(name: str):
    """Badge for a model family, painted in the application font so the plugin never shows a
    second typeface."""
    if name in _badge_cache:
        return _badge_cache[name]
    letters, color = _badges[name]
    size = 96
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    rect = QRectF(6, 18, 84, 60)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(_badge_colors[color]))
    painter.drawRoundedRect(rect, 16, 16)
    font = QGuiApplication.font()
    font.setBold(True)
    font.setPixelSize(34 if len(letters) > 2 else 42)
    painter.setFont(font)
    painter.setPen(QColor("#ffffff"))
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, letters)
    painter.end()
    _badge_cache[name] = QIcon(pixmap)
    return _badge_cache[name]


def icon(name: str):
    if name.startswith("sd-version-") and name[11:] in _badges:
        return model_badge(name[11:])
    theme = "dark" if is_dark else "light"
    path = icon_path / f"{name}-{theme}.svg"
    if not path.exists():
        path = path.with_suffix(".png")
    if not path.exists():
        log.error(f"Icon {name} not found for theme {theme}")
        return QIcon()
    if path.suffix == ".svg":
        path = _recolored_file(path, _icon_map)
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


class SelectDelegate(QStyledItemDelegate):
    """Items follow the stylesheet; a separator is a thin line (eg. under the recent styles)."""

    @staticmethod
    def _is_separator(index):
        return index.data(Qt.ItemDataRole.AccessibleDescriptionRole) == "separator"

    def sizeHint(self, option, index):
        if self._is_separator(index):
            return QSize(0, 9)
        return super().sizeHint(option, index)

    def paint(self, painter, option, index):
        if not self._is_separator(index) or painter is None:
            return super().paint(painter, option, index)
        y = option.rect.center().y()
        painter.save()
        painter.setPen(QPen(QColor(line_base), 1))
        painter.drawLine(option.rect.left() + 6, y, option.rect.right() - 6, y)
        painter.restore()


class HistoryDelegate(QStyledItemDelegate):
    """History list: results are rounded thumbnails with a blue ring when selected, and each
    group starts with a header that shows the time as a small tag next to the prompt."""

    def paint(self, painter, option, index):
        if painter is None:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        rect = QRectF(option.rect)
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if not isinstance(icon, QIcon) or icon.isNull():
            self._paint_header(painter, rect, option, str(index.data() or ""))
        else:
            self._paint_thumbnail(painter, rect, option, icon)
        painter.restore()

    def _paint_header(self, painter: QPainter, rect: QRectF, option, text: str):
        time, _sep, prompt = text.partition(" - ")
        if not prompt:
            time, prompt = "", text
        fm = option.fontMetrics
        x = rect.left() + 2
        if time:
            tag = QRectF(
                x,
                rect.center().y() - fm.height() / 2 - 1,
                fm.horizontalAdvance(time) + 12,
                fm.height() + 2,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(blue_fill))
            painter.drawRoundedRect(tag, tag.height() / 2, tag.height() / 2)
            painter.setPen(QColor(fill_text))
            painter.drawText(tag, Qt.AlignmentFlag.AlignCenter, time)
            x = tag.right() + 7
        painter.setPen(QColor(text_color))
        width = rect.right() - x - 4
        elided = fm.elidedText(prompt, Qt.TextElideMode.ElideRight, int(width))
        align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        painter.drawText(QRectF(x, rect.top(), width, rect.height()), align, elided)

    def _paint_thumbnail(self, painter: QPainter, rect: QRectF, option, icon: QIcon):
        ratio = painter.device().devicePixelRatioF() if painter.device() else 1.0
        size = option.decorationSize
        pixmap = icon.pixmap(QSize(int(size.width() * ratio), int(size.height() * ratio)))
        if pixmap.isNull():
            return
        inner = rect.adjusted(3, 3, -3, -3)
        scale = min(inner.width() / pixmap.width(), inner.height() / pixmap.height())
        w, h = pixmap.width() * scale, pixmap.height() * scale
        target = QRectF(inner.center().x() - w / 2, inner.center().y() - h / 2, w, h)
        clip = QPainterPath()
        clip.addRoundedRect(target, 6, 6)
        painter.setClipPath(clip)
        painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))
        painter.setClipping(False)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        if selected or hover:
            ring = QColor(accent_cool)
            ring.setAlpha(255 if selected else 120)
            painter.setPen(QPen(ring, 2.2 if selected else 1.4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(target.adjusted(-1.5, -1.5, 1.5, 1.5), 7, 7)


def style_select(combo: QComboBox):
    """Give a combo box the look of the docker's selectors (see select_stylesheet)."""
    combo.setItemDelegate(SelectDelegate(combo))  # a plain combo list ignores the stylesheet
    combo.setStyleSheet(select_stylesheet)


def accent_gradient(rect: QRectF, cool_end=False):
    """The accent fill across `rect`. With `cool_end` it runs on through the flag colors."""
    gradient = QLinearGradient(QPointF(rect.left(), 0), QPointF(rect.right(), 0))
    gradient.setColorAt(0, QColor(accent))
    if cool_end and len(ui_bands) > 1:
        for i, band in enumerate(ui_bands[1:], 1):
            gradient.setColorAt(i / (len(ui_bands) - 1), QColor(band))
    else:
        gradient.setColorAt(1, QColor(accent_warm))
    return gradient


def accent_at(t: float):
    """Color of the brand stroke at position t (0..1), through the colors of the look."""
    stops = [QColor(c) for c in ui_bands] if len(ui_bands) > 1 else [QColor(accent)] * 2
    t = min(max(t, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    lo, hi, f = stops[i], stops[i + 1], t - i
    return QColor(
        int(lo.red() + (hi.red() - lo.red()) * f),
        int(lo.green() + (hi.green() - lo.green()) * f),
        int(lo.blue() + (hi.blue() - lo.blue()) * f),
    )


def _lift(ground: str):
    """Color map for flag colors drawn on `ground`: dark colors are lifted off a dark ground,
    on a light window they stay as they are."""
    if _luminance(QColor(ground)) >= 0.18:
        return None
    return lambda c: _toward(c, ground, 2.0)


_flag_images: dict[tuple, QImage] = {}


def flag_image(width: int, height: int, ground: str | None = None):
    """The look's flag as an image of `width` x `height` (colors lifted for `ground`, the
    window by default), for sampling. None for the plain look."""
    if country is None:
        return None
    ground = ground or _window.name()
    key = (country.code, width, height, ground)
    if key not in _flag_images:
        image = QImage(max(1, width), max(1, height), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        flags.paint_flag(painter, QRectF(0, 0, width, height), country, 0, recolor=_lift(ground))
        painter.end()
        _flag_images[key] = image
    return _flag_images[key]


def paint_dissolve(
    painter: QPainter,
    rect: QRectF,
    step=6.0,
    seed=7,
    opacity=1.0,
    color: QColor | None = None,
    reverse=False,
    bands: tuple[str, ...] | None = None,
    outline_white: bool | None = None,
    ground: str | None = None,
):
    """The signature of the plugin: the flag dissolving into noise.
    A halftone of dots, full at the left of `rect`, shrinking and dropping out towards the
    right (`reverse` flips the direction). In the country look every dot takes the color of
    the flag at its place, so vertical tricolors, crosses and discs come through as they are;
    `ground` is what the dots sit on (the window by default). With `bands` the dots are rows
    of those colors, with `color` all dots get that one color (the plain look: the accent).
    Dots that would vanish into the ground (white on a light window) get a thin outline."""
    if outline_white is None:
        outline_white = color is None
    rng = random.Random(seed)
    ground = ground or _window.name()
    image = flag_image(int(rect.width() * 0.55), int(rect.height()), ground)
    if bands is None and image is None and color is None:
        color = QColor(accent)
    if bands is not None or color is not None:
        image = None
    n = len(bands) if bands else 3 if image is not None else 1
    rows = max(1, int(rect.height() / step))
    if n > 1:
        rows = max(n, n * round(rect.height() / step / n))
    row_step = rect.height() / rows
    cols = max(2, int(rect.width() / step))
    edge = QColor(white_edge if _luminance(QColor(ground)) >= 0.18 else grey)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    for col in range(cols):
        t = col / (cols - 1)
        for row in range(rows):
            if rng.random() < t * 0.6:
                continue  # noise takes over
            radius = min(step, row_step) * 0.42 * (1 - t) ** 1.2 + 0.35
            pos = 1 - t if reverse else t
            x = rect.left() + (rect.width() - step) * pos + (step / 2 if row % 2 else step / 4)
            y = rect.top() + (row + 0.5) * row_step
            if image is not None:
                # the whole flag fits in the solid part, its last column then dissolves
                fx = min(max((x - rect.left()) / rect.width() / 0.55, 0.0), 1.0)
                px = int(fx * (image.width() - 1))
                py = min(max(int(y - rect.top()), 0), image.height() - 1)
                dot = image.pixelColor(px, py)
            elif bands:
                dot = QColor(bands[row * n // rows])
            else:
                dot = QColor(color)
            alpha = opacity * (0.35 + 0.65 * (1 - t))
            faint = outline_white and color is None and contrast(dot, ground) < 1.3
            dot.setAlphaF(alpha)
            if faint:
                line = QColor(edge)
                line.setAlphaF(alpha)
                painter.setPen(QPen(line, 0.9))
                dot.setAlphaF(1.0)
            else:
                painter.setPen(Qt.PenStyle.NoPen)
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
    """A thin bar in the look: the flag stretched to fit (so a vertical tricolor shows as three
    segments side by side, a cross as a cross), or the one plain color."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    window = _window.name()
    if country is not None:
        flags.paint_flag(painter, rect, country, 1.5, recolor=_lift(window))
    else:
        clip = QPainterPath()
        clip.addRoundedRect(rect, 1.5, 1.5)
        painter.fillPath(clip, QColor(accent))
    if country is not None and any(_band_outline):  # keep a white band readable
        painter.setPen(QPen(QColor(white_edge if not is_dark else grey), 0.8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect.adjusted(-0.4, -0.4, 0.4, 0.4), 1.5, 1.5)
    painter.restore()


def paint_flag(painter: QPainter, rect: QRectF, of: Country | None = None, radius=2.0):
    """The flag of the look's country (or `of`) in `rect`. Nothing for the plain look."""
    of = of or country
    if of is not None:
        edge = grey if is_dark else white_edge
        flags.paint_flag(painter, rect, of, radius, outline=edge)


def flag_icon(of: Country | None, width=30, height=20):
    """The flag of `of` as an icon (for lists of countries); empty for None."""
    pixmap = QPixmap(width * 2, height * 2)
    pixmap.fill(Qt.GlobalColor.transparent)
    if of is not None:
        painter = QPainter(pixmap)
        flags.paint_flag(painter, QRectF(1, 1, width * 2 - 2, height * 2 - 2), of, 4, outline=grey)
        painter.end()
    return QIcon(pixmap)


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
        ground.setColorAt(1, QColor(banner_end))
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
            dots.adjust(0, 9, 0, -9)
            bands = None if country is not None else banner_bands
            paint_dissolve(
                painter, dots, step=7.5, seed=11, opacity=0.95, bands=bands, ground=flag_blue
            )
        if country is not None:  # the flag itself, small, in the corner
            painter.setClipping(False)
            flag_rect = QRectF(rect.right() - 12 - 27, 11, 27, 18)
            flags.paint_flag(painter, flag_rect, country, 2.5, outline="#55ffffff")


def _logo_shapes():
    """The stroke shapes of logo.svg (rects and dots with their opacity), in its 128 grid."""
    import re

    text = (icon_path / "logo.svg").read_text(encoding="utf-8")
    shapes = []
    for tag, attrs in re.findall(r"<(rect|circle)\s([^>]*)/>", text):
        a = dict(re.findall(r'(\w+)="([^"]*)"', attrs))
        if a.get("fill", "").startswith("url("):
            continue  # the tile
        path = QPainterPath()
        if tag == "rect":
            r = QRectF(float(a["x"]), float(a["y"]), float(a["width"]), float(a["height"]))
            path.addRoundedRect(r, float(a.get("rx", 0)), float(a.get("rx", 0)))
        else:
            c = QPointF(float(a["cx"]), float(a["cy"]))
            path.addEllipse(c, float(a["r"]), float(a["r"]))
        shapes.append((path, float(a.get("opacity", 1))))
    return shapes


_logo_cache: dict[tuple, QPixmap] = {}


def logo(size=128):
    """The logo in the colors of the look. In the country look the flag shows through the
    three strokes as it is (a vertical tricolor runs along them, a disc sits across them); in
    the plain look the strokes take the one color. Both on a deep ground of the look's hue."""
    if country is None:
        pixmap = QIcon(str(_logo)).pixmap(size, size) if _logo.suffix == ".svg" else QPixmap()
        return pixmap if not pixmap.isNull() else QPixmap(str(icon_path / "logo-128.png"))
    key = (country.code, size, flag_blue)
    if key in _logo_cache:
        return _logo_cache[key]
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.scale(size / 128, size / 128)
    tile = QLinearGradient(QPointF(4, 4), QPointF(124, 124))
    tile.setColorAt(0, QColor(flag_blue))
    tile.setColorAt(1, QColor(banner_end))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(tile)
    painter.drawRoundedRect(QRectF(4, 4, 120, 120), 28, 28)
    try:
        area = QRectF(18, 29, 102, 70)  # where the strokes are
        flag = QImage(int(area.width() * 4), int(area.height() * 4), QImage.Format.Format_ARGB32)
        flag.fill(Qt.GlobalColor.transparent)
        flag_painter = QPainter(flag)
        flags.paint_flag(flag_painter, QRectF(flag.rect()), country, 0, recolor=_lift(flag_blue))
        flag_painter.end()
        for path, opacity in _logo_shapes():
            painter.save()
            painter.setClipPath(path)
            painter.setOpacity(opacity)
            painter.drawImage(area, flag)
            painter.restore()
    except Exception as e:
        log.warning(f"Could not draw the logo strokes: {e}")
    painter.end()
    _logo_cache[key] = pixmap
    return pixmap


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


apply_look()
