"""Country looks: the flag of a country, drawn small, and the colors the interface takes from it.

Every country is a JSON file named after its two letter code (ISO 3166-1 alpha-2), eg. `NL.json`.
The plugin ships them in `ai_diffusion/countries`. Files in the `countries` folder of the user
data folder add countries or replace built-in ones. See `ai_diffusion/countries/README.md` for
the format and `scripts/flag_preview.py` to see a flag before using it.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..qt.QtCore import QLocale, QPointF, QRectF, Qt
from ..qt.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from ..util import client_logger as log
from ..util import plugin_dir, user_data_dir


@dataclass(frozen=True)
class Country:
    code: str
    name: str
    bands: tuple[str, ...]  # two or three flag colors, top to bottom: stripes, halftone, logo
    accent: str  # the primary action (Generate)
    cool: str | None  # state that is on, focus and selection (None = a deeper `accent`)
    flag: tuple[dict, ...] = field(default=(), compare=False)  # shapes, drawn in order
    source: str = field(default="", compare=False)  # file it was read from


builtin_dir = plugin_dir / "countries"
user_dir = user_data_dir / "countries"

# Shapes of a flag. Coordinates are fractions of the flag: x, y from the top left, w, h of the
# width and height. A radius `r` is a fraction of the height. Fields with a default are optional.
_number, _color, _colors = "number", "color", "colors"
_required = None  # a field without a default
shapes: dict[str, dict[str, tuple[str, object]]] = {
    "stripes": {"colors": (_colors, _required), "weights": ("numbers", ()), "vertical": ("bool", False)},
    "rect": {"color": (_color, _required), "x": (_number, 0), "y": (_number, 0), "w": (_number, 1), "h": (_number, 1)},
    "star": {"color": (_color, _required), "x": (_number, 0.5), "y": (_number, 0.5), "r": (_number, _required)},
    "disc": {"color": (_color, _required), "x": (_number, 0.5), "y": (_number, 0.5), "r": (_number, _required)},
    "ring": {"color": (_color, _required), "x": (_number, 0.5), "y": (_number, 0.5), "r": (_number, _required)},
    "crescent": {"color": (_color, _required), "x": (_number, 0.5), "y": (_number, 0.5), "r": (_number, _required)},
    "nordic-cross": {"color": (_color, _required), "thickness": (_number, _required), "x": (_number, 0.36)},
    "triangle": {"color": (_color, _required), "depth": (_number, _required)},
    "diamond": {"color": (_color, _required)},
    "union-jack": {"x": (_number, 0), "y": (_number, 0), "w": (_number, 1), "h": (_number, 1)},
    "taegeuk": {"top": (_color, _required), "bottom": (_color, _required), "r": (_number, _required)},
    "star-grid": {
        "color": (_color, _required), "x": (_number, 0), "y": (_number, 0), "w": (_number, 1),
        "h": (_number, 1), "columns": ("count", _required), "rows": ("count", _required),
    },
}  # fmt: skip

_hex_color = re.compile(r"^#([0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def _value_error(kind: str, value) -> str | None:
    if kind == _color:
        ok = isinstance(value, str) and _hex_color.match(value)
        return None if ok else f'{value!r} is not a color like "#ae1c28"'
    if kind == _colors:
        if not isinstance(value, list) or not value:
            return "needs a list of colors"
        return next((e for v in value if (e := _value_error(_color, v))), None)
    if kind == _number:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        return None if ok else f"{value!r} is not a number"
    if kind == "numbers":
        if not isinstance(value, list) or not value:
            return "needs a list of numbers"
        return next((e for v in value if (e := _value_error(_number, v))), None)
    if kind == "count":
        ok = isinstance(value, int) and not isinstance(value, bool) and value > 0
        return None if ok else f"{value!r} is not a whole number above 0"
    if kind == "bool":
        return None if isinstance(value, bool) else f"{value!r} is not true or false"
    return f"unknown kind {kind}"


def check(code: str, data) -> list[str]:
    """Everything wrong with the contents of a country file, empty if it is fine."""
    errors: list[str] = []
    if not (len(code) == 2 and code.isalpha() and code.isupper()):
        errors.append(f"file name: '{code}' is not a two letter country code in capitals (NL)")
    if not isinstance(data, dict):
        return errors + ["the file must contain one JSON object { ... }"]
    if not isinstance(data.get("name"), str) or not data["name"].strip():
        errors.append("name: missing")
    bands = data.get("bands")
    if e := _value_error(_colors, bands):
        errors.append(f"bands: {e}")
    elif not 2 <= len(bands) <= 3:
        errors.append("bands: needs two or three colors")
    if e := _value_error(_color, data.get("accent")):
        errors.append(f"accent: {e}")
    if data.get("cool") is not None and (e := _value_error(_color, data["cool"])):
        errors.append(f"cool: {e}")
    known = {"name", "bands", "accent", "cool", "flag", "comment"}
    errors += [f"{key}: unknown field" for key in data if key not in known]
    flag = data.get("flag")
    if not isinstance(flag, list) or not flag:
        return errors + ["flag: needs a list of shapes"]
    for i, layer in enumerate(flag, 1):
        where = f"flag, shape {i}"
        if not isinstance(layer, dict) or layer.get("shape") not in shapes:
            got = layer.get("shape") if isinstance(layer, dict) else layer
            errors.append(f"{where}: {got!r} is not one of {', '.join(shapes)}")
            continue
        spec = shapes[layer["shape"]]
        where += f" ({layer['shape']})"
        for key, (kind, default) in spec.items():
            if key not in layer:
                if default is _required:
                    errors.append(f"{where}: {key} is missing")
            elif e := _value_error(kind, layer[key]):
                errors.append(f"{where}: {key}: {e}")
        errors += [f"{where}: {k}: unknown field" for k in layer if k not in spec and k != "shape"]
        weights = layer.get("weights")
        if isinstance(weights, list) and weights and len(weights) != len(layer.get("colors") or []):
            errors.append(f"{where}: weights needs one number per color")
    return errors


def _complete(layer: dict) -> dict:
    """The layer with the defaults of its shape filled in."""
    spec = shapes[layer["shape"]]
    return {"shape": layer["shape"], **{k: d for k, (_, d) in spec.items()}, **layer}


def load_file(path: Path) -> Country:
    """The country in a file, raises ValueError with every problem if it is not valid."""
    code = path.stem
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"{path.name}: {e}") from e
    if errors := check(code, data):
        raise ValueError(f"{path.name}: " + "; ".join(errors))
    return Country(
        code=code,
        name=data["name"].strip(),
        bands=tuple(data["bands"]),
        accent=data["accent"],
        cool=data.get("cool"),
        flag=tuple(_complete(layer) for layer in data["flag"]),
        source=str(path),
    )


def load(folders: list[Path] | None = None) -> dict[str, Country]:
    """All countries: the built-in ones, then the user's, which replace a built-in one with the
    same code. A file that is not valid is skipped with a warning in the log."""
    result: dict[str, Country] = {}
    for folder in folders if folders is not None else [builtin_dir, user_dir]:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.json")):
            try:
                country = load_file(path)
            except ValueError as e:
                log.warning(f"Country look skipped, {folder}/{e}")
                continue
            result[country.code] = country
    return result


countries: dict[str, Country] = load()


def reload():
    """Read the country files again, eg. after the user added one."""
    global countries
    countries = load()


def _territory(locale_name: str) -> str:
    """The country of a locale name ("nl_NL.UTF-8@euro", "en-NL" -> "NL"), "" if it has none."""
    name = locale_name.split(".", 1)[0].split("@", 1)[0].replace("-", "_")
    parts = name.split("_")
    if len(parts) >= 2 and len(parts[-1]) == 2 and parts[-1].isalpha():
        return parts[-1].upper()
    return ""


def _windows_region() -> str:
    """The "Country or region" of Windows. It is a setting of its own: the regional format
    (what QLocale.system() reports) is often "English (United States)" anywhere."""
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    buffer = ctypes.create_unicode_buffer(16)
    try:  # Windows 10 1709 and later
        if kernel32.GetUserDefaultGeoName(buffer, len(buffer)) > 1:
            return buffer.value
    except AttributeError:
        pass
    geo_id = kernel32.GetUserGeoID(16)  # GEOCLASS_NATION
    if geo_id != -1 and kernel32.GetGeoInfoW(geo_id, 4, buffer, len(buffer), 0) > 1:  # ISO2
        return buffer.value
    return ""


# Linux: the categories that say most about where someone lives come first. LC_ALL overrides
# everything, LANG is the default for all, often only the language (en_US with Dutch formats).
_locale_variables = (
    "LC_ALL", "LC_ADDRESS", "LC_TELEPHONE", "LC_MONETARY", "LC_PAPER", "LC_MEASUREMENT",
    "LC_TIME", "LC_NUMERIC", "LANG",
)  # fmt: skip


def detect_country(platform: str | None = None, environ: dict[str, str] | None = None) -> str:
    """Country of this computer as a two letter code ("NL"). Empty if the system names none.
    Windows: its "Country or region" setting. Linux: the locale variables. macOS and the rest:
    the system locale, which there includes the region (eg. "en_NL")."""
    import os
    import sys

    platform = platform or sys.platform
    if platform.startswith("win"):
        try:
            region = _windows_region().upper()
            if len(region) == 2 and region.isalpha():
                return region
        except Exception as e:  # no ctypes or an old Windows: the system locale below
            log.warning(f"Could not read the Windows region: {e}")
    elif platform.startswith("linux"):
        environ = os.environ if environ is None else environ
        for variable in _locale_variables:
            if code := _territory(environ.get(variable, "")):
                return code
    return _territory(QLocale.system().name())


def resolve(code: str) -> Country | None:
    """The country for a setting value: "" means detect it."""
    return countries.get(code or detect_country())


def country_items():
    """(name, code) pairs for a selector, sorted by name."""
    return sorted(((c.name, c.code) for c in countries.values()), key=lambda x: x[0])


# Painting


def _star(cx: float, cy: float, r: float):
    points = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.4
        points.append(QPointF(cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return QPolygonF(points)


def _bands(painter: QPainter, r: QRectF, colors, weights, vertical: bool, col):
    weights = weights or [1] * len(colors)
    total = sum(weights)
    pos = 0.0
    for color, w in zip(colors, weights):
        if vertical:
            part = QRectF(
                r.left() + r.width() * pos / total, r.top(), r.width() * w / total + 0.4, r.height()
            )
        else:
            part = QRectF(
                r.left(),
                r.top() + r.height() * pos / total,
                r.width(),
                r.height() * w / total + 0.4,
            )
        painter.fillRect(part, col(color))
        pos += w


_navy_uk, _red_uk = "#012169", "#c8102e"


def _union(painter: QPainter, r: QRectF, col):
    painter.save()
    painter.setClipRect(r)
    painter.fillRect(r, col(_navy_uk))
    s = r.height() / 30  # the jack is 60 x 30 units
    a, b = r.topLeft(), r.bottomRight()
    c, d = r.bottomLeft(), r.topRight()
    for color, width in (("#ffffff", 6 * s), (_red_uk, 2 * s)):
        pen = QPen(col(color), width)
        painter.setPen(pen)
        painter.drawLine(a, b)
        painter.drawLine(c, d)
    painter.setPen(Qt.PenStyle.NoPen)
    cx, cy = r.center().x(), r.center().y()
    for color, half in (("#ffffff", 5 * s), (_red_uk, 3 * s)):
        painter.fillRect(QRectF(cx - half, r.top(), 2 * half, r.height()), col(color))
        painter.fillRect(QRectF(r.left(), cy - half, r.width(), 2 * half), col(color))
    painter.restore()


def paint_flag(
    painter: QPainter,
    rect: QRectF,
    country: Country,
    radius=2.0,
    outline: str = "",
    recolor: Callable[[str], str] | None = None,
):
    """The flag of `country` in `rect`, with rounded corners and an optional outline.
    `recolor` maps every flag color (eg. to lift dark colors off a dark ground)."""

    def col(c: str):
        return QColor(recolor(c) if recolor else c)

    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    clip = QPainterPath()
    clip.addRoundedRect(rect, radius, radius)
    painter.setClipPath(clip)
    painter.setPen(Qt.PenStyle.NoPen)
    x0, y0, w, h = rect.left(), rect.top(), rect.width(), rect.height()

    def sub(x, y, ww, hh):
        return QRectF(x0 + x * w, y0 + y * h, ww * w, hh * h)

    for layer in country.flag:
        shape = layer["shape"]
        if shape == "stripes":
            _bands(painter, rect, layer["colors"], layer["weights"], layer["vertical"], col)
        elif shape == "rect":
            painter.fillRect(
                sub(layer["x"], layer["y"], layer["w"], layer["h"]), col(layer["color"])
            )
        elif shape in ("star", "disc", "ring", "crescent"):
            cx, cy, r = x0 + layer["x"] * w, y0 + layer["y"] * h, layer["r"] * h
            color = col(layer["color"])
            if shape == "star":
                painter.setBrush(color)
                painter.drawPolygon(_star(cx, cy, r))
            elif shape == "disc" and r > 0:
                painter.setBrush(color)
                painter.drawEllipse(QPointF(cx, cy), r, r)
            elif shape == "ring":
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(color, max(0.8, r * 0.3)))
                painter.drawEllipse(QPointF(cx, cy), r, r)
                painter.setPen(Qt.PenStyle.NoPen)
            elif shape == "crescent":
                outer = QPainterPath()
                outer.addEllipse(QPointF(cx, cy), r, r)
                inner = QPainterPath()
                inner.addEllipse(QPointF(cx + r * 0.3, cy), r * 0.8, r * 0.8)
                painter.fillPath(outer.subtracted(inner), color)
        elif shape == "nordic-cross":
            th = layer["thickness"] * h
            color = col(layer["color"])
            painter.fillRect(QRectF(x0 + layer["x"] * w - th / 2, y0, th, h), color)
            painter.fillRect(QRectF(x0, y0 + h / 2 - th / 2, w, th), color)
        elif shape == "triangle":
            tri = QPolygonF([
                QPointF(x0, y0),
                QPointF(x0 + layer["depth"] * w, y0 + h / 2),
                QPointF(x0, y0 + h),
            ])
            painter.setBrush(col(layer["color"]))
            painter.drawPolygon(tri)
        elif shape == "diamond":
            m = 0.08
            pts = [(0.5, m), (1 - m, 0.5), (0.5, 1 - m), (m, 0.5)]
            painter.setBrush(col(layer["color"]))
            painter.drawPolygon(QPolygonF([QPointF(x0 + px * w, y0 + py * h) for px, py in pts]))
        elif shape == "taegeuk":
            r = layer["r"] * h
            box = QRectF(x0 + w / 2 - r, y0 + h / 2 - r, 2 * r, 2 * r)
            painter.setBrush(col(layer["top"]))
            painter.drawPie(box, 0, 180 * 16)
            painter.setBrush(col(layer["bottom"]))
            painter.drawPie(box, 180 * 16, 180 * 16)
        elif shape == "union-jack":
            _union(painter, sub(layer["x"], layer["y"], layer["w"], layer["h"]), col)
        elif shape == "star-grid":
            area = sub(layer["x"], layer["y"], layer["w"], layer["h"])
            cols, rows = layer["columns"], layer["rows"]
            r = min(area.width() / cols, area.height() / rows) * 0.22
            painter.setBrush(col(layer["color"]))
            for j in range(rows):
                for i in range(cols):
                    px = area.left() + (i + 0.5 + (0.5 if j % 2 else 0) * 0.5) * area.width() / cols
                    py = area.top() + (j + 0.5) * area.height() / rows
                    painter.drawEllipse(QPointF(px, py), r, r)
    painter.setClipping(False)
    if outline:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(outline), 0.8))
        painter.drawRoundedRect(rect.adjusted(-0.4, -0.4, 0.4, 0.4), radius, radius)
    painter.restore()
