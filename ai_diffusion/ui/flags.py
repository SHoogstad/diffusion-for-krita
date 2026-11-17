"""Country looks: the flag of a country, drawn small, and the colors the interface takes from it.

Every country has
- `bands`: two or three flag colors, top to bottom, for the brand stripes, the halftone and
  the logo strokes,
- `accent`: the color of the primary action (Generate),
- `cool`: the color for state that is on, focus and selection (None = a deeper `accent`),
- `layers`: how to draw the flag, simplified so it still reads at 16 pixels.

Layers (coordinates are fractions of the flag rectangle):
    ("h", colors[, weights])  horizontal bands      ("v", colors[, weights])  vertical bands
    ("rect", color, x, y, w, h)                     ("star", color, cx, cy, r)  r: of height
    ("disc", color, cx, cy, r)                      ("crescent", color, cx, cy, r)
    ("nordic", color, thickness[, x])               ("triangle", color, depth)
    ("diamond", color)                              ("union", x, y, w, h)  (Union Jack)
    ("taegeuk", top, bottom, r)
    ("dots", color, x, y, w, h, cols, rows)         ("ring", color, cx, cy, r)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

from ..qt.QtCore import QLocale, QPointF, QRectF, Qt
from ..qt.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF


class Country(NamedTuple):
    code: str
    name: str
    bands: tuple[str, ...]
    accent: str
    cool: str | None
    layers: tuple


def _c(code, name, bands, accent, cool, *layers):
    return Country(code, name, tuple(bands), accent, cool, tuple(layers))


_red_us, _blue_us = "#b22234", "#3c3b6e"
_navy_uk, _red_uk = "#012169", "#c8102e"
_southern_cross = [(0.75, 0.24, 0.07), (0.63, 0.47, 0.07), (0.86, 0.42, 0.07), (0.75, 0.82, 0.08)]

countries: dict[str, Country] = {
    c.code: c
    for c in [
        _c("NL", "Netherlands", ["#ae1c28", "#ffffff", "#21468b"], "#ae1c28", "#21468b",
           ("h", ["#ae1c28", "#ffffff", "#21468b"])),
        _c("BE", "Belgium", ["#000000", "#fdda24", "#ef3340"], "#ef3340", "#fdda24",
           ("v", ["#000000", "#fdda24", "#ef3340"])),
        _c("LU", "Luxembourg", ["#ed2939", "#ffffff", "#00a1de"], "#ed2939", "#00a1de",
           ("h", ["#ed2939", "#ffffff", "#00a1de"])),
        _c("FR", "France", ["#002654", "#ffffff", "#ce1126"], "#ce1126", "#002654",
           ("v", ["#002654", "#ffffff", "#ce1126"])),
        _c("DE", "Germany", ["#000000", "#dd0000", "#ffcc00"], "#dd0000", "#ffcc00",
           ("h", ["#000000", "#dd0000", "#ffcc00"])),
        _c("AT", "Austria", ["#c8102e", "#ffffff", "#c8102e"], "#c8102e", None,
           ("h", ["#c8102e", "#ffffff", "#c8102e"])),
        _c("CH", "Switzerland", ["#da291c", "#ffffff"], "#da291c", None,
           ("h", ["#da291c"]), ("rect", "#ffffff", 0.43, 0.2, 0.14, 0.6),
           ("rect", "#ffffff", 0.31, 0.4, 0.38, 0.2)),
        _c("IT", "Italy", ["#009246", "#ffffff", "#ce2b37"], "#ce2b37", "#009246",
           ("v", ["#009246", "#ffffff", "#ce2b37"])),
        _c("ES", "Spain", ["#aa151b", "#f1bf00", "#aa151b"], "#aa151b", "#f1bf00",
           ("h", ["#aa151b", "#f1bf00", "#aa151b"], [1, 2, 1])),
        _c("PT", "Portugal", ["#046a38", "#da291c"], "#da291c", "#046a38",
           ("v", ["#046a38", "#da291c"], [2, 3]), ("disc", "#ffe900", 0.4, 0.5, 0.2)),
        _c("IE", "Ireland", ["#169b62", "#ffffff", "#ff883e"], "#ff883e", "#169b62",
           ("v", ["#169b62", "#ffffff", "#ff883e"])),
        _c("GB", "United Kingdom", [_red_uk, "#ffffff", _navy_uk], _red_uk, _navy_uk,
           ("union", 0, 0, 1, 1)),
        _c("US", "United States", [_red_us, "#ffffff", _blue_us], _red_us, _blue_us,
           ("h", [_red_us, "#ffffff"] * 6 + [_red_us]), ("rect", _blue_us, 0, 0, 0.4, 7 / 13),
           ("dots", "#ffffff", 0, 0, 0.4, 7 / 13, 6, 5)),
        _c("CA", "Canada", ["#d80621", "#ffffff", "#d80621"], "#d80621", None,
           ("v", ["#d80621", "#ffffff", "#d80621"], [1, 2, 1]), ("star", "#d80621", 0.5, 0.5, 0.3)),
        _c("AU", "Australia", ["#e4002b", "#ffffff", "#012169"], "#e4002b", "#012169",
           ("h", ["#012169"]), ("union", 0, 0, 0.5, 0.5), ("star", "#ffffff", 0.25, 0.75, 0.13),
           *[("star", "#ffffff", x, y, r) for x, y, r in _southern_cross]),
        _c("NZ", "New Zealand", ["#c8102e", "#ffffff", "#012169"], "#c8102e", "#012169",
           ("h", ["#012169"]), ("union", 0, 0, 0.5, 0.5),
           *[("star", "#c8102e", x, y, r) for x, y, r in _southern_cross]),
        _c("SE", "Sweden", ["#006aa7", "#fecc02", "#006aa7"], "#fecc02", "#006aa7",
           ("h", ["#006aa7"]), ("nordic", "#fecc02", 0.2)),
        _c("NO", "Norway", ["#ba0c2f", "#ffffff", "#00205b"], "#ba0c2f", "#00205b",
           ("h", ["#ba0c2f"]), ("nordic", "#ffffff", 0.25), ("nordic", "#00205b", 0.125)),
        _c("DK", "Denmark", ["#c8102e", "#ffffff", "#c8102e"], "#c8102e", None,
           ("h", ["#c8102e"]), ("nordic", "#ffffff", 0.14)),
        _c("FI", "Finland", ["#ffffff", "#002f6c", "#ffffff"], "#002f6c", None,
           ("h", ["#ffffff"]), ("nordic", "#002f6c", 0.27)),
        _c("IS", "Iceland", ["#02529c", "#ffffff", "#dc1e35"], "#dc1e35", "#02529c",
           ("h", ["#02529c"]), ("nordic", "#ffffff", 0.24), ("nordic", "#dc1e35", 0.11)),
        _c("PL", "Poland", ["#ffffff", "#dc143c"], "#dc143c", None,
           ("h", ["#ffffff", "#dc143c"])),
        _c("CZ", "Czechia", ["#ffffff", "#d7141a", "#11457e"], "#d7141a", "#11457e",
           ("h", ["#ffffff", "#d7141a"]), ("triangle", "#11457e", 0.5)),
        _c("SK", "Slovakia", ["#ffffff", "#0b4ea2", "#ee1c25"], "#ee1c25", "#0b4ea2",
           ("h", ["#ffffff", "#0b4ea2", "#ee1c25"])),
        _c("HU", "Hungary", ["#ce2939", "#ffffff", "#477050"], "#ce2939", "#477050",
           ("h", ["#ce2939", "#ffffff", "#477050"])),
        _c("SI", "Slovenia", ["#ffffff", "#005da4", "#ed1c24"], "#ed1c24", "#005da4",
           ("h", ["#ffffff", "#005da4", "#ed1c24"])),
        _c("HR", "Croatia", ["#ff0000", "#ffffff", "#171796"], "#ff0000", "#171796",
           ("h", ["#ff0000", "#ffffff", "#171796"])),
        _c("RS", "Serbia", ["#c6363c", "#0c4076", "#ffffff"], "#c6363c", "#0c4076",
           ("h", ["#c6363c", "#0c4076", "#ffffff"])),
        _c("RO", "Romania", ["#002b7f", "#fcd116", "#ce1126"], "#ce1126", "#002b7f",
           ("v", ["#002b7f", "#fcd116", "#ce1126"])),
        _c("BG", "Bulgaria", ["#ffffff", "#00966e", "#d62612"], "#d62612", "#00966e",
           ("h", ["#ffffff", "#00966e", "#d62612"])),
        _c("GR", "Greece", ["#0d5eaf", "#ffffff", "#0d5eaf"], "#0d5eaf", None,
           ("h", ["#0d5eaf", "#ffffff"] * 4 + ["#0d5eaf"]), ("rect", "#0d5eaf", 0, 0, 0.37, 5 / 9),
           ("rect", "#ffffff", 0.148, 0, 0.074, 5 / 9), ("rect", "#ffffff", 0, 2 / 9, 0.37, 1 / 9)),
        _c("UA", "Ukraine", ["#0057b7", "#ffd700"], "#ffd700", "#0057b7",
           ("h", ["#0057b7", "#ffd700"])),
        _c("RU", "Russia", ["#ffffff", "#0039a6", "#d52b1e"], "#d52b1e", "#0039a6",
           ("h", ["#ffffff", "#0039a6", "#d52b1e"])),
        _c("LT", "Lithuania", ["#fdb913", "#006a44", "#c1272d"], "#c1272d", "#006a44",
           ("h", ["#fdb913", "#006a44", "#c1272d"])),
        _c("LV", "Latvia", ["#9e3039", "#ffffff", "#9e3039"], "#9e3039", None,
           ("h", ["#9e3039", "#ffffff", "#9e3039"], [2, 1, 2])),
        _c("EE", "Estonia", ["#0072ce", "#000000", "#ffffff"], "#0072ce", None,
           ("h", ["#0072ce", "#000000", "#ffffff"])),
        _c("TR", "Turkey", ["#e30a17", "#ffffff", "#e30a17"], "#e30a17", None,
           ("h", ["#e30a17"]), ("crescent", "#ffffff", 0.36, 0.5, 0.25),
           ("star", "#ffffff", 0.53, 0.5, 0.12)),
        _c("IL", "Israel", ["#0038b8", "#ffffff", "#0038b8"], "#0038b8", None,
           ("h", ["#ffffff", "#0038b8", "#ffffff", "#0038b8", "#ffffff"], [1.5, 1.5, 7, 1.5, 1.5]),
           ("ring", "#0038b8", 0.5, 0.5, 0.2)),
        _c("EG", "Egypt", ["#ce1126", "#ffffff", "#000000"], "#ce1126", None,
           ("h", ["#ce1126", "#ffffff", "#000000"]), ("disc", "#c09300", 0.5, 0.5, 0.1)),
        _c("AE", "United Arab Emirates", ["#00732f", "#ffffff", "#000000"], "#ff0000", "#00732f",
           ("h", ["#00732f", "#ffffff", "#000000"]), ("rect", "#ff0000", 0, 0, 0.25, 1)),
        _c("SA", "Saudi Arabia", ["#006c35", "#ffffff"], "#006c35", None,
           ("h", ["#006c35"]), ("rect", "#ffffff", 0.25, 0.62, 0.5, 0.06),
           ("rect", "#ffffff", 0.3, 0.3, 0.4, 0.14)),
        _c("ZA", "South Africa", ["#e03c31", "#007749", "#001489"], "#e03c31", "#007749",
           ("h", ["#e03c31", "#ffffff", "#007749", "#ffffff", "#001489"], [4, 1, 3, 1, 4]),
           ("triangle", "#ffffff", 0.42), ("triangle", "#007749", 0.36),
           ("triangle", "#ffb612", 0.3), ("triangle", "#000000", 0.25)),
        _c("NG", "Nigeria", ["#008751", "#ffffff", "#008751"], "#008751", None,
           ("v", ["#008751", "#ffffff", "#008751"])),
        _c("KE", "Kenya", ["#000000", "#bb0000", "#006600"], "#bb0000", "#006600",
           ("h", ["#000000", "#ffffff", "#bb0000", "#ffffff", "#006600"], [6, 1, 6, 1, 6])),
        _c("MA", "Morocco", ["#c1272d", "#006233"], "#c1272d", "#006233",
           ("h", ["#c1272d"]), ("star", "#006233", 0.5, 0.5, 0.26)),
        _c("JP", "Japan", ["#ffffff", "#bc002d", "#ffffff"], "#bc002d", None,
           ("h", ["#ffffff"]), ("disc", "#bc002d", 0.5, 0.5, 0.3)),
        _c("KR", "South Korea", ["#cd2e3a", "#ffffff", "#0047a0"], "#cd2e3a", "#0047a0",
           ("h", ["#ffffff"]), ("taegeuk", "#cd2e3a", "#0047a0", 0.25)),
        _c("CN", "China", ["#ee1c25", "#ffff00", "#ee1c25"], "#ee1c25", None,
           ("h", ["#ee1c25"]), ("star", "#ffff00", 0.17, 0.25, 0.15),
           ("star", "#ffff00", 0.33, 0.1, 0.05), ("star", "#ffff00", 0.4, 0.2, 0.05),
           ("star", "#ffff00", 0.4, 0.35, 0.05), ("star", "#ffff00", 0.33, 0.45, 0.05)),
        _c("TW", "Taiwan", ["#fe0000", "#000095", "#ffffff"], "#fe0000", "#000095",
           ("h", ["#fe0000"]), ("rect", "#000095", 0, 0, 0.5, 0.5),
           ("disc", "#ffffff", 0.25, 0.25, 0.13)),
        _c("HK", "Hong Kong", ["#de2910", "#ffffff"], "#de2910", None,
           ("h", ["#de2910"]), ("star", "#ffffff", 0.5, 0.5, 0.3)),
        _c("VN", "Vietnam", ["#da251d", "#ffff00", "#da251d"], "#da251d", None,
           ("h", ["#da251d"]), ("star", "#ffff00", 0.5, 0.5, 0.3)),
        _c("TH", "Thailand", ["#a51931", "#f4f5f8", "#2d2a4a"], "#a51931", "#2d2a4a",
           ("h", ["#a51931", "#f4f5f8", "#2d2a4a", "#f4f5f8", "#a51931"], [1, 1, 2, 1, 1])),
        _c("ID", "Indonesia", ["#ff0000", "#ffffff"], "#ff0000", None,
           ("h", ["#ff0000", "#ffffff"])),
        _c("MY", "Malaysia", ["#cc0001", "#ffffff", "#010066"], "#cc0001", "#010066",
           ("h", ["#cc0001", "#ffffff"] * 7), ("rect", "#010066", 0, 0, 0.5, 8 / 14),
           ("crescent", "#ffcc00", 0.2, 0.29, 0.18), ("star", "#ffcc00", 0.33, 0.29, 0.1)),
        _c("SG", "Singapore", ["#ef3340", "#ffffff"], "#ef3340", None,
           ("h", ["#ef3340", "#ffffff"]), ("crescent", "#ffffff", 0.17, 0.25, 0.17)),
        _c("PH", "Philippines", ["#0038a8", "#ffffff", "#ce1126"], "#ce1126", "#0038a8",
           ("h", ["#0038a8", "#ce1126"]), ("triangle", "#ffffff", 0.43),
           ("disc", "#fcd116", 0.14, 0.5, 0.1)),
        _c("IN", "India", ["#ff9933", "#ffffff", "#138808"], "#ff9933", "#138808",
           ("h", ["#ff9933", "#ffffff", "#138808"]), ("ring", "#000080", 0.5, 0.5, 0.13)),
        _c("PK", "Pakistan", ["#01411c", "#ffffff"], "#01411c", None,
           ("v", ["#ffffff", "#01411c"], [1, 3]), ("crescent", "#ffffff", 0.6, 0.5, 0.27),
           ("star", "#ffffff", 0.72, 0.38, 0.1)),
        _c("BD", "Bangladesh", ["#006a4e", "#f42a41"], "#f42a41", "#006a4e",
           ("h", ["#006a4e"]), ("disc", "#f42a41", 0.45, 0.5, 0.33)),
        _c("BR", "Brazil", ["#009c3b", "#ffdf00", "#002776"], "#009c3b", "#002776",
           ("h", ["#009c3b"]), ("diamond", "#ffdf00"), ("disc", "#002776", 0.5, 0.5, 0.25)),
        _c("AR", "Argentina", ["#74acdf", "#ffffff", "#74acdf"], "#74acdf", "#f6b40e",
           ("h", ["#74acdf", "#ffffff", "#74acdf"]), ("disc", "#f6b40e", 0.5, 0.5, 0.1)),
        _c("CL", "Chile", ["#0039a6", "#ffffff", "#d52b1e"], "#d52b1e", "#0039a6",
           ("h", ["#ffffff", "#d52b1e"]), ("rect", "#0039a6", 0, 0, 1 / 3, 0.5),
           ("star", "#ffffff", 1 / 6, 0.25, 0.13)),
        _c("CO", "Colombia", ["#fcd116", "#003893", "#ce1126"], "#ce1126", "#003893",
           ("h", ["#fcd116", "#003893", "#ce1126"], [2, 1, 1])),
        _c("PE", "Peru", ["#d91023", "#ffffff", "#d91023"], "#d91023", None,
           ("v", ["#d91023", "#ffffff", "#d91023"])),
        _c("VE", "Venezuela", ["#ffcc00", "#00247d", "#cf142b"], "#cf142b", "#00247d",
           ("h", ["#ffcc00", "#00247d", "#cf142b"])),
        _c("MX", "Mexico", ["#006847", "#ffffff", "#ce1126"], "#ce1126", "#006847",
           ("v", ["#006847", "#ffffff", "#ce1126"]), ("disc", "#8c6a2f", 0.5, 0.5, 0.12)),
        _c("CU", "Cuba", ["#002a8f", "#ffffff", "#cf142b"], "#cf142b", "#002a8f",
           ("h", ["#002a8f", "#ffffff"] * 2 + ["#002a8f"]), ("triangle", "#cf142b", 0.42),
           ("star", "#ffffff", 0.14, 0.5, 0.11)),
    ]
}  # fmt: skip


def detect_country() -> str:
    """Country of this computer, from the system locale (eg. "nl_NL" -> "NL"). Empty if the
    locale names no country."""
    name = QLocale.system().name()
    parts = name.replace("-", "_").split("_")
    if len(parts) >= 2 and len(parts[-1]) == 2 and parts[-1].isalpha():
        return parts[-1].upper()
    return ""


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

    for layer in country.layers:
        kind = layer[0]
        if kind in ("h", "v"):
            weights = layer[2] if len(layer) > 2 else None
            _bands(painter, rect, layer[1], weights, kind == "v", col)
        elif kind == "rect":
            painter.fillRect(sub(*layer[2:6]), col(layer[1]))
        elif kind == "star":
            _, color, cx, cy, r = layer
            painter.setBrush(col(color))
            painter.drawPolygon(_star(x0 + cx * w, y0 + cy * h, r * h))
        elif kind == "disc":
            _, color, cx, cy, r = layer
            if r > 0:
                painter.setBrush(col(color))
                painter.drawEllipse(QPointF(x0 + cx * w, y0 + cy * h), r * h, r * h)
        elif kind == "ring":
            _, color, cx, cy, r = layer
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(col(color), max(0.8, r * h * 0.3)))
            painter.drawEllipse(QPointF(x0 + cx * w, y0 + cy * h), r * h, r * h)
            painter.setPen(Qt.PenStyle.NoPen)
        elif kind == "crescent":
            _, color, cx, cy, r = layer
            outer = QPainterPath()
            outer.addEllipse(QPointF(x0 + cx * w, y0 + cy * h), r * h, r * h)
            inner = QPainterPath()
            inner.addEllipse(
                QPointF(x0 + cx * w + r * h * 0.3, y0 + cy * h), r * h * 0.8, r * h * 0.8
            )
            painter.fillPath(outer.subtracted(inner), col(color))
        elif kind == "nordic":
            color, t = layer[1], layer[2]
            cx = layer[3] if len(layer) > 3 else 0.36
            th = t * h
            painter.fillRect(QRectF(x0 + cx * w - th / 2, y0, th, h), col(color))
            painter.fillRect(QRectF(x0, y0 + h / 2 - th / 2, w, th), col(color))
        elif kind == "triangle":
            _, color, depth = layer
            tri = QPolygonF([
                QPointF(x0, y0),
                QPointF(x0 + depth * w, y0 + h / 2),
                QPointF(x0, y0 + h),
            ])
            painter.setBrush(col(color))
            painter.drawPolygon(tri)
        elif kind == "diamond":
            m = 0.08
            pts = [(0.5, m), (1 - m, 0.5), (0.5, 1 - m), (m, 0.5)]
            painter.setBrush(col(layer[1]))
            painter.drawPolygon(QPolygonF([QPointF(x0 + px * w, y0 + py * h) for px, py in pts]))
        elif kind == "taegeuk":
            _, top, bottom, r = layer
            box = QRectF(x0 + w / 2 - r * h, y0 + h / 2 - r * h, 2 * r * h, 2 * r * h)
            painter.setBrush(col(top))
            painter.drawPie(box, 0, 180 * 16)
            painter.setBrush(col(bottom))
            painter.drawPie(box, 180 * 16, 180 * 16)
        elif kind == "union":
            _union(painter, sub(*layer[1:5]), col)
        elif kind == "dots":
            _, color, x, y, ww, hh, cols, rows = layer
            area = sub(x, y, ww, hh)
            r = min(area.width() / cols, area.height() / rows) * 0.22
            painter.setBrush(col(color))
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
