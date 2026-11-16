"""Generates the icon set of the plugin into ai_diffusion/icons.

One drawing per icon on a 24 unit grid: round 2 unit lines in the theme's text color, with
details in red or blue (the brand colors). Each icon is written twice, for dark and for light
Krita themes. Run: python scripts/make_icons.py
"""

from pathlib import Path

icons_dir = Path(__file__).parent.parent / "ai_diffusion" / "icons"

themes = {
    "dark": {"c": "#d2d2d2", "r": "#f0505a", "b": "#7ea8f5"},
    "light": {"c": "#3d3d3d", "r": "#ae1c28", "b": "#21468b"},
}
warning_color = "#e0a030"


def p(d: str, color="c", width=2.0, extra=""):
    return f'<path d="{d}" stroke="{{{color}}}" stroke-width="{width}" {extra}/>'


def f(d: str, color="c", extra=""):
    return f'<path d="{d}" fill="{{{color}}}" stroke="none" {extra}/>'


def dot(x: float, y: float, r: float, color="c", extra=""):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{{{color}}}" stroke="none" {extra}/>'


def ring(x: float, y: float, r: float, color="c", extra=""):
    return f'<circle cx="{x}" cy="{y}" r="{r}" stroke="{{{color}}}" {extra}/>'


def box(x: float, y: float, w: float, h: float, rx=2.0, color="c", extra=""):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" stroke="{{{color}}}" {extra}/>'
    )


def fbox(x: float, y: float, w: float, h: float, rx=2.0, color="c", extra=""):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{{{color}}}"'
        f' stroke="none" {extra}/>'
    )


def star(cx: float, cy: float, r: float, color="c", outline=False):
    k = round(r * 0.14, 2)
    d = (
        f"M{cx} {cy - r}Q{cx + k} {cy - k} {cx + r} {cy}Q{cx + k} {cy + k} {cx} {cy + r}"
        f"Q{cx - k} {cy + k} {cx - r} {cy}Q{cx - k} {cy - k} {cx} {cy - r}Z"
    )
    return p(d, color, 1.6) if outline else f(d, color)


def plus(x: float, y: float, color="r", size=3.0):
    return p(f"M{x} {y - size}v{2 * size}M{x - size} {y}h{2 * size}", color)


dash = 'stroke-dasharray="2.2 3"'
selection = ring(12, 12, 8.5, extra=dash)
corners = p(
    "M4 9V5.5A1.5 1.5 0 0 1 5.5 4H9M15 4h3.5A1.5 1.5 0 0 1 20 5.5V9M20 15v3.5a1.5 1.5 0 0 1-1.5 1.5H15M9 20H5.5A1.5 1.5 0 0 1 4 18.5V15"
)
layers = p("M12 4l9 5-9 5-9-5z") + p("M3 14l9 5 9-5")
chain = p("M10.5 13.5a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1.2 1.2") + p(
    "M13.5 10.5a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1.2-1.2"
)
figure = ring(9, 5, 2.4) + p("M9 8v7M4 11.5h10M9 15l-4 6M9 15l4 6")
tray = p("M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3")
queue_lines = "M4 7h16", "M4 12h16", "M4 17h16"
document = p("M7 3h7l5 5v12a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z") + p("M14 3v5h5")
workspace = {
    "generation": star(10, 13.5, 9) + star(19.5, 5, 4, "r"),
    "upscaling": fbox(3, 13, 8, 8, 1.6)
    + p("M3 8.5V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-3.5")
    + p("M12.5 11.5L17 7M13 7h4v4", "r"),
    "live": f("M13.5 2L5 13.6h5.8L9.5 22 19 9.8h-6.2z", "r"),
    "animation": p("M8 4h11a2 2 0 0 1 2 2v8.5")
    + box(3, 8, 14, 12)
    + f("M8.3 11.2v5.6L13 14z", "r"),
    "custom": p("M6 12C12 12 11 5.5 17 5.5M6 12c6 0 5 6.5 11 6.5", "b")
    + dot(5, 12, 2.8)
    + dot(19, 5.5, 2.8)
    + dot(19, 18.5, 2.8),
}

icons: dict[str, str] = {
    "add-pose": figure + plus(19, 6),
    "alert": p("M12 3.5L21.5 20h-19z") + p("M12 10v4.5", "r") + dot(12, 17.2, 1.1, "r"),
    "apply": p("M4.5 12.5l5 5L20 6.5", "b", 2.4),
    "apply-layer": p("M12 9l9 4.5-9 4.5-9-4.5z") + p("M8 5.5l3 3 5.5-6", "b", 2.2),
    "cancel": p("M6 6l12 12M18 6L6 18", "r", 2.4),
    "comfyui": box(3, 4, 8, 6, 1.5) + box(13, 14, 8, 6, 1.5) + p("M11 7h3a3 3 0 0 1 3 3v4", "b"),
    "context": corners + dot(12, 12, 2),
    "context-automatic": corners + star(12, 12, 5, "r"),
    "context-image": corners + fbox(8, 8, 8, 8, 1.2),
    "context-layer": corners + p("M12 8.5l5 3.5-5 3.5-5-3.5z", width=1.6),
    "context-mask": corners + ring(12, 12, 4, "b", 'stroke-dasharray="1.6 2.6"'),
    "control-add": p("M11 6l8 4.5-8 4.5-8-4.5z")
    + p("M3 15l8 4.5 4-2.2")
    + plus(19.5, 17.5, size=2.6),
    "control-blur": dot(12, 12, 3)
    + ring(12, 12, 6, extra='opacity="0.6"')
    + ring(12, 12, 9.3, extra='opacity="0.3" stroke-dasharray="1 3"'),
    "control-canny_edge": p("M3 18L8 7l4 8 4-10 5 11", extra='stroke-linejoin="miter"'),
    "control-composition": box(3, 4, 18, 16) + p("M9 4v16M15 4v16M3 9.3h18M3 14.7h18", "b", 1.2),
    "control-depth": fbox(3, 11, 10, 10, 1.5)
    + box(8, 7, 10, 10, 1.5, extra='opacity="0.6"')
    + box(13, 3, 8, 8, 1.5, extra='opacity="0.35"'),
    "control-face": ring(12, 12, 9) + dot(9, 10, 1.2) + dot(15, 10, 1.2) + p("M8.5 14.5q3.5 3 7 0"),
    "control-generate": box(3, 10, 11, 10) + p("M3 17l3.5-3.5 4 4") + star(17.5, 7, 5, "r"),
    "control-hands": p(
        "M8 13V6.5a1.5 1.5 0 0 1 3 0V11M11 4.5a1.5 1.5 0 0 1 3 0V11M14 6a1.5 1.5 0 0 1 3 0v6"
        "M17 9.5a1.5 1.5 0 0 1 3 0V15a6 6 0 0 1-6 6h-1.6a6 6 0 0 1-4.5-2L4.6 15"
        "a1.6 1.6 0 0 1 2.5-2L8 14",
        width=1.7,
    ),
    "control-line_art": p("M3 18C7 6 10 20 13 11s6-5 8-6"),
    "control-normal": ring(10, 14, 7) + p("M12.5 11.5L20 4M15 4h5v5", "b"),
    "control-pose": ring(13, 4.5, 2.3) + p("M12 8l-1 7M5 9l7 1 6-3M11 15l-5 5M11 15l5 2 1 4"),
    "control-reference": box(3, 5, 18, 14) + p("M3 16l5-5 4 4 3-3 6 6") + dot(16, 9.5, 1.6, "r"),
    "control-scribble": p("M3 15c2-8 5-8 5-2s3 6 5-1 3-6 5 0 2 4 3 1"),
    "control-segmentation": f("M12 12V3a9 9 0 0 1 7.8 13.5z", "b")
    + ring(12, 12, 9)
    + p("M12 12V3M12 12l7.8 4.5M12 12l-7.8 4.5"),
    "control-soft_edge": p("M3 15Q8 5 12 12T21 9", width=6, extra='opacity="0.3"')
    + p("M3 15Q8 5 12 12T21 9"),
    "control-stencil": f(
        "M5 3h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"
        "M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10z",
        extra='fill-rule="evenodd"',
    ),
    "control-style": p("M20 4l-9 9")
    + f("M11.5 12.5c-3 0-5 1.6-5 4s-1.2 3.2-3.2 3.5c4 1.2 9.2 0 9.2-4.2 0-1.3-.3-2.4-1-3.3z", "r"),
    "copy-image": box(8, 8, 13, 13) + p("M5 16H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h11a1 1 0 0 1 1 1v1"),
    "discard": p("M6 7.5l1 12.5h10l1-12.5M10 11v6M14 11v6") + p("M4 7h16M9 7V4h6v3", "r"),
    "edit": p("M4 20l1-4L16 5l3 3L8 19z") + p("M13.5 7.5l3 3", "b"),
    "file-json": document
    + p("M10.5 11.5c-1.4 0-.9 1.7-1 2.3-.1.5-.8.7-.8.7s.7.2.8.7c.1.6-.4 2.3 1 2.3", "b", 1.5)
    + p("M13.5 11.5c1.4 0 .9 1.7 1 2.3.1.5.8.7.8.7s-.7.2-.8.7c-.1.6.4 2.3-1 2.3", "b", 1.5),
    "file-kra": document + dot(12.5, 14.5, 2.8, "r"),
    "fill": fbox(4, 4, 16, 16, 3),
    "fill-empty": box(4, 4, 16, 16, 3, extra=dash),
    "filter": p("M3 5h18l-7 8v6l-4 2v-8z"),
    "generate": workspace["generation"],
    "generate-region": star(12, 12, 7) + corners.replace("{c}", "{b}"),
    "import": tray + p("M12 3v11M8 10l4 4 4-4", "b"),
    "improve-prompt": p("M4 20L14.5 9.5")
    + star(17.5, 6.5, 5, "r")
    + dot(7, 7, 1.1)
    + dot(19, 16, 1.1),
    "inpaint-add_object": selection + plus(12, 12, size=3.5),
    "inpaint-automatic": selection + star(12, 12, 5, "r"),
    "inpaint-custom": selection
    + dot(8.5, 12, 1.3)
    + dot(12, 12, 1.3, "r")
    + dot(15.5, 12, 1.3, "b"),
    "inpaint-expand": box(8, 8, 8, 8, 1.2)
    + p(
        "M12 5V2M10 3.5L12 2l2 1.5M12 19v3M10 20.5l2 1.5 2-1.5M5 12H2M3.5 10L2 12l1.5 2M19 12h3M20.5 10l1.5 2-1.5 2",
        "b",
        1.5,
    ),
    "inpaint-fill": selection + dot(12, 12, 4.5),
    "inpaint-remove_object": selection + p("M8.5 12h7", "r"),
    "inpaint-replace_background": box(3, 3, 18, 18, extra=dash)
    + dot(12, 9.5, 3)
    + f("M6 19.5c0-4 2.8-6 6-6s6 2 6 6z"),
    "link": chain,
    "link-active": chain.replace("{c}", "{b}"),
    "link-disabled": f'<g opacity="0.35">{chain}</g>',
    "link-off": chain + p("M4 4l16 16", "r"),
    "more": dot(5, 12, 1.8) + dot(12, 12, 1.8) + dot(19, 12, 1.8),
    "pause": fbox(6, 5, 4, 14, 1.2) + fbox(14, 5, 4, 14, 1.2),
    "play": f("M7.5 4.2a1 1 0 0 1 1.5-.8l11 7.8a1 1 0 0 1 0 1.6l-11 7.8a1 1 0 0 1-1.5-.8z"),
    "queue-inactive": "".join(p(d, extra='opacity="0.75"') for d in queue_lines),
    "queue-active": p(queue_lines[0], "r", 2.4) + p(queue_lines[1]) + p(queue_lines[2]),
    "queue-waiting": p("M4 7h9M4 12h7M4 17h9")
    + ring(17.5, 12, 4.2, "b")
    + p("M17.5 10v2.2l1.4 1", "b", 1.4),
    "queue-upload": p("M4 20h16") + p("M12 16V4M7 9l5-5 5 5", "b"),
    "random": box(4, 4, 16, 16, 3)
    + dot(8.5, 8.5, 1.4)
    + dot(15.5, 15.5, 1.4)
    + dot(12, 12, 1.4, "r")
    + dot(15.5, 8.5, 1.4)
    + dot(8.5, 15.5, 1.4),
    "record": ring(12, 12, 8) + dot(12, 12, 3.5),
    "record-active": ring(12, 12, 8, "r") + dot(12, 12, 4.5, "r"),
    "refine": star(11, 13, 9, outline=True) + star(19.5, 5, 3.5, "r"),
    "refine-region": star(12, 12, 6.5, outline=True) + corners.replace("{c}", "{b}"),
    "region-add": box(3, 7, 14, 14, extra=dash) + plus(19, 6),
    "region-alpha": box(4, 4, 16, 16)
    + fbox(4, 4, 8, 8, 1.5, extra='opacity="0.55"')
    + fbox(12, 12, 8, 8, 1.5, extra='opacity="0.55"'),
    "region-alpha-active": box(4, 4, 16, 16, color="b")
    + fbox(4, 4, 8, 8, 1.5, "b")
    + fbox(12, 12, 8, 8, 1.5, "b"),
    "region-prompt": box(3, 4, 18, 16) + p("M8 9h8M12 9v7", "b"),
    "remove": p("M6 12h12", "r", 2.4),
    "reset": p("M4.5 12a7.5 7.5 0 1 0 2.3-5.4") + p("M4 4v4.5h4.5", "b"),
    "resolution-multiplier": box(3, 3, 18, 18, extra=dash) + fbox(3, 12, 9, 9, 1.6, "b"),
    "root": p("M4 11l8-7 8 7M6 10v10h12V10") + p("M10 20v-5h4v5", "r"),
    "save": p("M5 3h11l4 4v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z")
    + p("M8 3v5h7V3", "b")
    + p("M7 21v-7h10v7"),
    "seed": p("M12 21c-5 0-7-4-7-8 0-5 4-9 7-10 3 1 7 5 7 10 0 4-2 8-7 8z") + p("M12 21V11", "b"),
    "settings": p("M4 7h8M18 7h2M4 17h2M12 17h8") + ring(15, 7, 2.6, "r") + ring(9, 17, 2.6, "b"),
    "upload": tray + p("M12 14V3M8 7l4-4 4 4", "b"),
    "warning": f'<path d="M12 3.5L21.5 20h-19z" stroke="{warning_color}" stroke-width="2"/>'
    f'<path d="M12 10v4.5" stroke="{warning_color}" stroke-width="2"/>'
    f'<circle cx="12" cy="17.2" r="1.1" fill="{warning_color}" stroke="none"/>',
    "web-connection": ring(12, 12, 9)
    + p("M3 12h18", "b")
    + p("M12 3c3.5 3 3.5 15 0 18M12 3c-3.5 3-3.5 15 0 18"),
}
icons.update({f"workspace-{name}": body for name, body in workspace.items()})

# Model families: a badge in red or blue with white letters
badges = {
    "15": ("1.5", "b"), "xl": ("XL", "b"), "3": ("SD3", "b"), "illu": ("IL", "b"),
    "illu-v": ("IV", "b"), "anima": ("AN", "b"), "flux": ("FX", "r"), "flux-k": ("FK", "r"),
    "flux-2": ("F2", "r"), "chroma": ("CH", "r"), "krea2": ("K2", "r"), "qwen": ("QW", "b"),
    "z-image": ("ZI", "r"), "ernie": ("ER", "b"), "moss": ("MS", "r"), "fibo2": ("FB", "b"),
    "h3": ("H3", "r"),
}  # fmt: skip
badge_fill = {"r": "#c8242f", "b": "#2f5fc4"}
for name, (text, color) in badges.items():
    size = 8.5 if len(text) > 2 else 10.5
    icons[f"sd-version-{name}"] = (
        f'<rect x="1.5" y="4.5" width="21" height="15" rx="4" fill="{badge_fill[color]}"'
        ' stroke="none"/>'
        f'<text x="12" y="{12 + size * 0.36:.1f}" font-family="Arial, Helvetica, sans-serif"'
        f' font-size="{size}" font-weight="bold" text-anchor="middle" fill="#ffffff"'
        f' stroke="none">{text}</text>'
    )


def write_icons():
    for old in icons_dir.glob("*-dark.*"):
        old.unlink()
    for old in icons_dir.glob("*-light.*"):
        old.unlink()
    for name, body in icons.items():
        for theme, colors in themes.items():
            svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 24 24"'
                ' fill="none" stroke-linecap="round" stroke-linejoin="round" stroke-width="2">'
                f"{body.format(**colors)}</svg>\n"
            )
            (icons_dir / f"{name}-{theme}.svg").write_text(svg)
    print(f"{len(icons)} icons written to {icons_dir}")


if __name__ == "__main__":
    write_icons()
