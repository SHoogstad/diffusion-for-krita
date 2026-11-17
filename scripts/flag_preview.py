"""Check country files and render a preview of their flags and colors.

    python scripts/flag_preview.py                 check every built-in country file
    python scripts/flag_preview.py NL FR           check them and write flag_preview.png
    python scripts/flag_preview.py my/XX.json      same for a file of your own
    python scripts/flag_preview.py --all           a preview of every built-in country

The preview shows each flag large, at the sizes the plugin uses (24 x 16 and 16 x 11), and the
colors it gives the interface: the bands, the accent (Generate) and the cool color (on/focus).
Needs PyQt5 or PyQt6 (`pip install PyQt6`), not Krita.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "tests"))  # mock krita module

from ai_diffusion.qt.QtCore import QRectF, Qt
from ai_diffusion.qt.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter
from ai_diffusion.ui import flags


def find(arg: str) -> Path:
    path = Path(arg)
    if path.suffix == ".json":
        return path
    for folder in (flags.user_dir, flags.builtin_dir):
        if (folder / f"{arg.upper()}.json").exists():
            return folder / f"{arg.upper()}.json"
    raise SystemExit(f"No country file for '{arg}' in {flags.builtin_dir} or {flags.user_dir}")


def render(countries: list[flags.Country], out: Path):
    row_h, width = 92, 560
    image = QImage(width, 16 + row_h * len(countries), QImage.Format.Format_ARGB32)
    image.fill(QColor("#2b2b2b"))
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = QFont()
    font.setPixelSize(13)
    p.setFont(font)
    for i, c in enumerate(countries):
        y = 8 + i * row_h
        flags.paint_flag(p, QRectF(12, y + 8, 108, 72), c, 4)
        flags.paint_flag(p, QRectF(132, y + 8, 24, 16), c, 2)
        flags.paint_flag(p, QRectF(132, y + 32, 16, 11), c, 1.5)
        p.setPen(QColor("#e6e6e6"))
        p.drawText(QRectF(172, y + 6, 380, 20), Qt.AlignmentFlag.AlignLeft, f"{c.code}  {c.name}")
        swatches = [("bands", list(c.bands)), ("accent", [c.accent])]
        swatches.append(("cool", [c.cool or QColor(c.accent).darker(135).name()]))
        x = 172
        for label, colors in swatches:
            for color in colors:
                p.fillRect(QRectF(x, y + 32, 28, 28), QColor(color))
                x += 30
            p.setPen(QColor("#9a9a9a"))
            p.drawText(QRectF(x - 30 * len(colors), y + 62, 120, 18), 0, label)
            x += 24
    p.end()
    image.save(str(out))
    print(f"Preview written to {out}")


def main(args: list[str]):
    app = QGuiApplication(sys.argv[:1])  # noqa: F841 - fonts need it
    if not args or args == ["--all"]:
        paths = sorted(flags.builtin_dir.glob("*.json"))
    else:
        paths = [find(a) for a in args]
    ok: list[flags.Country] = []
    failed = 0
    for path in paths:
        try:
            ok.append(flags.load_file(path))
        except ValueError as e:
            failed += 1
            print(f"ERROR {e}")
    print(f"{len(ok)} valid, {failed} with errors")
    if ok and args:
        render(ok, Path.cwd() / "flag_preview.png")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
