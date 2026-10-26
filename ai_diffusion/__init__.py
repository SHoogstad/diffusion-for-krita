"""Generative AI plugin for Krita"""

__version__ = "1.55.3"
# Release tag on GitHub, used by the auto-updater. One build runs on Krita 5 and Krita 6.
__release__ = f"v{__version__}"

import importlib.util

if not importlib.util.find_spec(".websockets.src", "ai_diffusion"):
    raise ImportError(
        "Could not find websockets module. This indicates that it was not installed with the"
        " plugin. Please make sure to download a plugin release package (NOT just the source!). You"
        " can find the latest release package here:"
        " https://github.com/SHoogstad/diffusion-for-krita/releases"
    )

# The following imports depend on the code running inside Krita
if importlib.util.find_spec("krita"):
    import krita

    if not getattr(krita, "IS_MOCK", False):
        from .qt import QT6, binding

        krita_ver = krita.Krita.instance().version()
        if not krita_ver.startswith(("5", "6")):
            raise ImportError(
                f"This Plugin is for Krita 5 and 6, but you are using Krita {krita_ver}."
            )
        if krita_ver.startswith("6") != QT6:
            raise ImportError(f"Krita {krita_ver} does not match the Qt binding {binding}.")

        from .extension import AIToolsExtension as AIToolsExtension

# When not running inside Krita, try to import the development placeholder for Krita functions
else:
    import sys
    from pathlib import Path

    mock_dir = Path(__file__).parent.parent / "tests" / "mock"
    if mock_dir.exists():
        sys.path.append(str(mock_dir))
