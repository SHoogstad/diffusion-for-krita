"""The legacy look: the interface as it was before the redesign (InterfaceLook.legacy)."""

from ai_diffusion.settings import InterfaceLook
from ai_diffusion.ui import theme


def test_legacy_look():
    try:
        theme.apply_look(InterfaceLook.legacy, "", "#3d7be0")
        assert theme.legacy
        assert theme.docker_stylesheet == theme.primary_button_stylesheet == ""
        assert theme.red in ("#d07a40", "#c07630")  # the old error color
        for name in ("workspace-generation", "settings", "sd-version-flux-2", "logo-128"):
            old = list(theme.legacy_icon_path.glob(f"{name}*"))
            assert old, f"legacy icon {name} is missing"

        theme.apply_look(InterfaceLook.plain, "", "#3d7be0")
        assert not theme.legacy
        assert theme.docker_stylesheet != ""
    finally:
        theme.apply_look()  # back to the look from the settings
