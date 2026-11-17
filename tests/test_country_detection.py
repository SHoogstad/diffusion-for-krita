"""Which country the country look starts with (ui/flags.py detect_country)."""

from ai_diffusion.ui import flags


def test_territory_from_locale_names():
    assert flags._territory("nl_NL.UTF-8") == "NL"
    assert flags._territory("de_AT.UTF-8@euro") == "AT"
    assert flags._territory("en-NL") == "NL"
    assert flags._territory("C.UTF-8") == ""
    assert flags._territory("en") == ""


def test_linux_prefers_where_you_live_over_language():
    env = {"LANG": "en_US.UTF-8", "LC_TIME": "nl_NL.UTF-8", "LC_MONETARY": "nl_NL.UTF-8"}
    assert flags.detect_country("linux", env) == "NL"
    assert flags.detect_country("linux", {"LANG": "nl_NL.UTF-8"}) == "NL"
    assert flags.detect_country("linux", {"LC_ALL": "fr_FR.UTF-8", "LC_TIME": "nl_NL"}) == "FR"


def test_windows_uses_country_or_region(monkeypatch):
    # Regional format English (United States), Country or region Netherlands
    monkeypatch.setattr(flags, "_windows_region", lambda: "NL")
    assert flags.detect_country("win32") == "NL"
    monkeypatch.setattr(flags, "_windows_region", lambda: "001")  # "World": not a country
    assert flags.detect_country("win32") == flags._territory(flags.QLocale.system().name())
