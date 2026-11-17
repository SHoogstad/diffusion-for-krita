"""Country looks come from JSON files (ai_diffusion/countries, plus the user's own folder)."""

import json

from ai_diffusion.ui import flags

nl = {
    "name": "Netherlands",
    "bands": ["#ae1c28", "#ffffff", "#21468b"],
    "accent": "#ae1c28",
    "cool": "#21468b",
    "flag": [{"shape": "stripes", "colors": ["#ae1c28", "#ffffff", "#21468b"]}],
}


def test_every_builtin_file_is_valid():
    paths = sorted(flags.builtin_dir.glob("*.json"))
    assert len(paths) >= 60
    for path in paths:
        flags.load_file(path)  # raises with every problem in the file
    assert set(flags.countries) >= {p.stem for p in paths}


def test_check_reports_every_problem():
    bad = {
        "name": "",
        "bands": ["#ae1c28"],
        "accent": "red",
        "extra": 1,
        "flag": [
            {"shape": "stripes", "colors": ["#ffffff"], "weights": [1, 2]},
            {"shape": "star", "color": "#ffffff"},
            {"shape": "moon"},
        ],
    }
    errors = flags.check("nl", bad)
    text = "\n".join(errors)
    for expected in [
        "file name",
        "name: missing",
        "bands: needs two or three colors",
        "accent: 'red' is not a color",
        "extra: unknown field",
        "weights needs one number per color",
        "shape 2 (star): r is missing",
        "'moon' is not one of",
    ]:
        assert expected in text, f"{expected!r} not in {errors}"
    assert flags.check("NL", nl) == []


def test_defaults_are_filled_in(tmp_path):
    data = dict(nl, flag=[{"shape": "stripes", "colors": ["#ffffff"]}, {"shape": "disc", "color": "#bc002d", "r": 0.3}])  # fmt: skip
    (tmp_path / "JP.json").write_text(json.dumps(data))
    country = flags.load_file(tmp_path / "JP.json")
    assert country.code == "JP"
    assert country.flag[0]["vertical"] is False and country.flag[0]["weights"] == ()
    assert (country.flag[1]["x"], country.flag[1]["y"]) == (0.5, 0.5)


def test_user_files_add_and_replace_countries(tmp_path):
    builtin, user = tmp_path / "builtin", tmp_path / "user"
    builtin.mkdir()
    user.mkdir()
    (builtin / "NL.json").write_text(json.dumps(nl))
    (user / "NL.json").write_text(json.dumps(dict(nl, accent="#ff6600")))
    (user / "XK.json").write_text(json.dumps(dict(nl, name="Kosovo")))
    (user / "BR.json").write_text("{ not json")
    (user / "ZZ.json").write_text(json.dumps(dict(nl, flag=[])))

    countries = flags.load([builtin, user])
    assert countries["NL"].accent == "#ff6600"
    assert countries["XK"].name == "Kosovo"
    assert "BR" not in countries and "ZZ" not in countries  # skipped, not fatal
    assert flags.load([tmp_path / "missing"]) == {}
