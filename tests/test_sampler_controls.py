"""Sampler, scheduler and guider chosen in the style, without a custom preset in samplers.json."""

import json
from pathlib import Path

import pytest

from ai_diffusion.backend import workflow
from ai_diffusion.backend.comfy_workflow import ComfyWorkflow, ConditioningOutput, Output
from ai_diffusion.backend.resources import Arch
from ai_diffusion.style import SamplerPresets, Style


def make_style(**settings):
    style = Style(Path("default.json"))
    for name, value in settings.items():
        setattr(style, name, value)
    return style


def test_preset_is_the_default():
    style = make_style(sampler="Default - DPM++ 2M", live_sampler="Realtime - Hyper")
    preset = SamplerPresets.instance()["Default - DPM++ 2M"]
    assert style.get_sampler(is_live=False) == (preset.sampler, preset.scheduler, "auto")
    live = SamplerPresets.instance()["Realtime - Hyper"]
    assert style.get_sampler(is_live=True) == (live.sampler, live.scheduler, "auto")


def test_style_overrides_preset():
    style = make_style(
        sampler="Default - DPM++ 2M",
        sampler_method="er_sde",
        sampler_guider="basic",
        live_sampler_scheduler="beta",
        live_sampler_guider="cfg",
    )
    preset = SamplerPresets.instance()["Default - DPM++ 2M"]
    assert style.get_sampler(is_live=False) == ("er_sde", preset.scheduler, "basic")
    live = SamplerPresets.instance()[style.live_sampler]
    assert style.get_sampler(is_live=True) == (live.sampler, "beta", "cfg")

    sampling = workflow.sampling_from_style(style, 1.0, is_live=False)
    assert (sampling.sampler, sampling.scheduler, sampling.guider) == (
        "er_sde",
        preset.scheduler,
        "basic",
    )
    params = workflow._sampler_params(sampling, workflow.Extent(1024, 1024))
    assert params["sampler"] == "er_sde" and params["guider"] == "basic"


def test_save_and_load(tmp_path: Path):
    style = Style(tmp_path / "custom.json")
    style.sampler_method = "res_multistep"
    style.sampler_scheduler = "kl_optimal"
    style.sampler_guider = "cfg"
    style.preferred_megapixels = 16.0
    style.save()
    loaded = Style.load(tmp_path / "custom.json")
    assert loaded is not None
    assert loaded.sampler_method == "res_multistep" and loaded.sampler_scheduler == "kl_optimal"
    assert loaded.sampler_guider == "cfg" and loaded.live_sampler_guider == "auto"
    assert loaded.preferred_megapixels == pytest.approx(16.0)

    # a style from an older version has none of the new settings, a bad guider falls back
    cfg = json.loads((tmp_path / "custom.json").read_text())
    for name in ("sampler_method", "sampler_scheduler", "preferred_megapixels"):
        del cfg[name]
    cfg["sampler_guider"] = "nonsense"
    cfg["preferred_megapixels"] = 8  # whole numbers are fine
    (tmp_path / "old.json").write_text(json.dumps(cfg))
    old = Style.load(tmp_path / "old.json")
    assert old is not None
    assert old.sampler_method == "" and old.sampler_scheduler == "" and old.sampler_guider == "auto"
    assert old.preferred_megapixels == 8


def guider_of(arch: Arch, cfg: float, guider: str):
    w = ComfyWorkflow()
    cond = ConditioningOutput(Output(100, 0), Output(101, 0))
    w.sampler_custom_advanced(
        Output(1, 0), cond, Output(2, 0), arch, steps=4, cfg=cfg, seed=1, guider=guider
    )
    nodes = list(w.root.values())
    kinds = [n["class_type"] for n in nodes if n["class_type"].endswith("Guider")]
    assert len(kinds) == 1
    flux = [n["inputs"]["guidance"] for n in nodes if n["class_type"] == "FluxGuidance"]
    return kinds[0], (flux[0] if flux else None)


@pytest.mark.parametrize(
    "arch, cfg, guider, expected",
    [
        (Arch.sdxl, 7.0, "auto", "CFGGuider"),
        (Arch.sdxl, 1.0, "auto", "BasicGuider"),
        (Arch.sdxl, 7.0, "basic", "BasicGuider"),
        (Arch.sdxl, 1.0, "cfg", "CFGGuider"),
        (Arch.zimage, 1.0, "auto", "BasicGuider"),
        (Arch.zimage, 4.0, "cfg", "CFGGuider"),
    ],
)
def test_guider(arch: Arch, cfg: float, guider: str, expected: str):
    assert guider_of(arch, cfg, guider)[0] == expected


def test_guider_flux():
    # Flux: the scale is distilled guidance, unless a real CFG guider is asked for
    assert guider_of(Arch.flux, 2.5, "auto") == ("BasicGuider", pytest.approx(2.5))
    assert guider_of(Arch.flux, 2.5, "basic") == ("BasicGuider", pytest.approx(2.5))
    assert guider_of(Arch.flux, 2.5, "cfg") == ("CFGGuider", pytest.approx(3.5))


def test_negative_prompt_follows_guider():
    cond = workflow.ConditioningInput("a fox", negative="blurry")
    style = make_style(cfg_scale=5.0)
    sampling = workflow.sampling_from_style(style, 1.0, is_live=False)
    assert workflow.Conditioning.from_input(cond, sampling).negative is not None
    style.sampler_guider = "basic"
    sampling = workflow.sampling_from_style(style, 1.0, is_live=False)
    assert workflow.Conditioning.from_input(cond, sampling).negative is None


# Widgets need a QApplication, the test session only has a QCoreApplication: run it on its own.
_widget_script = """
import sys
from pathlib import Path

from ai_diffusion.qt.QtWidgets import QApplication

app = QApplication([])

from ai_diffusion.style import SamplerPresets, Style, StyleSettings
from ai_diffusion.ui.style import SamplerWidget


def make_style(**settings):
    style = Style(Path("default.json"))
    for name, value in settings.items():
        setattr(style, name, value)
    return style


widget = SamplerWidget("", "Quality", None)
style = make_style(sampler="Default - DPM++ 2M")
preset = SamplerPresets.instance()["Default - DPM++ 2M"]
widget.read(style)
assert widget._method.value == preset.sampler and widget._scheduler.value == preset.scheduler
assert widget._guider.value == "auto"

# unchanged: nothing extra is stored
widget.write(style)
assert style.sampler_method == "" and style.sampler_scheduler == ""

# pick a sampler and guider from the lists
widget._method.value = "er_sde"
widget._guider.value = "basic"
widget._change_sampler()  # what picking from the list triggers
widget.write(style)
assert style.sampler_method == "er_sde" and style.sampler_scheduler == ""
assert style.sampler_guider == "basic"
assert "changed below" in widget._sampler_info.text()

# a sampler the lists do not know (custom node) still shows
style.sampler_method = "my_custom_sampler"
widget.read(style)
assert widget._method.value == "my_custom_sampler"

# choosing a preset resets the choice
widget._preset.setCurrentText("Alternative - Euler A")
alt = SamplerPresets.instance()["Alternative - Euler A"]
assert widget._method.value == alt.sampler and widget._guider.value == "auto"
widget.write(style)
assert style.sampler == "Alternative - Euler A" and style.sampler_method == ""
assert style.sampler_guider == StyleSettings.sampler_guider.default
print("widget ok")
"""


def test_widget():
    import os
    import subprocess
    import sys

    root = Path(__file__).parent.parent
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "tests" / "mock")])
    cmd = [sys.executable, "-c", _widget_script]
    result = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True, cwd=root)
    assert "widget ok" in result.stdout, result.stdout + result.stderr
