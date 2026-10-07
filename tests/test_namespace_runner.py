"""The configured runner: one Namespace profile, and what it can and cannot build.

namespace-profile-execupack replaced the Blacksmith tiers on 2026-10-07. Its own CI run
measured 4 x86_64 CPUs, MemTotal 16,438,740 kB, a 101 GB overlay disk shared by / and /mnt,
and no swap, which the container refuses (swapon: Operation not permitted). These tests pin
those figures in config/pipeline.yaml and what follows from them, so a change to either is a
decision someone makes rather than a matrix that quietly shrinks or a job that runs out of
memory.
"""

import dataclasses

import pytest
from conftest import TOTAL_PARAMS, hf_config

from pipeline import export_mtk, export_xnnpack, families, settings

CFG = settings.load()
PROFILE = "namespace-profile-execupack"
MEM_TOTAL_BYTES = 16_438_740 * 1024
WINDOWS = (2048, 4096, 8192, 16384, 32768)


def arch(model_id):
    return families.architecture(hf_config(model_id), TOTAL_PARAMS[model_id])


def windows(model_id, backend):
    tiers = getattr(CFG, backend).runner_tiers
    return [w for w in WINDOWS if export_xnnpack.pick_runner(arch(model_id), w, tiers, backend)]


@pytest.mark.parametrize("backend", ["xnnpack", "vulkan", "mtk"])
def test_every_recipe_runs_on_the_namespace_profile_as_measured(backend):
    (tier,) = getattr(CFG, backend).runner_tiers
    assert tier.label == PROFILE
    assert tier.ram_bytes == MEM_TOTAL_BYTES
    # No swap: the container does not allow it, so nothing may be sized on swap.
    assert tier.swap_gib == 0


def test_no_workflow_runs_anywhere_else():
    for workflow in (settings.ROOT / ".github/workflows").glob("*.yml"):
        text = workflow.read_text(encoding="utf-8")
        assert "blacksmith" not in text, workflow.name
        for line in text.splitlines():
            if line.strip().startswith("runs-on:") and "${{" not in line:
                assert line.split("runs-on:")[1].strip() == PROFILE, f"{workflow.name}: {line.strip()}"


@pytest.mark.parametrize(
    ("model_id", "backend", "expected"),
    [
        # 15.7 GiB less the reserve, against host_need_bytes: the masks grow with the window
        # squared, so the small models keep up to 8k and lose 16k and 32k.
        ("Qwen/Qwen3-0.6B", "vulkan", [2048, 4096, 8192]),
        ("Qwen/Qwen3-0.6B", "xnnpack", [2048, 4096, 8192]),
        ("HuggingFaceTB/SmolLM2-360M-Instruct", "vulkan", [2048, 4096, 8192]),
        ("meta-llama/Llama-3.2-1B-Instruct", "vulkan", [2048, 4096, 8192]),
        ("meta-llama/Llama-3.2-1B-Instruct", "xnnpack", [2048, 4096]),
        # XNNPACK's 1.7x headroom on the fp32 weights puts the 1.5B to 4B models out of reach.
        ("Qwen/Qwen2.5-1.5B-Instruct", "vulkan", [2048, 4096]),
        ("Qwen/Qwen2.5-1.5B-Instruct", "xnnpack", []),
        ("Qwen/Qwen3-1.7B", "xnnpack", []),
        ("Qwen/Qwen3-4B", "vulkan", []),
        ("Qwen/Qwen3-4B", "xnnpack", []),
    ],
)
def test_what_the_profile_can_export(model_id, backend, expected):
    assert windows(model_id, backend) == expected


def test_mediatek_lfm2_does_not_fit_at_any_window():
    # By the current estimate: calibration_bytes takes 14.4 to 15.7 bytes a parameter, measured
    # on 4-chunk builds, about 17.4 GiB for the 1.2B. The 8-chunk 16k build peaked at 13.4 GiB
    # (finding 39), so this pins the estimate's answer, not a measured limit (finding 40).
    lfm = {
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "hidden_size": 2048,
    }
    for window in (512, 2048, 4096):
        recipe = dataclasses.replace(CFG.mtk, cache_size=window)
        assert export_mtk.pick_runner(lfm, recipe, 23, 1_170_340_608, streaming=True) is None
