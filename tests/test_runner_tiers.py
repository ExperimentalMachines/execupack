"""Per-window runner choice for the export_llm backends (export-vulkan.yml's matrix)."""

import pytest
from conftest import TOTAL_PARAMS, hf_config

from pipeline import export_xnnpack, families, settings, sizing

TIERS = settings.load().vulkan.runner_tiers
SMALL, MEDIUM, LARGE = (t.label for t in TIERS)


def arch(model_id):
    return families.architecture(hf_config(model_id), TOTAL_PARAMS[model_id])


def pick(model_id, window):
    tier = export_xnnpack.pick_runner(arch(model_id), window, TIERS)
    return tier.label if tier else None


def test_tiers_are_smallest_first():
    assert [t.ram_bytes for t in TIERS] == sorted(t.ram_bytes for t in TIERS)


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        # Qwen3-0.6B: 9.6 GiB at 2k up to 26.6 at 16k fit 32 GB of RAM; 32k needs 68.2.
        ("Qwen/Qwen3-0.6B", {2048: SMALL, 4096: SMALL, 8192: SMALL, 16384: SMALL, 32768: LARGE}),
        # Qwen3-4B's fp32 weights alone are 16 GB: 33.1 GiB at 2k, 108.4 at 32k.
        ("Qwen/Qwen3-4B", {2048: MEDIUM, 4096: MEDIUM, 8192: MEDIUM, 16384: MEDIUM, 32768: LARGE}),
        # Llama 3.2 1B attends on 16 layers, so even 32k (44.0 GiB) fits 64 GB.
        ("meta-llama/Llama-3.2-1B-Instruct", {2048: SMALL, 16384: SMALL, 32768: MEDIUM}),
    ],
)
def test_each_window_gets_the_smallest_runner_that_holds_it_in_ram(model_id, expected):
    for window, label in expected.items():
        assert pick(model_id, window) == label, window


def test_ram_is_preferred_over_swap():
    # Qwen3-0.6B at 32k needs 68.2 GiB: the 64 GB tier plus its 64 GiB of swap would hold
    # it, but at the pager's speed, so the 128 GB tier is chosen.
    need = sizing.host_need_bytes(arch("Qwen/Qwen3-0.6B"), 32768)
    medium = TIERS[1]
    assert need > medium.ram_bytes - export_xnnpack.RESERVE_BYTES
    assert need < medium.ram_bytes + medium.swap_gib * 2**30
    assert pick("Qwen/Qwen3-0.6B", 32768) == LARGE


def test_a_window_no_tier_can_carry_is_refused():
    huge = families.architecture(hf_config("Qwen/Qwen3-4B"), TOTAL_PARAMS["Qwen/Qwen3-4B"])
    assert export_xnnpack.pick_runner(huge, 131072, TIERS) is None
