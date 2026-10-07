"""Per-window runner choice for the export_llm backends (export-vulkan.yml's matrix), on a
three-size fleet; what the configured tier itself carries is test_namespace_runner.py."""

import pytest
from conftest import FLEET, TOTAL_PARAMS, hf_config

from pipeline import export_xnnpack, families, settings, sizing

TIERS = FLEET
SMALL, MEDIUM, LARGE = (t.label for t in TIERS)
GIB = 2**30


def arch(model_id):
    return families.architecture(hf_config(model_id), TOTAL_PARAMS[model_id])


def pick(model_id, window):
    tier = export_xnnpack.pick_runner(arch(model_id), window, TIERS, "vulkan")
    return tier.label if tier else None


def test_tiers_are_smallest_first():
    cfg = settings.load()
    for tiers in (TIERS, cfg.xnnpack.runner_tiers, cfg.vulkan.runner_tiers, cfg.mtk.runner_tiers):
        assert [t.ram_bytes for t in tiers] == sorted(t.ram_bytes for t in tiers)


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        # Measured peaks (2026-10-04): 4.8 GiB at 2k, 24.2 at 16k on 30.9 GiB, 80.1 at 32k.
        ("Qwen/Qwen3-0.6B", {2048: SMALL, 8192: SMALL, 16384: SMALL, 32768: LARGE}),
        # 16.9-21.8 GiB at 2k-8k fit the smallest tier; 39.3 at 16k does not.
        ("Qwen/Qwen3-4B", {2048: SMALL, 4096: SMALL, 8192: SMALL, 16384: MEDIUM, 32768: LARGE}),
        # Llama 3.2 1B attends on 16 layers: 40.0 GiB at 32k on a 62 GiB runner.
        ("meta-llama/Llama-3.2-1B-Instruct", {2048: SMALL, 16384: SMALL, 32768: MEDIUM}),
    ],
)
def test_each_window_gets_the_smallest_runner_that_holds_it(model_id, expected):
    for window, label in expected.items():
        assert pick(model_id, window) == label, window


@pytest.mark.parametrize(
    ("model_id", "window", "measured_gib"),
    [
        # The worst four of fifty Vulkan exports, against host.peak_in_use_bytes.
        ("Qwen/Qwen3-0.6B", 32768, 80.1),
        ("Qwen/Qwen3-1.7B", 32768, 86.4),
        ("Qwen/Qwen3-4B", 32768, 115.5),
        ("Qwen/Qwen3-0.6B", 16384, 24.2),
    ],
)
def test_the_vulkan_need_covers_what_the_runners_measured(model_id, window, measured_gib):
    assert sizing.host_need_bytes(arch(model_id), window, "vulkan") >= measured_gib * GIB


def test_the_vulkan_need_does_not_inherit_the_xnnpack_headroom_at_small_windows():
    # At 2k the estimate runs over what Vulkan uses (Qwen3-4B: 19.5 GiB estimated, 16.9
    # measured), so 1.7x would send it to a runner twice the size it needs.
    qwen = arch("Qwen/Qwen3-4B")
    assert sizing.host_need_bytes(qwen, 2048, "vulkan") < sizing.host_need_bytes(qwen, 2048, "xnnpack")


def test_when_nothing_holds_it_in_ram_the_most_ram_wins():
    # Qwen3-4B at 32k needs more than any tier's RAM: the largest tier with swap, never a
    # smaller tier paging more of it.
    qwen = arch("Qwen/Qwen3-4B")
    need = sizing.host_need_bytes(qwen, 32768, "vulkan")
    assert all(need > t.ram_bytes for t in TIERS)
    assert pick("Qwen/Qwen3-4B", 32768) == LARGE


def test_a_window_no_tier_can_carry_is_refused():
    assert export_xnnpack.pick_runner(arch("Qwen/Qwen3-4B"), 131072, TIERS, "vulkan") is None
