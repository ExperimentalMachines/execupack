import dataclasses

import pytest
from conftest import make_source

from pipeline import export_xnnpack, exporting, settings


class Downloaded(Exception):
    """Raised by the fake download: the run passed every pre-download check."""


@pytest.fixture
def qwen3_1_7b(monkeypatch):
    source = make_source("Qwen/Qwen3-1.7B")
    monkeypatch.setattr(export_xnnpack.hub, "fetch", lambda model_id, revision: source)
    monkeypatch.setattr(export_xnnpack.hub, "download", lambda *a, **k: (_ for _ in ()).throw(Downloaded()))
    # A 16.8 GB + 24 GiB hosted runner.
    monkeypatch.setattr(
        export_xnnpack,
        "host_info",
        lambda: {"nproc": 4, "mem_total_bytes": 16_766_414_848, "swap_total_bytes": 25_769_803_776},
    )
    return source


def run(tmp_path, context):
    return export_xnnpack.run("Qwen/Qwen3-1.7B", "main", tmp_path / "out", tmp_path / "work", context=context)


def test_windows_over_the_phone_budget_are_exported_anyway(tmp_path, qwen3_1_7b):
    # Qwen3-1.7B at 16k needs about 5.6 GB resident (over the 5 GB phone budget): exported,
    # the app and the benchmarker decide what fits (PLAN, "Context auto-fit").
    assert settings.load().device_budget_bytes == 5_000_000_000
    with pytest.raises(Downloaded):
        run(tmp_path, 16384)


def test_a_window_the_runner_cannot_build_is_a_skip_not_a_failure(tmp_path, qwen3_1_7b):
    # 28 layers x 32k^2 causal masks: 30 GB of masks alone, more than RAM + swap.
    with pytest.raises(exporting.SkipExport, match="causal masks alone"):
        run(tmp_path, 32768)
    assert issubclass(exporting.SkipExport, exporting.ExportError)


def test_no_window_given_means_the_largest_the_host_can_build(tmp_path, qwen3_1_7b, capsys):
    with pytest.raises(Downloaded):
        run(tmp_path, None)
    assert "window 16384" in capsys.readouterr().out


def test_vulkan_shares_the_recipe_and_only_swaps_the_delegate(tmp_path):
    import dataclasses
    from pathlib import Path

    from conftest import hf_config

    from pipeline import families, naming

    cfg = settings.load()
    plan = families.xnnpack_plan(families.family_for(hf_config("Qwen/Qwen3-1.7B")), hf_config("Qwen/Qwen3-1.7B"))
    args = (plan, Path("p.json"), Path("c.pth"), Path("o.pte"), 4096, cfg.prefill_chunk, cfg.vulkan, 1, [2])
    xnnpack = export_xnnpack.export_llm_config(*args, backend="xnnpack")
    vulkan = export_xnnpack.export_llm_config(*args, backend="vulkan")
    assert vulkan["backend"] == {"vulkan": {"enabled": True}}
    assert xnnpack["backend"] == {"xnnpack": {"enabled": True, "extended_ops": True}}
    assert {k: v for k, v in vulkan.items() if k != "backend"} == {k: v for k, v in xnnpack.items() if k != "backend"}
    # Same quantization recipe; only Vulkan carries per-window runner tiers.
    assert dataclasses.replace(cfg.vulkan, runner_tiers=()) == dataclasses.replace(cfg.xnnpack, runner_tiers=())
    # The app reads the backend from the name: the Vulkan file says so, the XNNPACK folder does.
    repo = naming.output_repo("Qwen/Qwen3-1.7B", cfg.hub_org, cfg.repo_suffix)
    name = naming.cpu_gpu_file("Qwen/Qwen3-1.7B", "vulkan", "8da4w", 4096)
    assert name == "Qwen3-1.7B-vulkan-8da4w-4k.pte"
    assert naming.check_app_rules(repo, f"vulkan/{name}", "vulkan") == []
    assert naming.app_backend(f"{repo}/vulkan/{name}") == "vulkan"


def test_unknown_backend_is_refused(tmp_path):
    with pytest.raises(exporting.ExportError, match="unknown backend"):
        export_xnnpack.run("Qwen/Qwen3-1.7B", "main", tmp_path, tmp_path, backend="coreml")


def test_the_fp32_recipe_leaves_the_linears_unquantized():
    from pathlib import Path

    from conftest import hf_config

    from pipeline import families

    cfg = settings.load()
    config = hf_config("HuggingFaceTB/SmolLM2-360M-Instruct")
    plan = families.xnnpack_plan(families.family_for(config), config)
    recipe = dataclasses.replace(cfg.xnnpack, qmode="fp32")
    built = export_xnnpack.export_llm_config(
        plan, Path("p.json"), Path("c.pth"), Path("o.pte"), 2048, cfg.prefill_chunk, recipe, 1, [2], backend="xnnpack"
    )
    assert built["quantization"]["qmode"] is None
    assert built["quantization"]["embedding_quantize"] == cfg.xnnpack.embedding_quantize
    # export_llm's own enum value for the default recipe passes through unchanged.
    assert export_xnnpack.QMODES["8da4w"] == "8da4w"
