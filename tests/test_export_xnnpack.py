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


def test_an_fp32_report_does_not_claim_int4_weights():
    cfg = settings.load()
    fp32 = export_xnnpack.recipe_fields(dataclasses.replace(cfg.xnnpack, qmode="fp32"), "fp32", False, "xnnpack", 2048)
    assert fp32["int4_codes"] is None and fp32["group_size"] is None
    assert "4-bit" not in fp32["description"] and "int4" not in fp32["label"]
    assert "not quantized" in fp32["description"] and fp32["embedding_quantize"] == cfg.xnnpack.embedding_quantize
    rtn = export_xnnpack.recipe_fields(cfg.xnnpack, "8da4w", False, "xnnpack", 2048)
    gptq = export_xnnpack.recipe_fields(cfg.xnnpack, "8da4w-gptq", True, "xnnpack", 2048)
    assert rtn["int4_codes"] == "round-to-nearest" and gptq["int4_codes"] == "gptq"
    assert "4-bit weights in groups of" in rtn["description"] and rtn["group_size"] == cfg.xnnpack.group_size


RECIPE = {
    "model_class": "phi_4_mini",
    "converter": "phi4",
    "params": {"dim": 3072},
    "export_llm_qmode": "8da4w",
    "embedding_hqq": False,
    "embedding_quantize": "8,0",
    "group_size": 32,
    "prefill_chunk": 2048,
    "output_head": "int8 per channel",
}
EXPECTED = {"source_id": "microsoft/Phi-4-mini-instruct", "executorch": "1.5.1", "recipe": RECIPE}


def _proxy(tmp_path, **changes):
    import json

    report = {
        "backend": "xnnpack",
        "source": {"id": "microsoft/Phi-4-mini-instruct", "sha": "a" * 40},
        "toolchain": {"executorch": "1.5.1"},
        "window": {"context": 8192},
        "recipe": {**RECIPE, "codes_sha256": "c0de"},
        "gate": {"passed": True, "graded": 8, "agreed": 8, "fp32_mean": 0.66, "export_mean": 0.53},
        "files": [{"path": "xnnpack/x-8k.pte", "sha256": "ff"}],
    }
    for key, value in changes.items():
        report[key] = value
    path = tmp_path / "export-report-8k.json"
    path.write_text(json.dumps(report))
    return path


def test_a_gate_proxy_counts_only_for_the_same_build(tmp_path):
    import copy

    from pipeline import export_xnnpack

    ok = export_xnnpack.check_gate_proxy(_proxy(tmp_path), "a" * 40, 8192, "c0de", EXPECTED)
    assert ok["gate"]["passed"] and ok["file"] == "xnnpack/x-8k.pte"
    four_bit_head = copy.deepcopy(EXPECTED)
    four_bit_head["recipe"]["output_head"] = None
    other_repo = {**EXPECTED, "source_id": "microsoft/Phi-4-mini-reasoning"}
    other_toolchain = {**EXPECTED, "executorch": "1.6.0"}
    for changes, args, match in (
        ({}, ("b" * 40, 8192, "c0de", EXPECTED), "another source revision"),
        ({}, ("a" * 40, 4096, "c0de", EXPECTED), "window is 8192"),
        ({}, ("a" * 40, 8192, "beef", EXPECTED), "other codes"),
        ({}, ("a" * 40, 8192, "c0de", four_bit_head), "output_head differs"),
        ({}, ("a" * 40, 8192, "c0de", other_repo), "another source repo"),
        ({}, ("a" * 40, 8192, "c0de", other_toolchain), "another ExecuTorch"),
        (
            {"recipe": {**RECIPE, "codes_sha256": "c0de", "embedding_quantize": "4,32"}},
            ("a" * 40, 8192, "c0de", EXPECTED),
            "embedding_quantize differs",
        ),
        ({"gate": {"passed": False}}, ("a" * 40, 8192, "c0de", EXPECTED), "did not pass"),
        ({"backend": "vulkan"}, ("a" * 40, 8192, "c0de", EXPECTED), "vulkan report"),
        ({"files": []}, ("a" * 40, 8192, "c0de", EXPECTED), "could not be read"),
    ):
        with pytest.raises(export_xnnpack.ExportError, match=match):
            export_xnnpack.check_gate_proxy(_proxy(tmp_path, **changes), *args)
    with pytest.raises(export_xnnpack.ExportError, match="needs --gate-proxy"):
        export_xnnpack.check_gate_proxy(None, "a" * 40, 8192, "c0de", EXPECTED)
    accepted = {"passed": False, "agreed": 7, "graded": 8, "override": {"reason": "near a tie"}}
    assert export_xnnpack.check_gate_proxy(_proxy(tmp_path, gate=accepted), "a" * 40, 8192, "c0de", EXPECTED)


def test_an_int8_head_is_priced_at_a_byte_a_weight(source):
    from pipeline import families, sizing

    arch = families.architecture(source("microsoft/Phi-4-mini-instruct").config, 3_836_021_760)
    four, eight = sizing.pte_bytes_estimate(arch, 2048), sizing.pte_bytes_estimate(arch, 2048, output_bits=8)
    head = 200_064 * 3072
    assert 0.4 * head < eight - four < 0.5 * head  # (1 - 0.5625) bytes per weight, plus overhead
    rows = sizing.choose_context(arch, (2048,), 5_000_000_000, 500_000_000, None, output_bits=8).table
    assert rows[0]["device_resident_bytes"] == sizing.device_resident_bytes(arch, 2048, 500_000_000, output_bits=8)
