"""pipeline/correct.py: an fp32 report's description and sizing corrected, nothing else, never guessed."""

import copy
import dataclasses

import pytest
from conftest import TOTAL_PARAMS, hf_config

from pipeline import correct, families, settings, sizing
from pipeline.export_xnnpack import recipe_fields

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def _published(context=8192):
    """A report as the fp32 recipe wrote it before the fix: 8da4w wording and sizing."""
    cfg = settings.load()
    arch = families.architecture(hf_config(MODEL), TOTAL_PARAMS[MODEL])
    old = recipe_fields(cfg.xnnpack, "8da4w", False, "xnnpack", 2048, executorch="1.4.0")
    old.update({"qmode": "fp32", "label": "fp32-g32, int8 embeddings, int4 codes rounded to nearest"})
    table = [
        {
            "context": c,
            "device_resident_bytes": sizing.device_resident_bytes(arch, c, cfg.runtime_overhead_bytes),
            "fits_device": True,
            "export_peak_bytes": 1,
            "fits_host": True,
        }
        for c in (32768, 16384, 8192, 4096, 2048)
    ]
    report = {
        "backend": "xnnpack",
        "toolchain": {"executorch": "1.4.0"},
        "source": {"id": MODEL, "sha": "a" * 40},
        "recipe": {"export_llm_qmode": None, "embedding_hqq": False, **old},
        "window": {
            "context": context,
            "device_resident_bytes": sizing.device_resident_bytes(arch, context, cfg.runtime_overhead_bytes),
            "phone_budget_bytes": cfg.device_budget_bytes,
            "fits_phone_budget": True,
            "table": table,
        },
        "estimates": {"pte_bytes_estimate": sizing.pte_bytes_estimate(arch, context)},
        "files": [{"path": "xnnpack/SmolLM2-360M-Instruct-fp32-8k.pte", "bytes": 1499480448, "sha256": "f" * 64}],
    }
    return report, arch, cfg


def test_an_fp32_report_is_corrected_and_only_its_described_fields_change():
    report, arch, cfg = _published()
    fixed = correct.corrected(report, arch, cfg, "abc1234", "2026-10-04")
    assert fixed["recipe"]["int4_codes"] is None and fixed["recipe"]["group_size"] is None
    assert "4-bit" not in fixed["recipe"]["description"] and "ExecuTorch 1.4.0" in fixed["recipe"]["description"]
    assert fixed["window"]["device_resident_bytes"] == sizing.device_resident_bytes(
        arch, 8192, cfg.runtime_overhead_bytes, "fp32"
    )
    assert fixed["estimates"]["pte_bytes_estimate"] == sizing.pte_bytes_estimate(arch, 8192, "fp32")
    assert fixed["files"] == report["files"]  # the .pte is never touched
    assert fixed["corrections"][-1]["by"] == "execupack abc1234"
    # Everything outside the listed fields is as it was.
    trimmed = copy.deepcopy(fixed)
    for key in ("group_size", "int4_codes", "label", "description"):
        trimmed["recipe"][key] = report["recipe"][key]
    trimmed["window"] = report["window"]
    trimmed["estimates"] = report["estimates"]
    del trimmed["corrections"]
    assert trimmed == report
    # Correcting twice changes nothing more.
    assert correct.corrected(fixed, arch, cfg, "abc1234", "2026-10-04") is None


def test_8da4w_reports_are_left_alone():
    report, arch, cfg = _published()
    report["recipe"]["qmode"] = "8da4w-gptq"
    assert correct.corrected(report, arch, cfg, "x", "d") is None


def test_figures_that_cannot_be_reproduced_are_refused():
    report, arch, cfg = _published()
    report["window"]["device_resident_bytes"] += 1
    with pytest.raises(correct.CannotCorrect, match="neither"):
        correct.corrected(report, arch, cfg, "x", "d")
    report, arch, cfg = _published()
    other = dataclasses.replace(cfg, runtime_overhead_bytes=cfg.runtime_overhead_bytes + 1)
    with pytest.raises(correct.CannotCorrect):
        correct.corrected(report, arch, other, "x", "d")
