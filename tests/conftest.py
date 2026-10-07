"""Shared fixtures. tests/fixtures/*.config.json are real HF config.json files;
et-*.params.json are the params files ExecuTorch v1.4.0 ships for the same models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.hub import SourceModel
from pipeline.settings import RunnerTier

FIXTURES = Path(__file__).parent / "fixtures"

# Real HF safetensors totals (Hub API, 2026-09-12).
TOTAL_PARAMS = {
    "Qwen/Qwen3-0.6B": 751_632_384,
    "Qwen/Qwen3-1.7B": 2_031_739_904,
    "Qwen/Qwen3-4B": 4_022_468_096,
    "Qwen/Qwen2.5-1.5B-Instruct": 1_543_714_304,
    "meta-llama/Llama-3.2-1B-Instruct": 1_235_814_400,
    "HuggingFaceTB/SmolLM2-360M-Instruct": 361_821_120,
    "Qwen/Qwen3-30B-A3B": 30_532_122_624,
    "google/gemma-3-1b-it": 999_885_952,
}
CONFIG_FILES = {
    "Qwen/Qwen3-0.6B": "qwen3-0.6b",
    "Qwen/Qwen3-1.7B": "qwen3-1.7b",
    "Qwen/Qwen3-4B": "qwen3-4b",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5-1.5b-instruct",
    "meta-llama/Llama-3.2-1B-Instruct": "llama-3.2-1b-instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct": "smollm2-360m-instruct",
    "Qwen/Qwen3-30B-A3B": "qwen3-30b-a3b",
    "google/gemma-3-1b-it": "gemma-3-1b-it",
}


def load_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def hf_config(model_id: str) -> dict:
    return load_json(f"{CONFIG_FILES[model_id]}.config.json")


def make_source(model_id: str, config: dict | None = None, **overrides) -> SourceModel:
    fields = dict(
        id=model_id,
        sha="0123456789abcdef0123456789abcdef01234567",
        created_at=None,
        pipeline_tag="text-generation",
        tags=[],
        gated=False,
        total_params=TOTAL_PARAMS.get(model_id, 1_000_000_000),
        config=config if config is not None else hf_config(model_id),
    )
    fields.update(overrides)
    return SourceModel(**fields)


@pytest.fixture
def source():
    return make_source


# A three-size fleet for testing the runner choice itself: the x86 runners execupack used until
# 2026-10-07, as their export reports measured them (30.9, 62.0 and 113.4 GiB of RAM). The
# configured tiers are one Namespace profile now (test_namespace_runner.py), but the choice
# has to keep working the day a larger profile is added beside it.
FLEET = (
    RunnerTier("small", 33_229_729_792, 160_000_000_000, 24),
    RunnerTier("medium", 66_577_858_560, 750_000_000_000, 64),
    RunnerTier("large", 121_766_014_976, 1_500_000_000_000, 96),
)
# The same fleet as config/pipeline.yaml's mtk tiers had it: nominal sizes, and the largest
# window each was given for a streaming (lfm2.py) calibration.
MTK_FLEET = (
    RunnerTier("small", 32_000_000_000, 160_000_000_000, 24, 4096),
    RunnerTier("medium", 64_000_000_000, 750_000_000_000, 64, 16384),
    RunnerTier("large", 128_000_000_000, 1_500_000_000_000, 96),
)
