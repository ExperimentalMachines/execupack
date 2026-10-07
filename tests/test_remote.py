"""The Modal side (pipeline/remote.py): what a window asks for, and that the exports go there.

Nothing here talks to Modal. The limits are the ones Modal enforced when probed on 2026-10-07:
a Sandbox may ask for 344064 MiB and 64 physical cores, and a larger request is refused.
"""

import json
import os
import subprocess

import pytest
from conftest import TOTAL_PARAMS, hf_config

from pipeline import export_xnnpack, families, remote, settings, sizing

CFG = settings.load()
WINDOWS = (2048, 4096, 8192, 16384, 32768)


def arch(model_id):
    return families.architecture(hf_config(model_id), TOTAL_PARAMS[model_id])


def test_modal_is_configured_with_the_limits_it_enforced():
    assert CFG.modal.max_memory_mib == 344064
    assert 0 < CFG.modal.cpu <= 64
    # A Sandbox lives at most 24 h, and the GitHub job that waits on it at most 350 min.
    assert CFG.modal.timeout_minutes < 350


@pytest.mark.parametrize("backend", ["xnnpack", "vulkan"])
def test_the_export_llm_backends_run_on_modal_up_to_its_ceiling(backend):
    (tier,) = getattr(CFG, backend).runner_tiers
    assert tier.label == settings.MODAL_TIER
    assert tier.ram_bytes == CFG.modal.max_memory_mib * 2**20
    assert tier.swap_gib == 0


@pytest.mark.parametrize(
    "model_id", ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "meta-llama/Llama-3.2-1B-Instruct"]
)
@pytest.mark.parametrize("backend", ["xnnpack", "vulkan"])
def test_every_window_up_to_32k_fits_a_sandbox(model_id, backend):
    # Everything Blacksmith ever built fits, Qwen3-4B at 32k (about 134 GiB) included.
    tiers = getattr(CFG, backend).runner_tiers
    for window in WINDOWS:
        assert export_xnnpack.pick_runner(arch(model_id), window, tiers, backend), window
        need = sizing.host_need_bytes(arch(model_id), window, backend)
        assert remote.memory_request(need, CFG.modal) is not None


def test_the_request_is_the_need_plus_the_margin_in_whole_mib():
    need = 10 * 2**30
    mib = remote.memory_request(need, CFG.modal)
    assert mib * 2**20 >= need * CFG.modal.memory_margin
    assert (mib - 1) * 2**20 < need * CFG.modal.memory_margin
    # Past the ceiling there is nothing to ask for.
    assert remote.memory_request(CFG.modal.max_memory_mib * 2**20, CFG.modal) is None


def test_torch_comes_from_the_cpu_index():
    pins = remote.cpu_pins(settings.ROOT / "requirements/export-xnnpack.txt")
    assert pins and all(p.startswith(("torch==", "torchvision==")) for p in pins)
    assert remote.CPU_INDEX == "https://download.pytorch.org/whl/cpu"


def test_the_image_carries_what_the_pipeline_reads():
    for name in remote.REPO_DIRS:
        assert (settings.ROOT / name).is_dir(), name
    assert {"pipeline", "config", "third_party", "requirements"} <= set(remote.REPO_DIRS)


@pytest.mark.parametrize("workflow", ["export-xnnpack.yml", "export-vulkan.yml"])
def test_the_workflow_sends_each_window_to_modal_with_its_own_memory(workflow):
    text = (settings.ROOT / ".github/workflows" / workflow).read_text(encoding="utf-8")
    assert "python -m pipeline modal-run --memory-mib" in text
    assert "MEMORY_MIB: ${{ matrix.entry.memory_mib }}" in text
    assert "MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}" in text
    # The heavy toolchain stays off the runner.
    assert "./.github/actions/setup-export" not in text


def test_the_matrix_entry_shape_the_workflow_reads():
    entry = {"context": 2048, "runner": "modal", "swap_gib": 0, "memory_mib": 6286}
    assert json.loads(json.dumps(entry))["memory_mib"] == 6286


@pytest.mark.parametrize(
    ("script", "status"),
    [("true", 0), ("exit 4", 4), ("false || exit $?", 1), ("echo report; exit 1", 1)],
)
def test_the_sandbox_script_keeps_the_steps_status_and_always_syncs(script, status, tmp_path):
    # The EXIT trap runs sync on every way out, an explicit `exit` included, and the step's
    # status survives it, exit 4 (a window that cannot be built) above all. sync is stood in
    # for by a marker file, since there is no Volume here.
    marker = tmp_path / "synced"
    wrapped = remote.wrap(script).replace(f"sync {remote.VOLUME_MOUNT}", f"touch {marker}")
    env = dict(os.environ, IN=str(tmp_path / "in"), OUT=str(tmp_path / "out"), WORK=str(tmp_path / "w"))
    assert subprocess.run(["bash", "-c", wrapped], env=env).returncode == status
    assert marker.exists()


def test_a_failed_sync_fails_a_step_that_had_succeeded(tmp_path):
    env = dict(os.environ, IN=str(tmp_path / "in"), OUT=str(tmp_path / "out"), WORK=str(tmp_path / "w"))
    failing = remote.wrap("true").replace(f"sync {remote.VOLUME_MOUNT}", "false")
    assert subprocess.run(["bash", "-c", failing], env=env, capture_output=True).returncode == 1
    # A step that already failed keeps its own status.
    skipped = remote.wrap("exit 4").replace(f"sync {remote.VOLUME_MOUNT}", "false")
    assert subprocess.run(["bash", "-c", skipped], env=env, capture_output=True).returncode == 4
