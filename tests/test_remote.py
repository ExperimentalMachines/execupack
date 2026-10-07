"""The Modal side (pipeline/remote.py): what a window asks for, and that the exports go there.

Nothing here talks to Modal. The limits are the ones Modal enforced when probed on 2026-10-07:
a Sandbox may ask for 344064 MiB and 64 physical cores, and a larger request is refused.
"""

import json
import os
import subprocess

import pytest
from conftest import TOTAL_PARAMS, hf_config

from pipeline import export_mtk, export_qnn, export_xnnpack, families, remote, settings, sizing

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


def test_the_mediatek_workflow_sends_each_window_with_its_memory_cores_and_minutes():
    text = (settings.ROOT / ".github/workflows/export-mtk.yml").read_text(encoding="utf-8")
    assert '--memory-mib "$MEMORY_MIB" --cpu "$CORES" --timeout-minutes "$SANDBOX_MINUTES"' in text
    assert "--mtk-tools requirements/mtk-tools.txt" in text
    assert "bash scripts/mtk-setup.sh &&" in text
    assert "./.github/actions/setup-export" not in text
    # The job outlives the longest Sandbox, within the 360 minutes a GitHub-hosted job may run.
    assert max(minutes for _, _, minutes in export_mtk.MODAL_SHAPES) < 350
    assert "timeout-minutes: 350" in text


def test_the_neuropilot_sdk_is_fetched_at_run_time_and_never_put_in_the_image():
    setup = (settings.ROOT / "scripts/mtk-setup.sh").read_text(encoding="utf-8")
    assert "sha256sum -c -" in setup and 'rm -rf "$sdk" "$sdk.tar.gz"' in setup
    image = (settings.ROOT / "pipeline/remote.py").read_text(encoding="utf-8").split("def image(")[1]
    assert "NEUROPILOT_SDK_URL" not in image.split("def _stream(")[0]


@pytest.mark.parametrize(
    ("window", "cores", "minutes"),
    [(2048, 8, 240), (4096, 8, 240), (8192, 32, 330), (16384, 32, 330), (32768, 64, 330)],
)
def test_a_mediatek_window_gets_more_cores_and_time_as_it_grows(window, cores, minutes):
    assert export_mtk.modal_shape(window) == (cores, minutes)
    assert cores <= 64


def test_a_window_past_every_shape_is_left_out_not_fatal():
    assert export_mtk.modal_shape(65536) is None


def test_the_qualcomm_workflow_sends_each_window_with_its_memory_cores_and_minutes():
    text = (settings.ROOT / ".github/workflows/export-qnn.yml").read_text(encoding="utf-8")
    assert '--memory-mib "$MEMORY_MIB" --cpu "$CORES" --timeout-minutes "$SANDBOX_MINUTES"' in text
    assert "--requirements requirements/export-qnn.txt" in text
    # QAIRT is downloaded inside each Sandbox now; no Actions cache keeps a copy.
    assert "actions/cache" not in text
    assert max(minutes for _, _, minutes in export_qnn.QNN_MODAL_SHAPES) < 350
    assert "timeout-minutes: 350" in text


@pytest.mark.parametrize(
    ("model_id", "cores"),
    [
        ("HuggingFaceTB/SmolLM2-360M-Instruct", 16),
        ("Qwen/Qwen3-0.6B", 16),
        ("meta-llama/Llama-3.2-1B-Instruct", 16),
        ("Qwen/Qwen3-1.7B", 64),
        ("Qwen/Qwen3-4B", 64),
    ],
)
def test_a_qualcomm_export_fits_a_sandbox_at_every_window_it_is_given(model_id, cores):
    for window in (2048, 4096, 8192):
        need, got_cores, minutes = export_qnn.modal_request(TOTAL_PARAMS[model_id], window)
        # Qwen3-0.6B measured 15.6 GB of RSS at 2k and 4k (findings 8 and 10): asked for above that.
        assert need >= 15_702_085_632
        assert remote.memory_request(need, CFG.modal) is not None
        assert got_cores == cores and minutes < 350  # inside the GitHub-hosted job that waits


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


def test_the_run_that_built_a_file_reaches_the_sandbox():
    assert {"GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"} <= set(remote.PASSTHROUGH)


def test_the_container_peak_is_read_from_its_cgroup(tmp_path):
    from pipeline.exporting import MemorySampler

    current = tmp_path / "memory.current"
    current.write_text("3221225472\n")
    sampler = MemorySampler(meminfo=tmp_path / "missing", cgroup=(tmp_path / "absent", current))
    sampler.sample()
    current.write_text("1073741824\n")
    sampler.sample()
    assert sampler.result()["peak_cgroup_bytes"] == 3221225472
    assert MemorySampler(meminfo=tmp_path / "missing", cgroup=()).result()["peak_cgroup_bytes"] is None
