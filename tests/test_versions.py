import re

from pipeline import settings


def pins(path):
    found = {}
    for line in (settings.ROOT / path).read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", line.strip())
        if match:
            found[match.group(1).lower()] = match.group(2)
    return found


def test_executorch_pin_matches_versions_env():
    # The CPU/GPU exporter's pin. What may be published is APP_RUNTIME_VERSION's to decide
    # (publish.require_app_runtime), and the exporter is never behind the apps.
    cfg = settings.load()
    assert pins("requirements/export-xnnpack.txt")["executorch"] == cfg.executorch_version
    assert settings.release(cfg.app_runtime_version) <= settings.release(cfg.executorch_version)


def test_torch_pin_matches_executorch_1_6_0():
    # ExecuTorch v1.6.0 torch_pin.py: TORCH_VERSION = "2.15.0" (the wheel requires torch==2.15.0);
    # install_requirements.py: TORCHAO_MIN_VERSION 0.18.0, still the newest torchao.
    export = pins("requirements/export-xnnpack.txt")
    assert export["torch"] == "2.15.0"
    assert export["torchao"] == "0.18.0"
    assert export["pytorch-tokenizers"] == "1.6.0"


def minor_behind(newer: str, older: str) -> bool:
    (a, b), (c, d) = (tuple(int(x) for x in v.split(".")[:2]) for v in (newer, older))
    return a == c and b - d == 1


def test_npu_exports_stay_on_the_release_before():
    # The NPU toolchains keep ExecuTorch 1.5.1 with its torch and torchao (config/versions.env):
    # Qualcomm because 1.6.0 compiles with QAIRT 2.51, newer than the apps' QNN runtime, and
    # MediaTek because mtk_converter is cp310-only and 1.6.0 has no cp310 wheel. Exactly one
    # minor behind, the oldest release a 1.6.0 runtime is promised to load.
    versions = settings.read_env_file(settings.CONFIG_DIR / "versions.env")
    runtime = settings.load().executorch_version
    for path, key in (
        ("requirements/export-qnn.txt", "QNN_EXECUTORCH_VERSION"),
        ("requirements/mtk-tools.txt", "MTK_EXECUTORCH_VERSION"),
    ):
        tools = pins(path)
        assert tools["executorch"] == versions[key] == "1.5.1", path
        assert (tools["torch"], tools["torchao"]) == ("2.14.0", "0.18.0"), path
        assert minor_behind(runtime, tools["executorch"]), path
    # QAIRT is 1.5.1's (qnn_config.sh at v1.5.1), not 1.6.0's 2.51.0.260929.
    assert versions["QAIRT_VERSION"] == "2.37.0.250724"
    assert pins("requirements/export-qnn.txt")["torchvision"] == "0.29.0"  # built against torch 2.14.0
    # The examples/mediatek source at v1.5.1's commit, so the scripts match that wheel.
    assert versions["EXECUTORCH_COMMIT"] == "3b60683923245cf472b7323426920e15623ba361"


def test_mediatek_tools_keep_transformers_4():
    # examples/mediatek imports transformers.tokenization_utils (4.x), which caps
    # huggingface_hub below 1.0: hence a separate environment from the pipeline's.
    tools = pins("requirements/mtk-tools.txt")
    assert tools["transformers"].startswith("4.")
    assert tools["huggingface_hub"].startswith("0.")
    versions = settings.read_env_file(settings.CONFIG_DIR / "versions.env")
    assert versions["MTK_PYTHON_VERSION"] == "3.10"  # mtk_converter is cp310-only
    assert re.fullmatch(r"[0-9a-f]{64}", versions["NEUROPILOT_SDK_SHA256"])


def test_huggingface_hub_pin_is_shared():
    assert pins("requirements/dev.txt")["huggingface_hub"] == pins("requirements/export-xnnpack.txt")["huggingface_hub"]


def test_backend_requirements_share_their_core_pins():
    # Deliberate differences: transformers (the Qualcomm scripts run on ExecuTorch's example
    # pin, 5.0.0rc1; the smoke test's runner import on a current release) and the runtime the
    # Qualcomm toolchain stays on (test_npu_exports_stay_on_the_release_before).
    runtime = {"executorch", "torch", "torchao", "pytorch-tokenizers"}
    xnnpack, qnn = pins("requirements/export-xnnpack.txt"), pins("requirements/export-qnn.txt")
    for package in set(xnnpack) & set(qnn) - {"transformers"} - runtime:
        assert xnnpack[package] == qnn[package], package
    assert xnnpack["executorch"] == settings.load().executorch_version


def test_every_requirement_is_pinned():
    files = ("dev.txt", "watch.txt", "modal.txt", "export-xnnpack.txt", "export-qnn.txt", "export-mtk.txt")
    files += ("mtk-tools.txt",)
    for path in files:
        for line in (settings.ROOT / "requirements" / path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            assert "==" in line or "@ git+" in line, f"{path}: {line!r} is not pinned"
