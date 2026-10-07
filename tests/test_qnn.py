import dataclasses
import json
import os
import subprocess
from pathlib import Path

import pytest
from conftest import load_json, make_source

from pipeline import export_qnn, families, manifest, naming, publish, settings

torch = pytest.importorskip("torch")
CFG = settings.load()


def flag(command, name):
    return command[command.index(name) + 1]


def test_compile_only_command_for_a_registered_checkpoint(tmp_path):
    command = export_qnn.llama_command("qwen3-0_6b", "SM8650", CFG.qnn, tmp_path / "art", None)
    assert command[1:3] == ["-m", "pipeline.qnn_launch"]
    assert "--compile_only" in command
    assert flag(command, "--decoder_model") == "qwen3-0_6b"
    assert flag(command, "--soc_model") == "SM8650"
    assert flag(command, "--model_mode") == "hybrid"
    assert flag(command, "--max_context_len") == flag(command, "--max_seq_len") == "2048"
    assert flag(command, "--prefill_ar_len") == "128"
    assert flag(command, "--calib_tasks") == "wikitext"
    assert flag(command, "--build_folder")  # llama.py realpath()s it even when compiling only
    assert "--checkpoint" not in command
    # The wheel lacks the registry's params .json, so it is always handed over.
    params = Path(flag(command, "--params"))
    assert params == settings.ROOT / "third_party/executorch/examples/models/qwen3/config/0_6b_config.json"


def test_a_forced_window_reaches_the_command_and_the_file_name(tmp_path):
    recipe = dataclasses.replace(CFG.qnn, max_context_len=4096)
    command = export_qnn.llama_command("qwen3-0_6b", "SM8750", recipe, tmp_path / "art", None)
    assert flag(command, "--max_context_len") == flag(command, "--max_seq_len") == "4096"
    assert naming.qnn_file("Qwen/Qwen3-0.6B", recipe.model_mode, 4096) == "Qwen3-0.6B-qnn-hybrid-4k.pte"


def test_a_window_below_the_prefill_length_is_refused_before_any_download(tmp_path):
    with pytest.raises(export_qnn.ExportError, match="below the prefill length"):
        export_qnn.run("Qwen/Qwen3-0.6B", "main", "SM8750", tmp_path / "out", tmp_path / "work", context=64)


def test_llama_gets_metas_original_checkpoint(tmp_path):
    meta = tmp_path / "original"
    command = export_qnn.llama_command("llama3_2-1b_instruct", "SM8750", CFG.qnn, tmp_path / "art", meta)
    assert flag(command, "--checkpoint") == str(meta / "consolidated.00.pth")
    assert flag(command, "--params") == str(meta / "params.json")
    assert flag(command, "--tokenizer_model") == str(meta / "tokenizer.model")
    assert command.count("--params") == 1


def test_every_registered_checkpoint_has_params_or_metas_checkpoint():
    for decoder in families.QNN_DECODERS.values():
        if decoder in families.QNN_META_CHECKPOINT:
            assert decoder not in families.QNN_PARAMS
        else:
            json.loads(export_qnn.params_file(decoder).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "decoder, hf_fixture",
    [
        ("qwen3-0_6b", "qwen3-0.6b"),
        ("qwen3-1_7b", "qwen3-1.7b"),
        ("qwen2_5-1_5b", "qwen2.5-1.5b"),
        ("gemma3-1b", "gemma-3-1b-it"),
        ("smollm2_135m", "smollm2-135m"),
    ],
)
def test_copied_params_describe_the_hf_checkpoint(decoder, hf_fixture):
    params = json.loads(export_qnn.params_file(decoder).read_text(encoding="utf-8"))
    arch = families.architecture(load_json(f"{hf_fixture}.config.json"), 1)
    assert params["n_layers"] == arch.n_layers
    assert params["n_heads"] == arch.n_heads
    assert params["n_kv_heads"] == arch.n_kv_heads
    assert params["dim"] == arch.dim
    assert params["hidden_dim"] == arch.intermediate
    assert params["vocab_size"] == arch.vocab_size
    assert params.get("head_dim", arch.dim // arch.n_heads) == arch.head_dim


SDK = {
    "version": "2.37.0.250724",
    "root": "/home/runner/.cache/executorch/qnn/sdk-2.37.0.250724",
    "libcxx_dir": "/home/runner/.cache/executorch/qnn/libcxx-14.0.0",
    "libcxx_files": ["libc++.so.1.0", "libc++abi.so.1.0", "libunwind.so.1"],
}


def installed_sdk(root: Path) -> dict:
    """SDK with its paths under `root`, and the files qnn_sdk checks for created there."""
    sdk = {**SDK, "root": str(root / "sdk"), "libcxx_dir": str(root / "libcxx")}
    htp = root / "sdk" / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"
    htp.parent.mkdir(parents=True)
    htp.touch()
    (root / "libcxx").mkdir()
    for name in SDK["libcxx_files"]:
        (root / "libcxx" / name).touch()
    return sdk


def test_the_sdk_is_probed_in_a_child_so_this_process_keeps_its_environment(monkeypatch, tmp_path):
    calls = []
    sdk = installed_sdk(tmp_path)

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(sdk) + "\n", stderr="Loaded libc++.so.1.0\n")

    monkeypatch.setattr(export_qnn.subprocess, "run", fake_run)
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    assert export_qnn.qnn_sdk() == sdk
    assert calls and calls[0][1] == "-c"
    # ExecuTorch 1.5.1 downloads only when asked: the package import no longer does it.
    assert "setup_qnn_sdk()" in calls[0][2]
    assert "QNN_SDK_ROOT" not in os.environ


def test_a_probe_that_downloaded_nothing_fails_before_the_script_runs(monkeypatch, tmp_path):
    # What the first Modal run hit: the paths came back, the files were never fetched.
    sdk = {**SDK, "root": str(tmp_path / "sdk"), "libcxx_dir": str(tmp_path / "libcxx")}
    monkeypatch.setattr(
        export_qnn.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout=json.dumps(sdk) + "\n", stderr=""),
    )
    with pytest.raises(export_qnn.ExportError, match="libQnnHtp.so"):
        export_qnn.qnn_sdk()


def test_sonames():
    assert export_qnn.soname("libc++.so.1.0") == "libc++.so.1"
    assert export_qnn.soname("libc++abi.so.1.0") == "libc++abi.so.1"
    assert export_qnn.soname("libunwind.so.1") == "libunwind.so.1"


def test_the_script_starts_with_the_sdk_and_libcxx_on_the_loader_path(tmp_path):
    if os.name == "nt":
        pytest.skip("symlinks need privileges on Windows; the export runs on Linux")
    links = tmp_path / "links"
    env = export_qnn.qnn_env(SDK, links, {"LD_LIBRARY_PATH": "/opt/x", "HOME": "/h"})
    assert env["QNN_SDK_ROOT"] == SDK["root"]
    assert env["LD_LIBRARY_PATH"] == f"{SDK['root']}/lib/x86_64-linux-clang:{links}:/opt/x"
    assert env["HOME"] == "/h" and env["PYTHONUNBUFFERED"] == "1"
    assert os.readlink(links / "libc++.so.1") == f"{SDK['libcxx_dir']}/libc++.so.1.0"
    assert sorted(p.name for p in links.iterdir()) == ["libc++.so.1", "libc++abi.so.1", "libunwind.so.1"]
    export_qnn.qnn_env(SDK, links, {})  # re-running over existing links is fine


def test_decoder_pte_is_found_by_mode(tmp_path):
    (tmp_path / "qwen3-0_6b_hybrid_llama_qnn.pte").write_bytes(b"x")
    assert export_qnn.decoder_pte(tmp_path, "hybrid").name == "qwen3-0_6b_hybrid_llama_qnn.pte"
    (tmp_path / "hybrid_llama_qnn.pte").write_bytes(b"x")
    with pytest.raises(export_qnn.ExportError, match="expected one hybrid"):
        export_qnn.decoder_pte(tmp_path, "hybrid")


def test_a_forced_revision_behind_main_is_refused(tmp_path, monkeypatch):
    source = make_source("Qwen/Qwen3-0.6B", sha="a" * 40)
    export_qnn.check_script_revision(source, "a" * 40)
    with pytest.raises(export_qnn.ExportError, match="compiles main"):
        export_qnn.check_script_revision(source, "b" * 40)
    # Wired into run(): refused right after the metadata fetch, before the SDK or any download.
    monkeypatch.setattr(export_qnn.hub, "fetch", lambda model_id, revision: source)
    monkeypatch.setattr(export_qnn.hub, "head_sha", lambda model_id: "b" * 40)
    monkeypatch.setattr(export_qnn, "qnn_sdk", lambda: pytest.fail("reached the SDK probe"))
    with pytest.raises(export_qnn.ExportError, match="compiles main"):
        export_qnn.run("Qwen/Qwen3-0.6B", "a" * 40, "SM8750", tmp_path / "out", tmp_path / "work")


def tiny_program(tmp_path, delegate_id=None, methods=("forward",)):
    """A .pte with one add graph per method; with ``delegate_id``, the adds run through
    that (fake) delegate."""
    from executorch.exir import to_edge
    from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
    from executorch.exir.backend.compile_spec_schema import CompileSpec
    from executorch.exir.backend.partitioner import DelegationSpec, Partitioner, PartitionResult
    from torch.export import export

    class Add(torch.nn.Module):
        def forward(self, x):
            return x + x

    edge = to_edge({name: export(Add(), (torch.ones(2),)) for name in methods})
    if delegate_id:
        # to_backend finds the backend class by name among BackendDetails' subclasses, which
        # are weak references: keep the class alive until the lowering is done.
        fake_backend = type(
            delegate_id,
            (BackendDetails,),
            {"preprocess": staticmethod(lambda program, specs: PreprocessResult(processed_bytes=b"fake-blob"))},
        )

        class All(Partitioner):
            def partition(self, exported_program):
                spec = DelegationSpec(delegate_id, [CompileSpec("k", b"v")])
                tags = {}
                for node in exported_program.graph.nodes:
                    if node.op == "call_function":
                        node.meta["delegation_tag"] = "t0"
                        tags["t0"] = spec
                return PartitionResult(tagged_exported_program=exported_program, partition_tags=tags)

        edge = edge.to_backend(All())
        del fake_backend
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "tiny.pte"
    path.write_bytes(edge.to_executorch().buffer)
    return path


def test_structural_check_reads_delegates_from_the_program(tmp_path):
    pytest.importorskip("executorch")
    plain = tiny_program(tmp_path / "plain")
    assert export_qnn.delegates(plain) == {"forward": {"backends": [], "delegate_calls": 0}}
    check = export_qnn.structural_check(plain, "kv")
    assert not check["passed"] and "missing decoder method 'kv_forward'" in check["problems"][0]

    delegated = tiny_program(tmp_path / "qnn", delegate_id=export_qnn.QNN_BACKEND_ID)
    assert export_qnn.delegates(delegated) == {"forward": {"backends": ["QnnBackend"], "delegate_calls": 1}}

    hybrid = tiny_program(tmp_path / "hybrid", delegate_id=export_qnn.QNN_BACKEND_ID, methods=export_qnn.HYBRID_METHODS)
    check = export_qnn.structural_check(hybrid, "hybrid")
    assert check["passed"], check["problems"]
    assert set(check["delegates"]) == set(export_qnn.HYBRID_METHODS)
    half = tiny_program(tmp_path / "half", delegate_id=export_qnn.QNN_BACKEND_ID, methods=("kv_forward",))
    assert "missing decoder method 'prefill_forward'" in export_qnn.structural_check(half, "hybrid")["problems"][0]
    # The same check serves Vulkan files (one "forward" method on VulkanBackend).
    from pipeline import export_xnnpack, smoke

    gpu = tiny_program(tmp_path / "gpu", delegate_id=export_xnnpack.DELEGATE_ID["vulkan"])
    assert smoke.structural_check(gpu, ("forward",), "VulkanBackend")["passed"]
    assert not smoke.structural_check(plain, ("forward",), "VulkanBackend")["passed"]
    undelegated = tiny_program(tmp_path / "cpu", methods=export_qnn.HYBRID_METHODS)
    problems = export_qnn.structural_check(undelegated, "hybrid")["problems"]
    assert len(problems) == 2 and all("does not run on QnnBackend" in p for p in problems)
    not_parsed = tmp_path / "junk.pte"
    not_parsed.write_bytes(b"QnnBackend kv_forward prefill_forward")  # the old byte grep would pass this
    check = export_qnn.structural_check(not_parsed, "hybrid")
    assert not check["passed"] and check["problems"][0].startswith("program did not load")


def test_every_registered_qnn_checkpoint_gets_names_the_app_reads_as_qnn():
    for model_id in families.QNN_DECODERS:
        repo = naming.output_repo(model_id, CFG.hub_org, CFG.repo_suffix)
        for soc in CFG.qnn.socs:
            path = f"{naming.qnn_folder(soc)}/{naming.qnn_file(model_id, CFG.qnn.model_mode, 2048)}"
            assert naming.check_app_rules(repo, path, "qnn") == [], (repo, path)


def qnn_report(soc="sm8650"):
    return {
        "backend": "qnn",
        "target": soc,
        "target_name": export_qnn.SOC_NAMES[soc.upper()],
        "tokenizer": "tokenizer.json",
        "output_repo": "experimentalmachines/Qwen3-0.6B-ExecuTorch",
        "source": {"id": "Qwen/Qwen3-0.6B", "sha": "c1899de289a0", "license": {"license": "apache-2.0"}},
        "toolchain": {"executorch": "1.4.0", "qairt": "2.37.0.250724"},
        "recipe": {"label": "QNN HTP", "description": "recipe."},
        "window": {"context": 2048, "kv_cache_bytes_per_token": None},
        "files": [{"path": f"qnn/{soc}/Qwen3-0.6B-qnn-hybrid-2k.pte", "bytes": 700_000_000, "sha256": "ab"}],
        "metadata": {},
        "smoke": {"kind": "structural", "passed": True, "problems": []},
        "run": {},
    }


def test_readme_and_config_for_qnn():
    config = manifest.backend_config(qnn_report())
    assert config["qnn_sdk_version"] == "2.37.0.250724"
    assert config["target"] == "sm8650"
    text = manifest.readme("experimentalmachines/Qwen3-0.6B-ExecuTorch", [qnn_report(), qnn_report("sm8750")], [], [])
    assert "SM8650 (Snapdragon 8 Gen 3)" in text and "SM8750 (Snapdragon 8 Elite)" in text
    assert "structure checked (no host NPU runtime)" in text
    assert "QAIRT) 2.37.0.250724" in text and "AI Stack License" in text
    assert "- qnn" in text


def test_publish_refuses_a_tokenizer_inside_a_backend_folder(tmp_path):
    folder = tmp_path / "qnn" / "sm8650"
    folder.mkdir(parents=True)
    (folder / "export-report.json").write_text(json.dumps(qnn_report()))
    (folder / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match="repo root"):
        publish.publish_hf(tmp_path, "qnn", "SM8650")


def test_the_launcher_runs_the_qualcomm_script():
    from pipeline import qnn_launch

    assert qnn_launch.SCRIPT == "executorch.examples.qualcomm.oss_scripts.llama.llama"


def test_a_tokenizer_without_bos_gets_the_end_of_text_token_as_prefix():
    # Qwen3: tokenizer_config has bos_token null, so pytorch_tokenizers sets bos_id = None.
    # eager_eval imports lm_eval and transformers, which the export image has and CI does not.
    try:
        from executorch.examples.models.llama.evaluate import eager_eval
    except (ImportError, AttributeError) as error:
        pytest.skip(f"eager_eval needs requirements/export-qnn.txt: {error}")

    from pipeline import qnn_launch

    qnn_launch.patch()

    class Tokenizer:
        bos_id = None
        eos_id = 151645

    wrapper = eager_eval.EagerEvalWrapper.__new__(eager_eval.EagerEvalWrapper)
    wrapper._tokenizer = Tokenizer()
    assert wrapper.prefix_token_id == 151645
    Tokenizer.bos_id = 1
    assert wrapper.prefix_token_id == 1
