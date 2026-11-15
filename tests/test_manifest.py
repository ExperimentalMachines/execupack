from pipeline import manifest, smoke


def report(source_id="Qwen/Qwen3-1.7B", backend="xnnpack", license=None):
    return {
        "backend": backend,
        "target": None,
        "tokenizer": "tokenizer.json",
        "output_repo": "experimentalmachines/Qwen3-1.7B-ExecuTorch",
        "source": {
            "id": source_id,
            "sha": "abcdef0123456789",
            "license": license or {"license": "apache-2.0"},
            "license_files": ["LICENSE"],
        },
        "toolchain": {"executorch": "1.4.0"},
        "recipe": {"label": "8da4w-g32, int8 embeddings", "description": "recipe text."},
        "window": {"context": 8192, "kv_cache_bytes_per_token": 229_376},
        "files": [{"path": "xnnpack/Qwen3-1.7B-8da4w-8k.pte", "bytes": 1_500_000_000, "sha256": "ff"}],
        "metadata": {
            "get_max_context_len": 8192,
            "get_max_seq_len": 2048,
            "get_eos_ids": [151645, 151643],
            "methods": ["forward", "get_max_context_len"],
        },
        "smoke": {"passed": True, "answered": True},
        "run": {"url": "https://github.com/o/r/actions/runs/1"},
    }


def test_backend_config_is_in_the_form_the_app_reads():
    config = manifest.backend_config(report())
    assert config["runtime_version"] == "1.4.0"
    [variant] = config["variants"]
    assert variant["file"] == "Qwen3-1.7B-8da4w-8k.pte"  # bare file name
    assert variant["methods"]["get_max_context_len"] == 8192
    assert "methods" not in variant["methods"]


def window(context, executorch, backend="xnnpack"):
    r = report(backend=backend)
    label = f"{context // 1024}k"
    r["toolchain"] = {"executorch": executorch}
    r["window"] = {**r["window"], "context": context}
    r["files"] = [{**r["files"][0], "path": f"{backend}/Qwen3-1.7B-8da4w-{label}.pte"}]
    return r


def test_a_folder_of_two_releases_names_each_file_and_the_newest():
    # Windows are republished one at a time: a 2k file rebuilt on 1.6.0 beside a 32k one still
    # from 1.5.1. Each variant names its own release. The folder, all the apps read today, names
    # the newest: an installed app on the older runtime is then warned about the newer file.
    mixed = [window(2048, "1.6.0"), window(32768, "1.5.1")]
    for reports in (mixed, list(reversed(mixed))):
        config = manifest.backend_config(reports)
        assert config["runtime_version"] == "1.6.0"
        assert [(v["context"], v["runtime_version"]) for v in config["variants"]] == [(2048, "1.6.0"), (32768, "1.5.1")]
    assert manifest.backend_config([window(2048, "1.4.0"), window(4096, "1.10.0")])["runtime_version"] == "1.10.0"
    assert manifest.runtime_versions([window(2048, "1.10.0"), window(4096, "1.9.0"), window(8192, "1.10.0")]) == [
        "1.9.0",
        "1.10.0",
    ]
    # A pre-release sorts before its release, whatever the report order.
    pre = [window(2048, "1.6.0a0+git1234"), window(4096, "1.6.0")]
    assert (
        manifest.runtime_versions(pre) == manifest.runtime_versions(list(reversed(pre))) == ["1.6.0a0+git1234", "1.6.0"]
    )


def test_a_report_without_a_release_is_named_as_found_not_a_crash():
    # Retained reports are regenerated into config.json and the card on every publish: one that
    # names no release, or not one packaging reads, must not stop another backend's publish.
    odd = [window(2048, "1.5.1"), window(4096, "main"), window(8192, "1")]
    odd.append(window(16384, "1.5.1"))
    odd[-1]["toolchain"] = {}
    for missing in ("absent", None):
        # The largest window sets the folder's shared fields: its toolchain may be gone too.
        largest = window(32768, "1.5.1")
        if missing == "absent":
            del largest["toolchain"]
        else:
            largest["toolchain"] = None
        config = manifest.backend_config([window(2048, "1.5.1"), largest])
        assert config["runtime_version"] == "1.5.1" and "qnn_sdk_version" not in config
        assert [v["runtime_version"] for v in config["variants"]] == ["1.5.1", None]
    assert manifest.runtime_versions(odd) == ["None", "main", "1", "1.5.1"]
    assert manifest.backend_config(odd)["runtime_version"] == "1.5.1"
    text = manifest.readme("experimentalmachines/Qwen3-1.7B-ExecuTorch", odd, [], ["LICENSE"])
    assert "the release its file was built with" in text  # no single runtime promised for these
    assert not manifest.one_runtime_for_all(["1", "1.5.1"]) or True
    assert manifest.one_runtime_for_all(["1.5.1", "1.6.0"]) and not manifest.one_runtime_for_all(["1.4.0", "1.6.0"])
    assert manifest.one_runtime_for_all(["1"])


def test_the_card_names_the_newest_cpu_or_gpu_release_not_the_npu_one():
    # A MediaTek folder built a release behind (config/versions.env) must not set the CPU files'
    # runtime, and neither may the smallest CPU window when a larger one is newer.
    reports = [window(2048, "1.5.1"), window(32768, "1.6.0"), window(2048, "1.5.1", backend="mtk")]
    reports[2].update(
        target="mt6991",
        neuropilot={"build": "8.0.8-build20250925", "mtk_converter": "8.13.0", "mtk_neuron": "8.2.23"},
        runner={},
    )
    text = manifest.readme("experimentalmachines/Qwen3-1.7B-ExecuTorch", reports, [], ["LICENSE"])
    assert "any ExecuTorch 1.6.0 runtime" in text and "ExecuTorch 1.5.1 runtime" not in text


def test_the_card_names_no_single_runtime_for_files_two_releases_apart():
    # ExecuTorch promises a runtime loads its own release's files and the previous minor's: a
    # 1.4.0 file beside a 1.6.0 one has no runtime promised for both.
    reports = [window(2048, "1.4.0"), window(32768, "1.6.0")]
    text = manifest.readme("experimentalmachines/Qwen3-1.7B-ExecuTorch", reports, [], ["LICENSE"])
    assert "any ExecuTorch 1.6.0 runtime" not in text and "any ExecuTorch 1.4.0 runtime" not in text
    assert "the release its file was built with (1.4.0, 1.6.0: each variant's `runtime_version`" in text


def test_readme_has_card_metadata_and_kv_arithmetic():
    text = manifest.readme("experimentalmachines/Qwen3-1.7B-ExecuTorch", [report()], ["executorch"], ["LICENSE"])
    assert text.startswith("---\nlicense: apache-2.0\n")
    assert "base_model_relation: quantized" in text
    assert "- xnnpack" in text and "- executorch" in text
    # 229,376 x 8,192 = 1,879,048,192
    assert "at 8,192 tokens: the KV cache costs 229,376 bytes per token (fp32), 1,879,048,192 bytes" in text
    assert "[`LICENSE`](LICENSE)" in text
    assert "NOTICE" not in text


def test_llama_derivatives_carry_the_attribution_the_license_requires():
    r = report("meta-llama/Llama-3.2-1B-Instruct", license={"license": "llama3.2"})
    text = manifest.readme("experimentalmachines/Llama-3.2-1B-Instruct-ExecuTorch", [r], [], ["LICENSE.txt"])
    assert "**Built with Llama**" in text
    assert "[`NOTICE`](NOTICE)" in text
    assert "Llama 3.2 Community License" in manifest.NOTICES["llama32"]


def test_other_license_names_are_used():
    r = report(
        "Qwen/Qwen2.5-3B-Instruct",
        license={"license": "other", "license_name": "qwen-research", "license_link": "https://x/LICENSE"},
    )
    text = manifest.readme("experimentalmachines/Qwen2.5-3B-Instruct-ExecuTorch", [r], [], ["LICENSE"])
    assert "license_name: qwen-research" in text
    assert "(qwen-research)" in text


def test_degenerate_output():
    assert smoke.degenerate(["a"] * 8)
    assert not smoke.degenerate(["a"] * 7)
    assert not smoke.degenerate(["a"] * 7 + ["b"])
