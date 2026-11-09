import dataclasses
import json
import subprocess
from pathlib import Path

import pytest
from conftest import MTK_FLEET, TOTAL_PARAMS, hf_config, load_json

from pipeline import eligibility, export_mtk, families, manifest, naming, publish, settings

CFG = settings.load()
# The runner choice is tested on a three-size fleet (conftest); the configured tier is one.
FLEET = dataclasses.replace(CFG.mtk, runner_tiers=MTK_FLEET)


def flag(command, name):
    return command[command.index(name) + 1]


def plan(model_id="Qwen/Qwen3-0.6B", config=None):
    config = config or hf_config(model_id)
    return families.mtk_plan(families.family_for(config), config, CFG.mtk.max_chunks)


@pytest.mark.parametrize(("layers", "chunks"), [(28, 4), (24, 4), (36, 4), (30, 3), (26, 2), (1, 1)])
def test_chunks_split_the_layers_evenly(layers, chunks):
    assert families.mtk_chunks(layers, 4) == chunks


def test_qwen_families_use_mediateks_qwen_script_and_their_chat_templates():
    assert plan() == families.MtkPlan("qwen.py", "qwen3.json", 4)
    qwen25 = load_json("qwen2.5-1.5b-instruct.config.json")
    assert families.mtk_plan(families.family_for(qwen25), qwen25, 4) == families.MtkPlan("qwen.py", "qwen.json", 4)


def test_lfm2_uses_lfm2_script_and_chatml_preformatter():
    cfg_1_2b = {"architectures": ["Lfm2ForCausalLM"], "model_type": "lfm2", "num_hidden_layers": 16}
    plan_1_2b = families.mtk_plan(families.family_for(cfg_1_2b), cfg_1_2b, CFG.mtk.max_chunks)
    assert plan_1_2b == families.MtkPlan("lfm2.py", "qwen3.json", 4)

    cfg_2_6b = {"architectures": ["Lfm2ForCausalLM"], "model_type": "lfm2", "num_hidden_layers": 30}
    plan_2_6b = families.mtk_plan(families.family_for(cfg_2_6b), cfg_2_6b, CFG.mtk.max_chunks)
    assert plan_2_6b == families.MtkPlan("lfm2.py", "qwen3.json", 3)


def test_rope_scaling_and_other_model_types_are_refused():
    config = hf_config("Qwen/Qwen3-0.6B") | {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}}
    with pytest.raises(families.UnsupportedModel, match="RoPE scaling"):
        plan(config=config)
    with pytest.raises(families.UnsupportedModel, match="model_type"):
        plan(config=hf_config("Qwen/Qwen3-0.6B") | {"model_type": "qwen2"})


def test_export_command_matches_mediateks_shell_scripts(tmp_path):
    command = export_mtk.export_command("/venv/bin/python", plan(), CFG.mtk, "MT6989", tmp_path / "config.json")
    assert command[:3] == ["/venv/bin/python", "model_export_scripts/qwen.py", str(tmp_path / "config.json")]
    assert flag(command, "--precision") == "A16W4"
    assert flag(command, "--num_chunks") == "4"
    assert flag(command, "--dataset") == "aot_utils/llm_utils/prompts/alpaca.txt"
    assert flag(command, "--preformatter") == "aot_utils/llm_utils/preformatter_templates/qwen3.json"
    shapes = command.index("-shapes")
    assert command[shapes + 1 : shapes + 3] == ["128t512c", "1t512c"]
    assert flag(command, "--response_cap") == "9"
    assert flag(command, "--platform") == "DX3"
    assert flag(export_mtk.export_command("p", plan(), CFG.mtk, "MT6991", tmp_path), "--platform") == "DX4"


def test_output_names_follow_the_scripts(tmp_path):
    exp = export_mtk.exp_name(tmp_path / "Qwen3-0.6B", "A16W4", 4)
    assert exp == "Qwen3-0.6B_A16W4_4_chunks"
    assert export_mtk.method_names(exp, CFG.mtk, 3) == [
        "Qwen3-0.6B_A16W4_4_chunks_128t512c_3",
        "Qwen3-0.6B_A16W4_4_chunks_1t512c_3",
    ]


def test_every_chunk_name_reads_as_neuropilot_in_the_app():
    for model_id in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B", "Qwen/Qwen2.5-1.5B-Instruct"):
        repo = naming.output_repo(model_id, CFG.hub_org, CFG.repo_suffix)
        for soc in CFG.mtk.socs:
            for i in range(4):
                path = f"{naming.mtk_folder(soc)}/{naming.mtk_chunk_file(model_id, 'A16W4', 2048, i, 4)}"
                assert naming.check_app_rules(repo, path, "mtk") == [], path
    assert (
        naming.mtk_chunk_file("Qwen/Qwen3-0.6B", "A16W4", 2048, 0, 4) == "Qwen3-0.6B-neuropilot-a16w4-2k-chunk1of4.pte"
    )


def test_runner_settings_for_qwen3_0_6b():
    runner = export_mtk.runner_settings(hf_config("Qwen/Qwen3-0.6B"), CFG.mtk, 151643, [151645, 151643])
    assert runner["hidden_size"] == 1024
    assert runner["num_head"] == 16
    assert runner["num_layer"] == 28
    assert runner["head_dim"] == 128
    assert runner["rot_emb_base"] == 1000000.0
    assert runner["cache_size"] == 512 and runner["prompt_token_batch_size"] == 128
    assert runner["eos_token"] == 151645 and runner["eos_tokens"] == [151645, 151643]
    assert runner["vocab_size"] == 151936
    assert runner["tokenizer_type"] == "hf" and runner["cache_type"] == "fp32"


def test_a_forced_window_changes_the_shapes(tmp_path):
    recipe = dataclasses.replace(CFG.mtk, cache_size=4096)
    command = export_mtk.export_command("p", plan(), recipe, "MT6991", tmp_path)
    shapes = command.index("-shapes")
    assert command[shapes + 1 : shapes + 3] == ["128t4096c", "1t4096c"]


def test_unknown_chip_and_short_window_are_refused_before_any_download(tmp_path):
    with pytest.raises(export_mtk.ExportError, match="unknown MediaTek chip"):
        export_mtk.run("Qwen/Qwen3-0.6B", "main", "MT6878", tmp_path, tmp_path, "p", tmp_path)
    with pytest.raises(export_mtk.ExportError, match="below the prompt length"):
        export_mtk.run("Qwen/Qwen3-0.6B", "main", "MT6991", tmp_path, tmp_path, "p", tmp_path, context=64)


def test_the_precision_comes_from_the_environment_and_only_as_a_name_mediatek_knows(tmp_path, monkeypatch):
    monkeypatch.setenv("MTK_PRECISION", "A16W3")
    with pytest.raises(export_mtk.ExportError, match="unknown NeuroPilot precision"):
        export_mtk.run("Qwen/Qwen3-0.6B", "main", "MT6991", tmp_path, tmp_path, "p", tmp_path)
    # And a known one names the files, so an A16W8 export sits beside an A16W4 one.
    assert "-neuropilot-a16w8-4k-" in naming.mtk_chunk_file("Qwen/Qwen3-0.6B", "A16W8", 4096, 0, 4)


def test_the_prompt_batch_comes_from_the_environment_and_must_be_positive(tmp_path, monkeypatch):
    monkeypatch.setenv("MTK_PROMPT_TOKENS", "0")
    with pytest.raises(export_mtk.ExportError, match="MTK_PROMPT_TOKENS must be positive"):
        export_mtk.run("Qwen/Qwen3-0.6B", "main", "MT6991", tmp_path, tmp_path, "p", tmp_path)
    # A window below the batch is still refused against the batch actually used.
    monkeypatch.setenv("MTK_PROMPT_TOKENS", "64")
    with pytest.raises(export_mtk.ExportError, match="below the prompt length 64"):
        export_mtk.run("Qwen/Qwen3-0.6B", "main", "MT6991", tmp_path, tmp_path, "p", tmp_path, context=32)


def mtk_report(soc="mt6991"):
    chunks = [f"mtk/{soc}/Qwen3-0.6B-neuropilot-a16w4-2k-chunk{i}of4.pte" for i in range(1, 5)]
    embedding = f"mtk/{soc}/Qwen3-0.6B-neuropilot-embedding-fp32.bin"
    return {
        "backend": "mtk",
        "target": soc,
        "target_name": export_mtk.SOC_NAMES[soc.upper()],
        "tokenizer": "tokenizer.json",
        "output_repo": "experimentalmachines/Qwen3-0.6B-ExecuTorch",
        "source": {"id": "Qwen/Qwen3-0.6B", "sha": "c1899de289a0", "license": {"license": "apache-2.0"}},
        "toolchain": {"executorch": "1.4.0"},
        "neuropilot": {
            "name": "NeuroPilot Express SDK",
            "build": "8.0.8-build20250925",
            "mtk_converter": "8.13.0+public",
            "mtk_neuron": "8.2.23",
        },
        "recipe": {"label": "NeuroPilot A16W4, 4 chunks", "description": "recipe."},
        "window": {"context": 2048, "kv_cache_bytes_per_token": None},
        "runner": {"token_embedding_path": embedding.rsplit("/", 1)[-1], "cache_size": 2048},
        "files": [{"path": p, "bytes": 150_000_000, "sha256": "ab"} for p in chunks]
        + [{"path": embedding, "bytes": 622_329_856, "sha256": "cd"}],
        "metadata": {},
        "smoke": {"kind": "structural", "passed": True, "problems": []},
        "run": {},
    }


def test_config_describes_one_model_in_several_files():
    config = manifest.backend_config(mtk_report())
    assert len(config["variants"]) == 1
    variant = config["variants"][0]
    assert variant["files"] == [f"Qwen3-0.6B-neuropilot-a16w4-2k-chunk{i}of4.pte" for i in range(1, 5)]
    assert variant["embedding"] == "Qwen3-0.6B-neuropilot-embedding-fp32.bin"
    assert variant["size_bytes"] == 4 * 150_000_000 + 622_329_856
    # MediaTek's runner flags differ per window (cache size, file names): they live on the variant.
    assert variant["runner"]["cache_size"] == 2048 and variant["context"] == 2048
    assert "runner" not in config
    assert config["neuropilot_sdk"]["build"] == "8.0.8-build20250925"


def test_readme_credits_mediatek_without_claiming_the_sdk_is_included():
    text = manifest.readme("experimentalmachines/Qwen3-0.6B-ExecuTorch", [mtk_report(), mtk_report("mt6989")], [], [])
    assert "MT6991 (Dimensity 9400)" in text and "MT6989 (Dimensity 9300)" in text
    assert "MediaTek NeuroPilot Express SDK 8.0.8-build20250925 (mtk_converter" in text and "MediaTek Inc." in text
    assert "No MediaTek SDK or runtime library is included" in text
    assert "neuropilot-embedding-fp32.bin" in text
    assert "- mtk" in text


def test_the_card_states_each_backends_windows_from_its_reports():
    small, large = mtk_report(), mtk_report()
    small["window"] = {**small["window"], "context": 512}
    large["window"] = {**large["window"], "context": 8192}
    assert manifest.window_spans([small, large]) == "MediaTek NeuroPilot MT6991 (Dimensity 9400) at 512 to 8k"
    assert manifest.window_spans([small]) == "MediaTek NeuroPilot MT6991 (Dimensity 9400) at 512"


def test_the_card_names_the_runner_a_per_layer_build_needs_and_only_then():
    uniform = manifest.readme("experimentalmachines/Qwen3-0.6B-ExecuTorch", [mtk_report()], [], [])
    assert "state_layout: per-layer" not in uniform
    lfm = mtk_report()
    lfm["runner"] = {**lfm["runner"], "state_layout": "per-layer"}
    text = manifest.readme("experimentalmachines/LFM2.5-1.2B-Instruct-ExecuTorch", [lfm, mtk_report()], [], [])
    assert "`state_layout: per-layer`" in text and "cannot load them" in text
    assert "tools/npu/patches/executorch-release-1.4-pd.patch" in text


def test_publish_refuses_a_tokenizer_inside_a_mediatek_folder(tmp_path):
    folder = tmp_path / "mtk" / "mt6991"
    folder.mkdir(parents=True)
    (folder / "export-report.json").write_text(json.dumps(mtk_report()))
    (folder / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match="repo root"):
        publish.publish_hf(tmp_path, "mtk", "MT6991")


def test_calibration_memory_estimate_covers_the_measured_peak_and_refuses_what_did_not_fit():
    # Qwen3-0.6B: 28 layers x K,V x 8 KV heads x 128 x 4 B = 229,376 B of fp32 cache per token.
    # MediaTek's alpaca.txt holds 9 prompts (8 newlines: its last line has none).
    config = hf_config("Qwen/Qwen3-0.6B")
    params = TOTAL_PARAMS["Qwen/Qwen3-0.6B"]
    weights = 751_632_384 * 6

    # The peak is one prompt's steps, not the corpus. This is the property the
    # writer_batch_size=1 in the MediaTek patches buys, and the reason a bigger calibration
    # set is now free: nine prompts cost what one does.
    at_512 = export_mtk.calibration_bytes(config, CFG.mtk, 9, params)
    assert at_512 == export_mtk.calibration_bytes(config, CFG.mtk, 512, params)
    assert at_512 == int(10 * 229_376 * 512 * export_mtk.CALIBRATION_OVERHEAD) + weights

    # Both runs that were measured held every prompt at once, so their peaks bound the old
    # behaviour and the new estimate has to come in well under them, or the patch did
    # nothing. Qwen3-0.6B run 34760462220: 29,592,731,648 B. LFM2.5-1.2B 2026-09-13:
    # 18,383,785,984 B. Nothing has yet measured the peak *with* writer_batch_size=1, which
    # is why CALIBRATION_OVERHEAD keeps the larger of the two solved factors.
    assert at_512 < 29_592_731_648 / 3

    # LFM2.5-1.2B and 2.6B are what this matrix exports: 16 and 30 layers, 8 KV heads, head
    # dim 64. Every window from 2k to 32k had to fit the 128 GB runner used until 2026-10-07,
    # with the 64 GiB swap the workflow added, or the window could not be built there.
    budget = 128_000_000_000 + 64 * 1024**3 - 1_000_000_000
    for layers, model_params in ((16, 1_170_340_608), (30, 2_600_000_000)):
        lfm = {
            "num_hidden_layers": layers,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "hidden_size": 2048,
        }
        for window in (2048, 4096, 8192, 16384, 32768):
            recipe = dataclasses.replace(CFG.mtk, cache_size=window)
            assert export_mtk.calibration_bytes(lfm, recipe, 9, model_params) < budget


def test_calibration_disk_is_what_the_runner_tier_has_to_carry():
    # writer_batch_size=1 took the prompt count out of memory by putting every prompt's
    # steps in the Arrow cache instead, so disk is now the constraint that picks the runner.
    lfm = {
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "hidden_size": 2048,
    }
    at_32k = dataclasses.replace(CFG.mtk, cache_size=32768)
    cache = export_mtk.calibration_disk_bytes(lfm, at_32k, 9)
    assert cache == 9 * 10 * 65_536 * 32_768  # 193 GB

    # The small fleet runner's 160 GB disk cannot hold it; the medium one's 750 GB can.
    assert cache + export_mtk.DISK_MARGIN_BYTES > 160 * 10**9
    assert cache + export_mtk.DISK_MARGIN_BYTES < 750 * 10**9

    # It scales with the window, which is the whole point of checking per window.
    at_2k = dataclasses.replace(CFG.mtk, cache_size=2048)
    assert export_mtk.calibration_disk_bytes(lfm, at_2k, 9) * 16 == cache


def test_the_runner_is_sized_by_the_window_and_both_limits_bind():
    lfm12 = {
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "hidden_size": 2048,
    }
    lfm26 = dict(lfm12, num_hidden_layers=30)

    def pick(config, window, params):
        tier = export_mtk.pick_runner(config, dataclasses.replace(FLEET, cache_size=window), 9, params)
        return tier.label if tier else None

    # The window sets the size, not the model: the same model wants three different tiers.
    assert pick(lfm12, 2048, 1_170_340_608) == "small"
    assert pick(lfm12, 8192, 1_170_340_608) == "medium"
    assert pick(lfm12, 32768, 1_170_340_608) == "large"

    # And the model matters at a fixed window, because both terms carry its shape.
    assert pick(lfm26, 8192, 2_600_000_000) == "large"

    # Tiers are tried smallest first, so a window that fits the smallest never gets a
    # bigger one: that is the saving this exists for.
    smallest = FLEET.runner_tiers[0]
    assert pick(lfm12, 2048, 1_170_340_608) == smallest.label

    # Nothing is handed a runner that cannot hold it. The 2.6B at 32k needs about 157 GiB
    # against the largest tier's 128 GB, so it is the one window placed on swap, and it is
    # still placed rather than dropped.
    biggest = FLEET.runner_tiers[-1]
    assert pick(lfm26, 32768, 2_600_000_000) == biggest.label
    at_32k = dataclasses.replace(CFG.mtk, cache_size=32768)
    need = export_mtk.calibration_bytes(lfm26, at_32k, 9, 2_600_000_000)
    assert need > biggest.ram_bytes
    assert need < biggest.ram_bytes + biggest.swap_gib * 1024**3

    # A window past every tier's RAM goes to the tier with the most RAM, not to the smallest
    # that swap can stretch to: Qwen2.5-3B at 32k (36 layers, 2 KV heads of 128) needs about
    # 110 GB, which the 64 GB tier with 64 GiB of swap would also "carry".
    qwen25_3b = {
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "hidden_size": 2048,
    }
    assert pick(qwen25_3b, 32768, 3_085_938_688) == biggest.label


def test_no_runner_tier_is_arm_because_the_toolchain_is_x86_only():
    # mtk_converter is a cp310 manylinux x86_64 wheel and mtk_neuron is tagged
    # linux_x86_64. An ARM runner cannot install either, whatever its size.
    for tier in CFG.mtk.runner_tiers:
        assert "arm" not in tier.label


def test_the_calibration_patch_adds_what_the_export_checks_for():
    patch = settings.ROOT / "third_party/executorch/patches/mediatek-calibration-as-arrays.patch"
    added = [line[1:].strip() for line in patch.read_text(encoding="utf-8").splitlines() if line.startswith("+ ")]
    assert export_mtk.PATCH_MARKER in added
    # One family script per patch target: every mapped script must be covered.
    targets = {line.split("/")[-1] for line in patch.read_text(encoding="utf-8").splitlines() if line.startswith("+++")}
    assert {plan().script} <= targets


def test_the_export_script_starts_with_mtk_neurons_lib_on_the_loader_path(monkeypatch):
    lib = "/w/mtk-venv/lib/python3.10/site-packages/mtk_neuron/lib"

    def fake_run(command, **kwargs):
        assert command[1] == "-c" and "mtk_neuron" in command[2]
        return subprocess.CompletedProcess(command, 0, stdout=lib + "\n", stderr="")

    monkeypatch.setattr(export_mtk.subprocess, "run", fake_run)
    env = export_mtk.tool_env("/w/mtk-venv/bin/python", {"LD_LIBRARY_PATH": "/opt/x", "HOME": "/h"})
    assert env["LD_LIBRARY_PATH"] == f"{lib}:/opt/x"
    assert env["HOME"] == "/h" and env["PYTHONUNBUFFERED"] == "1"
    assert export_mtk.tool_env("p", {})["LD_LIBRARY_PATH"] == lib


LFM12 = {
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "hidden_size": 2048,
}
LFM26 = dict(LFM12, num_hidden_layers=30)


def test_the_corpus_fills_every_window_and_leaves_room_for_the_replies():
    from pipeline import mtk_corpus

    for window in (512, 2048, 32768):
        planned = mtk_corpus.plan(window, CFG.mtk.response_cap, CFG.mtk.long_samples, 9)
        assert [kind for kind, _ in planned[:9]] == ["short"] * 9
        long_targets = [target for kind, target in planned if kind != "short"]
        # The longest sample reaches the last position a prompt can hold, so every position
        # the window has is observed; nothing is longer, or the replies would not fit.
        assert max(long_targets) == window - CFG.mtk.response_cap - 1
        assert long_targets == sorted(long_targets)
        kinds = [kind for kind, _ in planned[9:]]
        assert kinds.count("chat") == kinds.count("document") == CFG.mtk.long_samples // 2


def test_the_corpus_description_names_its_sources_and_what_it_filled():
    from pipeline import mtk_corpus

    summary = {
        "samples": 23,
        "kinds": {"short": 9, "chat": 7, "document": 7},
        "fill_min": 15,
        "fill_max": 32758,
        "tokens_total": 247000,
    }
    text = mtk_corpus.describe(summary)
    for needle in ("UltraChat 200k", "PG19", "alpaca.txt", "32,758", "own chat template"):
        assert needle in text


def test_the_lfm2_patch_carries_the_streaming_calibration_the_export_checks_for():
    patch = settings.ROOT / "third_party/executorch/patches/mediatek-lfm2.patch"
    added = [line[1:] for line in patch.read_text(encoding="utf-8").splitlines() if line.startswith("+")]
    assert any(line.startswith(export_mtk.STREAMING_MARKER) for line in added)
    lfm = {"architectures": ["Lfm2ForCausalLM"], "model_type": "lfm2", "num_hidden_layers": 16}
    assert plan(config=lfm).script in export_mtk.STREAMING_SCRIPTS
    # Qwen keeps MediaTek's own script and its Arrow round trip, with the arrays patch.
    assert plan().script not in export_mtk.STREAMING_SCRIPTS


def test_lfm2_conv_layers_carry_their_own_state_and_the_padding_inputs():
    patch = settings.ROOT / "third_party/executorch/patches/mediatek-lfm2.patch"
    text = patch.read_text(encoding="utf-8")
    modeling = text[text.index("modeling_lfm2.py b/") :]
    added = [line[1:] for line in modeling.splitlines() if line.startswith("+")]
    script = text[text.index("model_export_scripts/lfm2.py b/") : text.index("configuration_lfm2.py b/")]
    # The export checks the script for the marker before it runs it, as run() reads it.
    assert export_mtk.PER_LAYER_STATE_MARKER in script
    assert "def state_shapes(" in modeling and "def conv_inputs(" in modeling
    # The export checks for the marker; the conv layer takes its own state, the valid mask and
    # the selector, and no longer rides a K cache.
    assert any("def forward(self, hidden_states, conv_valid, conv_select, conv_state):" in line for line in added)
    assert not any("past_key[:, :, :8, :]" in line for line in added)
    # Grouped-query attention by regrouping the queries: no K/V copied four times per layer.
    assert "class Lfm2Attention(Attention):" in modeling
    assert export_mtk.state_layout("lfm2.py") == "per-layer"
    # Qwen keeps MediaTek's uniform caches, which the stock runner reads.
    assert export_mtk.state_layout(plan().script) == "uniform"


def test_a_rendered_corpus_is_passed_without_mediateks_preformatter(tmp_path):
    corpus = str(tmp_path / "calibration-4096.jsonl")
    command = export_mtk.export_command("p", plan(), CFG.mtk, "MT6991", tmp_path, corpus)
    assert flag(command, "--dataset") == corpus
    assert "--preformatter" not in command
    # The prompt file still goes through the preformatter, as MediaTek's scripts expect.
    assert "--preformatter" in export_mtk.export_command("p", plan(), CFG.mtk, "MT6991", tmp_path)


def test_streaming_calibration_keeps_nothing_on_disk_and_barely_grows_with_the_window():
    at = {w: dataclasses.replace(CFG.mtk, cache_size=w) for w in (512, 32768)}
    for config, params in ((LFM12, 1_170_340_608), (LFM26, 2_697_198_592)):
        assert export_mtk.calibration_disk_bytes(config, at[32768], 23, streaming=True) == 0
        small = export_mtk.calibration_bytes(config, at[512], 23, params, streaming=True)
        large = export_mtk.calibration_bytes(config, at[32768], 23, params, streaming=True)
        # Lowering sets the floor, measured at 15.7 and 14.4 bytes per parameter.
        assert small >= params * 14.4
        assert large < small * 1.2
        # Against MediaTek's Arrow round trip at the same window: 88.6 and 169 GB.
        assert large < export_mtk.calibration_bytes(config, at[32768], 9, params) / 2


def test_streaming_windows_go_to_the_cores_that_finish_them():
    def pick(config, window, params):
        recipe = dataclasses.replace(FLEET, cache_size=window)
        tier = export_mtk.pick_runner(config, recipe, 23, params, streaming=True)
        return tier.label if tier else None

    assert pick(LFM12, 2048, 1_170_340_608) == "small"
    assert pick(LFM12, 4096, 1_170_340_608) == "small"
    assert pick(LFM12, 16384, 1_170_340_608) == "medium"
    assert pick(LFM12, 32768, 1_170_340_608) == "large"
    # The 2.6B's lowering needs more than the smallest tier holds in RAM at any window, and
    # its 32k window is a time question, not a memory one: it no longer needs swap.
    assert pick(LFM26, 2048, 2_697_198_592) == "medium"
    biggest = FLEET.runner_tiers[-1]
    at_32k = dataclasses.replace(CFG.mtk, cache_size=32768)
    assert export_mtk.calibration_bytes(LFM26, at_32k, 23, 2_697_198_592, streaming=True) < biggest.ram_bytes


def test_llama_family_uses_mediateks_llama_script_with_the_fast_tokenizer():
    llama32 = load_json("llama-3.2-1b-instruct.config.json")
    # Llama 3's frequency scaling is implemented by mediatek-rope-theta.patch.
    assert families.mtk_plan(families.family_for(llama32), llama32, 4) == families.MtkPlan(
        "llama.py", "llama3.json", 4, "pretrained_fast"
    )
    smollm2 = load_json("smollm2-135m.config.json")
    # SmolLM2 is plain RoPE and speaks ChatML, so it calibrates in that template.
    assert families.mtk_plan(families.family_for(smollm2), smollm2, 4) == families.MtkPlan(
        "llama.py",
        "qwen.json",
        3,
        "pretrained_fast",  # 30 layers: three chunks of ten
    )


def test_the_patches_carry_what_the_export_checks_for():
    patches = settings.ROOT / "third_party/executorch/patches"
    calibration = (patches / "mediatek-calibration-as-arrays.patch").read_text(encoding="utf-8")
    for script in ("qwen.py", "llama.py"):
        assert f"+++ b/examples/mediatek/model_export_scripts/{script}" in calibration
    added = [line[1:].strip() for line in calibration.splitlines() if line.startswith("+")]
    assert export_mtk.WRITER_MARKER in added
    rope = (patches / "mediatek-rope-theta.patch").read_text(encoding="utf-8")
    assert sum(export_mtk.ROPE_MARKER in line for line in rope.splitlines() if line.startswith("+")) == 2
    for config in export_mtk.ROPE_CONFIGS.values():
        assert f"+++ b/examples/mediatek/models/llm_models/{config}" in rope
    gemma = (patches / "mediatek-gemma.patch").read_text(encoding="utf-8")
    for relative, marker in export_mtk.GEMMA3_MARKERS.items():
        assert f"+++ b/examples/mediatek/{relative}" in gemma
        assert sum(line[1:].strip() == marker for line in gemma.splitlines() if line.startswith("+")) == 1


def test_gemma3_config_is_written_in_mediateks_spelling():
    mtk = families.mtk_model_config(hf_config("google/gemma-3-1b-it"))
    assert mtk["model_type"] == "gemma3"  # HF says gemma3_text; configuration_gemma reads gemma3
    assert mtk["sliding_window_attention_size"] == 512  # MediaTek's default is 1024
    assert mtk["global_local_attention_pattern"] == ["SLIDING_LOCAL"] * 5 + ["GLOBAL"]
    assert mtk["use_res_clamp"] is False  # HF clamps no residual; MediaTek's default does
    assert families.mtk_model_config(hf_config("Qwen/Qwen3-0.6B")) == hf_config("Qwen/Qwen3-0.6B")


def test_a_gemma3_layer_pattern_that_does_not_repeat_is_refused():
    config = {**hf_config("google/gemma-3-1b-it"), "sliding_window_pattern": None}
    config["layer_types"] = ["sliding_attention"] * 5 + ["full_attention"] + ["full_attention"] * 20
    with pytest.raises(families.UnsupportedModel, match="one period"):
        families.mtk_model_config(config)


def test_gemma3_runs_gemma_py_and_its_runner_config_names_both_rope_tables():
    config = hf_config("google/gemma-3-1b-it")
    assert families.family_for(config).key == "gemma3"
    plan = families.mtk_plan(families.family_for(config), config, CFG.mtk.max_chunks)
    assert (plan.script, plan.preformatter, plan.num_chunks) == ("gemma.py", "gemma.json", 2)  # 26 layers
    three = export_mtk.runner_settings(hf_config("google/gemma-3-1b-it"), CFG.mtk, 2, [1, 106])
    assert (three["rot_emb_base"], three["rot_emb_local_base"], three["window_size"]) == (1e6, 10000.0, 512)
    assert three["rot_emb_inputs"] == ["global", "local"]
    two = export_mtk.runner_settings(hf_config("google/gemma-2-2b-it"), CFG.mtk, 2, [1, 107])
    assert two["rot_emb_base"] == 10000.0 and "rot_emb_local_base" not in two and "window_size" not in two


def test_a_head_over_mtk_converters_2gb_protobuf_is_refused_at_any_chunk_count():
    gemma2 = hf_config("google/gemma-2-2b-it")
    assert families.family_for(gemma2).key == "gemma2"
    assert families.mtk_head_bytes(gemma2) == 256000 * 2304 * 4  # measured refused, 2026-11-09
    for chunks in (2, 13, 26):
        with pytest.raises(families.UnsupportedModel, match="2 GB"):
            families.mtk_plan(families.family_for(gemma2), gemma2, chunks)
    assert families.mtk_head_bytes(hf_config("google/gemma-3-1b-it")) == 262144 * 1152 * 4  # 1.2 GB, built


def gemma_report(model, window, **runner):
    report = mtk_report()
    name = model.rsplit("/", 1)[-1]
    report["source"] = {"id": model, "sha": "0" * 12, "license": {"license": "gemma"}}
    report["output_repo"] = f"experimentalmachines/{name}-ExecuTorch"
    report["window"] = {"context": window, "kv_cache_bytes_per_token": None}
    report["runner"] = {**report["runner"], **runner}
    limits = families.mtk_limits(hf_config(model), window)
    if limits:
        report["limits"] = limits
    return report


def test_a_mediatek_only_family_ahead_of_the_app_is_not_promised_to_any_runtime():
    text = manifest.readme(
        "experimentalmachines/gemma-2-2b-it-ExecuTorch", [gemma_report("google/gemma-2-2b-it", 2048)], [], []
    )
    assert "no chat template for this model family yet" in text
    assert "Any ExecuTorch runtime" not in text  # the MediaTek files need MediaTek's runner
    assert "or any ExecuTorch" not in text  # nor in the opening sentence


def test_an_npu_only_card_promises_no_generic_runtime_and_a_portable_one_does():
    gemma3 = [gemma_report("google/gemma-3-1b-it", 512)]
    opening = manifest.readme("experimentalmachines/gemma-3-1b-it-ExecuTorch", gemma3, [], [])
    assert "for phone NPUs only" in opening and "or any ExecuTorch" not in opening
    assert "does not read `config.json`" in opening  # MediaTek's runner takes flags
    # With CPU or GPU files beside the NPU ones, only those are promised to any runtime.
    mixed = [*gemma3, {**gemma3[0], "backend": "xnnpack"}]
    text = manifest.readme("experimentalmachines/gemma-3-1b-it-ExecuTorch", mixed, [], [])
    assert "The CPU (XNNPACK) and GPU (Vulkan) files also run on any ExecuTorch" in text
    assert "the NPU files need the runner their section below names" in text
    assert "Android app or any ExecuTorch" not in text


def test_gemma2_past_its_sliding_window_says_what_it_computes_differently():
    assert families.mtk_limits(hf_config("google/gemma-2-2b-it"), 4096) == []
    reports = [gemma_report("google/gemma-2-2b-it", 4096), gemma_report("google/gemma-2-2b-it", 8192)]
    text = manifest.readme("experimentalmachines/gemma-2-2b-it-ExecuTorch", reports, [], [])
    assert text.count("over the whole 8,192-token window in MediaTek's graph") == 1
    assert "prompts up to 4,096 tokens are computed as Hugging Face computes them" in text


def test_the_gemma3_card_says_its_runner_must_feed_both_rope_tables():
    runner = export_mtk.runner_settings(hf_config("google/gemma-3-1b-it"), CFG.mtk, 2, [1, 106])
    report = gemma_report(
        "google/gemma-3-1b-it",
        2048,
        **{k: runner[k] for k in ("rot_emb_base", "rot_emb_local_base", "window_size", "rot_emb_inputs")},
    )
    text = manifest.readme("experimentalmachines/gemma-3-1b-it-ExecuTorch", [report], [], [])
    assert "`rot_emb_inputs: [global, local]`" in text
    assert "rot_emb_local_base` (10000)" in text and "`window_size` (512)" in text
    assert "cannot run them correctly" in text
    plain = manifest.readme("experimentalmachines/Qwen3-0.6B-ExecuTorch", [mtk_report()], [], [])
    assert "rot_emb_inputs" not in plain


def test_each_run_builds_in_its_own_copy_of_the_scripts(tmp_path):
    # Two windows of one model on one host shared pte/<exp>, which has no window in its name,
    # and the first run to finish deleted the other's chunk (2026-11-09).
    examples = tmp_path / "examples" / "mediatek"
    (examples / "model_export_scripts").mkdir(parents=True)
    (examples / "model_export_scripts" / "gemma.py").write_text("# script\n")
    (examples / "pte" / "gemma-3-1b-it_A16W8_2_chunks").mkdir(parents=True)
    (examples / "pte" / "gemma-3-1b-it_A16W8_2_chunks" / "stale_0.pte").write_bytes(b"x")
    (examples / "model_export_scripts" / "__pycache__").mkdir()
    runs = [export_mtk.private_examples(examples, tmp_path / f"work-{w}") for w in (8192, 16384)]
    assert runs[0] != runs[1] and all(r.is_absolute() for r in runs)
    for run in runs:
        assert (run / "model_export_scripts" / "gemma.py").read_text() == "# script\n"
        assert not (run / "pte").exists() and not (run / "model_export_scripts" / "__pycache__").exists()
    (runs[0] / "leftover").write_text("from a crashed run")
    again = export_mtk.private_examples(examples, tmp_path / "work-8192")
    assert again == runs[0] and not (again / "leftover").exists()
    assert (examples / "pte" / "gemma-3-1b-it_A16W8_2_chunks" / "stale_0.pte").exists()  # never touched


def test_gemma3_is_refused_on_scripts_without_the_two_table_patch(tmp_path, monkeypatch, source):
    examples = tmp_path / "mediatek"
    (examples / "model_export_scripts").mkdir(parents=True)
    (examples / "models" / "llm_models").mkdir(parents=True)
    calibrated = f"{export_mtk.PATCH_MARKER}\n{export_mtk.WRITER_MARKER}\n"
    (examples / "model_export_scripts" / "gemma.py").write_text(calibrated)
    (examples / "models" / "llm_models" / "modeling_gemma.py").write_text("def forward(self, *args): ...\n")
    monkeypatch.setattr(export_mtk.hub, "fetch", lambda model_id, revision: source(model_id))
    with pytest.raises(export_mtk.ExportError, match="modeling_gemma.py lacks the patch.*mediatek-gemma.patch"):
        export_mtk.run("google/gemma-3-1b-it", "main", "MT6991", tmp_path, tmp_path, "p", examples)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"layer_types": ["sliding_attention"] * 26, "sliding_window_pattern": None}, "no full_attention"),
        ({"layer_types": ["chunked_attention"] * 26, "sliding_window_pattern": None}, "no MediaTek layer"),
        ({"sliding_window": None}, "no sliding_window"),
    ],
)
def test_a_gemma3_config_mediatek_cannot_translate_is_a_verdict_not_a_crash(change, reason, source):
    # Anything but UnsupportedModel would end the watcher's whole pass (codex review of #30).
    config = {**hf_config("google/gemma-3-1b-it"), **change}
    with pytest.raises(families.UnsupportedModel, match=reason):
        families.mtk_model_config(config)
    verdict = eligibility.evaluate(source("google/gemma-3-1b-it", config), CFG)
    assert reason in verdict.backends["mtk"]


def test_the_script_gets_absolute_paths_though_it_runs_in_the_runs_own_copy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tool, config = export_mtk.script_paths("venv/bin/python", Path("work/weights/m"))
    assert tool == str(tmp_path / "venv/bin/python") and config == tmp_path / "work/weights/m/config.json"
    assert export_mtk.script_paths("python3", Path("w"))[0] == "python3"  # found on PATH, left alone
