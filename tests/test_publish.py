"""Publishing one window into a repo that already holds other windows and backends."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline import manifest, publish


def report(context, backend="xnnpack", target=None, sha="abcdef0123456789", run="https://gh/run/1"):
    label = f"{context // 1024}k"
    folder = backend if target is None else f"{backend}/{target}"
    return {
        "backend": backend,
        "target": target,
        "tokenizer": "tokenizer.json",
        "output_repo": "experimentalmachines/Qwen3-1.7B-ExecuTorch",
        "source": {"id": "Qwen/Qwen3-1.7B", "sha": sha, "license": {"license": "apache-2.0"}, "license_files": []},
        "toolchain": {"executorch": "1.4.0"},
        "recipe": {"label": "8da4w-g32, int8 embeddings", "description": "recipe text."},
        "window": {"context": context, "kv_cache_bytes_per_token": 229_376, "fits_phone_budget": context <= 8192},
        # The tests write b"pte" as the file; publish checks the bytes against this.
        "files": [
            {
                "path": f"{folder}/Qwen3-1.7B-8da4w-{label}.pte",
                "bytes": 1_000 * context,
                "sha256": hashlib.sha256(b"pte").hexdigest(),
            }
        ],
        "metadata": {"get_max_context_len": context, "get_max_seq_len": min(2048, context)},
        "smoke": {"passed": True, "answered": True},
        "run": {"url": run},
    }


def test_superseded_only_touches_this_window_and_the_pre_matrix_files():
    folder, label = "xnnpack", "2k"
    keep = [
        "xnnpack/Qwen3-1.7B-8da4w-32k.pte",
        "xnnpack/export-report-32k.json",
        "qnn/sm8650/Qwen3-1.7B-qnn-hybrid-2k.pte",
    ]
    replace = [
        "xnnpack/Qwen3-1.7B-8da4w-2k.pte",
        "xnnpack/export-report-2k.json",
        "xnnpack/config.json",
        "xnnpack/export-report.json",
    ]
    assert [p for p in keep if publish.superseded(p, folder, label)] == []
    assert [p for p in replace if publish.superseded(p, folder, label)] == replace
    # "2k" is not a substring match on "32k", and a chip folder is not its backend's folder.
    assert not publish.superseded("xnnpack/Qwen3-1.7B-8da4w-32k.pte", "xnnpack", "2k")
    assert not publish.superseded("qnn/sm8650/config.json", "qnn", "2k")
    assert publish.superseded("mtk/mt6991/Qwen3-1.7B-neuropilot-a16w4-2k-chunk1of4.pte", "mtk/mt6991", "2k")
    assert not publish.superseded("mtk/mt6991/Qwen3-1.7B-neuropilot-embedding-fp32.bin", "mtk/mt6991", "2k")
    # A model whose own name carries a window-like token (a real HuggingFaceTB repo).
    name = "xnnpack/SmolLM2-1.7B-Instruct-16k-8da4w-4k.pte"
    assert publish.window_of(name.split("/")[1]) == "4k"
    assert not publish.superseded(name, "xnnpack", "16k")
    assert publish.superseded(name, "xnnpack", "4k")
    assert publish.window_of("export-report-16k.json") == "16k"
    assert publish.window_of("config.json") is None


def test_release_tags_are_per_window_and_chip():
    assert publish.release_tag(report(4096)) == "Qwen3-1.7B-xnnpack-4k-abcdef0"
    assert publish.release_tag(report(2048, backend="qnn", target="sm8650")) == "Qwen3-1.7B-qnn-sm8650-2k-abcdef0"


def test_publish_hf_retries_once_on_a_stale_parent_commit(tmp_path, monkeypatch):
    from huggingface_hub.errors import HfHubHTTPError

    out = tmp_path / "out"
    (out / "xnnpack").mkdir(parents=True)
    (out / "tokenizer.json").write_text("{}")
    (out / "xnnpack" / "Qwen3-1.7B-8da4w-2k.pte").write_bytes(b"pte")
    (out / "xnnpack" / "export-report-2k.json").write_text(json.dumps(report(2048)))
    hub = FakeHub({"README.md": None})
    calls = []

    def create_commit(repo_id, operations, commit_message, parent_commit):
        calls.append(parent_commit)
        if len(calls) == 1:  # another window's job committed in between
            import httpx

            response = httpx.Response(412, request=httpx.Request("POST", "https://huggingface.co/api"))
            raise HfHubHTTPError("412 Precondition Failed", response=response)
        return SimpleNamespace(commit_url="https://hf/commit/2")

    hub.create_commit = create_commit
    monkeypatch.setattr(publish.hub, "api", lambda: hub)
    monkeypatch.setattr(publish.time, "sleep", lambda s: None)
    assert publish.publish_hf(out, "xnnpack") == "https://hf/commit/2"
    assert calls == ["rev1", "rev1"]  # re-read the repo and rebuilt the commit before retrying


def test_a_concurrent_commit_is_retried_too(tmp_path, monkeypatch):
    """409, not just 412. Five windows publish at once and the Hub answers either way.

    Run 35373841386 published four of five windows and lost 8192 to a 409 that was not
    retried, so the repository carried four of the five files it should have.
    """
    from huggingface_hub.errors import HfHubHTTPError

    out = tmp_path / "out"
    (out / "xnnpack").mkdir(parents=True)
    (out / "tokenizer.json").write_text("{}")
    (out / "xnnpack" / "Qwen3-1.7B-8da4w-2k.pte").write_bytes(b"pte")
    (out / "xnnpack" / "export-report-2k.json").write_text(json.dumps(report(2048)))
    hub = FakeHub({"README.md": None})
    calls = []

    def create_commit(repo_id, operations, commit_message, parent_commit):
        calls.append(parent_commit)
        if len(calls) == 1:
            import httpx

            response = httpx.Response(409, request=httpx.Request("POST", "https://huggingface.co/api"))
            raise HfHubHTTPError("409 Conflict", response=response)
        return SimpleNamespace(commit_url="https://hf/commit/2")

    hub.create_commit = create_commit
    monkeypatch.setattr(publish.hub, "api", lambda: hub)
    monkeypatch.setattr(publish.time, "sleep", lambda s: None)
    assert publish.publish_hf(out, "xnnpack") == "https://hf/commit/2"
    assert calls == ["rev1", "rev1"]


def test_a_status_that_is_not_a_race_still_raises(tmp_path, monkeypatch):
    # Retrying a 403 or a 404 would turn a real failure into six slow ones and then the
    # same failure, with the cause five minutes further from the top of the log.
    from huggingface_hub.errors import HfHubHTTPError

    out = tmp_path / "out"
    (out / "xnnpack").mkdir(parents=True)
    (out / "tokenizer.json").write_text("{}")
    (out / "xnnpack" / "Qwen3-1.7B-8da4w-2k.pte").write_bytes(b"pte")
    (out / "xnnpack" / "export-report-2k.json").write_text(json.dumps(report(2048)))
    hub = FakeHub({"README.md": None})
    calls = []

    def create_commit(repo_id, operations, commit_message, parent_commit):
        import httpx

        calls.append(parent_commit)
        response = httpx.Response(403, request=httpx.Request("POST", "https://huggingface.co/api"))
        raise HfHubHTTPError("403 Forbidden", response=response)

    hub.create_commit = create_commit
    monkeypatch.setattr(publish.hub, "api", lambda: hub)
    monkeypatch.setattr(publish.time, "sleep", lambda s: None)
    with pytest.raises(HfHubHTTPError):
        publish.publish_hf(out, "xnnpack")
    assert len(calls) == 1


def test_backend_config_merges_every_window_in_the_folder():
    config = manifest.backend_config([report(8192), report(2048), report(32768)])
    assert [v["context"] for v in config["variants"]] == [2048, 8192, 32768]
    assert [v["fits_phone_budget"] for v in config["variants"]] == [True, True, False]
    assert config["variants"][0]["file"] == "Qwen3-1.7B-8da4w-2k.pte"
    assert config["variants"][0]["methods"]["get_max_context_len"] == 2048


class FakeHub:
    """Enough of HfApi for publish_hf: a repo with files, and a create_commit that records."""

    def __init__(self, files: dict[str, dict]):
        self.files = files  # path -> report json for report files
        self.token = None
        self.commits = []

    def create_repo(self, repo_id, repo_type, exist_ok):
        pass

    def model_info(self, repo_id, expand):
        return SimpleNamespace(sha="rev1", siblings=[SimpleNamespace(rfilename=p) for p in self.files])

    def create_commit(self, repo_id, operations, commit_message, parent_commit):
        self.commits.append((operations, commit_message, parent_commit))
        return SimpleNamespace(commit_url="https://hf/commit/1")


def test_publish_hf_keeps_the_other_windows_and_regenerates_config_and_readme(tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "xnnpack").mkdir(parents=True)
    (out / "tokenizer.json").write_text("{}")
    new = report(4096)
    (out / "xnnpack" / "Qwen3-1.7B-8da4w-4k.pte").write_bytes(b"pte")
    (out / "xnnpack" / "export-report-4k.json").write_text(json.dumps(new))
    (out / "xnnpack" / "config.json").write_text(json.dumps(manifest.backend_config(new)))

    remote = {
        "README.md": None,
        "tokenizer.json": None,
        "xnnpack/Qwen3-1.7B-8da4w-16k.pte": None,
        "xnnpack/export-report-16k.json": report(16384, run="https://gh/run/0"),
        "xnnpack/config.json": None,
        # A pre-matrix export: unlabelled report, its window's file, an older 4k that is replaced.
        "xnnpack/export-report.json": report(8192),
        "xnnpack/Qwen3-1.7B-8da4w-8k.pte": None,
        "xnnpack/Qwen3-1.7B-8da4w-4k.pte": None,
        "qnn/sm8650/Qwen3-1.7B-qnn-hybrid-2k.pte": None,
        "qnn/sm8650/export-report-2k.json": report(2048, backend="qnn", target="sm8650"),
        "qnn/sm8650/config.json": None,
    }
    hub = FakeHub(remote)
    monkeypatch.setattr(publish.hub, "api", lambda: hub)

    def fake_download(repo_id, path, revision, token):
        local = tmp_path / "dl" / path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(remote[path]))
        return str(local)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    assert publish.publish_hf(out, "xnnpack") == "https://hf/commit/1"
    [(operations, message, parent)] = hub.commits
    assert parent == "rev1" and message.startswith("xnnpack 4k: Qwen/Qwen3-1.7B@")
    deleted = sorted(op.path_in_repo for op in operations if type(op).__name__ == "CommitOperationDelete")
    added = {op.path_in_repo: op for op in operations if type(op).__name__ == "CommitOperationAdd"}
    # Only the pre-matrix report goes, moved to its labelled name; the 8k and 16k windows and
    # the QNN folder stay.
    assert deleted == ["xnnpack/export-report.json"]
    assert json.loads(added["xnnpack/export-report-8k.json"].path_or_fileobj)["window"]["context"] == 8192
    assert "xnnpack/Qwen3-1.7B-8da4w-4k.pte" in added and "tokenizer.json" in added
    config = json.loads(added["xnnpack/config.json"].path_or_fileobj)
    assert [v["context"] for v in config["variants"]] == [4096, 8192, 16384]
    readme = added["README.md"].path_or_fileobj.decode()
    for name in (
        "xnnpack/Qwen3-1.7B-8da4w-4k.pte",
        "8da4w-8k.pte",
        "8da4w-16k.pte",
        "qnn/sm8650/Qwen3-1.7B-8da4w-2k.pte",
    ):
        assert name in readme, name
    assert "[run 1](https://gh/run/0)" in readme and "[run 2](https://gh/run/1)" in readme


def test_load_report_wants_exactly_one_report(tmp_path):
    folder = tmp_path / "xnnpack"
    folder.mkdir()
    with pytest.raises(ValueError, match="expected one export report"):
        publish.load_report(tmp_path, "xnnpack")
    (folder / "export-report-2k.json").write_text(json.dumps(report(2048)))
    assert publish.load_report(tmp_path, "xnnpack")["window"]["context"] == 2048
    # Several windows exported into one folder by hand: --context picks.
    (folder / "export-report-8k.json").write_text(json.dumps(report(8192)))
    with pytest.raises(ValueError, match="pass --context"):
        publish.load_report(tmp_path, "xnnpack")
    assert publish.load_report(tmp_path, "xnnpack", context=8192)["window"]["context"] == 8192


def test_a_report_that_failed_its_decision_gate_is_never_published(tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "xnnpack").mkdir(parents=True)
    (out / "tokenizer.json").write_text("{}")
    (out / "xnnpack" / "Qwen3-1.7B-8da4w-2k.pte").write_bytes(b"pte")
    failed = {**report(2048), "gate": {"passed": False, "problems": ["kept fp32's choice on 7 of 8 calling rows"]}}
    (out / "xnnpack" / "export-report-2k.json").write_text(json.dumps(failed))
    monkeypatch.setattr(publish.hub, "api", lambda: pytest.fail("reached the Hub"))
    with pytest.raises(ValueError, match="failed its decision gate: kept fp32's choice on 7 of 8"):
        publish.publish_hf(out, "xnnpack")


def test_a_gptq_file_needs_its_own_gate_or_for_vulkan_a_passing_proxy():
    gptq = {"int4_codes": "gptq", "label": "x", "description": "x"}
    cpu = {**report(2048), "recipe": gptq}
    with pytest.raises(ValueError, match="no decision gate"):
        publish.require_publishable(cpu)
    publish.require_publishable({**cpu, "gate": {"passed": True}})
    gpu = {**report(2048, backend="vulkan"), "recipe": gptq}
    with pytest.raises(ValueError, match="no passing gate proxy"):
        publish.require_publishable(gpu)
    with pytest.raises(ValueError, match="no passing gate proxy"):
        publish.require_publishable({**gpu, "gate_proxy": {"gate": {"passed": False}}})
    publish.require_publishable({**gpu, "gate_proxy": {"gate": {"passed": True}}})
    # Round to nearest has no fp32 reference to be gated against, as before.
    publish.require_publishable({**report(2048, backend="vulkan"), "recipe": {"int4_codes": "round-to-nearest"}})


def test_a_failed_gate_publishes_only_with_an_operators_recorded_reason():
    gptq = {"int4_codes": "gptq", "label": "x", "description": "x"}
    failed = {"passed": False, "agreed": 7, "graded": 8, "problems": ["7 of 8"]}
    cpu = {**report(2048), "recipe": gptq, "gate": failed}
    with pytest.raises(ValueError, match="failed its decision gate"):
        publish.require_publishable(cpu)
    with pytest.raises(ValueError, match="failed its decision gate"):
        publish.require_publishable({**cpu, "gate": {**failed, "override": {"reason": ""}}})
    publish.require_publishable({**cpu, "gate": {**failed, "override": {"reason": "fp32 is near a tie"}}})
    gpu = {**report(2048, backend="vulkan"), "recipe": gptq}
    publish.require_publishable({**gpu, "gate_proxy": {"gate": {**failed, "override": {"reason": "near a tie"}}}})


def test_a_file_that_is_not_the_one_its_report_describes_is_never_published(tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "xnnpack").mkdir(parents=True)
    (out / "tokenizer.json").write_text("{}")
    (out / "xnnpack" / "Qwen3-1.7B-8da4w-2k.pte").write_bytes(b"rebuilt after verify")
    (out / "xnnpack" / "export-report-2k.json").write_text(json.dumps(report(2048)))
    monkeypatch.setattr(publish.hub, "api", lambda: pytest.fail("reached the Hub"))
    with pytest.raises(ValueError, match="not the file its report describes"):
        publish.publish_hf(out, "xnnpack")


def test_a_model_file_the_report_does_not_list_is_never_uploaded_from_the_root(tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "xnnpack").mkdir(parents=True)
    (out / "tokenizer.json").write_text("{}")
    (out / "stray.pte").write_bytes(b"not verified")
    (out / "xnnpack" / "Qwen3-1.7B-8da4w-2k.pte").write_bytes(b"pte")
    (out / "xnnpack" / "export-report-2k.json").write_text(json.dumps(report(2048)))
    monkeypatch.setattr(publish.hub, "api", lambda: pytest.fail("reached the Hub"))
    with pytest.raises(ValueError, match="stray.pte.*not in the report"):
        publish.publish_hf(out, "xnnpack")


def test_a_release_never_attaches_a_model_file_its_report_does_not_list(tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "xnnpack").mkdir(parents=True)
    (out / "tokenizer.json").write_text("{}")
    (out / "xnnpack" / "Qwen3-1.7B-8da4w-2k.pte").write_bytes(b"pte")
    (out / "xnnpack" / "renamed.pte").write_bytes(b"not verified")
    (out / "xnnpack" / "export-report-2k.json").write_text(json.dumps(report(2048)))
    monkeypatch.setattr(publish.subprocess, "run", lambda *a, **k: pytest.fail("reached gh"))
    with pytest.raises(ValueError, match="renamed.pte.*not in the report"):
        publish.publish_release(out, "xnnpack")


def publish_over(tmp_path, monkeypatch, remote, new_sha=None):
    """Publish a new xnnpack 4k window over ``remote``; return (deleted, added, message)."""
    out = tmp_path / "out"
    (out / "xnnpack").mkdir(parents=True)
    (out / "tokenizer.json").write_text("{}")
    new = report(4096)
    if new_sha:
        new["files"][0]["sha256"] = new_sha
    (out / "xnnpack" / "Qwen3-1.7B-8da4w-4k.pte").write_bytes(b"pte")
    (out / "xnnpack" / "export-report-4k.json").write_text(json.dumps(new))
    (out / "xnnpack" / "config.json").write_text(json.dumps(manifest.backend_config(new)))
    hub = FakeHub(remote)
    monkeypatch.setattr(publish.hub, "api", lambda: hub)

    def fake_download(repo_id, path, revision, token):
        local = tmp_path / "dl" / path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(remote[path]))
        return str(local)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(publish, "require_same_bytes", lambda out_dir, report: None)
    publish.publish_hf(out, "xnnpack")
    [(operations, message, _)] = hub.commits
    deleted = sorted(op.path_in_repo for op in operations if type(op).__name__ == "CommitOperationDelete")
    added = {op.path_in_repo: op for op in operations if type(op).__name__ == "CommitOperationAdd"}
    return deleted, added, message


def gpu(context, twin_sha):
    r = report(context, backend="vulkan")
    r["gate_proxy"] = {"file": f"xnnpack/Qwen3-1.7B-8da4w-{context // 1024}k.pte", "sha256": twin_sha}
    return r


def remote_with_gpu(proxy_4k):
    old = report(4096)
    old["files"][0]["sha256"] = "old-xnnpack-4k"
    return {
        "README.md": None,
        "xnnpack/Qwen3-1.7B-8da4w-4k.pte": None,
        "xnnpack/export-report-4k.json": old,
        "xnnpack/config.json": None,
        "vulkan/Qwen3-1.7B-8da4w-4k.pte": None,
        "vulkan/export-report-4k.json": gpu(4096, proxy_4k),
        "vulkan/Qwen3-1.7B-8da4w-8k.pte": None,
        "vulkan/export-report-8k.json": gpu(8192, "xnnpack-8k"),
        "vulkan/config.json": None,
    }


def test_replacing_an_xnnpack_window_withdraws_the_vulkan_window_gated_through_it(tmp_path, monkeypatch):
    deleted, added, message = publish_over(tmp_path, monkeypatch, remote_with_gpu("old-xnnpack-4k"))
    assert deleted == ["vulkan/Qwen3-1.7B-8da4w-4k.pte", "vulkan/export-report-4k.json"]
    assert "withdraws vulkan 4k" in message
    # The other Vulkan window stays, and the folder's config and the README forget the 4k one.
    config = json.loads(added["vulkan/config.json"].path_or_fileobj)
    assert [v["context"] for v in config["variants"]] == [8192]
    readme = added["README.md"].path_or_fileobj.decode()
    assert "vulkan/Qwen3-1.7B-8da4w-8k.pte" in readme and "vulkan/Qwen3-1.7B-8da4w-4k.pte" not in readme


@pytest.mark.parametrize("case", ["unpublished-twin", "same-file"])
def test_a_vulkan_window_whose_twin_survives_the_commit_stays(tmp_path, monkeypatch, case):
    if case == "unpublished-twin":
        # SmolLM2-360M: gated through a GPTQ build that was never on the Hub (its CPU files are fp32).
        remote = remote_with_gpu("never-published")
        remote["vulkan/export-report-4k.json"]["gate_proxy"]["file"] = "xnnpack/Qwen3-1.7B-8da4w-gptq-4k.pte"
        new_sha = None
    else:
        # The very file being republished: same path, same bytes.
        remote, new_sha = remote_with_gpu("same-bytes"), "same-bytes"
    deleted, added, message = publish_over(tmp_path, monkeypatch, remote, new_sha=new_sha)
    assert not any(p.startswith("vulkan/") for p in deleted) and "withdraws" not in message
    assert "vulkan/config.json" not in added


def publish_gpu(tmp_path, monkeypatch, remote, **kwargs):
    """Publish a GPTQ Vulkan 4k window gated through xnnpack 4k sha 'twin-4k' over ``remote``."""
    out = tmp_path / "gpu-out"
    (out / "vulkan").mkdir(parents=True, exist_ok=True)
    new = gpu(4096, "twin-4k")
    (out / "vulkan" / "Qwen3-1.7B-8da4w-4k.pte").write_bytes(b"pte")
    (out / "vulkan" / "export-report-4k.json").write_text(json.dumps(new))
    (out / "vulkan" / "config.json").write_text(json.dumps(manifest.backend_config(new)))
    hub = FakeHub(remote)
    monkeypatch.setattr(publish.hub, "api", lambda: hub)

    def fake_download(repo_id, path, revision, token):
        local = tmp_path / "dl" / path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(remote[path]))
        return str(local)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return publish.publish_hf(out, "vulkan", **kwargs), hub


def remote_twin(sha):
    twin = report(4096)
    twin["files"][0]["sha256"] = sha
    return {"xnnpack/Qwen3-1.7B-8da4w-4k.pte": None, "xnnpack/export-report-4k.json": twin, "README.md": None}


def test_a_vulkan_file_publishes_only_beside_the_twin_its_proxy_names(tmp_path, monkeypatch):
    url, hub = publish_gpu(tmp_path, monkeypatch, remote_twin("twin-4k"))
    assert url == "https://hf/commit/1" and len(hub.commits) == 1


def test_a_vulkan_file_whose_twin_was_replaced_meanwhile_is_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="not the XNNPACK file of this window on the Hub"):
        publish_gpu(tmp_path, monkeypatch, remote_twin("someone-else-replaced-it"))
    with pytest.raises(ValueError, match="publish that twin first"):
        publish_gpu(tmp_path, monkeypatch, {"README.md": None})
    # A twin that is deliberately not published (SmolLM2-360M) is said out loud.
    url, _ = publish_gpu(tmp_path, monkeypatch, {"README.md": None}, allow_unpublished_twin=True)
    assert url == "https://hf/commit/1"


def test_a_pre_matrix_xnnpack_report_still_withdraws_the_vulkan_window_it_gated(tmp_path, monkeypatch):
    remote = remote_with_gpu("old-xnnpack-4k")
    old = remote.pop("xnnpack/export-report-4k.json")
    remote["xnnpack/export-report.json"] = old  # the unlabelled report of an older layout
    deleted, _, message = publish_over(tmp_path, monkeypatch, remote)
    assert "vulkan/Qwen3-1.7B-8da4w-4k.pte" in deleted and "withdraws vulkan 4k" in message


def test_a_replaced_twin_takes_its_vulkan_window_with_it_even_without_its_report(tmp_path, monkeypatch):
    # Judged on the tree the commit produces, not on old reports: the proxy's file is replaced.
    remote = remote_with_gpu("old-xnnpack-4k")
    del remote["xnnpack/export-report-4k.json"]
    deleted, _, message = publish_over(tmp_path, monkeypatch, remote)
    assert "vulkan/Qwen3-1.7B-8da4w-4k.pte" in deleted and "withdraws vulkan 4k" in message


def test_the_same_bytes_under_another_name_still_withdraw_the_vulkan_window(tmp_path, monkeypatch):
    # The proxy names a path this commit deletes; matching the sha alone would have kept it.
    remote = remote_with_gpu(hashlib.sha256(b"pte").hexdigest())
    remote["vulkan/export-report-4k.json"]["gate_proxy"]["file"] = "xnnpack/Qwen3-1.7B-8da4w-old-name-4k.pte"
    remote["xnnpack/Qwen3-1.7B-8da4w-old-name-4k.pte"] = None
    deleted, _, message = publish_over(tmp_path, monkeypatch, remote)
    assert "xnnpack/Qwen3-1.7B-8da4w-old-name-4k.pte" in deleted and "vulkan/export-report-4k.json" in deleted


class FakeGh:
    """`gh release view/download/delete` for one Vulkan release holding one report."""

    def __init__(self, gpu_report=None):
        self.gpu_report = gpu_report
        self.calls = []

    def run(self, command, **kwargs):
        self.calls.append(command[:3])
        if command[:3] == ["gh", "release", "view"]:
            if self.gpu_report == "api-error":
                return SimpleNamespace(returncode=1, stdout="", stderr="HTTP 502: Bad Gateway")
            if self.gpu_report is None:
                return SimpleNamespace(returncode=1, stdout="", stderr="release not found")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[:3] == ["gh", "release", "download"]:
            directory = command[command.index("--dir") + 1]
            (Path(directory) / "export-report-4k.json").write_text(json.dumps(self.gpu_report))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[:3] == ["gh", "release", "delete"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(command)


@pytest.mark.parametrize(
    ("gpu_report", "withdrawn"),
    [
        (None, False),  # no GPU release for this window
        (gpu(4096, hashlib.sha256(b"pte").hexdigest()), False),  # gated through these very bytes
        (gpu(4096, "older-bytes"), True),  # gated through this file before it was re-released
        ({**gpu(4096, "older-bytes"), "gate_proxy": {"file": "xnnpack/other.pte", "sha256": "x"}}, False),
    ],
)
def test_re_releasing_an_xnnpack_window_withdraws_only_the_gpu_release_it_gated(monkeypatch, gpu_report, withdrawn):
    gh = FakeGh(gpu_report)
    monkeypatch.setattr(publish.subprocess, "run", gh.run)
    tag = publish.withdraw_orphaned_gpu_release(report(4096))
    assert (tag == "Qwen3-1.7B-vulkan-4k-abcdef0") is withdrawn
    assert (["gh", "release", "delete"] in gh.calls) is withdrawn


def test_a_gpu_release_needs_its_twin_on_the_hub(tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "vulkan").mkdir(parents=True)
    (out / "tokenizer.json").write_text("{}")
    (out / "vulkan" / "Qwen3-1.7B-8da4w-4k.pte").write_bytes(b"pte")
    (out / "vulkan" / "export-report-4k.json").write_text(json.dumps(gpu(4096, "twin-4k")))
    checked = []

    def missing_twin(report):
        checked.append(report["files"][0]["path"])
        raise ValueError("not the XNNPACK file of this window on the Hub")

    monkeypatch.setattr(publish, "require_twin_on_hub_now", missing_twin)
    with pytest.raises(ValueError, match="not the XNNPACK file"):
        publish.publish_release(out, "vulkan")
    assert checked == ["vulkan/Qwen3-1.7B-8da4w-4k.pte"]
    # Said out loud, it goes past the check (to gh, which this test stops at).
    monkeypatch.setattr(publish.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""))
    assert publish.publish_release(out, "vulkan", allow_unpublished_twin=True) == "Qwen3-1.7B-vulkan-4k-abcdef0"


def test_an_unreadable_gpu_release_is_an_error_not_an_absent_one(monkeypatch):
    monkeypatch.setattr(publish.subprocess, "run", FakeGh("api-error").run)
    with pytest.raises(RuntimeError, match="gh release view"):
        publish.withdraw_orphaned_gpu_release(report(4096))


def mtk_window(context, embedding_sha):
    r = report(context, backend="mtk", target="mt6991")
    label = f"{context // 1024}k"
    r["files"] = [
        {
            "path": f"mtk/mt6991/Qwen3-1.7B-neuropilot-a16w4-{label}-chunk1of1.pte",
            "bytes": 3,
            "sha256": r["files"][0]["sha256"],
        },
        {"path": "mtk/mt6991/Qwen3-1.7B-neuropilot-embedding-fp32.bin", "bytes": 3, "sha256": embedding_sha},
    ]
    return r


def test_a_shared_file_must_match_what_the_other_windows_reports_describe():
    same, other = hashlib.sha256(b"emb").hexdigest(), hashlib.sha256(b"newer weights").hexdigest()
    publish.require_shared_files_agree(mtk_window(16384, same), [mtk_window(8192, same)])  # a rerun from one revision
    with pytest.raises(ValueError, match=r"embedding-fp32.bin \(the 8k window's report has"):
        publish.require_shared_files_agree(mtk_window(16384, other), [mtk_window(8192, same)])


def test_publish_hf_refuses_to_swap_the_embedding_other_mediatek_windows_run_on(tmp_path, monkeypatch):
    # Publishing a 16k export built from a newer revision would have replaced the folder's one
    # embedding table under the published 8k chunks (codex review of #30).
    out = tmp_path / "out"
    (out / "mtk" / "mt6991").mkdir(parents=True)
    (out / "tokenizer.json").write_text("{}")
    new = mtk_window(16384, hashlib.sha256(b"newer weights").hexdigest())
    (out / "mtk/mt6991/Qwen3-1.7B-neuropilot-a16w4-16k-chunk1of1.pte").write_bytes(b"pte")
    (out / "mtk/mt6991/Qwen3-1.7B-neuropilot-embedding-fp32.bin").write_bytes(b"newer weights")
    (out / "mtk/mt6991/export-report-16k.json").write_text(json.dumps(new))
    remote = {
        "README.md": None,
        "mtk/mt6991/Qwen3-1.7B-neuropilot-a16w4-8k-chunk1of1.pte": None,
        "mtk/mt6991/Qwen3-1.7B-neuropilot-embedding-fp32.bin": None,
        "mtk/mt6991/export-report-8k.json": mtk_window(8192, hashlib.sha256(b"emb").hexdigest()),
    }
    hub = FakeHub(remote)
    monkeypatch.setattr(publish.hub, "api", lambda: hub)

    def fake_download(repo_id, path, revision, token):
        local = tmp_path / "dl" / path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(remote[path]))
        return str(local)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    with pytest.raises(ValueError, match="other published windows depend on"):
        publish.publish_hf(out, "mtk", "mt6991")
    assert hub.commits == []
