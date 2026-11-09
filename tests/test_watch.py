import json
import subprocess

import pytest
from conftest import hf_config, make_source

from pipeline import settings, watch

CFG = settings.load()
QWEN3 = hf_config("Qwen/Qwen3-1.7B")


class FakeHub:
    """Listings per org, and the config each model id resolves to."""

    def __init__(self, listings, configs=None, failing=()):
        self.listings = listings
        self.configs = configs or {}
        self.failing = set(failing)
        self.fetched = []
        self.dispatched = []

    def lister(self, org, limit):
        return [watch.Listed(m, f"sha-{m}", "2026-09-01T00:00:00+00:00") for m in self.listings.get(org, [])][:limit]

    def fetch(self, model_id, revision):
        self.fetched.append(model_id)
        if model_id in self.failing:
            raise RuntimeError("503 from the Hub")
        return make_source(model_id, config=self.configs.get(model_id, QWEN3), sha=f"sha-{model_id}")

    def dispatch(self, workflow, model_id, revision):
        self.dispatched.append((workflow, model_id, revision))


def run(tmp_path, hub, **kwargs):
    summary = watch.run(
        tmp_path / "seen.json",
        CFG,
        lister=hub.lister,
        fetch=hub.fetch,
        dispatcher=kwargs.pop("dispatcher", hub.dispatch),
        in_flight=kwargs.pop("in_flight", lambda workflow: False),
        **kwargs,
    )
    if kwargs.get("dispatch", True):
        watch.save_state(tmp_path / "seen.json", summary["state"])
    return summary


def test_first_run_seeds_and_exports_nothing(tmp_path):
    hub = FakeHub({"Qwen": ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"], "google": ["google/gemma-3-1b-it"]})
    summary = run(tmp_path, hub)
    assert summary["seeded"] == 3
    assert hub.fetched == [] and hub.dispatched == []
    state = json.loads((tmp_path / "seen.json").read_text())
    assert {e["status"] for e in state["models"].values()} == {"seeded"}


def test_new_models_are_checked_once_and_dispatched(tmp_path):
    hub = FakeHub({"Qwen": ["Qwen/Qwen3-0.6B"]})
    run(tmp_path, hub)  # seed
    hub.listings["Qwen"] = [
        "Qwen/Qwen3-1.7B-Instruct-2609",
        "Qwen/Qwen3-1.7B-FP8",
        "Qwen/Qwen3.8-1B",
        "Qwen/Qwen3-0.6B",
    ]
    summary = run(tmp_path, hub)
    models = summary["state"]["models"]
    # Excluded by name: never fetched.
    assert "Qwen/Qwen3-1.7B-FP8" not in hub.fetched
    assert models["Qwen/Qwen3-1.7B-FP8"]["status"] == "skipped"
    # Same architecture class, later generation: refused by the family name pattern.
    assert models["Qwen/Qwen3.8-1B"]["status"] == "skipped"
    assert "not named like a qwen3 release" in models["Qwen/Qwen3.8-1B"]["reasons"][0]
    # The eligible one goes to XNNPACK at the revision the watcher saw.
    assert hub.dispatched == [
        ("export-xnnpack.yml", "Qwen/Qwen3-1.7B-Instruct-2609", "sha-Qwen/Qwen3-1.7B-Instruct-2609")
    ]
    entry = models["Qwen/Qwen3-1.7B-Instruct-2609"]
    assert entry["status"] == "pending"  # its MediaTek stage is still to come
    assert entry["backends"]["xnnpack"]["status"] == "dispatched"
    # Vulkan is built in the XNNPACK run, from the same codes: it goes out with it.
    assert entry["backends"]["vulkan"]["status"] == "dispatched"
    assert entry["backends"]["vulkan"]["built_with"] == "xnnpack"
    assert entry["backends"]["qnn"]["status"] == "skipped"
    assert entry["backends"]["mtk"]["status"] == "pending"
    # Seeded models are not re-checked.
    assert "Qwen/Qwen3-0.6B" not in summary["new"]
    # The next pass finds nothing new; the idle XNNPACK stage hands over to MediaTek.
    hub.dispatched.clear()
    summary = run(tmp_path, hub)
    assert summary["new"] == [] and [d[0] for d in hub.dispatched] == ["export-mtk.yml"]
    assert summary["state"]["models"]["Qwen/Qwen3-1.7B-Instruct-2609"]["status"] == "dispatched"


def test_orgs_are_watched_for_their_own_families_only(tmp_path):
    # Seen in a real dry run: HuggingFaceTB's GSM8K fine-tune of Qwen3 passes every other rule.
    hub = FakeHub({"HuggingFaceTB": []})
    run(tmp_path, hub)
    hub.listings["HuggingFaceTB"] = ["HuggingFaceTB/qwen3-1.7b-gsm8k-sft"]
    entry = run(tmp_path, hub)["state"]["models"]["HuggingFaceTB/qwen3-1.7b-gsm8k-sft"]
    assert entry["status"] == "skipped"
    assert entry["reasons"] == ["not one of HuggingFaceTB's own families (smollm2, smollm3)"]
    assert hub.fetched == [] and hub.dispatched == []
    # Asking for it explicitly still works.
    run(tmp_path, hub, backfill=("HuggingFaceTB/qwen3-1.7b-gsm8k-sft",))
    assert [d[1] for d in hub.dispatched] == ["HuggingFaceTB/qwen3-1.7b-gsm8k-sft"]


def test_dispatch_budget_carries_over(tmp_path):
    hub = FakeHub({"Qwen": []})
    run(tmp_path, hub)
    hub.listings["Qwen"] = ["Qwen/Qwen3-0.6B-A", "Qwen/Qwen3-0.6B-B", "Qwen/Qwen3-0.6B-C"]
    run(tmp_path, hub, max_dispatch=2)
    assert len(hub.dispatched) == 2
    summary = run(tmp_path, hub, max_dispatch=2)
    assert len(hub.dispatched) == 3
    assert {e["backends"]["xnnpack"]["status"] for e in summary["state"]["models"].values()} == {"dispatched"}


def test_fetch_errors_are_retried_then_given_up(tmp_path):
    hub = FakeHub({"Qwen": []}, failing={"Qwen/Qwen3-0.6B-X"})
    run(tmp_path, hub)
    hub.listings["Qwen"] = ["Qwen/Qwen3-0.6B-X"]
    for attempt in range(1, watch.MAX_ATTEMPTS):
        entry = run(tmp_path, hub)["state"]["models"]["Qwen/Qwen3-0.6B-X"]
        assert (entry["status"], entry["attempts"]) == ("retry", attempt)
    entry = run(tmp_path, hub)["state"]["models"]["Qwen/Qwen3-0.6B-X"]
    assert entry["status"] == "skipped"
    assert len(hub.fetched) == watch.MAX_ATTEMPTS


def test_backfill_exports_a_seeded_model(tmp_path):
    hub = FakeHub({"Qwen": ["Qwen/Qwen3-1.7B"]})
    run(tmp_path, hub)
    summary = run(tmp_path, hub, backfill=("Qwen/Qwen3-1.7B",))
    assert summary["backfilled"] == ["Qwen/Qwen3-1.7B"]
    # Qwen3-1.7B is also in ExecuTorch's Qualcomm registry, but stages go one backend at a
    # time: XNNPACK (and Vulkan with it) now, QNN on the pass after that stage is idle.
    assert hub.dispatched == [("export-xnnpack.yml", "Qwen/Qwen3-1.7B", "sha-Qwen/Qwen3-1.7B")]
    assert summary["stage"] == "xnnpack"
    summary = run(tmp_path, hub)
    assert hub.dispatched[-1] == ("export-qnn.yml", "Qwen/Qwen3-1.7B", "sha-Qwen/Qwen3-1.7B")
    assert summary["stage"] == "qnn"


def test_backfill_on_the_first_run_still_seeds_everything_else(tmp_path):
    hub = FakeHub({"Qwen": ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B"]})
    summary = run(tmp_path, hub, backfill=("Qwen/Qwen3-1.7B",))
    assert summary["seeded"] == 2
    assert {d[1] for d in hub.dispatched} == {"Qwen/Qwen3-1.7B"}
    assert summary["state"]["models"]["Qwen/Qwen3-0.6B"]["status"] == "seeded"


def test_dry_run_dispatches_nothing(tmp_path):
    hub = FakeHub({"Qwen": []})
    run(tmp_path, hub)
    hub.listings["Qwen"] = ["Qwen/Qwen3-0.6B-A"]
    summary = run(tmp_path, hub, dispatch=False)
    assert hub.dispatched == []
    assert summary["dispatched"] == ["Qwen/Qwen3-0.6B-A -> xnnpack"]
    assert "Checked 1 model(s)" in watch.markdown(summary)


def test_a_failed_dispatch_stays_pending_and_the_pass_goes_on(tmp_path):
    hub = FakeHub({"Qwen": []})
    run(tmp_path, hub)
    hub.listings["Qwen"] = ["Qwen/Qwen3-0.6B-A", "Qwen/Qwen3-0.6B-B"]
    real = hub.dispatch

    def flaky(workflow, model_id, revision):
        if model_id.endswith("-A"):
            raise RuntimeError("HTTP 502")
        real(workflow, model_id, revision)

    summary = run(tmp_path, hub, dispatcher=flaky)
    assert [d[1] for d in hub.dispatched] == ["Qwen/Qwen3-0.6B-B"]
    assert summary["failed"] == ["Qwen/Qwen3-0.6B-A -> xnnpack: HTTP 502"]
    a = summary["state"]["models"]["Qwen/Qwen3-0.6B-A"]
    assert a["status"] == "pending" and a["backends"]["xnnpack"]["status"] == "pending"
    assert "HTTP 502" in a["backends"]["xnnpack"]["last_error"]
    assert "Could not dispatch" in watch.markdown(summary)
    # The next pass retries it and clears the error.
    hub.dispatched.clear()
    summary = run(tmp_path, hub)
    assert [d[1] for d in hub.dispatched] == ["Qwen/Qwen3-0.6B-A"]
    assert "last_error" not in summary["state"]["models"]["Qwen/Qwen3-0.6B-A"]["backends"]["xnnpack"]


def test_gh_dispatch_treats_a_run_in_flight_as_dispatched(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command[:3])
        if command[:3] == ["gh", "run", "list"]:
            runs = [{"displayTitle": "XNNPACK: Qwen/Qwen3-0.6B@abc", "status": "in_progress"}]
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(runs), stderr="")
        raise AssertionError("must not dispatch again")

    monkeypatch.setattr(watch.subprocess, "run", fake_run)
    watch.gh_dispatch("export-xnnpack.yml", "Qwen/Qwen3-0.6B", "abc")
    assert calls == [["gh", "run", "list"]]
    assert not watch.run_in_flight("export-xnnpack.yml", "Qwen/Qwen3-0.6B-Base", None)


def test_stages_wait_for_the_previous_backend_to_finish(tmp_path):
    hub = FakeHub({"Qwen": []})
    run(tmp_path, hub)
    hub.listings["Qwen"] = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]  # 0.6B/1.7B also QNN
    summary = run(tmp_path, hub)
    assert [d[0] for d in hub.dispatched] == ["export-xnnpack.yml"] * 3 and summary["stage"] == "xnnpack"
    # XNNPACK runs still in flight: nothing new, the QNN stage waits.
    hub.dispatched.clear()
    summary = run(tmp_path, hub, in_flight=lambda wf: wf == "export-xnnpack.yml")
    assert hub.dispatched == [] and summary["stage"] == "xnnpack"
    # XNNPACK idle (Vulkan was built in those runs): QNN for the two in the registry.
    summary = run(tmp_path, hub)
    assert sorted(d[1] for d in hub.dispatched) == ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]
    assert {d[0] for d in hub.dispatched} == {"export-qnn.yml"} and summary["stage"] == "qnn"
    # QNN idle: MediaTek, the last stage, for every model with a recipe (all three here).
    hub.dispatched.clear()
    summary = run(tmp_path, hub)
    assert {d[0] for d in hub.dispatched} == {"export-mtk.yml"} and len(hub.dispatched) == 3
    assert summary["stage"] == "mtk"
    # Everything dispatched: nothing left, no stage.
    hub.dispatched.clear()
    summary = run(tmp_path, hub)
    assert hub.dispatched == [] and "stage" not in summary
    assert {e["status"] for e in summary["state"]["models"].values()} == {"dispatched"}


def test_requeue_puts_cancelled_dispatches_back(tmp_path):
    hub = FakeHub({"Qwen": []})
    run(tmp_path, hub)
    hub.listings["Qwen"] = ["Qwen/Qwen3-4B"]
    run(tmp_path, hub)
    assert len(hub.dispatched) == 1
    summary = run(tmp_path, hub, requeue=True)  # the run was cancelled: dispatch it again
    assert len(hub.dispatched) == 2
    entry = summary["state"]["models"]["Qwen/Qwen3-4B"]
    assert entry["backends"]["xnnpack"]["status"] == "dispatched" and "requeued" in entry["backends"]["xnnpack"]


def test_runs_in_flight_ignores_ghost_runs(monkeypatch):
    runs = [
        {"displayTitle": "Export XNNPACK", "name": "Export XNNPACK", "status": "queued"},  # ghost
        {"displayTitle": "XNNPACK: Qwen/Qwen3-4B@abc", "name": "Export XNNPACK", "status": "completed"},
    ]

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(runs), stderr="")

    monkeypatch.setattr(watch.subprocess, "run", fake_run)
    assert not watch.runs_in_flight("export-xnnpack.yml")
    runs[1]["status"] = "in_progress"
    assert watch.runs_in_flight("export-xnnpack.yml")


def test_state_version_is_checked(tmp_path):
    (tmp_path / "seen.json").write_text(json.dumps({"version": 99, "models": {}}))
    with pytest.raises(ValueError, match="state version"):
        watch.load_state(tmp_path / "seen.json")


def test_vulkan_is_never_dispatched_on_its_own_and_comes_back_with_a_requeue(tmp_path):
    # Vulkan files are built in the XNNPACK run from the same solved codes; a workflow of its
    # own could only solve again or round to nearest.
    assert "vulkan" not in watch.WORKFLOWS and "vulkan" not in watch.STAGES
    hub = FakeHub({"Qwen": []})
    run(tmp_path, hub)
    hub.listings["Qwen"] = ["Qwen/Qwen3-1.7B"]
    run(tmp_path, hub)
    assert hub.dispatched == [("export-xnnpack.yml", "Qwen/Qwen3-1.7B", "sha-Qwen/Qwen3-1.7B")]
    # Cancelled runs go back to the queue together, and out again together.
    hub.dispatched.clear()
    summary = run(tmp_path, hub, requeue=True)
    assert hub.dispatched == [("export-xnnpack.yml", "Qwen/Qwen3-1.7B", "sha-Qwen/Qwen3-1.7B")]
    vulkan = summary["state"]["models"]["Qwen/Qwen3-1.7B"]["backends"]["vulkan"]
    assert vulkan["status"] == "dispatched" and vulkan["built_with"] == "xnnpack"


def test_a_vulkan_stage_left_pending_by_the_old_staging_is_skipped_with_the_way_to_build_it(tmp_path):
    hub = FakeHub({"Qwen": []})
    run(tmp_path, hub)
    hub.listings["Qwen"] = ["Qwen/Qwen3-1.7B"]
    run(tmp_path, hub)
    # A state written before 2026-11-09: XNNPACK out, Vulkan waiting for a stage that is gone.
    state = json.loads((tmp_path / "seen.json").read_text())
    state["models"]["Qwen/Qwen3-1.7B"]["backends"]["vulkan"] = {"status": "pending"}
    (tmp_path / "seen.json").write_text(json.dumps(state))
    hub.dispatched.clear()
    summary = run(tmp_path, hub)
    vulkan = summary["state"]["models"]["Qwen/Qwen3-1.7B"]["backends"]["vulkan"]
    assert vulkan["status"] == "skipped" and "backfill" in vulkan["reason"]
    assert all(d[0] != "export-vulkan.yml" for d in hub.dispatched)


def test_vulkan_is_skipped_when_the_xnnpack_run_that_builds_it_is_not_made(monkeypatch):
    class Verdict:
        reasons, family = [], "qwen3"
        backends = {"xnnpack": "not here", "vulkan": None, "qnn": None, "mtk": None}

    monkeypatch.setattr(watch.eligibility, "evaluate", lambda *a, **k: Verdict())
    entry = {"first_seen": "x"}
    source = make_source("Qwen/Qwen3-1.7B", config=QWEN3)
    watch._evaluate("Qwen/Qwen3-1.7B", "main", CFG, lambda m, r: source, entry)
    assert entry["backends"]["vulkan"]["status"] == "skipped"
    assert "xnnpack run" in entry["backends"]["vulkan"]["reason"]


def test_an_entry_from_before_vulkan_existed_records_it_when_its_xnnpack_run_goes_out_again(tmp_path):
    # The real shape on the state branch: a dispatched model whose backends have no vulkan line.
    hub = FakeHub({"Qwen": []})
    run(tmp_path, hub)
    hub.listings["Qwen"] = ["Qwen/Qwen3-1.7B"]
    run(tmp_path, hub)
    state = json.loads((tmp_path / "seen.json").read_text())
    del state["models"]["Qwen/Qwen3-1.7B"]["backends"]["vulkan"]
    (tmp_path / "seen.json").write_text(json.dumps(state))
    hub.dispatched.clear()
    summary = run(tmp_path, hub, requeue=True)
    assert hub.dispatched == [("export-xnnpack.yml", "Qwen/Qwen3-1.7B", "sha-Qwen/Qwen3-1.7B")]
    vulkan = summary["state"]["models"]["Qwen/Qwen3-1.7B"]["backends"]["vulkan"]
    assert vulkan["status"] == "dispatched" and vulkan["built_with"] == "xnnpack" and "predates" in vulkan["note"]
