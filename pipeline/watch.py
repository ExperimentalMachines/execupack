"""Watch Hugging Face orgs for new models and dispatch the export workflows.

State lives in one JSON file (committed to the ``state`` branch by the workflow):

    {"version": 1, "seeded_at": ..., "models": {"<model id>": {
        "status": "seeded" | "skipped" | "pending" | "dispatched" | "retry",
        "sha": ..., "created_at": ..., "first_seen": ..., "reasons": [...],
        "backends": {"xnnpack": {"status": "pending" | "dispatched" | "skipped", ...}}}}}

The first run records every listed model as ``seeded`` and exports nothing; existing
models are exported by backfill. After that, each model not yet in the state is checked
once: by name first (no network), then against its config. Eligible models get their
backends dispatched, at most ``max_dispatch`` workflow runs per watcher run; the rest stay
``pending`` for the next run.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pipeline import eligibility, hub, naming
from pipeline.settings import Settings

STATE_VERSION = 1
# Backend → the workflow that exports it (one run covers every target chip). MediaTek
# joins in phase 4.
WORKFLOWS = {
    "xnnpack": "export-xnnpack.yml",
    "qnn": "export-qnn.yml",
    "mtk": "export-mtk.yml",
}
# Backends with no workflow of their own, built in another backend's run. Vulkan files come
# out of the XNNPACK workflow's window jobs (2026-11-09): the same solved codes, gated through
# the XNNPACK file of the same window, since the host has no Vulkan runtime to gate them. A
# workflow of its own could only solve again or round to nearest, which every published Vulkan
# file was until then.
BUILT_WITH = {"vulkan": "xnnpack"}
# Dispatch order: the CPU (and with it GPU) exports of every model first, then each NPU in turn.
STAGES = ("xnnpack", "qnn", "mtk")
# A model whose metadata could not be read is retried this many runs before it is skipped.
MAX_ATTEMPTS = 5


@dataclass(frozen=True)
class Listed:
    id: str
    sha: str | None
    created_at: str | None


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_org(org: str, limit: int) -> list[Listed]:
    """The org's newest ``limit`` models. Sorted by creation, so older repos never move up."""
    models = hub.api().list_models(author=org, sort="created_at", limit=limit, expand=["createdAt", "sha"])
    return [Listed(m.id, m.sha, m.created_at.isoformat() if m.created_at else None) for m in models]


_IN_FLIGHT = ("queued", "in_progress", "waiting", "pending", "requested")


def run_in_flight(workflow: str, model_id: str, repo: str | None) -> bool:
    """Whether ``workflow`` already has a queued or running run for ``model_id`` (run names
    are "<backend>: <model id>@<revision>…")."""
    command = ["gh", "run", "list", "--workflow", workflow, "--limit", "100", "--json", "displayTitle,status"]
    if repo:
        command += ["-R", repo]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return any(r["status"] in _IN_FLIGHT and f" {model_id}@" in r["displayTitle"] for r in json.loads(result.stdout))


def runs_in_flight(workflow: str) -> bool:
    """Whether ``workflow`` has any queued or running run (the stage is still busy).

    A dispatched run is titled "<backend>: <model>@<revision>"; a run whose title is just
    the workflow's name is a ghost GitHub created while its API was failing (2026-09-13),
    stuck "queued" and impossible to cancel, and does not count.
    """
    repo = os.environ.get("GITHUB_REPOSITORY")
    command = ["gh", "run", "list", "--workflow", workflow, "--limit", "100", "--json", "displayTitle,name,status"]
    if repo:
        command += ["-R", repo]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gh run list {workflow} failed: {result.stderr.strip()[-300:]}")
    return any(r["status"] in _IN_FLIGHT and "@" in r["displayTitle"] for r in json.loads(result.stdout))


def gh_dispatch(workflow: str, model_id: str, revision: str, attempts: int = 4) -> None:
    """Start one export run, once: the dispatch API has answered 502 after creating the run,
    so a run already in flight for this model counts as dispatched, and transient failures
    are retried."""
    repo = os.environ.get("GITHUB_REPOSITORY")
    if run_in_flight(workflow, model_id, repo):
        print(f"    {workflow} already has a run in flight for {model_id}")
        return
    command = ["gh", "workflow", "run", workflow, "-f", f"model_id={model_id}", "-f", f"revision={revision}"]
    if repo:
        command += ["-R", repo]
    for attempt in range(attempts):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return
        time.sleep(10)
        if run_in_flight(workflow, model_id, repo):
            return
        if attempt < attempts - 1:
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"gh workflow run {workflow} for {model_id} failed: {result.stderr.strip()[-300:]}")


def load_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("version") != STATE_VERSION:
        raise ValueError(f"{path} has state version {state.get('version')}, expected {STATE_VERSION}")
    return state


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def org_reason(model_id: str, org: str, settings: Settings) -> str | None:
    """Why a watched org's repo is not one of that org's own families, if it is not."""
    own = settings.org_families.get(org, ())
    token = naming.app_family(naming.source_name(model_id))
    if own and token not in own:
        return f"not one of {org}'s own families ({', '.join(own)})"
    return None


def _evaluate(
    model_id: str,
    revision: str,
    settings: Settings,
    fetch: Callable[[str, str], hub.SourceModel],
    entry: dict,
    org: str | None = None,
) -> None:
    """Fill ``entry`` with the model's verdict: skipped, pending, or retry on a fetch error."""
    reasons = eligibility.name_reasons(model_id, settings)
    if org is not None and not reasons:
        reason = org_reason(model_id, org, settings)
        reasons = [reason] if reason else []
    if reasons:
        entry.update(status="skipped", reasons=reasons, backends={})
        return
    try:
        source = fetch(model_id, revision)
    except Exception as error:  # network or Hub errors: try again next run
        attempts = entry.get("attempts", 0) + 1
        if attempts >= MAX_ATTEMPTS:
            entry.update(status="skipped", reasons=[f"metadata unreadable after {attempts} runs: {error}"])
        else:
            entry.update(status="retry", attempts=attempts, reasons=[f"metadata fetch failed: {error}"])
        return
    verdict = eligibility.evaluate(source, settings)
    entry["sha"] = source.sha
    entry["family"] = verdict.family
    entry["reasons"] = verdict.reasons
    entry["backends"] = {}
    for backend, why in verdict.backends.items():
        if not verdict.reasons and why is None and (backend in WORKFLOWS or backend in BUILT_WITH):
            entry["backends"][backend] = {"status": "pending"}
        else:
            entry["backends"][backend] = {"status": "skipped", "reason": why or "no export workflow yet"}
    for backend, host in BUILT_WITH.items():
        info, host_info = entry["backends"].get(backend), entry["backends"].get(host)
        if info and info["status"] == "pending" and (host_info or {}).get("status") != "pending":
            entry["backends"][backend] = {
                "status": "skipped",
                "reason": f"built in the {host} run, which this model does not get",
            }
    pending = any(b["status"] == "pending" for b in entry["backends"].values())
    entry["status"] = "pending" if pending else "skipped"
    entry.pop("attempts", None)


def run(
    state_path: Path,
    settings: Settings,
    *,
    dispatch: bool = True,
    backfill: tuple[str, ...] = (),
    limit_per_org: int = 200,
    max_dispatch: int = 6,
    requeue: bool = False,
    lister: Callable[[str, int], list[Listed]] = list_org,
    fetch: Callable[[str, str], hub.SourceModel] = hub.fetch,
    dispatcher: Callable[[str, str, str], None] = gh_dispatch,
    in_flight: Callable[[str], bool] = runs_in_flight,
) -> dict:
    """One watcher pass. Returns a summary; the caller saves ``summary["state"]``."""
    state = load_state(state_path)
    stamp = now()
    summary = {"seeded": 0, "new": [], "backfilled": [], "dispatched": [], "failed": [], "state": None}

    seeding = state is None
    if seeding:
        # Everything that exists today is recorded, not exported; a backfill in the same run
        # still exports the models it names.
        state = {"version": STATE_VERSION, "seeded_at": stamp, "models": {}}
        for org in settings.watch_orgs:
            for m in lister(org, limit_per_org):
                state["models"][m.id] = {
                    "status": "seeded",
                    "sha": m.sha,
                    "created_at": m.created_at,
                    "first_seen": stamp,
                }
        summary["seeded"] = len(state["models"])
    models = state["models"]

    for org in settings.watch_orgs if not seeding else ():
        for m in lister(org, limit_per_org):
            entry = models.get(m.id)
            if entry is not None and entry["status"] != "retry":
                continue
            entry = entry or {"first_seen": stamp, "created_at": m.created_at}
            _evaluate(m.id, m.sha or "main", settings, fetch, entry, org=org)
            models[m.id] = entry
            summary["new"].append(m.id)

    for model_id in backfill:
        entry = {"first_seen": models.get(model_id, {}).get("first_seen", stamp), "backfill": stamp}
        _evaluate(model_id, "main", settings, fetch, entry)
        models[model_id] = entry
        summary["backfilled"].append(model_id)

    if requeue:
        # Runs were cancelled: everything recorded as dispatched goes back to the queue.
        for entry in models.values():
            for info in entry.get("backends", {}).values():
                if info["status"] == "dispatched":
                    info.update(status="pending", requeued=stamp)
                    info.pop("at", None)
            if any(i["status"] == "pending" for i in entry.get("backends", {}).values()):
                entry["status"] = "pending"

    # Stages: every model's XNNPACK exports before any Qualcomm one, then MediaTek. A later
    # stage starts only when the earlier one has nothing pending and nothing in flight.
    budget = max_dispatch
    by_first_seen = sorted(models.items(), key=lambda kv: kv[1].get("first_seen") or "")
    for backend in STAGES:
        queue = [
            (model_id, entry)
            for model_id, entry in by_first_seen
            if entry.get("status") == "pending" and entry["backends"].get(backend, {}).get("status") == "pending"
        ]
        if not queue and not (dispatch and in_flight(WORKFLOWS[backend])):
            continue  # this stage is finished; the next one may start
        summary["stage"] = backend
        for model_id, entry in queue:
            if budget == 0:
                break
            info = entry["backends"][backend]
            if dispatch:
                try:
                    dispatcher(WORKFLOWS[backend], model_id, entry["sha"])
                except Exception as error:  # GitHub API trouble: stays pending, the pass goes on
                    info.update(last_error=f"{stamp}: {error}"[:400])
                    summary["failed"].append(f"{model_id} -> {backend}: {error}")
                    budget -= 1
                    continue
                info.update(status="dispatched", at=stamp)
                info.pop("last_error", None)
                # The backends this run builds too (Vulkan with XNNPACK) go out with it. An entry
                # recorded before the backend existed has no line for it at all (26 such models on
                # the state branch, 2026-11-09): the run's targets job decides whether the model
                # can take it, and the state says it went out with the run.
                for built, host in BUILT_WITH.items():
                    if host != backend:
                        continue
                    other = entry["backends"].get(built)
                    if other is None:
                        entry["backends"][built] = {
                            "status": "dispatched",
                            "at": stamp,
                            "built_with": backend,
                            "note": "entry predates this backend; the run decides whether the model can take it",
                        }
                    elif other["status"] == "pending":
                        other.update(status="dispatched", at=stamp, built_with=backend)
            summary["dispatched"].append(f"{model_id} -> {backend}")
            budget -= 1
        break  # later stages wait for this one
    # A model whose host run went out before its backend was built there (a Vulkan stage
    # recorded under the old staging) would wait forever: say what would build it instead.
    for entry in models.values():
        for built, host in BUILT_WITH.items():
            info, host_info = entry.get("backends", {}).get(built), entry.get("backends", {}).get(host)
            if info and info["status"] == "pending" and (host_info or {}).get("status") not in (None, "pending"):
                info.update(
                    status="skipped",
                    reason=f"its {host} run predates {built} being built there; backfill the model to build it",
                )
    for entry in models.values():
        if entry.get("status") == "pending" and all(
            i["status"] != "pending" for i in entry.get("backends", {}).values()
        ):
            entry["status"] = "dispatched"

    summary["state"] = state
    return summary


def markdown(summary: dict) -> str:
    """Job summary for $GITHUB_STEP_SUMMARY."""
    state = summary["state"]
    lines = ["### Hugging Face watcher", ""]
    if summary["seeded"]:
        lines += [
            f"First run: recorded {summary['seeded']} existing models from the watched orgs without "
            "exporting them. Export any of them with the backfill input.",
            "",
        ]
    checked = summary["new"] + summary["backfilled"]
    stage = f" (stage: {summary['stage']})" if summary.get("stage") else ""
    lines.append(f"Checked {len(checked)} model(s); dispatched {len(summary['dispatched'])} export run(s){stage}.")
    if summary.get("failed"):
        lines += ["", "Could not dispatch (still pending, retried next run):", *[f"- {f}" for f in summary["failed"]]]
    if checked:
        lines += ["", "| Model | Result | Why |", "|---|---|---|"]
        for model_id in checked:
            entry = state["models"][model_id]
            why = "; ".join(entry.get("reasons") or [])
            if not why:
                skipped = [f"{b}: {i['reason']}" for b, i in entry.get("backends", {}).items() if i.get("reason")]
                why = "; ".join(skipped)
            lines.append(f"| `{model_id}` | {entry['status']} | {why.replace('|', '/')} |")
    if summary["dispatched"]:
        lines += ["", "Dispatched:", *[f"- {d}" for d in summary["dispatched"]]]
    return "\n".join(lines) + "\n"
