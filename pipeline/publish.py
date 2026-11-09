"""Publishing an export: Hugging Face Hub, GitHub Releases, and the job summary."""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

from pipeline import exporting, hub, manifest, naming, settings

# GitHub rejects release assets of 2 GiB or more.
RELEASE_ASSET_LIMIT = 2 * 1024**3


def _folder(backend: str, target: str | None) -> str:
    return backend if target is None else f"{backend}/{target.lower()}"


def load_report(out_dir: Path, backend: str, target: str | None = None, context: int | None = None) -> dict:
    """The export report in the local backend folder: one run exports one window, so there
    is one unless several windows were exported into the same folder by hand, in which
    case ``context`` picks."""
    folder = out_dir / _folder(backend, target)
    found = sorted(folder.glob("export-report*.json"))
    if context is not None:
        found = [p for p in found if p.name == exporting.report_file(context)]
    if len(found) != 1:
        raise ValueError(f"expected one export report in {folder}, found {[p.name for p in found]}; pass --context")
    return json.loads(found[0].read_text(encoding="utf-8"))


def is_report(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name.startswith("export-report") and name.endswith(".json")


# The window label sits right before the extension (or before "-chunk<i>of<n>" for MediaTek),
# so a model whose own name carries a window-like token ("SmolLM2-1.7B-Instruct-16k") is not
# read as that window.
_WINDOW_FILE = re.compile(r"-(?P<label>\d+k|\d+)(?:\.pte|-chunk\d+of\d+\.pte)$")


def window_of(name: str) -> str | None:
    """The window label a published file name carries, or None (config, embedding table)."""
    if name.startswith("export-report-") and name.endswith(".json"):
        return name[len("export-report-") : -len(".json")]
    match = _WINDOW_FILE.search(name)
    return match.group("label") if match else None


def superseded(path: str, folder: str, label: str) -> bool:
    """Whether a file already in the repo is replaced by this run: the same window in the
    same backend folder, or the folder's pre-matrix files that carried no window in their
    name. Other windows' files stay: the folder holds every window side by side."""
    if not path.startswith(f"{folder}/") or "/" in path[len(folder) + 1 :]:
        return False
    name = path[len(folder) + 1 :]
    if name in ("config.json", "export-report.json"):
        return True
    return window_of(name) == label


def require_same_bytes(out_dir: Path, report: dict) -> None:
    """Every file the report lists is, byte for byte, the file it describes: a verdict is
    about the bytes it measured, and a .pte rebuilt or replaced after verify is not that file
    (codex review, 2026-11-09)."""
    for entry in report["files"]:
        local = out_dir / entry["path"]
        if not local.is_file():
            raise ValueError(f"{entry['path']} is listed in the report but not in {out_dir}")
        if exporting.sha256(local) != entry.get("sha256"):
            raise ValueError(f"{entry['path']} is not the file its report describes (sha256 differs)")


def require_vouched(report: dict, paths: list[str]) -> None:
    """Every model file about to be uploaded is one the report lists: an extra or renamed
    .pte or .bin, in the backend folder or at the repo root, has no verdict and no digest
    behind it (codex review, rounds five and six)."""
    listed = {entry["path"] for entry in report["files"]}
    unvouched = sorted(path for path in paths if path.endswith((".pte", ".bin")) and path not in listed)
    if unvouched:
        raise ValueError(f"{unvouched} would be uploaded but are not in the report; nothing vouches for them")


def shared_file_clashes(report: dict, retained: list[dict]) -> list[tuple[dict, str]]:
    """The other windows whose reports list a file this run uploads (MediaTek's embedding
    table, one per folder) with different bytes, each with what differs.

    Replacing that file from another source revision would leave those windows' chunks running
    on weights their reports never measured, under digests that no longer match (codex review
    of #30). A rerun from the same revision writes the same bytes and clashes with nothing.
    """
    ours = {entry["path"]: entry.get("sha256") for entry in report["files"]}
    clashes = []
    for other in retained:
        differ = [
            f"{entry['path']} ({str(entry.get('sha256'))[:12]} in the "
            f"{naming.window_label(other['window']['context'])} report, {str(ours[entry['path']])[:12]} in this run's)"
            for entry in other.get("files", [])
            if entry["path"] in ours and entry.get("sha256") != ours[entry["path"]]
        ]
        if differ:
            clashes.append((other, "; ".join(differ)))
    return clashes


def require_publishable(report: dict) -> None:
    """Refuse a report that has not passed every check its recipe calls for.

    The workflow runs the smoke test in its own job and fails before reaching publish, but
    publishing is also a command a person can run by hand, against a directory exported with
    --skip-smoke. Refusing here is what makes "published" mean "ran and answered" rather than
    "the workflow happened to be wired correctly that day". A report that was gated must have
    passed it. A GPTQ file always has an fp32 reference to be gated against (the solve writes
    it), so it must carry a passing gate of its own, or for Vulkan, which the host cannot run,
    a gate proxy that passed: a GPTQ report with neither is one whose gate never ran or
    crashed (codex review, 2026-11-09). Any other report with no gate (round-to-nearest
    Vulkan) is published on its smoke test, as before.
    """
    name = report["files"][0]["path"]
    smoke = report.get("smoke")
    if smoke is None:
        raise ValueError(
            f"{name} has no smoke result: run `python -m pipeline verify` on this directory before publishing it"
        )
    if not smoke.get("passed"):
        raise ValueError(f"{name} failed its smoke test: {'; '.join(smoke.get('problems', []))}")
    verdict = report.get("gate")
    if verdict is not None and not verdict.get("passed") and not (verdict.get("override") or {}).get("reason"):
        raise ValueError(f"{name} failed its decision gate: {'; '.join(verdict.get('problems', []))}")
    if (report.get("recipe") or {}).get("int4_codes") == "gptq":
        if report.get("backend") == "vulkan":
            proxy = (report.get("gate_proxy") or {}).get("gate") or {}
            if not proxy.get("passed") and not (proxy.get("override") or {}).get("reason"):
                raise ValueError(f"{name} has GPTQ codes and no passing gate proxy (the XNNPACK file from its codes)")
        elif verdict is None:
            raise ValueError(f"{name} has GPTQ codes and no decision gate: verify it with --gate before publishing")


def require_twin_on_hub(
    report: dict, by_path: dict[str, dict], existing: list[str], allow_unpublished_twin: bool = False
) -> None:
    """A GPTQ Vulkan file publishes only into a head that holds the XNNPACK file its gate proxy
    names: the same path, and a report of the same window giving the same sha256. Checked
    against the very revision the commit is pinned to, on every retry, so another publisher
    replacing the twin in between is caught (codex review, CI round 2). A Vulkan file gated
    through a twin that is deliberately not published (SmolLM2-360M ships fp32 on XNNPACK)
    needs ``allow_unpublished_twin``, said out loud."""
    proxy = report.get("gate_proxy")
    if report.get("backend") != "vulkan" or not proxy or allow_unpublished_twin:
        return
    context = report["window"]["context"]
    present = any(
        path.startswith("xnnpack/")
        and other["window"]["context"] == context
        and other["files"][0]["sha256"] == proxy.get("sha256")
        and other["files"][0]["path"] == proxy.get("file")
        for path, other in by_path.items()
    )
    if not present or proxy.get("file") not in existing:
        raise ValueError(
            f"{report['files'][0]['path']} is gated through {proxy.get('file')} ({str(proxy.get('sha256'))[:12]}), "
            "which is not the XNNPACK file of this window on the Hub: publish that twin first, "
            "or pass --allow-unpublished-twin if it is deliberately not published"
        )


def orphaned_gpu_window(
    existing: list[str], deleted: set[str], by_path: dict[str, dict], report: dict, label: str
) -> list[str]:
    """The Vulkan files of this window that the XNNPACK commit now being built leaves without
    their gate, judged on the tree the commit produces (codex review, CI rounds 1 to 3): a
    Vulkan report's gate proxy names its twin by path and sha256, and if that twin was on the
    Hub and after this commit is gone, or there under other bytes, the Vulkan window goes with
    it. A proxy naming a file that was never on the Hub (a twin deliberately not published,
    as SmolLM2-360M's) is not this commit's to judge, and stays."""
    context = report["window"]["context"]
    new_path, new_sha = report["files"][0]["path"], report["files"][0]["sha256"]
    gpu_paths = [
        p
        for p in (f"vulkan/{exporting.report_file(context)}", "vulkan/export-report.json")
        if p in by_path and by_path[p]["window"]["context"] == context
    ]
    orphaned = []
    for gpu_path in gpu_paths:
        proxy = by_path[gpu_path].get("gate_proxy") or {}
        twin, twin_sha = proxy.get("file"), proxy.get("sha256")
        if not twin or twin not in existing:
            continue
        if twin == new_path:
            still_there = twin_sha == new_sha
        else:
            still_there = twin not in deleted
        if not still_there:
            orphaned.append(gpu_path)
    if not orphaned:
        return []
    files = [
        p for p in existing if p.startswith("vulkan/") and p.count("/") == 1 and window_of(p.split("/")[1]) == label
    ]
    return sorted(set(files) | set(orphaned))


def publish_hf(
    out_dir: Path,
    backend: str,
    target: str | None = None,
    context: int | None = None,
    attempts: int = 6,
    allow_unpublished_twin: bool = False,
    replace_shared: bool = False,
) -> str:
    """Commit this run's window into its backend folder beside the other windows, plus the
    shared root files; regenerate the folder's config.json and the repo README.md from
    every report in the repo.

    A run whose shared file (MediaTek's embedding) differs from the one the folder's other
    windows were built on is refused; with ``replace_shared`` those windows are withdrawn in
    the same commit, so the folder never mixes tables, and are republished from this run's
    revision afterwards, one run each.

    Runs publish on their own, possibly at the same moment as another (other windows, other
    backends), so the commit is pinned to the revision the README was rendered from and
    retried on conflict.
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    cfg = settings.load()
    report = load_report(out_dir, backend, target, context)
    require_publishable(report)
    repo_id = report["output_repo"]
    folder = _folder(backend, target)
    label = naming.window_label(report["window"]["context"])
    local_folder = out_dir / folder
    # This window's files, plus the folder's shared ones (config.json, MediaTek's embedding).
    new_paths = {
        f"{folder}/{p.name}": p for p in local_folder.iterdir() if p.is_file() and window_of(p.name) in (None, label)
    }
    root_files = {p.name: p for p in out_dir.iterdir() if p.is_file()}
    scoped = [path for path in new_paths if path.rsplit("/", 1)[-1] in exporting.TOKENIZER_FILES]
    if scoped:
        # A tokenizer inside any backend folder makes the app stop lending the root one to
        # every other folder (HuggingFaceClient.tokenizerFor), breaking the other backends.
        raise ValueError(f"tokenizer files must stay at the repo root, found {scoped}")
    require_vouched(report, [*new_paths, *root_files])
    require_same_bytes(out_dir, report)

    hf = hub.api()
    hf.create_repo(repo_id, repo_type="model", exist_ok=True)

    for attempt in range(attempts):
        info = hf.model_info(repo_id, expand=["sha", "siblings"])
        existing = [s.rfilename for s in info.siblings or []]
        stale = [path for path in existing if superseded(path, folder, label)]
        reports = [report]
        migrated: dict[str, bytes] = {}
        by_path: dict[str, dict] = {}
        for path in existing:
            if not is_report(path) or (path in stale and not path.endswith("/export-report.json")):
                continue
            local = hf_hub_download(repo_id, path, revision=info.sha, token=hf.token)
            other = json.loads(Path(local).read_text(encoding="utf-8"))
            by_path[path] = other
            if path in stale:
                # A pre-matrix report carries no window in its name. Its window's file stays
                # unless it is the one this run replaces, so the report moves to the labelled name.
                if naming.window_label(other["window"]["context"]) == label:
                    continue
                migrated[f"{folder}/{exporting.report_file(other['window']['context'])}"] = (
                    json.dumps(other, indent=2).encode() + b"\n"
                )
            reports.append(other)
        if folder == "vulkan":
            require_twin_on_hub(report, by_path, existing, allow_unpublished_twin)
        # Replacing an XNNPACK window takes with it the Vulkan window whose gate proxy names the
        # file being replaced: left up, that GPU file would cite a gate no longer on the Hub (a
        # failed or skipped Vulkan build, an fp32 or rounded rerun). Its new twin, when one is
        # built, publishes right after this commit (export-xnnpack.yml).
        orphaned = (
            orphaned_gpu_window(existing, set(stale) - set(new_paths), by_path, report, label)
            if folder == "xnnpack"
            else []
        )
        if orphaned:
            dropped = {id(by_path[p]) for p in orphaned if p in by_path}
            reports = [r for r in reports if id(r) not in dropped]
        others = [r for r in reports if r is not report]
        # Only this folder's windows can be withdrawn (its config.json is the one rebuilt here); a
        # clash with another backend's report cannot be settled by this run and is refused outright
        # (codex review of #30, round four).
        foreign = shared_file_clashes(report, [r for r in others if _folder(r["backend"], r.get("target")) != folder])
        if foreign:
            raise ValueError(
                "this run's files differ from what another backend's reports record: "
                + "; ".join(why for _, why in foreign)
                + ". Withdrawing windows in this folder cannot settle that."
            )
        clashes = shared_file_clashes(report, [r for r in others if _folder(r["backend"], r.get("target")) == folder])
        withdrawn: list[str] = []
        if clashes and not replace_shared:
            raise ValueError(
                "this run's shared files differ from those other published windows were built on: "
                + "; ".join(why for _, why in clashes)
                + ". Pass --replace-shared to withdraw those windows in the same commit, then republish "
                "each from this run's revision."
            )
        if clashes:
            gone = {id(other) for other, _ in clashes}
            reports = [r for r in reports if id(r) not in gone]
            # Only each withdrawn window's own files go: those in this folder that carry its window
            # in their name, and its report. A file any kept report (this run's included) still
            # lists, an unlabelled shared file, or a path in another folder is never touched, so a
            # withdrawal cannot take a file a window that stays depends on (codex review of #30,
            # round three).
            kept = {entry["path"] for r in reports for entry in r.get("files", [])}
            withdrawn = [path for path, other in by_path.items() if id(other) in gone]
            for other, _ in clashes:
                own = naming.window_label(other["window"]["context"])
                withdrawn += [
                    entry["path"]
                    for entry in other["files"]
                    if entry["path"].startswith(f"{folder}/")
                    and "/" not in entry["path"][len(folder) + 1 :]
                    and window_of(entry["path"].rsplit("/", 1)[-1]) == own
                ]
                migrated.pop(f"{folder}/{exporting.report_file(other['window']['context'])}", None)
            withdrawn = sorted(
                {p for p in withdrawn if p in existing and p not in stale and p not in new_paths and p not in kept}
            )
        folder_reports = [r for r in reports if _folder(r["backend"], r.get("target")) == folder]
        config = json.dumps(manifest.backend_config(folder_reports), indent=2) + "\n"
        license_files = sorted({f for r in reports for f in r["source"].get("license_files", [])})
        readme = manifest.readme(repo_id, reports, list(cfg.hub_tags), license_files)

        operations = [CommitOperationDelete(path_in_repo=path) for path in stale if path not in new_paths]
        operations += [CommitOperationDelete(path_in_repo=path) for path in withdrawn]
        operations += [
            CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v))
            for k, v in new_paths.items()
            if k != f"{folder}/config.json"
        ]
        operations.append(CommitOperationAdd(path_in_repo=f"{folder}/config.json", path_or_fileobj=config.encode()))
        operations += [CommitOperationAdd(path_in_repo=k, path_or_fileobj=v) for k, v in migrated.items()]
        operations += [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in root_files.items()]
        operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode("utf-8")))
        message = f"{folder} {label}: {report['source']['id']}@{report['source']['sha'][:12]}"
        if clashes:
            labels = sorted({naming.window_label(other["window"]["context"]) for other, _ in clashes})
            message += f"; withdraws {', '.join(labels)}, built on another shared file"
        if orphaned:
            operations += [CommitOperationDelete(path_in_repo=p) for p in orphaned]
            gpu_reports = [r for r in reports if _folder(r["backend"], r.get("target")) == "vulkan"]
            if gpu_reports:
                gpu_config = json.dumps(manifest.backend_config(gpu_reports), indent=2) + "\n"
                operations.append(
                    CommitOperationAdd(path_in_repo="vulkan/config.json", path_or_fileobj=gpu_config.encode())
                )
            elif "vulkan/config.json" in existing:
                operations.append(CommitOperationDelete(path_in_repo="vulkan/config.json"))
            message += f"; withdraws vulkan {label}, whose gate was the file replaced"
        try:
            commit = hf.create_commit(
                repo_id,
                operations,
                commit_message=message,
                parent_commit=info.sha,
            )
            return commit.commit_url
        except HfHubHTTPError as error:
            status = getattr(error.response, "status_code", None)
            # 412 is the parent_commit precondition failing; 409 is the Hub refusing a
            # concurrent commit to the same branch. Both mean another window's job got
            # there first, and both are fixed by rebuilding the commit against the head
            # this loop re-reads at the top. Five windows publish at once, so this is the
            # normal case, not an exceptional one: run 35373841386 lost the 8192 window to
            # an unretried 409 while the other four went through.
            if status not in (409, 412) or attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def release_tag(report: dict) -> str:
    """One release per backend, chip, window and source revision: the window jobs of one
    model finish at the same time and must not race on one tag."""
    name = report["source"]["id"].rsplit("/", 1)[-1]
    target = f"-{report['target']}" if report.get("target") else ""
    label = naming.window_label(report["window"]["context"])
    return f"{name}-{report['backend']}{target}-{label}-{report['source']['sha'][:7]}"


def release_notes(report: dict, hf_url: str | None) -> str:
    lines = [
        f"{manifest.BACKEND_TITLES[report['backend']]} export of "
        f"[{report['source']['id']}](https://huggingface.co/{report['source']['id']}) "
        f"at `{report['source']['sha'][:12]}`, window {report['window']['context']:,} tokens, "
        f"ExecuTorch {report['toolchain']['executorch']}.",
        "",
    ]
    if hf_url:
        lines += [f"Hugging Face: {hf_url}", ""]
    too_big = [f for f in report["files"] if f["bytes"] >= RELEASE_ASSET_LIMIT]
    for f in too_big:
        lines.append(
            f"- `{f['path']}` is {f['bytes']:,} bytes, over GitHub's 2 GiB asset limit: download it from Hugging Face."
        )
    return "\n".join(lines) + "\n"


def require_twin_on_hub_now(report: dict) -> None:
    """``require_twin_on_hub`` against the Hub's current head: a release is a mirror of the
    Hub, so a GPU release goes out only while the twin its gate cites is there (codex review,
    CI round 3)."""
    from huggingface_hub import hf_hub_download

    hf = hub.api()
    info = hf.model_info(report["output_repo"], expand=["sha", "siblings"])
    existing = [s.rfilename for s in info.siblings or []]
    by_path = {}
    for path in existing:
        if path.startswith("xnnpack/") and is_report(path):
            local = hf_hub_download(report["output_repo"], path, revision=info.sha, token=hf.token)
            by_path[path] = json.loads(Path(local).read_text(encoding="utf-8"))
    require_twin_on_hub(report, by_path, existing)


def withdraw_orphaned_gpu_release(report: dict) -> str | None:
    """After an XNNPACK release is (re)published: the Vulkan release of the same window and
    source revision, if its gate proxy names this very file under other bytes, cites a gate
    this release no longer holds, and is deleted (codex review, CI round 3). Returns its tag."""
    import tempfile

    tag = release_tag({**report, "backend": "vulkan", "target": None})
    view = subprocess.run(["gh", "release", "view", tag, "--json", "tagName"], capture_output=True, text=True)
    if view.returncode != 0:
        # Only a release that is confirmed missing means there is nothing to withdraw; an API,
        # network or auth failure must not pass for one (codex review, CI round 4).
        if "release not found" in (view.stderr + view.stdout).lower():
            return None
        raise RuntimeError(f"gh release view {tag} failed: {view.stderr.strip()[-500:]}")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["gh", "release", "download", tag, "--pattern", "export-report-*.json", "--dir", tmp],
            capture_output=True,
            check=True,
        )
        found = sorted(Path(tmp).glob("export-report-*.json"))
        proxy = (json.loads(found[0].read_text(encoding="utf-8")).get("gate_proxy") or {}) if found else {}
    twin = report["files"][0]
    if proxy.get("file") != twin["path"] or proxy.get("sha256") == twin["sha256"]:
        return None
    subprocess.run(["gh", "release", "delete", tag, "--yes", "--cleanup-tag"], check=True)
    return tag


def publish_release(
    out_dir: Path,
    backend: str,
    target: str | None = None,
    context: int | None = None,
    attempts: int = 5,
    allow_unpublished_twin: bool = False,
) -> str:
    """Create (or update) a GitHub release with this window's files under the 2 GiB limit."""
    report = load_report(out_dir, backend, target, context)
    require_publishable(report)
    require_same_bytes(out_dir, report)
    if backend == "vulkan" and report.get("gate_proxy") and not allow_unpublished_twin:
        require_twin_on_hub_now(report)
    folder = out_dir / _folder(backend, target)
    tag = release_tag(report)
    label = naming.window_label(report["window"]["context"])
    hf_url = f"https://huggingface.co/{report['output_repo']}"
    candidates = [p for p in sorted(folder.iterdir()) if window_of(p.name) in (None, label)]
    require_vouched(report, [str(p.relative_to(out_dir)) for p in candidates])
    candidates.append(out_dir / report["tokenizer"])
    assets = [str(p) for p in candidates if p.is_file() and p.stat().st_size < RELEASE_ASSET_LIMIT]
    notes = release_notes(report, hf_url)
    create = ["gh", "release", "create", tag, "--title", tag, "--notes", notes, *assets]
    # GitHub's API answers 5xx now and then (a whole evening of them on 2026-09-13); a
    # release is the last step of a multi-hour job, so it is retried, not failed.
    for attempt in range(attempts):
        created = subprocess.run(create, capture_output=True, text=True)
        if created.returncode == 0:
            if backend == "xnnpack":
                withdraw_orphaned_gpu_release(report)
            return tag
        output = created.stderr + created.stdout
        if "already exists" in output:
            break
        if not re.search(r"HTTP 5\d\d", output) or attempt == attempts - 1:
            raise RuntimeError(f"gh release create failed: {created.stderr.strip()[-2000:]}")
        time.sleep(30 * (attempt + 1))
    # The release exists (a re-run of the same window, or a create that answered 5xx after
    # creating it): replace its assets and notes, and with them the GPU release they gated.
    subprocess.run(["gh", "release", "upload", tag, "--clobber", *assets], check=True)
    try:
        subprocess.run(["gh", "release", "edit", tag, "--notes", notes], check=True)
    finally:
        # The assets are replaced already: the GPU release they gated goes even if the notes
        # edit failed (codex review, CI round 4).
        if backend == "xnnpack":
            withdraw_orphaned_gpu_release(report)
    return tag


def summary(out_dir: Path, backend: str, target: str | None = None, context: int | None = None) -> str:
    """Markdown for $GITHUB_STEP_SUMMARY."""
    report = load_report(out_dir, backend, target, context)
    smoke = report.get("smoke") or {}
    host = report["host"]
    estimates = report.get("estimates") or {}
    rows = [
        ("Source", f"{report['source']['id']} @ `{report['source']['sha'][:12]}`"),
        ("Family / variant", f"{report['source']['family']} / {report['source']['variant']}"),
        ("Parameters", f"{report['source']['total_params']:,}"),
        ("Window", f"{report['window']['context']:,} ({report['window']['reason']})"),
        ("Files", ", ".join(f"`{f['path']}` {f['bytes']:,} B" for f in report["files"])),
        ("Peak RSS export", f"{host.get('peak_rss_export_bytes') or 0:,} B"),
        ("Peak swap in use", f"{host.get('peak_swap_used_bytes') or 0:,} B"),
        ("Peak RAM + swap in use", f"{host.get('peak_in_use_bytes') or 0:,} B at {host.get('peak_in_use_at') or '?'}"),
        ("CPU", host.get("cpu_model") or "?"),
        ("Export time", f"{host.get('export_seconds')} s of {host.get('total_seconds')} s"),
    ]
    if estimates:
        rows += [
            (".pte estimate vs actual", f"{estimates['pte_bytes_estimate']:,} / {estimates['pte_bytes_actual']:,} B"),
            ("Export peak estimate", f"{estimates['export_peak_bytes_estimate']:,} B"),
        ]
    if not smoke:
        rows.append(("Check", "not run"))
    elif smoke.get("kind") == "structural":
        rows.append(
            ("Check", "structure " + ("passed" if smoke["passed"] else "failed: " + "; ".join(smoke["problems"])))
        )
    else:
        rows.append(("Smoke test", f"{'passed' if smoke.get('passed') else 'failed'}: {smoke.get('reply', '')!r}"))
        if smoke.get("template_error"):
            rows.append(("Chat template", f"could not be rendered, completion prompt used: {smoke['template_error']}"))
    measuring = report.get("measuring_gate") or {}
    if measuring.get("prefill_tok_per_sec") is not None:
        rows.append(("Prefill throughput", f"{measuring['prefill_tok_per_sec']:.2f} tok/s"))
    if measuring.get("decode_tok_per_sec") is not None:
        rows.append(("Decode throughput", f"{measuring['decode_tok_per_sec']:.2f} tok/s"))
    title = manifest.BACKEND_TITLES[backend] + (f" {manifest.target(report)}" if report.get("target") else "")
    lines = [f"### {title}: {report['output_repo']}", "", "| | |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines) + "\n"
