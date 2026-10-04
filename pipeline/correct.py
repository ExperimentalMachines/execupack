"""Correct what published export reports say about an fp32 export, without touching any .pte.

Exports made with the fp32 recipe before it was described correctly (fixed in the commit that
added ``export_xnnpack.recipe_fields`` and the per-recipe sizing) carry the 8da4w wording and
sizing: int4 codes, a group size, "4-bit weights", and a phone estimate for 4-bit linears. The
files themselves are right. This rewrites only those report fields, regenerates the folder's
config.json (the app reads its ``quantization`` label and ``fits_phone_budget``) and the README
from the corrected reports, and commits them pinned to the revision it read, so the .pte files,
their hashes and every pin against them stay as they are.

A report is corrected only when the old figures can be reproduced exactly from the 8da4w
sizing with the configured runtime overhead: that proves which inputs the export used, so
the fp32 figures replace them like for like. Anything else is refused, not guessed.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import json
from pathlib import Path

from pipeline import families, hub, manifest, settings, sizing
from pipeline.export_xnnpack import recipe_fields

FIELDS = (
    "recipe.group_size",
    "recipe.int4_codes",
    "recipe.label",
    "recipe.description",
    "window.device_resident_bytes",
    "window.fits_phone_budget",
    "window.table[].device_resident_bytes",
    "window.table[].fits_device",
    "estimates.pte_bytes_estimate",
)


class CannotCorrect(ValueError):
    pass


def corrected(report: dict, arch: sizing.Architecture, cfg, commit: str, today: str) -> dict | None:
    """The report with its fp32 description and sizing fixed, or None when it needs nothing."""
    recipe = report.get("recipe") or {}
    if report.get("backend") != "xnnpack" or recipe.get("qmode") != "fp32":
        return None
    window = report["window"]
    overhead = cfg.runtime_overhead_bytes
    # The old figures must be exactly the 8da4w sizing of this architecture and overhead.
    rows = [window, *window["table"]]
    for row in rows:
        expected = sizing.device_resident_bytes(arch, row["context"], overhead, "8da4w")
        fp32 = sizing.device_resident_bytes(arch, row["context"], overhead, "fp32")
        if row["device_resident_bytes"] not in (expected, fp32):
            raise CannotCorrect(
                f"{report['files'][0]['path']}: resident bytes at {row['context']} are "
                f"{row['device_resident_bytes']:,}, neither the 8da4w figure {expected:,} nor the fp32 one {fp32:,}"
            )
    new = copy.deepcopy(report)
    base = dataclasses.replace(
        cfg.xnnpack, qmode="fp32", embedding_quantize=recipe["embedding_quantize"], group_size=recipe["group_size"]
    )
    new["recipe"].update(
        recipe_fields(
            base, "fp32", False, "xnnpack", recipe["prefill_chunk"], executorch=report["toolchain"]["executorch"]
        )
    )
    budget = window["phone_budget_bytes"]
    for row in (new["window"], *new["window"]["table"]):
        row["device_resident_bytes"] = sizing.device_resident_bytes(arch, row["context"], overhead, "fp32")
    new["window"]["fits_phone_budget"] = new["window"]["device_resident_bytes"] <= budget
    for row in new["window"]["table"]:
        row["fits_device"] = row["device_resident_bytes"] <= budget
    if "estimates" in new:
        new["estimates"]["pte_bytes_estimate"] = sizing.pte_bytes_estimate(arch, window["context"], "fp32")
    if new == report:
        return None
    new.setdefault("corrections", []).append(
        {
            "date": today,
            "by": f"execupack {commit}",
            "fields": list(FIELDS),
            "reason": "fp32 export described and sized as 8da4w; the .pte is unchanged",
        }
    )
    return new


def correct_repo(repo_id: str, commit: str, dry_run: bool = True) -> dict:
    """Correct every fp32 report in one Hub repo; return what changed (and the commit URL)."""
    from huggingface_hub import CommitOperationAdd, hf_hub_download

    cfg = settings.load()
    hf = hub.api()
    info = hf.model_info(repo_id, expand=["sha", "siblings"])
    from pipeline.publish import is_report

    paths = sorted(s.rfilename for s in info.siblings or [] if is_report(s.rfilename))
    reports = {
        p: json.loads(Path(hf_hub_download(repo_id, p, revision=info.sha, token=hf.token)).read_text()) for p in paths
    }
    today = datetime.date.today().isoformat()
    changed: dict[str, dict] = {}
    archs: dict[tuple, sizing.Architecture] = {}
    for path, report in reports.items():
        key = (report["source"]["id"], report["source"]["sha"])
        if report.get("recipe", {}).get("qmode") == "fp32" and key not in archs:
            source = hub.fetch(*key)
            archs[key] = families.architecture(source.config, source.total_params)
        fixed = corrected(report, archs[key], cfg, commit, today) if key in archs else None
        if fixed is not None:
            changed[path] = fixed
    result = {"repo": repo_id, "revision": info.sha, "corrected": sorted(changed), "commit_url": None}
    if not changed or dry_run:
        return result
    reports.update(changed)
    all_reports = list(reports.values())
    operations = [
        CommitOperationAdd(path_in_repo=p, path_or_fileobj=(json.dumps(r, indent=2) + "\n").encode())
        for p, r in sorted(changed.items())
    ]
    for folder in sorted({p.rsplit("/", 1)[0] for p in changed}):
        folder_reports = [r for p, r in reports.items() if p.rsplit("/", 1)[0] == folder]
        config = json.dumps(manifest.backend_config(folder_reports), indent=2) + "\n"
        operations.append(CommitOperationAdd(path_in_repo=f"{folder}/config.json", path_or_fileobj=config.encode()))
    license_files = sorted({f for r in all_reports for f in r["source"].get("license_files", [])})
    readme = manifest.readme(repo_id, all_reports, list(cfg.hub_tags), license_files)
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode("utf-8")))
    commit_info = hf.create_commit(
        repo_id,
        operations,
        commit_message=f"correct fp32 report fields ({len(changed)} reports); no .pte changed",
        parent_commit=info.sha,
    )
    result["commit_url"] = commit_info.commit_url
    return result
