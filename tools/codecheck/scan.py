"""Byte-check published .pte files: sha256 against the report and the Hub's LFS record, then the
int4 codes against round to nearest of the fp32 source the report names.

    python scan.py tasks.json results.jsonl dl_dir work_dir

tasks.json: [[repo, path_in_repo], ...]. One result line per file; a file already in results is
skipped, so the scan resumes.
"""

import hashlib
import json
import shutil
import sys
import traceback
from pathlib import Path

import check
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from ptecodes import load_pte, vulkan_int4, xnnpack_int4

tasks_path, results_path, dl, work = (Path(p) for p in sys.argv[1:5])
# Optional: {repo: [source_id, source_sha]} for a repo whose own source is gone from the Hub (the
# heretic LFM2.5 checkpoints): its files are compared with the base model instead, and every
# linear's row is kept, since abliterated matrices differ from the base by more than rounding.
overrides = json.loads(Path(sys.argv[5]).read_text()) if len(sys.argv) > 5 else {}
tasks = json.loads(tasks_path.read_text())
done = set()
if results_path.exists():
    for line in results_path.read_text().splitlines():
        r = json.loads(line)
        done.add((r["repo"], r["path"]))
api = HfApi()
lfs = {}
state_cache = {}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


for repo, path in tasks:
    if (repo, path) in done:
        continue
    result = {"repo": repo, "path": path}
    try:
        if repo not in lfs:
            info = api.model_info(repo, files_metadata=True)
            lfs[repo] = {
                "sha": info.sha,
                "files": {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings},
            }
        result["repo_sha"] = lfs[repo]["sha"]
        folder, name = path.rsplit("/", 1)
        window = name.rsplit("-", 1)[1].removesuffix(".pte")
        report = json.loads(
            Path(hf_hub_download(repo, f"{folder}/export-report-{window}.json", revision=lfs[repo]["sha"])).read_text()
        )
        rfile = next(f for f in report["files"] if f["path"] == path)
        recipe = report.get("recipe", {})
        result.update(
            claimed_codes=recipe.get("int4_codes"),
            claimed_label=recipe.get("label"),
            executorch=(report.get("toolchain") or {}).get("executorch"),
            gate_passed=(report.get("gate") or {}).get("passed") if isinstance(report.get("gate"), dict) else None,
            gate_proxy=bool(report.get("gate_proxy")),
            source=report["source"]["id"],
            source_sha=report["source"]["sha"],
        )
        local = Path(hf_hub_download(repo, path, revision=lfs[repo]["sha"], local_dir=dl / "pte"))
        digest = sha256(local)
        result.update(
            sha256=digest,
            sha_matches_report=digest == rfile["sha256"],
            sha_matches_hub=digest == lfs[repo]["files"].get(path),
            bytes=local.stat().st_size,
            bytes_match_report=local.stat().st_size == rfile["bytes"],
        )
        src_key = (report["source"]["id"], report["source"]["sha"])
        if repo in overrides:
            src_key = tuple(overrides[repo])
            result["compared_with"] = list(src_key)
        src_dir = dl / "src" / f"{src_key[0].replace('/', '--')}@{src_key[1][:12]}"
        if not (src_dir / "config.json").exists():
            snapshot_download(
                src_key[0], revision=src_key[1], allow_patterns=["*.safetensors", "*.json"], local_dir=src_dir
            )
        if src_key not in state_cache:
            state_cache.clear()
            state_cache[src_key] = check.et_state(src_dir, work)
        state = state_cache[src_key]
        pte, named = load_pte(local)
        tensors = xnnpack_int4(pte, named) + vulkan_int4(pte, named)
        if tensors:
            low_first, rows, not_int4 = check.analyse(tensors, state)
        else:
            low_first, rows, not_int4 = None, [], sorted(state)
        result.update(check.summarise(local, tensors, low_first, rows, not_int4, None))
        if repo in overrides:
            result["rows"] = [
                {k: r[k] for k in ("key", "class", "rel_err", "agree_rtn", "scale_at_amax")} for r in rows
            ]
        del pte, named, tensors
        local.unlink()
    except Exception as error:  # record and move on: one bad file must not stop the scan
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()[-2000:]
    with open(results_path, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(repo, path, result.get("verdict"), result.get("error", ""), flush=True)
shutil.rmtree(dl / "pte", ignore_errors=True)
print("SCAN DONE", flush=True)
