"""After the rollout: is every file in the rolled-out repos what its report says it is?

    python verify_hub.py <models.json> <collected-dir> <out.json> [--skip MODEL ...]

Per repo, from the Hub, against the byte-check records collected from the boxes
(<collected>/<box>/gptq-<slug>/codesmatch-*.json and codes.sha256):
- xnnpack/ and vulkan/ each hold exactly the five windows: one report per window label, and the
  .pte files the reports name are exactly the .pte files in the folder, one each;
- every report: int4_codes gptq, ExecuTorch 1.5.1, the pinned source revision, a file sha256
  equal to the Hub's LFS sha256, and a measured, passing gate (XNNPACK: graded > 0, fp32 and
  export means present, no override) or a Vulkan gate proxy whose verdict equals the published
  XNNPACK report's of the same window and whose sha256 is that XNNPACK file;
- the report's codes_sha256 is the hash of this model's codes.pt as it was on the boxes, and
  the .pte was byte-checked against that codes.pt (an ok codesmatch record in this model's own
  runs directory with the file's sha256);
- README.md and vulkan/config.json carry no round-to-nearest claim.
Exit 1 if any repo has a problem.
"""

import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

WINDOWS = {"2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768}
GATE_FIELDS = ("passed", "graded", "agreed", "fp32_mean", "export_mean")

args = sys.argv[1:]
skip = set(args[args.index("--skip") + 1 :]) if "--skip" in args else set()
models = [m for m in json.loads(Path(args[0]).read_text()) if m["model"] not in skip]
collected, out_path = Path(args[1]), Path(args[2])


def slug(model_id):
    return model_id.split("/")[1].lower().replace(".", "")


# Decided exceptions (user, 2026-11-09): Qwen3-0.6B ships fp32 linears on XNNPACK and no Vulkan
# folder; SmolLM2-360M keeps its fp32 XNNPACK files and ships GPTQ Vulkan files whose gate proxy
# is an XNNPACK twin that was gated but not published (its reports are kept as evidence).
FP32_REBUILT = {"Qwen/Qwen3-0.6B"}
FP32_KEPT = {"HuggingFaceTB/SmolLM2-360M-Instruct"}
NO_VULKAN = {"Qwen/Qwen3-0.6B"}

evidence = {}
for runs_dir in collected.glob("*/gptq-*"):
    ev = evidence.setdefault(
        runs_dir.name.removeprefix("gptq-"), {"codes": set(), "checked": {}, "gate_ref": set(), "twins": {}}
    )
    for t in (runs_dir / "twin-reports").glob("export-report-*.json"):
        d = json.loads(t.read_text())
        ev["twins"][d["files"][0]["sha256"]] = d
    if (runs_dir / "gate.json").exists():
        ev["gate_ref"].add(hashlib.sha256((runs_dir / "gate.json").read_bytes()).hexdigest())
    if (runs_dir / "codes.sha256").exists():
        ev["codes"].add((runs_dir / "codes.sha256").read_text().strip())
    for f in runs_dir.glob("codesmatch-*.json"):
        try:
            d = json.loads(f.read_text())
        except ValueError:
            continue
        fp32_record = f.name.startswith("codesmatch-xnnpack-fp32-")
        sound = d.get("ok") is True and (
            (fp32_record and d.get("int4_linears") == 0)
            or (
                not fp32_record
                and d.get("int4_linears", 0) > 0
                and d.get("matched_exactly") == d.get("int4_linears")
                and d.get("order_checked") is True
                and f.name.startswith(f"codesmatch-{d.get('backend')}-")
            )
        )
        if sound:
            ev["checked"][d["sha256"]] = {
                "file": d["file"],
                "matched": d["matched_exactly"],
                "of": d["int4_linears"],
                "backend": "xnnpack" if fp32_record else d.get("backend"),
                "record": str(f),
            }

api = HfApi()
out, failed = [], False
for m in models:
    repo, rev_src = m["repo"], m["main_sha"]
    ev = evidence.get(slug(m["model"]), {"codes": set(), "checked": {}, "gate_ref": set(), "twins": {}})
    fp32_rebuilt, fp32_kept, no_vulkan = m["model"] in FP32_REBUILT, m["model"] in FP32_KEPT, m["model"] in NO_VULKAN
    info = api.model_info(repo, files_metadata=True)
    files = {s.rfilename: s for s in info.siblings}
    problems, rows = [], {}
    if len(ev["codes"]) != 1 and not fp32_rebuilt:
        problems.append(
            f"codes.pt hash for {slug(m['model'])}: {sorted(ev['codes']) or 'none collected'}, expected exactly one"
        )
    codes_sha = next(iter(ev["codes"])) if len(ev["codes"]) == 1 else None
    for folder in ("xnnpack", "vulkan"):
        if folder == "vulkan" and no_vulkan:
            left = sorted(f for f in files if f.startswith("vulkan/"))
            if left:
                problems.append(f"vulkan/ should be withdrawn, still holds {left}")
            continue
        fp32 = folder == "xnnpack" and (fp32_rebuilt or fp32_kept)
        ptes = sorted(f for f in files if f.startswith(folder + "/") and f.endswith(".pte"))
        reports = sorted(f for f in files if f.startswith(folder + "/export-report-"))
        labels = [r.rsplit("-", 1)[1].removesuffix(".json") for r in reports]
        if sorted(labels) != sorted(WINDOWS):
            problems.append(f"{folder}: report labels {labels}, expected exactly {list(WINDOWS)}")
        named = []
        for r, label in zip(reports, labels, strict=True):
            d = json.loads(Path(hf_hub_download(repo, r, revision=info.sha)).read_text())
            rec, f0 = d["recipe"], d["files"][0]
            if len(d["files"]) != 1:
                problems.append(f"{r}: names {len(d['files'])} files")
            named.append(f0["path"])
            lfs = files.get(f0["path"])
            gate = d.get("gate") if folder == "xnnpack" else (d.get("gate_proxy") or {}).get("gate")
            row = {
                "report": r,
                "file": f0["path"],
                "sha256": f0["sha256"],
                "codes_sha256": rec.get("codes_sha256"),
                "gate": {k: (gate or {}).get(k) for k in GATE_FIELDS},
            }
            rows[(folder, label)] = row
            if d["window"]["context"] != WINDOWS.get(label):
                problems.append(f"{r}: window {d['window']['context']} under label {label}")
            kind = "fp32" if fp32 else "gptq"
            if (
                not f0["path"].startswith(folder + "/")
                or f"-{kind}-" not in f0["path"]
                or not f0["path"].endswith(f"-{label}.pte")
            ):
                problems.append(f"{r}: file name {f0['path']} is not this folder's {kind} {label} file")
            if lfs is None or lfs.lfs is None or lfs.lfs.sha256 != f0["sha256"]:
                problems.append(
                    f"{r}: names {f0['path']} sha {f0['sha256'][:12]}, "
                    f"Hub has {getattr(getattr(lfs, 'lfs', None), 'sha256', None)}"
                )
            if d["source"]["sha"] != rev_src:
                problems.append(f"{r}: source revision {d['source']['sha'][:12]} != {rev_src[:12]}")
            if fp32:
                if rec.get("int4_codes") is not None or rec.get("qmode") != "fp32":
                    problems.append(
                        f"{r}: expected fp32 linears, recipe says {rec.get('qmode')} / {rec.get('int4_codes')}"
                    )
                if fp32_kept:
                    continue  # the deliberate fp32 files built on 1.4.0, untouched by this rollout
            elif rec.get("int4_codes") != "gptq":
                problems.append(f"{r}: int4_codes {rec.get('int4_codes')}")
            if d["toolchain"]["executorch"] != "1.5.1":
                problems.append(f"{r}: ExecuTorch {d['toolchain']['executorch']}")
            measured = (
                (gate or {}).get("passed") is True
                and not (gate or {}).get("override")
                and isinstance((gate or {}).get("graded"), int)
                and gate["graded"] > 0
                and all((gate or {}).get(k) is not None for k in ("agreed", "fp32_mean", "export_mean"))
            )
            if not measured:
                problems.append(f"{r}: gate {'proxy ' if folder == 'vulkan' else ''}is not a measured pass: {gate}")
            if folder == "vulkan":
                proxy = d.get("gate_proxy") or {}
                row["proxy_sha256"] = proxy.get("sha256")
                row["proxy_file"], row["proxy_report"] = proxy.get("file"), proxy.get("report")
            elif len(ev["gate_ref"]) != 1 or (gate or {}).get("reference_sha256") not in ev["gate_ref"]:
                # The verdict must have been measured against this model's own fp32 reference.
                problems.append(
                    f"{r}: gate measured against {str((gate or {}).get('reference_sha256'))[:12]}, "
                    f"this model's gate.json is {sorted(ev['gate_ref']) or 'not collected'}"
                )
            chk = ev["checked"].get(f0["sha256"])
            if chk and (chk["file"] != f0["path"].split("/")[-1] or chk["backend"] != folder):
                problems.append(f"{r}: byte-check record {chk['record']} is for {chk['backend']}/{chk['file']}")
            if fp32:
                if f0["sha256"] not in ev["checked"]:
                    problems.append(f"{r}: {f0['path']} sha {f0['sha256'][:12]} has no ok fp32 byte check")
                else:
                    row["byte_check"] = ev["checked"][f0["sha256"]]
                continue
            if not codes_sha or rec.get("codes_sha256") != codes_sha:
                problems.append(
                    f"{r}: codes_sha256 {str(rec.get('codes_sha256'))[:12]} "
                    f"is not this model's codes.pt {str(codes_sha)[:12]}"
                )
            if f0["sha256"] not in ev["checked"]:
                problems.append(f"{r}: {f0['path']} sha {f0['sha256'][:12]} has no ok byte check in this model's runs")
            else:
                row["byte_check"] = ev["checked"][f0["sha256"]]
        if sorted(named) != ptes or len(set(named)) != len(named):
            problems.append(f"{folder}: reports name {sorted(named)}, the folder holds {ptes}")
        cfg_doc = json.loads(Path(hf_hub_download(repo, f"{folder}/config.json", revision=info.sha)).read_text())
        want = sorted((v["file"].split("/")[-1], v["sha256"], WINDOWS[k[1]]) for k, v in rows.items() if k[0] == folder)
        have = sorted((v["file"], v["sha256"], v["context"]) for v in cfg_doc.get("variants", []))
        if want != have:
            problems.append(f"{folder}/config.json variants {have} differ from the reports {want}")
        for v in cfg_doc.get("variants", []):
            lfs = files.get(f"{folder}/{v['file']}")
            if lfs is None or lfs.size != v.get("size_bytes"):
                problems.append(
                    f"{folder}/config.json: {v['file']} size {v.get('size_bytes')} != Hub {getattr(lfs, 'size', None)}"
                )
    for label in WINDOWS:
        x, v = rows.get(("xnnpack", label)), rows.get(("vulkan", label))
        if v and fp32_kept:
            twin = ev["twins"].get(v.get("proxy_sha256"))
            if not twin:
                problems.append(
                    f"vulkan {label}: proxy sha {str(v.get('proxy_sha256'))[:12]} matches no kept twin report"
                )
                continue
            tg = {k: (twin.get("gate") or {}).get(k) for k in GATE_FIELDS}
            if (
                tg != v["gate"]
                or twin["recipe"].get("codes_sha256") != v["codes_sha256"]
                or twin["window"]["context"] != WINDOWS[label]
            ):
                problems.append(f"vulkan {label}: proxy does not match its kept twin report")
            if twin["files"][0]["sha256"] not in ev["checked"] or v.get("proxy_file") != twin["files"][0]["path"]:
                problems.append(
                    f"vulkan {label}: kept twin {twin['files'][0]['path']} "
                    "lacks a byte check or is not the file the proxy names"
                )
            tgate = twin.get("gate") or {}
            if not (
                tgate.get("passed") is True
                and not tgate.get("override")
                and (tgate.get("graded") or 0) > 0
                and tgate.get("reference_sha256") in ev["gate_ref"]
                and len(ev["gate_ref"]) == 1
            ):
                problems.append(f"vulkan {label}: kept twin is not a measured pass against this model's gate.json")
            continue
        if not (x and v):
            continue
        if v.get("proxy_file") != x["file"] or v.get("proxy_report") != x["report"].split("/")[-1]:
            problems.append(
                f"vulkan {label}: proxy names {v.get('proxy_file')} / {v.get('proxy_report')}, the twin is {x['file']}"
            )
        if v.get("proxy_sha256") != x["sha256"]:
            problems.append(
                f"vulkan {label}: proxy sha {str(v.get('proxy_sha256'))[:12]} "
                f"is not the published xnnpack file {x['sha256'][:12]}"
            )
        if v["gate"] != x["gate"]:
            problems.append(f"vulkan {label}: proxy verdict {v['gate']} differs from the xnnpack report's {x['gate']}")
        if v["codes_sha256"] != x["codes_sha256"]:
            problems.append(f"{label}: xnnpack and vulkan name different codes")
    readme = Path(hf_hub_download(repo, "README.md", revision=info.sha)).read_text()
    cfg = "" if no_vulkan else Path(hf_hub_download(repo, "vulkan/config.json", revision=info.sha)).read_text()
    if "rounded to nearest" in cfg or "round-to-nearest" in cfg:
        problems.append("vulkan/config.json still says rounded to nearest")
    import re

    for link in sorted(set(re.findall(r"(?:xnnpack|vulkan)/[A-Za-z0-9._-]+\.pte", readme))):
        if link not in files:
            problems.append(f"README links {link}, which is not in the repo")
    for ln in readme.splitlines():
        if "rounded to nearest" in ln or "round-to-nearest" in ln:
            problems.append(f"README: {ln.strip()[:200]}")
    failed |= bool(problems)
    out.append(
        {
            "repo": repo,
            "sha": info.sha,
            "ok": not problems,
            "problems": problems,
            "rows": {f"{k[0]}/{k[1]}": v for k, v in rows.items()},
        }
    )
    print(repo, "OK" if not problems else f"{len(problems)} problems", flush=True)
    for p in problems:
        print("   ", p, flush=True)
out_path.write_text(json.dumps(out, indent=1))
sys.exit(1 if failed else 0)
