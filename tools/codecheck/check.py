"""Are a .pte's int4 codes rounded to nearest, or chosen (GPTQ)? Judged from the bytes alone.

    python check.py <file.pte> <hf-model-dir> <work-dir> [--codes codes.pt] [--json out.json]

Every int4 linear in the file (XNNPACK qbint4 or Vulkan linear_dq8ca_q4gsw) is matched to the
source linear whose fp32 weights its dequantisation is nearest, then measured two ways:

- agree_rtn: the share of its codes equal to torchao's round to nearest (scale amax/7.5,
  codes clamp(round(w/scale), -8, 7)), the codes an unsolved export carries;
- scale_at_amax: the share of its groups whose scale is amax/7.5 (within bf16 rounding).

Calibrated on SmolLM2-135M: the published RTN Vulkan file reads 0.990-0.992 and 1.0 on every
linear; the published GPTQ XNNPACK file reads 0.67-0.87 and 0.09-0.36. A linear is "rtn" at
>= 0.98 on both, "solved" below 0.95 or 0.90, and "ambiguous" otherwise.

With --codes, every matched linear must also carry exactly the solve's codes and scales.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, __import__("os").environ.get("EXECUPACK", "/Users/alpha/execupack-gptq"))
from ptecodes import _unpack_nibbles, load_pte, vulkan_int4, xnnpack_int4  # noqa: E402

from pipeline import convert as pconvert  # noqa: E402
from pipeline import families

RTN_AGREE, RTN_SCALE = 0.98, 0.98
SOLVED_AGREE, SOLVED_SCALE = 0.95, 0.90


def et_state(model_dir: Path, work: Path) -> dict:
    config = json.loads((model_dir / "config.json").read_text())
    plan = families.xnnpack_plan(families.family_for(config), config)
    out = work / f"{model_dir.name}-et.pth"
    if not out.exists():
        pconvert.convert(model_dir, out, plan.converter, config)
    sd = torch.load(out, map_location="cpu", weights_only=True)
    return {k: v.float().numpy() for k, v in sd.items() if v.ndim == 2 and not k.endswith("embeddings.weight")}


def rtn(w, scale, group):
    return np.clip(np.round(w / np.repeat(scale, group, axis=1)), -8, 7)


def classify(agree, at_amax):
    if agree >= RTN_AGREE and at_amax >= RTN_SCALE:
        return "rtn"
    if agree < SOLVED_AGREE or at_amax < SOLVED_SCALE:
        return "solved"
    return "ambiguous"


def decode(t, low_first):
    codes = _unpack_nibbles(t["raw"], low_first).reshape(t["dims"]).astype(np.int32) - 8
    return codes


def analyse(tensors, state, solved_codes=None):
    by_shape = {}
    for k, v in state.items():
        by_shape.setdefault(v.shape, []).append(k)

    def nearest(t, low_first, sample=64):
        deq = decode(t, low_first)[:sample] * np.repeat(t["scale"][:sample], t["group"], axis=1)
        return sorted(
            (float(np.linalg.norm(deq - state[k][:sample]) / np.linalg.norm(state[k][:sample])), k)
            for k in by_shape.get(t["dims"], [])
        )

    # Nibble order, per file: the order whose dequantisation lands nearer a source weight.
    low_first = min((True, False), key=lambda lf: nearest(tensors[0], lf)[0][0])
    rows, used = [], set()
    for t in tensors:
        codes = decode(t, low_first)
        errs = nearest(t, low_first)
        (err, key), runner = errs[0], (errs[1][0] if len(errs) > 1 else float("inf"))
        w = state[key]
        g = t["group"]
        amax = np.abs(w.reshape(w.shape[0], -1, g)).max(-1)
        ratio = t["scale"] / np.maximum(amax / 7.5, 1e-12)
        agree = float((codes == rtn(w, (amax / 7.5).astype(np.float32), g)).mean())
        at_amax = float((np.abs(ratio - 1) < 0.006).mean())
        deq = codes * np.repeat(t["scale"], g, axis=1)
        row = {
            "key": key,
            "dims": list(t["dims"]),
            "rel_err": float(np.linalg.norm(deq - w) / np.linalg.norm(w)),
            "match_margin": float(runner / err) if err > 0 else float("inf"),
            "agree_rtn": agree,
            "agree_own_scale": float((codes == rtn(w, t["scale"], g)).mean()),
            "scale_at_amax": at_amax,
            "class": classify(agree, at_amax),
            "duplicate": key in used,
        }
        if t.get("sums") is not None:
            row["sums_ok"] = bool((codes.reshape(t["dims"][0], -1, g).sum(-1) == t["sums"]).all())
        if solved_codes is not None:
            entry = solved_codes.get(key[: -len(".weight")])
            if entry is None:
                row["codes_exact"] = None
            else:
                q = entry["qdata"].to(torch.int32).numpy()
                s = entry["scale"].float().numpy()
                row["codes_exact"] = bool(q.shape == codes.shape and (q == codes).all())
                row["scales_exact"] = bool(s.shape == t["scale"].shape and np.array_equal(s, t["scale"]))
        rows.append(row)
        used.add(key)
    return low_first, rows, sorted(set(state) - used)


def summarise(path: Path, tensors, low_first, rows, not_int4, solved_codes):
    classes = {c: sum(r["class"] == c for r in rows) for c in ("rtn", "solved", "ambiguous")}
    sums_bad = any(r.get("sums_ok") is False for r in rows)
    duplicates = sum(r["duplicate"] for r in rows)
    if not rows:
        verdict = "no-int4"
    elif not_int4 or duplicates or sums_bad:
        # A verdict about the codes needs every source linear found once, and a Vulkan file's
        # sums consistent with its codes (codex review: a partial file must not read "gptq").
        verdict = "incomplete"
    elif classes["solved"] == len(rows):
        verdict = "gptq"
    elif classes["rtn"] == len(rows):
        verdict = "rtn"
    else:
        verdict = "mixed"
    out = {
        "file": path.name,
        "backends": sorted({"vulkan" if "sums" in t else "xnnpack" for t in tensors}),
        "verdict": verdict,
        "int4_linears": len(rows),
        "classes": classes,
        "source_linears_not_int4": not_int4,
        "duplicate_matches": sum(r["duplicate"] for r in rows),
        "match_margin_min": min((r["match_margin"] for r in rows), default=None),
        "agree_rtn_max": max((r["agree_rtn"] for r in rows), default=None),
        "agree_rtn_median": float(np.median([r["agree_rtn"] for r in rows])) if rows else None,
        "scale_at_amax_max": max((r["scale_at_amax"] for r in rows), default=None),
        "rel_err_max": max((r["rel_err"] for r in rows), default=None),
        "low_nibble_first": low_first,
    }
    if any("sums_ok" in r for r in rows):
        out["vulkan_sums_ok"] = all(r.get("sums_ok", True) for r in rows)
    if solved_codes is not None:
        out["codes_exact"] = sum(r.get("codes_exact") is True for r in rows)
        out["scales_exact"] = sum(r.get("scales_exact") is True for r in rows)
        out["codes_missing"] = sorted(r["key"] for r in rows if r.get("codes_exact") is None)
        out["codes_unused"] = sorted(set(solved_codes) - {r["key"][: -len(".weight")] for r in rows})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pte")
    ap.add_argument("model_dir")
    ap.add_argument("work")
    ap.add_argument("--codes")
    ap.add_argument("--json")
    ap.add_argument("--rows", action="store_true", help="print every linear")
    a = ap.parse_args(argv)
    path = Path(a.pte)
    pte, named = load_pte(path)
    tensors = xnnpack_int4(pte, named) + vulkan_int4(pte, named)
    state = et_state(Path(a.model_dir), Path(a.work))
    solved = torch.load(a.codes, map_location="cpu", weights_only=True) if a.codes else None
    if not tensors:
        low_first, rows, not_int4 = None, [], sorted(state)
    else:
        low_first, rows, not_int4 = analyse(tensors, state, solved)
    out = summarise(path, tensors, low_first, rows, not_int4, solved)
    if a.rows:
        out["rows"] = rows
    text = json.dumps(out, indent=1)
    if a.json:
        Path(a.json).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
