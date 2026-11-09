"""Does a built .pte carry exactly the solve's codes? Read from the file's bytes, before publish.

    python codesmatch.py <file.pte> <codes.pt> [--allow-unused output]

Every int4 linear in the file (XNNPACK qbint4 or Vulkan linear_dq8ca_q4gsw) must equal one
entry of codes.pt exactly: the same int4 codes and the same scales (bf16 in an XNNPACK file,
fp32 in a Vulkan one; the solve picks scales bf16 can hold, so both are exact). Every codes.pt
entry must be used once, except names given with --allow-unused (an int8 output head has no
int4 codes to carry). Matching is by content, so no fp32 source and no layer names are needed.

Exit 0 when the file carries the codes and nothing else at int4; 1 otherwise, with the reasons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ptecodes import _unpack_nibbles, load_pte, vulkan_int4, xnnpack_int4  # noqa: E402

# Execution order inside one transformer layer of ExecuTorch's llama_transformer (attention
# wq, wk, wv, then wo; feed-forward w2(silu(w1(x)) * w3(x))). Both delegates serialize their
# int4 linears in this order, layers ascending and the head last, so a file whose matched
# names follow it carries each module's codes where that module's computation reads them.
LAYER_ORDER = [
    "attention.wq",
    "attention.wk",
    "attention.wv",
    "attention.wo",
    "feed_forward.w1",
    "feed_forward.w3",
    "feed_forward.w2",
]


def expected_order(names):
    """codes.pt names in execution order, or None if a name falls outside the known layout."""
    import re

    def key(name):
        if name == "output":
            return (10**9, 0)
        m = re.fullmatch(r"layers\.(\d+)\.(.+)", name)
        if not m or m.group(2) not in LAYER_ORDER:
            raise KeyError(name)
        return (int(m.group(1)), LAYER_ORDER.index(m.group(2)))

    try:
        return sorted(names, key=key)
    except KeyError:
        return None


def digest(codes: np.ndarray) -> str:
    c = np.ascontiguousarray(codes.astype(np.int8))
    return hashlib.sha256(repr(c.shape).encode() + c.tobytes()).hexdigest()


def bf16(x: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(x)).to(torch.bfloat16).float().numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pte")
    ap.add_argument("codes")
    ap.add_argument("--allow-unused", action="append", default=[])
    a = ap.parse_args(argv)

    solved = torch.load(a.codes, map_location="cpu", weights_only=True)
    index = {}
    for name, entry in solved.items():
        index.setdefault(digest(entry["qdata"].to(torch.int8).numpy()), []).append(name)

    pte, named = load_pte(Path(a.pte))
    xnn, vk = xnnpack_int4(pte, named), vulkan_int4(pte, named)
    tensors = xnn + vk
    problems, used, matched = [], {}, 0
    if not tensors:
        problems.append("the file has no int4 linear")
    # Nibble order is a property of the backend's packing: decide it once, on whichever order
    # finds the first tensor in codes.pt.
    low_first = True
    if tensors:
        first = tensors[0]
        hits = {
            lf: digest(_unpack_nibbles(first["raw"], lf).reshape(first["dims"]).astype(np.int16) - 8) in index
            for lf in (True, False)
        }
        low_first = hits[True] or not hits[False]
    # The names the file should hold, in execution order: a tensor whose codes appear under
    # several names takes the one expected at its position (codex review, round 2).
    want_order = expected_order([n for n in solved if n not in a.allow_unused])
    for i, t in enumerate(tensors):
        codes = _unpack_nibbles(t["raw"], low_first).reshape(t["dims"]).astype(np.int16) - 8
        names = [n for n in index.get(digest(codes), []) if n not in used]
        if not names:
            problems.append(f"int4 tensor {i} {t['dims']} matches no unused codes.pt entry")
            continue
        have = t["scale"]

        def stored(n, t=t):
            w = solved[n]["scale"].float().numpy()
            return bf16(w) if "sums" not in t else w

        exact = [n for n in names if stored(n).shape == have.shape and np.array_equal(stored(n), have)]
        expected_here = want_order[i] if want_order and i < len(want_order) else None
        pool = exact or names
        name = expected_here if expected_here in pool else pool[0]
        if not exact:
            want = stored(name)
            diff = float(np.abs(want - have).max()) if want.shape == have.shape else None
            problems.append(f"{name}: scales differ (max abs diff {diff})")
        if t.get("sums") is not None and not (codes.reshape(t["dims"][0], -1, t["group"]).sum(-1) == t["sums"]).all():
            problems.append(f"{name}: Vulkan per-group code sums disagree with the codes")
        used[name] = i
        matched += 1
    order = [n for n, _ in sorted(used.items(), key=lambda kv: kv[1])]
    if a.allow_unused:
        want_order = expected_order([n for n in solved if n not in a.allow_unused or n in used])
    if want_order is None:
        problems.append("codes.pt has a module outside the known layer layout; order not checkable")
    elif order != want_order:
        first = next(
            (k for k, (x, y) in enumerate(zip(order, want_order, strict=False)) if x != y),
            min(len(order), len(want_order)),
        )
        problems.append(
            f"linears are not in execution order: position {first} holds "
            f"{order[first] if first < len(order) else None}, expected "
            f"{want_order[first] if first < len(want_order) else None}"
        )
    unused = sorted(set(solved) - set(used) - set(a.allow_unused))
    if unused:
        problems.append(f"codes.pt entries not in the file: {unused[:8]}{' ...' if len(unused) > 8 else ''}")
    sha = hashlib.sha256()
    with open(a.pte, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            sha.update(chunk)
    result = {
        "file": Path(a.pte).name,
        "sha256": sha.hexdigest(),
        "backend": "vulkan" if vk else "xnnpack",
        "int4_linears": len(tensors),
        "matched_exactly": matched,
        "codes_entries": len(solved),
        "unused_allowed": sorted(set(a.allow_unused) & (set(solved) - set(used))),
        "low_nibble_first": low_first,
        "order_checked": want_order is not None,
        "ok": not problems,
        "problems": problems,
    }
    print(json.dumps(result, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
