"""Static back-compat: parse every published .pte's program with this interpreter's ExecuTorch schema.

Range-requests the header, then the program flatbuffer only (segments are not fetched), and
parses it with exir._serialize._program._flatbuffer_to_program. Records methods, delegate ids,
delegate calls per method, and whether any instruction or value type is unknown.
"""

import json
import sys
from collections import Counter
from importlib import metadata
from pathlib import Path

import requests
from executorch.exir._serialize._program import _flatbuffer_to_program, _get_extended_header
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers

rows = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
only = sys.argv[3] if len(sys.argv) > 3 else None
headers = build_hf_headers()
results = json.loads(out.read_text()) if out.exists() else {}
sess = requests.Session()


def get(url, a, b):
    h = dict(headers, Range=f"bytes={a}-{b}")
    r = sess.get(url, headers=h, allow_redirects=True, timeout=600)
    r.raise_for_status()
    return r.content


for r in rows:
    key = f"{r['repo']}/{r['path']}"
    if key in results or (only and not r["path"].startswith(only)):
        continue
    url = hf_hub_url(r["repo"], r["path"], revision=r["sha"])
    rec = {"bytes": r["bytes"]}
    try:
        head = get(url, 0, 63)
        rec["file_identifier"] = head[4:8].decode("ascii", "replace")
        eh = _get_extended_header(head)
        size = eh.program_size if eh and eh.is_valid() else r["bytes"]
        rec["program_size"] = size
        if size > 200 * 2**20:
            raise ValueError(f"program flatbuffer is {size} bytes; not fetched")
        prog = _flatbuffer_to_program(get(url, 0, size - 1))
        plans = {}
        for plan in prog.execution_plan:
            instr = Counter(type(i.instr_args).__name__ for c in plan.chains for i in c.instructions)
            plans[plan.name] = {
                "delegates": sorted({d.id for d in plan.delegates}),
                "delegate_calls": instr.get("DelegateCall", 0),
                "instructions": dict(instr),
                "values": dict(Counter(type(v.val).__name__ for v in plan.values)),
            }
        rec.update(ok=True, version=prog.version, methods=sorted(plans), plans=plans, segments=len(prog.segments))
    except Exception as e:
        rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:500])
    results[key] = rec
    out.write_text(json.dumps(results, indent=0))
    print(("ok  " if rec.get("ok") else "FAIL") + f" {key} {rec.get('error', '')}", flush=True)
print(
    "DONE",
    metadata.version("executorch"),
    sum(1 for v in results.values() if v.get("ok")),
    "of",
    len(results),
    flush=True,
)
