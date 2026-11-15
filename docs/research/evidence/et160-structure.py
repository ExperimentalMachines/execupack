"""Structure of an XNNPACK .pte: operators, instructions, delegate calls, XNNPACK node and value types."""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/codecheck"))
from ptecodes import ET, _flatc_json, delegates, load_pte  # noqa: E402

pte, named = load_pte(Path(sys.argv[1]))
prog = pte.program
plan = next(p for p in prog.execution_plan if p.name == "forward")
ops = [o.name + ("." + o.overload if o.overload else "") for o in plan.operators]
instr = Counter(type(i.instr_args).__name__ for c in plan.chains for i in c.instructions)
kernels = Counter(
    ops[i.instr_args.op_index]
    for c in plan.chains
    for i in c.instructions
    if type(i.instr_args).__name__ == "KernelCall"
)
nodes, values, sizes = Counter(), Counter(), []
schema = ET / "backends/xnnpack/serialization/schema.fbs"
for name, did, blob in delegates(pte):
    if name != "forward" or did != "XnnpackBackend":
        continue
    from executorch.backends.xnnpack.serialization.xnnpack_graph_serialize import XNNHeader

    h = XNNHeader.from_bytes(blob[: XNNHeader.EXPECTED_LENGTH])
    g = _flatc_json(schema, blob[h.flatbuffer_offset : h.flatbuffer_offset + h.flatbuffer_size])
    sizes.append(len(blob))
    nodes.update(n.get("xnode_union_type") for n in g.get("xnodes", []))
    values.update(v.get("xvalue_union_type") for v in g.get("xvalues", []))
out = {
    "operators": sorted(set(ops)),
    "instructions": dict(instr),
    "kernel_calls": dict(kernels),
    "xnnpack_delegates": len(sizes),
    "xnnpack_blob_bytes": sum(sizes),
    "xnnpack_nodes": dict(nodes),
    "xnnpack_values": dict(values),
    "named_data": len(named),
    "named_bytes": sum(len(v) for v in named.values()),
    "values": dict(Counter(type(v.val).__name__ for v in plan.values)),
}
print(json.dumps(out, indent=1, sort_keys=True))
