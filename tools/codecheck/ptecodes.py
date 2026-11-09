"""Pull the int4 linears (codes and scales) out of an ExecuTorch .pte, for XNNPACK and Vulkan.

Reads only the file: no runtime, no reports. Each linear comes back as
{"codes": int8 [out, in] in -8..7, "scale": float32 [out, in // group]} so it can be compared
with the fp32 source weights.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import executorch
import numpy as np
from executorch.exir._serialize._flatbuffer import _get_flatc_path
from executorch.exir._serialize._program import deserialize_pte_binary

ET = Path(list(executorch.__path__)[0])


def _flatc_json(schema: Path, blob: bytes) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        bin_path = Path(tmp) / "g.bin"
        bin_path.write_bytes(blob)
        subprocess.run(
            [_get_flatc_path(), "--json", "--strict-json", "--raw-binary", "-o", tmp, str(schema), "--", str(bin_path)],
            check=True,
            capture_output=True,
        )
        return json.loads((Path(tmp) / "g.json").read_text())


def load_pte(path: Path):
    data = Path(path).read_bytes()
    pte = deserialize_pte_binary(data)
    named = {}
    if pte.named_data is not None:
        store = pte.named_data
        for key, entry in store.pte_data.items():
            named[key] = bytes(store.buffers[entry.buffer_index])
    return pte, named


def delegates(pte):
    """Delegate payloads in the order the program calls them (its DelegateCall instructions),
    not the order of the delegates table: position in this list is position in execution."""
    from executorch.exir.schema import DelegateCall

    program = pte.program
    out = []
    for plan in program.execution_plan:
        for chain in plan.chains:
            for instr in chain.instructions:
                call = instr.instr_args
                if not isinstance(call, DelegateCall):
                    continue
                d = plan.delegates[call.delegate_index]
                loc = d.processed
                if loc.location.name != "INLINE":
                    raise RuntimeError(f"segment delegate data not restored for {d.id}")
                out.append((plan.name, d.id, bytes(program.backend_delegate_data[loc.index].data)))
    return out


if __name__ == "__main__":
    import sys

    pte, named = load_pte(Path(sys.argv[1]))
    print("named data keys:", len(named), list(named)[:3])
    for plan, did, blob in delegates(pte):
        print(plan, did, len(blob), blob[4:8])


def _bf16_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (u16 << 16).view(np.float32)


def _unpack_nibbles(raw: bytes, low_first: bool) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8)
    lo, hi = (b & 0x0F).astype(np.int16), (b >> 4).astype(np.int16)
    pair = (lo, hi) if low_first else (hi, lo)
    return np.stack(pair, axis=-1).reshape(-1)


def xnnpack_int4(pte, named) -> list[dict]:
    """The qbint4 weight of every XNNPACK fully-connected node, in execution order: delegate
    calls in program order, nodes in graph order, each node's filter resolved by value id.
    A qbint4 value that no fully-connected node uses is an error, not a silent extra."""
    from executorch.backends.xnnpack.serialization.xnnpack_graph_serialize import XNNHeader

    schema = ET / "backends/xnnpack/serialization/schema.fbs"
    out = []
    for plan, did, blob in delegates(pte):
        if did != "XnnpackBackend":
            continue
        h = XNNHeader.from_bytes(blob[: XNNHeader.EXPECTED_LENGTH])
        g = _flatc_json(schema, blob[h.flatbuffer_offset : h.flatbuffer_offset + h.flatbuffer_size])
        consts = g.get("constant_data", [])

        def read(idx, consts=consts, h=h, blob=blob):
            c = consts[idx]
            if c.get("named_key"):
                return named[c["named_key"]]
            start = h.constant_data_offset + int(c["offset"])
            return blob[start : start + int(c["size"])]

        by_id, int4_ids = {}, set()
        for v in g.get("xvalues", []):
            u = v.get("xvalue_union", {})
            tv = u.get("tensor_value", u)
            if "id_out" in tv:
                by_id[tv["id_out"]] = v
            if v.get("xvalue_union_type") == "XNNQuantizedTensorValue" and tv.get("datatype") == "xnn_datatype_qbint4":
                int4_ids.add(tv["id_out"])
        used = []
        for node in g.get("xnodes", []):
            if node.get("xnode_union_type") != "XNNFullyConnected":
                continue
            fid = node["xnode_union"]["filter_id"]
            if fid not in int4_ids:
                continue
            u = by_id[fid]["xvalue_union"]
            tv, qp = u["tensor_value"], u["quant_params"]
            if u.get("quant_params_type") != "PerChannelGroupQuant":
                raise RuntimeError(f"qbint4 filter {fid} without per-channel-group quantization")
            rows, cols = tv["dims"]
            group = qp["group_size"]
            raw = read(tv["constant_buffer_idx"])
            assert len(raw) * 2 == rows * cols, (len(raw), rows, cols)
            scale = _bf16_to_f32(read(qp["scale_buffer_idx"])).reshape(rows, cols // group)
            out.append({"plan": plan, "dims": (rows, cols), "group": group, "raw": raw, "scale": scale})
            used.append(fid)
        if sorted(used) != sorted(int4_ids):
            raise RuntimeError(f"XNNPACK delegate: int4 values {sorted(int4_ids)} used by FC nodes as {sorted(used)}")
    return out


def vulkan_int4(pte, named) -> list[dict]:
    """Every et_vk.linear_dq8ca_q4gsw weight: packed nibbles [out, in/2], fp32 scales [groups, out]
    (returned transposed to [out, groups]), and the int32 per-group code sums the shader uses."""
    from executorch.backends.vulkan.serialization.vulkan_graph_serialize import VulkanDelegateHeader as H

    schema = ET / "backends/vulkan/serialization/schema.fbs"
    out = []
    for plan, did, blob in delegates(pte):
        if did != "VulkanBackend":
            continue
        h = H.from_bytes(blob[: H.EXPECTED_LENGTH])
        g = _flatc_json(schema, blob[h.flatbuffer_offset : h.flatbuffer_offset + h.flatbuffer_size])
        vals, consts = g["values"], g.get("constants", [])

        def tensor(i, vals=vals, consts=consts, blob=blob, h=h):
            v = vals[i]["value"]
            c = consts[v["constant_id"]]
            raw = (
                named[c["named_key"]]
                if c.get("named_key")
                else blob[h.bytes_offset + int(c["offset"]) :][: int(c["length"])]
            )
            return v, raw

        for call in g["chain"]:
            if call["name"] != "et_vk.linear_dq8ca_q4gsw.default":
                continue
            a = call["args"]
            wv, wraw = tensor(a[3])
            # One linear (the head, in SmolLM2) computes its sums at run time: no constant to read.
            has_sums = vals[a[4]]["value"].get("constant_id", -1) >= 0
            sv, sraw = tensor(a[4]) if has_sums else (None, None)
            cv, craw = tensor(a[5])
            group = vals[a[6]]["value"]["int_val"]
            rows, half = wv["dims"]
            assert wv["datatype"] == "UINT8" and cv["datatype"] == "FLOAT32"
            scale = np.frombuffer(craw, dtype="<f4").reshape(cv["dims"]).T.copy()
            sums = np.frombuffer(sraw, dtype="<i4").reshape(sv["dims"]).T.copy() if has_sums else None
            out.append(
                {"plan": plan, "dims": (rows, half * 2), "group": group, "raw": wraw, "scale": scale, "sums": sums}
            )
    return out
