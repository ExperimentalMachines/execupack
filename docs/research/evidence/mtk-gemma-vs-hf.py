"""fp32 MediaTek Gemma graph against Hugging Face, before any quantization (docs/research, finding 46).

    python mtk-gemma-vs-hf.py <gemma3|gemma2> <execupack checkout> <examples/mediatek dir> <out.json>

MediaTek's chunk classes (examples/mediatek/models/llm_models/modeling_gemma.py) run on the CPU
in fp32 with mtk_converter stubbed (as in docs/research/evidence/mtk-conv-state-and-padding.py),
from the config execupack's families.mtk_model_config writes. They see one prompt in one feed
with an empty cache of WINDOW slots, masked the way MediaTek's MaskBuilder masks: 0 where
visible, MASK where not, and for Gemma 3's sliding layers a second mask that also hides keys
more than sliding_window tokens back.

Scored against transformers' fp32 model (eager attention, which applies Gemma 2's softcapping)
on a ~900-token passage (every position: KL, top-1, perplexity) and on the decision gate's
16 prompts (the last position). Variants:
  gemma3: RoPE at 10000 on every layer (MediaTek's master table), at 1e6 on every layer (its
          runner sample's rot_emb_base), and per layer (10000 sliding, 1e6 global: what a
          two-table graph would compute); each with mask -100 and with -1e9
  gemma2: RoPE at 10000 (every layer, as HF), mask -100 and -1e9
"""

import json
import os
import sys
import types
from pathlib import Path

for name in ("mtk_converter", "mtk_converter.python", "mtk_converter.python.converters", "mtk_neuron"):
    sys.modules[name] = types.ModuleType(name)
pytorch_mod = types.ModuleType("mtk_converter.python.converters.pytorch")
pytorch_mod.importer_v2 = types.ModuleType("importer_v2")
sys.modules["mtk_converter.python.converters.pytorch"] = pytorch_mod

which, checkout, examples, out_path = sys.argv[1], Path(sys.argv[2]).resolve(), os.path.abspath(sys.argv[3]), Path(sys.argv[4]).resolve()
sys.path.insert(0, str(checkout))
import torch  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from pipeline import families, gate  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "8")))
REPO = {"gemma3": "google/gemma-3-1b-it", "gemma2": "google/gemma-2-2b-it"}[which]
SHA = HfApi().model_info(REPO).sha
snap = Path(snapshot_download(REPO, revision=SHA, allow_patterns=["*.json", "*.safetensors", "*.model", "README.md"]))
stage = Path(os.environ.get("STAGE", f"/tmp/gemma-stage-{which}"))
stage.mkdir(parents=True, exist_ok=True)
for f in snap.iterdir():
    if f.name != "config.json" and not (stage / f.name).exists():
        (stage / f.name).symlink_to(f.resolve())
hf_config = json.loads((snap / "config.json").read_text())
(stage / "config.json").write_text(json.dumps(families.mtk_model_config(hf_config), indent=2))

os.chdir(examples)
sys.path.insert(0, examples)
from aot_utils.llm_utils.utils import get_embedding_layer, get_master_rot_emb, load_checkpoints, resolve_model_classes  # noqa: E402

config, weight_dir, _, chunk_class = resolve_model_classes(str(stage / "config.json"))
print("MediaTek config:", type(config).__name__, chunk_class.__name__, flush=True)
state_dict = load_checkpoints(weight_dir)
embedding = get_embedding_layer(config, weight_dir, state_dict)
n_layers = config.num_hidden_layers
chunk = chunk_class(config, n_layers, chunk_idx=0, dtype=torch.float32, include_tail=True, jit_trace=True)
chunk = chunk.load_weights(state_dict, 0).eval()
del state_dict
head_dim, kv_heads = int(config.head_dim), config.num_key_value_heads

tok = AutoTokenizer.from_pretrained(snap)
readme = (snap / "README.md").read_text() if (snap / "README.md").exists() else ""
body = readme.split("---", 2)[-1]
passage = tok(body, add_special_tokens=True)["input_ids"][:900]
# Gemma 2's template refuses a system turn; fold it into the first user turn, as Gemma 3's
# template does itself (the app has no Gemma 2 template to copy).
_apply = tok.apply_chat_template


def _folded(messages, *args, **kwargs):
    if messages and messages[0]["role"] == "system":
        first, rest = messages[1], messages[2:]
        messages = [{"role": "user", "content": messages[0]["content"] + "\n\n" + first["content"]}, *rest]
    return _apply(messages, *args, **kwargs)


if which == "gemma2":
    tok.apply_chat_template = _folded
prompts = gate.prompts(tok)
gate_ids = {q: tok(t, add_special_tokens=False)["input_ids"] for q, t in prompts.items()}
WINDOW = max(len(passage), *map(len, gate_ids.values())) + 8
print("passage", len(passage), "tokens; gate prompts", min(map(len, gate_ids.values())), "-",
      max(map(len, gate_ids.values())), "; window", WINDOW, flush=True)


def rot_table(base):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(WINDOW + 8, dtype=torch.float32), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return torch.stack((emb.cos(), emb.sin()))[None]  # (1, 2, len, head_dim)


master = get_master_rot_emb(config, dtype=torch.float32)
for base in (10000.0, 1e6):
    ref = rot_table(base)
    n = min(ref.shape[2], master.shape[2])
    print(f"MediaTek master table vs base {base:g}: max |diff| {float((master[:, :, :n] - ref[:, :, :n]).abs().max()):.2e}",
          flush=True)


def masks(n, mask_value, sliding):
    full = torch.full((1, 1, n, WINDOW + n), mask_value)
    local = full.clone()
    for r in range(n):
        full[0, 0, r, WINDOW : WINDOW + r + 1] = 0.0
        first = max(0, r - sliding + 1) if sliding else 0
        local[0, 0, r, WINDOW + first : WINDOW + r + 1] = 0.0
    return full, local


def run(ids, mask_value, rope):
    n = len(ids)
    hidden = embedding(torch.tensor([ids], dtype=torch.int32))
    cache = [torch.zeros(1, kv_heads, WINDOW, head_dim) for _ in range(2 * n_layers)]
    if which == "gemma2":
        full, _ = masks(n, mask_value, 0)
        pos = rope["global"][:, :, :n]
        with torch.no_grad():
            return chunk(hidden, full, pos, *cache)[0][0]
    full, local = masks(n, mask_value, config.sliding_window_attention_size)
    # The patched Gemma3ModelChunk (mediatek-gemma.patch) takes the global table, then the
    # sliding one; a single-table variant hands it the same table twice.
    pos_g = rope["global"][:, :, :n]
    pos_l = rope.get("local", rope["global"])[:, :, :n]
    with torch.no_grad():
        return chunk(hidden, full, local, pos_g, pos_l, *cache)[0][0]


hf = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32, attn_implementation="eager").eval()
with torch.no_grad():
    hf_passage = hf(torch.tensor([passage])).logits[0].double()
    hf_gate = {q: hf(torch.tensor([ids])).logits[0, -1].double() for q, ids in gate_ids.items()}
del hf


def compare(a, b, ids=None):
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    kl = (la.exp() * (la - lb)).sum(-1)
    out = {"kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "top1": float((a.argmax(-1) == b.argmax(-1)).double().mean())}
    if ids is not None:
        target = torch.tensor(ids[1:])
        out["ppl_hf"] = float(torch.exp(-la[:-1].gather(1, target[:, None]).mean()))
        out["ppl_mtk"] = float(torch.exp(-lb[:-1].gather(1, target[:, None]).mean()))
    return out


def graded(mtk_gate):
    ref, measured = {"rows": {}}, {}
    for q in gate_ids:
        p, m = torch.softmax(hf_gate[q], -1), torch.softmax(mtk_gate[q], -1)
        t = int(p.argmax())
        ref["rows"][q] = {"token": t, "p": float(p[t])}
        measured[q] = {"top": int(m.argmax()), "p": float(m[t])}
    v = gate.check(ref, measured)
    v.pop("rows")
    return {k: v[k] for k in ("passed", "agreed", "graded", "fp32_mean", "export_mean", "quiet_rows_answered_like_fp32",
                              "quiet_rows", "problems")}


if which == "gemma2":
    ropes = [{"name": "10000 everywhere", "global": rot_table(10000.0)}]
else:
    ropes = [
        {"name": "10000 everywhere", "global": rot_table(10000.0)},
        {"name": "1e6 everywhere", "global": rot_table(1e6)},
        {"name": "per-layer", "local": rot_table(float(config.rope_local_base_freq)), "global": rot_table(1e6)},
    ]
results = {"repo": REPO, "sha": SHA, "passage_tokens": len(passage), "variants": []}
for rope in ropes:
    for mask_value in (-100.0,):
        mtk_passage = run(passage, mask_value, rope).double()
        mtk_gate = {q: run(ids, mask_value, rope)[-1].double() for q, ids in gate_ids.items()}
        row = {"rope": rope["name"], "mask": mask_value, "passage": compare(hf_passage, mtk_passage, passage),
               "gate_last_token": compare(torch.stack(list(hf_gate.values())), torch.stack(list(mtk_gate.values()))),
               "gate": graded(mtk_gate)}
        results["variants"].append(row)
        print(json.dumps(row), flush=True)
out_path.write_text(json.dumps(results, indent=1))
