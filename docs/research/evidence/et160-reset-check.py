"""LFM2 conv state: does a prompt at position 0 read zeros whatever the previous prompt left?

    python -I reset_check.py <file.pte> <tokenizer.json>
Prompt B after prompt A in one runner, against prompt B in a fresh one: equal logits mean the
state was cleared in the graph (pipeline/lfm2_state.py).
"""

import json
import sys
import tempfile
import types
from pathlib import Path

import torch
from executorch.examples.models.llama.runner.native import NativeLlamaRunner
from executorch.runtime import Runtime
from tokenizers import Tokenizer

pte, tok = sys.argv[1], sys.argv[2]
prog = Runtime.get().load_program(pte)
methods = sorted(prog.method_names)
reset = prog.load_method("get_state_reset_at_zero").execute([])[0] if "get_state_reset_at_zero" in methods else None
vocab = int(prog.load_method("get_vocab_size").execute([])[0])
del prog
params = Path(tempfile.mkdtemp()) / "p.json"
params.write_text(json.dumps({"vocab_size": vocab}))
mk = lambda: NativeLlamaRunner(
    types.SimpleNamespace(
        pte=pte, params=str(params), tokenizer=tok, tokenizer_config=None, max_len=2048, kv_cache=True
    )
)
t = Tokenizer.from_file(tok)
a = t.encode("Write a haiku about the sea and the moon over the harbor.", add_special_tokens=False).ids
b = t.encode("What is the capital of Italy? Answer in one word.", add_special_tokens=False).ids


def f(r, ids):
    out = r.forward(torch.tensor([ids]), torch.tensor([0]))
    last = out[0, -1] if out.dim() == 3 else out[-1]  # [1, vocab] when only the last position comes back
    assert last.dim() == 1 and last.numel() == vocab, tuple(out.shape)
    return last.float()


r = mk()
f(r, a)
after_a = f(r, b)
fresh = f(mk(), b)
print(
    json.dumps(
        {
            "get_state_reset_at_zero": reset if reset is None else bool(reset),
            "max_abs_diff_b_after_a_vs_fresh": float((after_a - fresh).abs().max()),
            "top_after_a": int(after_a.argmax()),
            "top_fresh": int(fresh.argmax()),
            "a_alone_vs_b_fresh": float((f(mk(), a) - fresh).abs().max()),
        }
    )
)
