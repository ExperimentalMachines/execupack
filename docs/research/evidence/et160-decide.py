"""Per gate row: the next-token decision of one .pte on this interpreter's runtime.

    python -I decide.py <file.pte> <tokenizer.json> <rows.json> <out.json>
rows.json: {question: [token ids]}. Reloads the runner per row, as pipeline/gate.py does.
"""

import json
import sys
import tempfile
import types
from importlib import metadata
from pathlib import Path

import torch
from executorch.examples.models.llama.runner.native import NativeLlamaRunner
from executorch.runtime import Runtime

pte, tok, rows_path, out = sys.argv[1:5]
rows = json.loads(Path(rows_path).read_text())
vocab = Runtime.get().load_program(pte).load_method("get_vocab_size").execute([])[0]
params = Path(tempfile.mkdtemp()) / "params.json"
params.write_text(json.dumps({"vocab_size": int(vocab)}))
res = {"executorch": metadata.version("executorch"), "rows": {}}
for q, ids in rows.items():
    runner = NativeLlamaRunner(
        types.SimpleNamespace(
            pte=pte, params=str(params), tokenizer=tok, tokenizer_config=None, max_len=2048, kv_cache=True
        )
    )
    logits = runner.forward(torch.tensor([ids], dtype=torch.long), torch.tensor([0], dtype=torch.long))
    last = (logits[0, -1] if logits.dim() == 3 else logits[-1]).float()
    p = torch.softmax(last, -1)
    top = torch.topk(p, 3)
    res["rows"][q] = {
        "top": int(top.indices[0]),
        "p": float(top.values[0]),
        "top3": [int(i) for i in top.indices],
        "p3": [round(float(v), 5) for v in top.values],
    }
Path(out).write_text(json.dumps(res, indent=1))
print("ok", len(rows), "rows on", res["executorch"])
