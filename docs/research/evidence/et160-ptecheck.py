"""Run one XNNPACK .pte on this interpreter's ExecuTorch runtime and record what it does.

    python -I ptecheck.py <file.pte> <tokenizer.json> <out.json> [--decode N]

Loads through NativeLlamaRunner (as pipeline/gate.py does, so the custom ops are registered),
prefills a fixed prompt from position 0, then decodes greedily one token at a time. Records the
program's methods and metadata, the prefill logits (as .npy beside out.json), and the decoded
ids, so two runtimes can be compared on the same file.
"""

import json
import sys
import tempfile
import time
import types
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

PROMPT = (
    "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg "
    "combined movable metal type with an oil-based ink and a screw press. Explain in two sentences "
    "why this mattered:"
)


def main():
    pte, tok_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    decode = int(sys.argv[sys.argv.index("--decode") + 1]) if "--decode" in sys.argv else 24
    from executorch.examples.models.llama.runner.native import NativeLlamaRunner
    from executorch.runtime import Runtime
    from tokenizers import Tokenizer

    record = {"executorch": metadata.version("executorch"), "pte": str(pte)}
    started = time.time()
    try:
        program = Runtime.get().load_program(str(pte))
        record["methods"] = sorted(program.method_names)
        meta = {}
        for name in record["methods"]:
            if name in ("forward",):
                continue
            try:
                values = program.load_method(name).execute([])
                meta[name] = [v.tolist() if hasattr(v, "tolist") else v for v in values]
            except Exception as error:  # noqa: BLE001
                meta[name] = f"error: {error}"
        record["metadata"] = meta
        del program

        ids = Tokenizer.from_file(str(tok_path)).encode(PROMPT, add_special_tokens=False).ids
        vocab = meta.get("get_vocab_size", [None])[0]
        params = Path(tempfile.mkdtemp()) / "params.json"
        params.write_text(json.dumps({"vocab_size": vocab}))
        runner = NativeLlamaRunner(
            types.SimpleNamespace(
                pte=str(pte),
                params=str(params),
                tokenizer=str(tok_path),
                tokenizer_config=None,
                max_len=2048,
                kv_cache=True,
            )
        )
        logits = runner.forward(torch.tensor([ids], dtype=torch.long), torch.tensor([0], dtype=torch.long))
        last = (logits[0, -1] if logits.dim() == 3 else logits[-1]).float()
        np.save(out.with_suffix(".npy"), last.numpy())
        generated = [int(last.argmax())]
        pos = len(ids)
        for _ in range(decode - 1):
            step = runner.forward(
                torch.tensor([[generated[-1]]], dtype=torch.long), torch.tensor([pos], dtype=torch.long)
            )
            last = (step[0, -1] if step.dim() == 3 else step[-1]).float()
            generated.append(int(last.argmax()))
            pos += 1
        record.update(prompt_tokens=len(ids), generated=generated, ok=True)
    except Exception as error:  # noqa: BLE001
        record.update(ok=False, error=f"{type(error).__name__}: {error}")
    record["seconds"] = round(time.time() - started, 1)
    out.write_text(json.dumps(record, indent=1))
    print(json.dumps({k: record.get(k) for k in ("ok", "error", "seconds", "prompt_tokens")}))


if __name__ == "__main__":
    main()
