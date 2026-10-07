"""Runs ExecuTorch's Qualcomm llama script with the one upstream fix the pinned wheel lacks.

    python -m pipeline.qnn_launch --decoder_model qwen3-0_6b ...   (export_qnn.llama_command)

The script calibrates on wikitext through lm_eval, using ExecuTorch's EagerEvalWrapper as the
model (examples/qualcomm/oss_scripts/llama/dataset/loaders.py, collect_lm_eval_tokens). In
1.5.1, EagerEvalWrapper.prefix_token_id returns the tokenizer's bos_id whenever the attribute
exists, even when it is None, and Qwen3's tokenizer_config has no bos_token. lm_eval then
puts None at the head of the first rolling window and fails building the tensor ("'NoneType'
object cannot be interpreted as an integer"): every Qwen3 window failed this way on
2026-10-07. ExecuTorch main falls back to the end-of-text token when bos_id is None
(examples/models/llama/evaluate/eager_eval.py), as lm_eval's own HFLM does; this is that
property, set before the script runs. Drop this module when the ExecuTorch pin has the fix.
"""

from __future__ import annotations

import runpy

SCRIPT = "executorch.examples.qualcomm.oss_scripts.llama.llama"


def _prefix_token_id(self):
    bos_id = getattr(self._tokenizer, "bos_id", None)
    if bos_id is not None:
        return bos_id
    return self.eot_token_id


def patch() -> None:
    from executorch.examples.models.llama.evaluate import eager_eval

    eager_eval.EagerEvalWrapper.prefix_token_id = property(_prefix_token_id)


if __name__ == "__main__":
    patch()
    runpy.run_module(SCRIPT, run_name="__main__", alter_sys=True)
