# codecheck: is a published .pte what its report says it is?

Byte-level checks used for the 2026-11-09 GPTQ rollout (docs/PLAN.md, Known limits; research
findings 43-45). Not wired into the pipeline or CI yet.

- `ptecodes.py`: reads every int4 linear out of a `.pte` (ExecuTorch 1.5.1), in execution order:
  XNNPACK `qbint4` filters of fully-connected nodes (delegate calls in program order), Vulkan
  `et_vk.linear_dq8ca_q4gsw` weights, scales and per-group code sums.
- `codesmatch.py <file.pte> <codes.pt>`: exit 0 only if every int4 linear equals the solve's codes
  and scales exactly, every codes.pt entry is used, and the order is the transformer's execution
  order (wq, wk, wv, wo, w1, w3, w2 per layer, head last). Ran before every publish.
- `check.py <file.pte> <hf-model-dir> <work>`: no codes needed; matches each int4 linear to the
  fp32 source and calls it round-to-nearest or solved (thresholds calibrated on SmolLM2-135M).
- `scan.py`, `make_tasks.py`: `check.py` over every XNNPACK/Vulkan file in an org, sharded.
- `verify_hub.py`: ties every Hub file of the rolled-out repos to its byte check, codes hash,
  gate reference and (Vulkan) gate proxy; per-model exceptions are listed at the top.

Paths in `check.py` (`EXECUPACK`) and in `verify_hub.py`'s inputs assume the rollout's layout.
