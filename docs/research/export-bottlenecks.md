# Where export time goes (Qwen3-0.6B, 2026-09-12/13)

Question: why do the NPU exports take 40-100 minutes on GitHub's hosted runners, and what is
the main bottleneck?

**Answer, with the evidence below:** for the Qualcomm (QNN) exports, one calibration pass
takes 38-61% of the export and the HTP compile 32-48%; together 87-93%. Inside that pass,
ExecuTorch's recipe runs SeqMSE for this model, which by operation count (estimate, not
measured separately) is about 99% of the pass's arithmetic. How the pass scales with the
window is not settled: on the same chip it grew 3.9x from 2k to 4k, but the evidence cannot
separate window scaling from runner-to-runner speed and swapping.

Every number here comes from the files in [`evidence/`](evidence/) or from ExecuTorch's
source at commit `3dd7ccd1d863fad22639dd2d918ae34a41ce45f0` (v1.4.0). Measured values are
marked as measured; estimates show their arithmetic.

## 1. Measured stage times (QNN, Qwen3-0.6B)

Host: GitHub `ubuntu-latest`, 4 vCPU, 16,766,414,848 B RAM, 25,769,799,680 B swap added by
`scripts/ci/prepare-host.sh`. "Export" is the Qualcomm script's run time
(`host.export_seconds` in the export report). Stage times are ExecuTorch's own log lines.

| Run | Chip | Window | Calibration loop | Other quantize | HTP compile | Rest | Export |
|---|---|---|---|---|---|---|---|
| [34706269492](https://github.com/ExperimentalMachines/execupack/actions/runs/34706269492) | SM8650 | 2,048 | 1,761 s (48.4%) | 280 s (7.7%) | 1,538 s (42.3%) | 56 s (1.6%) | 3,637 s |
| [34706269492](https://github.com/ExperimentalMachines/execupack/actions/runs/34706269492) | SM8750 | 2,048 | 936 s (38.5%) | 257 s (10.5%) | 1,172 s (48.2%) | 69 s (2.8%) | 2,434 s |
| [34741690464](https://github.com/ExperimentalMachines/execupack/actions/runs/34741690464) | SM8750 | 4,096 | 3,674 s (61.4%) | 328 s (5.5%) | 1,905 s (31.8%) | 80 s (1.3%) | 5,987 s |

Evidence: [`qnn-run6-2k-sm8650-timings.log`](evidence/qnn-run6-2k-sm8650-timings.log),
[`qnn-run6-2k-sm8750-timings.log`](evidence/qnn-run6-2k-sm8750-timings.log),
[`qnn-4k-sm8750-timings.log`](evidence/qnn-4k-sm8750-timings.log),
[`qnn-4k-sm8750-export-report.json`](evidence/qnn-4k-sm8750-export-report.json); export
seconds of the 2k runs from their published `export-report.json`
(`experimentalmachines/Qwen3-0.6B-ExecuTorch`, `qnn/<chip>/`).

- *Calibration loop*: the tqdm line of the one-batch calibration (`1/1 [29:21 ...]`,
  `1/1 [15:36 ...]`, `1/1 [1:01:13 ...]`).
- *Other quantize*: `MultiModalManager::quantize completed in` minus the loop: exporting
  and preparing the three decoder graphs.
- *HTP compile*: `HybridTextDecoder::compile completed in`, QAIRT building the prefill and
  decode context binaries.
- *Rest*: export minus quantize minus compile (model load, dataset, file writing).

Peak RSS of the export was 15.66-15.70 GB in all three runs (reports), just under physical
memory, so it does not show how much swap was used.

## 2. Why the calibration loop is expensive: SeqMSE

ExecuTorch registers a SeqMSE search for two of its Qualcomm LLM entries only:

- `examples/qualcomm/oss_scripts/llama/__init__.py:268`: `Llama3_2_1B_Instruct`,
  `seq_mse_candidates = 50`.
- `examples/qualcomm/oss_scripts/llama/__init__.py:501`: `Qwen3_0_6B`,
  `seq_mse_candidates = 50`. Every other entry in the file has `seq_mse_candidates = 0`.
- The run logs confirm it: `| seq_mse_candidates | 50 |` in the config table of all three
  runs.

What it does:

1. Calibration runs one forward pass of the whole sequence
   (`wrappers/llm_wrappers.py:717-724`, "Full AR sequence with KV cache; used only for
   quantization"; `inference/decoder.py:136`, `predict_step` is a single forward). The
   sequence is `max_context_len` tokens (`Calibration: collected 1 sequences
   (max_context_length=2048|4096, limit=1)` in the logs).
2. For the first batch with `seq_mse_candidates` set, `quantize/ptq.py:65-66` runs a second
   forward inside `SeqMSE(...)`.
3. `backends/qualcomm/_passes/seq_mse.py:148` attaches a search to every `conv2d`. The
   Qualcomm wrapper runs every linear layer as a 1x1 conv: attention and feed-forward
   through `prepare_attention_conv` / `prepare_feedfoward_conv`
   (`wrappers/llm_wrappers.py:308-312`, defined at `model/static_llama.py:343` and `:539`),
   then every remaining `nn.Linear`, the output projection included, through
   `convert_linear_to_conv2d` (`wrappers/llm_wrappers.py:314`,
   `backends/qualcomm/utils/utils.py:153`). For Qwen3-0.6B that is 28 x 7 + 1 = 197 layers.
4. Each search (`seq_mse.py:107-135`) evaluates the layer once at the observer's scale, at
   50 coarse scales, then at 50 fine scales around the best one: 101 full evaluations of
   the layer on that pass's whole input. Each evaluation fake-quantizes the weight and
   recomputes the conv over all tokens.

### Operation count (estimate)

Weights in those 197 convs, from Qwen3-0.6B's config.json (hidden 1,024, 16 query heads and
8 KV heads of 128, intermediate 3,072, vocabulary 151,936):

- per layer: q 1,024x2,048 + k 1,024x1,024 + v 1,024x1,024 + o 2,048x1,024 + gate, up,
  down 3 x 1,024x3,072 = 15,728,640
- 28 layers: 440,401,920; output projection 1,024 x 151,936 = 155,582,464; total
  595,984,384.

A conv evaluation costs 2 x 595,984,384 = 1,191,968,768 FLOP per token.

| Window | Plain forward (convs) | SeqMSE (101 evaluations) | Implied rate, if SeqMSE is all of the loop |
|---|---|---|---|
| 2,048 | 2,441,152,036,864 FLOP | 246,556,355,723,264 FLOP | 140 GFLOP/s (SM8650 run), 263 GFLOP/s (SM8750 run) |
| 4,096 | 4,882,304,073,728 FLOP | 493,112,711,446,528 FLOP | 134 GFLOP/s |

SeqMSE is 101 times the plain pass's conv arithmetic, so if time follows arithmetic it is
about 99% of the calibration loop. This is an estimate: the log times the loop as a whole
(`predict_step` plus the SeqMSE pass), and attention, observers and fake-quantization are
not counted. The implied rates are plausible for fp32 on 4 vCPUs.

## 3. What the evidence does not settle

- **Runner-to-runner speed.** The two 2k runs did identical calibration work (the chip only
  matters at compile time) and took 1,761 s and 936 s, a factor of 1.88. The reports did not
  record the CPU model; `exporting.host_info` records it (`cpu_model`) from now on.
- **Window scaling.** On the same chip (SM8750) the loop went from 936 s at 2k to 3,674 s at
  4k, a factor of 3.92, where the operation count predicts 2. That is within what runner
  variance (1.88x) times the predicted 2x could produce (3.76x), or the 4k run swapped: the
  output projection alone produces 4,096 x 151,936 x 4 B = 2,489,319,424 B per evaluation at
  4k, and peak RSS sat at the physical-memory ceiling in both runs. Exports now record the
  peak swap in use (`exporting.MemorySampler`, `peak_swap_used_bytes`).
- **The HTP compile** (1,172-1,905 s) is QAIRT's own offline preparation of two graphs;
  nothing here breaks it down further.

A run with `seq_mse_candidates` set to 0 on the same runner type would measure SeqMSE's share
directly. It is not done: it changes ExecuTorch's recipe for this model, and there is no
on-device way here to check the quality cost.

## 4. Other backends

- **XNNPACK:** 600 s (2k) and 721 s (16k) for Qwen3-0.6B in the phase-0 probes
  ([PLAN.md](../PLAN.md), phase 0). Not a bottleneck; its limit is memory, see the 32k probe
  ([`xnnpack-probe-32k-runner-killed.log`](evidence/xnnpack-probe-32k-runner-killed.log)).
- **MediaTek:** the first run was limited by memory, not time: preparing calibration inputs
  at a 2,048-token cache took the runner down
  ([`mtk-run1-calibration-memory.log`](evidence/mtk-run1-calibration-memory.log);
  `export_mtk.calibration_bytes` estimates 75,161,927,680 B; corrected to 84,557,168,640 B
  for the 9 prompts the file really holds, see [finding 18](README.md)). No timing yet.
  *Added 2026-09-13:* the second run (512-token cache) was limited by reading the prepared
  calibration inputs back as Python lists, about 35 minutes per prompt
  ([finding 17](README.md)).
- **Every job:** 2-4 minutes of setup (disk cleanup, swap, pip installs) before the export.

## 5. Options (not decided)

1. Keep ExecuTorch's recipe. The worst case measured, 5,987 s, is inside the 6-hour job
   limit.
2. Fewer SeqMSE candidates, or none, for faster exports; the quality effect is unmeasured.
3. Larger (paid) runners; both calibration and compile are CPU-bound.
