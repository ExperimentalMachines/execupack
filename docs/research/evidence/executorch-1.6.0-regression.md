# ExecuTorch 1.6.0: regression against 1.5.1 (2026-11-13/14)

What was measured before moving the CPU (XNNPACK) and GPU (Vulkan) exporter to ExecuTorch 1.6.0
(torch 2.15.0, torchao 0.18.0, pytorch-tokenizers 1.6.0). Host: macOS arm64 (Apple M5), the
executorch PyPI wheels' Python runtime in Python 3.11 virtualenvs: 1.4.0 (torch 2.13.0), 1.5.1
(torch 2.14.0) and 1.6.0 (torch 2.15.0). Gate references are the fp32 `gate.json` files the solves
wrote on the Linux x86 boxes on 2026-11-09 (a Mac's fp32 reference is wrong, finding 44). The
probe scripts are beside this file (`et160-*.py`); only their import paths and formatting were
changed from what ran.

Not covered: Linux (the Modal image's install, the solve on torch 2.15, the x86 C++ runner of the
verify job), any GPU or phone, the apps' own Android builds of the runtime, and Phi-4-mini's export
(its 3.8B fp32 conversion does not fit this host). QNN and MediaTek keep the 1.5.1 toolchain, so
their export is unchanged.

A correction made before this was written up: the first pass of the gate comparison handed
`verify` the published report, whose recorded gate was against the same reference file, so for 12
of the 16 files `verify` answered "already verified" with the published verdict and measured
nothing. Every gate below was measured: the report's recorded smoke test and gate are removed
before each run, and each run's log shows the gate executing (no "already verified").

## Every published file parses under 1.6.0's program schema

`et160-static-parse.py`: the header and program flatbuffer of all 368 `.pte` files in the
experimentalmachines org (100 XNNPACK, 65 Vulkan, 23 QNN, 178 MediaTek chunks, 2 others), fetched
by range request and parsed with 1.6.0's `exir._serialize._program._flatbuffer_to_program`: 368 of
368 parse, 66,927 kernel calls and 34,825 delegate calls in all. This checks the program's
structure against 1.6.0's schema, not that any delegate runs.

## Published CPU files on the 1.5.1 and 1.6.0 runtimes

One file per repo, its 2k window, from the Hub. Probe (`et160-ptecheck.py`): a fixed raw-text
prompt (no chat template) prefilled at position 0, then 24 greedy tokens; KL is the next-token
distribution after the prefill, 1.5.1 runtime against 1.6.0. Gate: `python -m pipeline verify
--gate` with the run's archived fp32 reference, both runtimes the same day; verdict,
agreed/graded, and the export's mean probability of fp32's token. "Gate as published" is the
verdict recorded at publication (x86 Linux, 1.5.1 or 1.4.0 runtime).

| Repo | Built with | Linears | Loads on 1.5.1 / 1.6.0 | Same methods and metadata | Greedy tokens equal before the first difference (of 24) | KL | Gate as published | Gate, 1.5.1 runtime | Gate, 1.6.0 runtime |
|---|---|---|---|---|---|---|---|---|---|
| SmolLM2-135M-Instruct | 1.5.1 | int4 GPTQ | yes / yes | yes | 24 | 0.0140 | pass 4/4 0.5078 | pass 4/4 0.5008 | pass 4/4 0.5218 |
| Qwen2.5-0.5B-Instruct | 1.5.1 | int4 GPTQ | yes / yes | yes | 0 | 0.0531 | pass 9/9 0.6564 | pass 9/9 0.6667 | pass 9/9 0.7202 |
| LFM2.5-1.2B-Instruct | 1.4.0 | int4 GPTQ | yes / yes | yes | 14 | 0.0180 | no archived reference | | |
| LFM2.5-1.2B-Instruct-heretic | 1.4.0 | int4 GPTQ | yes / yes | yes | 24 | 0.0078 | no archived reference | | |
| Llama-3.2-1B-Instruct | 1.5.1 | int4 GPTQ | yes / yes | yes | 22 | 0.0025 | pass 12/12 0.8293 | pass 12/12 0.7518 | pass 12/12 0.7333 |
| Qwen2.5-1.5B-Instruct | 1.5.1 | int4 GPTQ | yes / yes | yes | 0 | 0.0566 | pass 12/12 0.7435 | pass 12/12 0.8599 | pass 12/12 0.8786 |
| Qwen2.5-Math-1.5B-Instruct | 1.5.1 | int4 GPTQ | yes / yes | yes | 15 | 0.0118 | pass 3/3 0.7408 | pass 3/3 0.7169 | pass 3/3 0.7427, smoke FAIL |
| LFM3-1.7B | 1.5.1 | int4 GPTQ | yes / yes | yes | 0 | 0.0098 | pass 12/12 0.9991 | pass 12/12 0.998 | pass 12/12 0.9992 |
| LFM3-1.7B-Base | 1.5.1 | int4 GPTQ | yes / yes | yes | 24 | 0.0033 | pass 12/12 0.9786 | pass 12/12 0.9799 | pass 12/12 0.9832 |
| Qwen3-1.7B | 1.5.1 | int4 GPTQ | yes / yes | yes | 0 | 0.0315 | pass 12/12 0.9946 | pass 12/12 0.9897 | pass 12/12 0.9745 |
| SmolLM2-360M-Instruct | 1.4.0 | fp32 | yes / yes | yes | 24 | 0.0000 | pass 2/2 0.5673 | pass 2/2 0.5674 | pass 2/2 0.5674 |
| LFM2.5-2.6B | 1.4.0 | int4 GPTQ | yes / yes | yes | 18 | 0.0152 | no archived reference | | |
| LFM2.5-2.6B-heretic | 1.4.0 | int4 GPTQ | yes / yes | yes | 24 | 0.0187 | no archived reference | | |
| Qwen2.5-3B-Instruct | 1.5.1 | int4 GPTQ | yes / yes | yes | 2 | 0.0148 | pass 12/12 0.9968 | pass 12/12 0.9966 | pass 12/12 0.9968 |
| Llama-3.2-3B-Instruct | 1.5.1 | int4 GPTQ | yes / yes | yes | 17 | 0.0095 | pass 12/12 0.9111 | pass 12/12 0.8796 | pass 12/12 0.8633 |
| Qwen3-0.6B | 1.5.1 | fp32 | yes / yes | yes | 24 | 0.0000 | pass 4/4 0.6177 | pass 4/4 0.6178 | pass 4/4 0.6178 |
| Qwen3-4B-Thinking-2507 | 1.5.1 | int4 GPTQ | yes / yes | yes | 7 | 0.0167 | pass 12/12 0.9838 | pass 12/12 0.9813 | pass 12/12 0.987 |
| Qwen3-4B-Instruct-2507 | 1.5.1 | int4 GPTQ | yes / yes | yes | 0 | 0.0644 | pass 12/12 0.9954 | pass 12/12 0.9942 | pass 12/12 0.9956 |
| Qwen3-4B | 1.5.1 | int4 GPTQ | yes / yes | yes | 6 | 0.0223 | pass 12/12 0.9999 | pass 12/12 0.9999 | pass 12/12 0.9999 |
| Phi-4-mini-instruct-3.8B | 1.5.1 | int4 GPTQ | yes / yes | yes | 13 | 0.0226 | FAIL 7/8 0.5066 (published with an override) | FAIL 7/8 0.4275 | pass 8/8 0.5179 |

- The int4 files' outputs move between the runtimes (KL 0.0025 to 0.064 on the probe; export means
  by up to 0.090 on the gate, Phi-4-mini, and 0.054 for the rest); the two fp32 files' do not (KL
  under 1e-7).
- Verdicts and agreed counts on 1.6.0 equal those on 1.5.1 for 15 of the 16 gated files. Phi-4-mini
  keeps fp32's choice on all 8 rows on 1.6.0 and on 7 on 1.5.1.
- Llama 3.2's chat template carries the date, so its rows today are not the ones its published gate
  read; both runtimes here read the same ones.
- **Qwen2.5-Math-1.5B fails the smoke test on 1.6.0.** Asked the capital of France, it answers "The
  capital of France is ** parallelogram** … **lyle**", where on 1.5.1 it answers "** parallele** …
  **paris**" and at publication (x86) answered "\"paris\"". `verify` stops at the smoke test, so its
  gate on 1.6.0 was measured on its own (`gate.measure` and `gate.check`): pass 3/3 0.7427. Replayed
  step by step (`replay.py` in the session's tools, not committed), the two runtimes agree for seven
  tokens; at the answer both put little weight on any token (" **" at 0.052 / 0.034, then
  " paralle" at 0.026 / 0.039), and the eighth token splits "le" 0.293 over "log" 0.267 on 1.5.1
  against "log" 0.256 over "le" 0.240 on 1.6.0. The smoke test requires "paris" from every model
  of 500 million parameters or more (`smoke.ANSWER_REQUIRED_FROM_PARAMS`). A 1.6.0 export of the
  same model from the same codes (below) opens with "** parallelogram**" too and passes the smoke
  test because its reply reaches "Paris" later. Only these two Mac runs were measured; how often
  this model's greedy reply reaches "paris" elsewhere is not known.

## Files built with 1.4.0, on the 1.4.0, 1.5.1 and 1.6.0 runtimes

A 1.6.0 runtime is promised to load 1.5.1 files, not 1.4.0 ones, and five CPU files are still 1.4.0
builds. `et160-decide.py` compares runtimes on each file's current gate prompts (`gate.prompts`
today, the model's chat template; the heretic finetunes' from their base model, their source repos
being gone): one forward from position 0 per row, the runner reloaded per row as the gate does,
and the next token compared with the 1.4.0 runtime's. These are not the prompts the files were
gated with at publication (`gate.prompts` has changed since: LFM2.5-2.6B's published gate has fp32
at 0.9999 on the row that differs below, where today's prompt gives 0.38), so this is first-token
agreement between runtimes, not a re-run of the published gate.

| File | Rows | Same token as on 1.4.0: 1.5.1 runtime | 1.6.0 runtime | Rows that differ | Largest change in the top token's probability (1.5.1 / 1.6.0) |
|---|---|---|---|---|---|
| xnnpack/LFM2.5-1.2B-Instruct-8da4w-gptq-2k.pte | 16 | 16/16 | 16/16 | none | 0.3293 / 0.2727 |
| xnnpack/LFM2.5-1.2B-Instruct-heretic-8da4w-gptq-2k.pte | 16 | 16/16 | 16/16 | none | 0.1886 / 0.1954 |
| xnnpack/LFM2.5-2.6B-8da4w-gptq-2k.pte | 16 | 15/16 | 15/16 | Who wrote Pride and Prejudice? | 0.0886 / 0.1041 |
| xnnpack/LFM2.5-2.6B-heretic-8da4w-gptq-2k.pte | 16 | 15/16 | 16/16 | Who wrote Pride and Prejudice? (1.5.1 only) | 0.0988 / 0.1019 |
| xnnpack/SmolLM2-360M-Instruct-fp32-2k.pte | 16 | 16/16 | 16/16 | none | 0.0 / 0.0 |

On the differing row LFM2.5-2.6B's first token is "Pr" 0.384 over "The" 0.206 on 1.4.0, "The"
0.296 over "Pr" 0.248 on 1.5.1, and "The" 0.311 over "Pr" 0.242 on 1.6.0. The largest probability
move, LFM2.5-1.2B on "In what country is Gdynia?", is 0.954 on 1.4.0, 0.624 on 1.5.1 and 0.816 on
1.6.0, `<|tool_call_start|>` each time.

MediaTek: 145 of the 178 published chunks (39 of 48 windows, in 12 repos) are 1.4.0 builds, by
their export reports. They parse (above); there is no host NeuroPilot runtime to run them. Between
1.5.1 and 1.6.0 only build files changed under `backends/mediatek` and `examples/mediatek`.

## New exports with 1.6.0

`python -m pipeline export-xnnpack` / `export-vulkan` from this tree at the published source
revision, window 2k, with the archived GPTQ codes, then `verify --gate` with the archived reference
on the 1.6.0 runtime (these reports had no gate yet, so each was measured). "Codes" is
`tools/codecheck/codesmatch.py`: every int4 linear read back from the file's bytes against
codes.pt.

| Model | Recipe | Bytes, 1.6.0 | Bytes, published (1.5.1, x86) | Codes | Checked by | Result, 1.6.0 export | Gate as published |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M-Instruct | XNNPACK 8da4w GPTQ | 105,990,784 | 106,018,048 | 211/211 exact | gate | pass 4/4 0.5237 | pass 4/4 0.5078 |
| SmolLM2-135M-Instruct | Vulkan 8da4w GPTQ | 132,503,552 | 132,513,280 | 211/211 exact | gate proxy (the row above) and structure | `forward` is one `VulkanBackend` call; not run on a GPU | proxy |
| Qwen2.5-0.5B-Instruct | XNNPACK 8da4w GPTQ | 417,477,120 | 417,495,296 | 169/169 exact | gate | pass 9/9 0.6735 | pass 9/9 0.6564 |
| Qwen3-0.6B | XNNPACK fp32 linears | 2,545,152,640 | 2,545,178,112 | none (fp32) | gate | pass 4/4 0.6178 | pass 4/4 0.6177 |
| Llama-3.2-1B-Instruct | XNNPACK 8da4w GPTQ | 959,958,272 | 959,972,864 | 113/113 exact | gate | pass 12/12 0.8229 | pass 12/12 0.8293 |
| LFM3-1.7B | XNNPACK 8da4w GPTQ (`pipeline/lfm3_model.py`) | 1,239,187,968 | 1,239,210,240 | 180/180 exact (codesmatch has no layer order for LFM3, as before) | gate | pass 12/12 0.9994, quiet 3/3 | pass 12/12 0.9991 |
| LFM2.5-1.2B-Instruct | XNNPACK 8da4w rounded (no archived codes) | 795,678,720 | 795,686,784 (GPTQ, 1.4.0) | none | smoke only (no reference) | "Paris" | pass 12/12 0.8989 |
| Qwen2.5-Math-1.5B-Instruct | XNNPACK 8da4w GPTQ | 1,107,592,704 | 1,107,613,952 | 197/197 exact | gate | pass 3/3 0.7296; smoke passes, its reply opening "** parallelogram**" and reaching "**Paris**" 271 characters in | pass 3/3 0.7408 |

Every file has the same methods and metadata as its published twin. The LFM files keep
`get_state_reset_at_zero`, and their graph clamps the position (`aten::clamp`, from
`pipeline/lfm2_state.py`), not 1.6.0's own `input_pos == 0` reset, which this repository's patch
still replaces. `et160-reset-check.py`: a second prompt after a first one in the same runner gives
exactly the logits of a fresh runner (largest difference 0.0, against 11.2 for LFM3 and 15.4 for
LFM2.5 between the two prompts' own logits).

## The same model, two builds and two runtimes

SmolLM2-135M-Instruct, 2k, the same codes and reference; export mean of fp32's token on the 4
graded rows (fp32: 0.5834), every cell measured and passing 4/4:

| File | 1.5.1 runtime | 1.6.0 runtime |
|---|---|---|
| Published, built with 1.5.1 on x86 | 0.5008 | 0.5218 |
| Built with 1.5.1 on this Mac | 0.5192 | 0.5177 |
| Built with 1.6.0 on this Mac | 0.4885 | 0.5237 |

The build and the runtime each move the mean by up to about 0.035, and no cell changes a decision.
The two Mac builds (`et160-structure.py`, `et160-constdiff.py`):

- Same operators, kernel calls (759) and XNNPACK nodes (304 delegate calls; 211 fully connected).
  The 1.6.0 file is 27,264 B smaller: the program's value table (Int values 1,606 to 821, IntList
  421 to 242) and the XNNPACK delegate blobs (347,056 B to 346,576 B in all).
- All 422 named-data entries (the int4 codes and scales, the int8 embedding) are byte-identical.
  The only constants that differ are the two RoPE tables (131,072 fp32 values each), by at most
  1.19e-7. `examples/models/llama/rope.py` is the same at both tags and torch moved from 2.14 to
  2.15; which of the two toolchains' parts makes the difference was not isolated.

## A 1.6.0 file on the 1.5.1 runtime

The apps ship 1.5.1, and ExecuTorch promises nothing for a newer file on an older runtime. The
1.6.0 SmolLM2-135M file loads on 1.5.1 with the same methods and metadata and passes its gate there
(the table above: 4/4, 0.4885 against 0.5237 on 1.6.0). One model on the Python runtime is not a
promise, so `publish` refuses 1.6.0 files until the apps ship 1.6.0 (`APP_RUNTIME_VERSION`).
