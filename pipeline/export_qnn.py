"""Export one HF model to a Qualcomm QNN (HTP) .pte for one Snapdragon chip.

Drives ExecuTorch 1.4.0's own Qualcomm LLM script (examples/qualcomm/oss_scripts/llama)
in compile-only mode: it downloads the checkpoint, calibrates the quantization recipe it
registers for that model, and compiles hybrid prefill + decode graphs into QNN context
binaries for the chip. Produces, under ``out_dir``:

    tokenizer.json | tokenizer.model, LICENSE…, NOTICE   (repo root, shared with other backends)
    qnn/<soc>/<name>-qnn-hybrid-<window>.pte
    qnn/<soc>/config.json
    qnn/<soc>/export-report-<window>.json

There is no host runtime for HTP binaries here, so the check after export is structural:
the program loads, carries the decoder graphs, and delegates to QnnBackend.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pipeline import eligibility, families, hub, manifest, naming, settings, smoke
from pipeline.exporting import (
    ExportError,
    MemorySampler,
    children_peak_rss,
    copy_side_files,
    host_info,
    report_file,
    run_tool,
    sha256,
    source_report,
    toolchain,
)

BACKEND = "qnn"
# examples/qualcomm/oss_scripts/llama/decoder_constants.py: DECODER_GRAPH_NAMES
HYBRID_METHODS = ("kv_forward", "prefill_forward")
# backends/qualcomm/partition/qnn_partitioner.py: QnnBackend.__name__ is the delegate id.
QNN_BACKEND_ID = "QnnBackend"
META_FILES = ("original/consolidated.00.pth", "original/params.json", "original/tokenizer.model")
SOC_NAMES = {"SM8650": "Snapdragon 8 Gen 3", "SM8750": "Snapdragon 8 Elite"}

# The Modal Sandbox a window is given (pipeline/remote.py). Only Qwen3-0.6B has been measured:
# 15.6 GB of peak RSS at both 2k and 4k (findings 8 and 10), on 16 GB runners with swap, so
# that figure is a floor on its need rather than the need. Modal treats a plain memory request
# as a minimum, not a limit (the Sandbox may use more) and bills the higher of request and
# use, so this asks for a modest share and lets the peak burst past it. Refit it against each
# report's peak_rss_export_bytes and peak_cgroup_bytes; on Modal peak_in_use_bytes is the host's.
QNN_BYTES_PER_PARAM = 20
QNN_FLOOR_BYTES = 24 * 2**30
# The window enters through calibration and the two graphs' attention, at most half again by 8k.
QNN_WINDOW_SCALE = 16384
# Time grows with the model and faster than the window: Qwen3-0.6B took 2,434 s at 2k and
# 5,987 s at 4k on a 4 vCPU runner (findings 8 and 10). (largest parameter count, physical
# cores, Sandbox minutes); the minutes cost nothing unless a run hangs.
QNN_MODAL_SHAPES = ((1_300_000_000, 16, 720), (4_500_000_000, 32, 1380))


def modal_request(params: int, window: int) -> tuple[int, int, int]:
    """Memory in bytes, physical cores and minutes for a QNN export on Modal."""
    need = max(QNN_FLOOR_BYTES, params * QNN_BYTES_PER_PARAM) * (1 + window / QNN_WINDOW_SCALE)
    for largest, cores, minutes in QNN_MODAL_SHAPES:
        if params <= largest:
            return int(need), cores, minutes
    raise ValueError(f"no Modal shape for a {params:,}-parameter QNN export")


EXECUTORCH_SOURCE_FILES = settings.ROOT / "third_party" / "executorch"


def params_file(decoder: str) -> Path:
    """The registry's params file for ``decoder`` (copied from ExecuTorch; the wheel lacks it)."""
    path = EXECUTORCH_SOURCE_FILES / families.QNN_PARAMS[decoder]
    if not path.is_file():
        raise ExportError(f"missing {path}: copy it from the ExecuTorch source at the pinned version")
    return path


def llama_command(
    decoder: str,
    soc: str,
    recipe: settings.QnnRecipe,
    artifact: Path,
    meta_dir: Path | None,
) -> list[str]:
    window = recipe.max_context_len
    command = [
        sys.executable,
        "-m",
        "executorch.examples.qualcomm.oss_scripts.llama.llama",
        "--decoder_model",
        decoder,
        "--soc_model",
        soc,
        "--compile_only",
        "--model_mode",
        recipe.model_mode,
        "--max_seq_len",
        str(window),
        "--max_context_len",
        str(window),
        "--prefill_ar_len",
        str(recipe.prefill_ar_len),
        "--calib_tasks",
        *recipe.calib_tasks,
        "--calib_limit",
        str(recipe.calib_limit),
        "--artifact",
        str(artifact),
        # main() calls os.path.realpath(args.build_folder) even in compile-only mode, where the
        # on-device runner it points at is never used.
        "--build_folder",
        str(artifact.parent / "build-android"),
        # Required by the script; used only when it runs on a device, which compile-only skips.
        "--prompt",
        "What is the capital of France?",
        "--temperature",
        "0",
    ]
    if meta_dir is not None:
        command += [
            "--checkpoint",
            str(meta_dir / "consolidated.00.pth"),
            "--params",
            str(meta_dir / "params.json"),
            "--tokenizer_model",
            str(meta_dir / "tokenizer.model"),
        ]
    else:
        # Every reader of the params prefers --params over the registry's params_path, which
        # points into the source tree the wheel does not carry.
        command += ["--params", str(params_file(decoder))]
    return command


def check_script_revision(source: hub.SourceModel, head: str) -> None:
    """The Qualcomm script downloads its registry entry's repo at ``main``, whatever revision
    was asked for; refuse when that would compile weights other than the ones recorded."""
    if head != source.sha:
        raise ExportError(
            f"{source.id} was requested at {source.sha[:12]} but the Qualcomm script compiles "
            f"main, now at {head[:12]}; re-run with --revision main (or the revision recorded "
            "will not match the weights)"
        )


def decoder_pte(artifact: Path, model_mode: str) -> Path:
    """The text decoder the script wrote: ``[<decoder>_]<mode>_llama_qnn.pte``."""
    matches = sorted(artifact.glob(f"*{model_mode}_llama_qnn.pte"))
    if len(matches) != 1:
        found = sorted(p.name for p in artifact.glob("*.pte"))
        raise ExportError(f"expected one {model_mode} decoder .pte in {artifact}, found {found}")
    return matches[0]


delegates = smoke.delegates  # kept for callers and tests


def structural_check(pte: Path, model_mode: str) -> dict:
    """The program loads, has the decoder graphs, and each of them runs on the QNN delegate."""
    wanted = HYBRID_METHODS if model_mode == "hybrid" else ("kv_forward",)
    return smoke.structural_check(pte, wanted, QNN_BACKEND_ID)


_SDK_PROBE = """
import json
# ExecuTorch 1.5.1 moved the download out of the package's __init__, which now has no side
# effects (backends/qualcomm/utils/qnn_sdk_setup.py): it happens here or not at all.
from executorch.backends.qualcomm.utils.qnn_sdk_setup import setup_qnn_sdk
setup_qnn_sdk()
from executorch.backends.qualcomm.scripts import download_qnn_sdk as d
print(json.dumps({
    "version": d.QNN_VERSION,
    "root": str(d._get_sdk_dir()),
    "libcxx_dir": str(d._get_staging_dir(f"libcxx-{d.LLVM_VERSION}")),
    "libcxx_files": list(d.REQUIRED_LIBCXX_LIBS),
}))
"""


def qnn_sdk() -> dict:
    """Where the executorch wheel keeps the QAIRT SDK it pins, and the libc++ that SDK needs.

    Asked of a separate interpreter, whose import downloads both on first use: importing
    executorch.backends.qualcomm edits os.environ and preloads libraries in the importing
    process, which must not leak into this one.
    """
    # The first import downloads the SDK archive (about a gigabyte); a stalled download
    # should fail the job here, not at the 6-hour limit.
    result = subprocess.run([sys.executable, "-c", _SDK_PROBE], capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise ExportError(f"could not set up the QAIRT SDK:\n{result.stderr[-4000:]}")
    sdk = json.loads(result.stdout.strip().splitlines()[-1])
    # The paths are computed, not found: check the download put the libraries there, so a
    # missing SDK fails here and not as a dlopen error deep in the Qualcomm script.
    wanted = [Path(sdk["root"]) / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"]
    wanted += [Path(sdk["libcxx_dir"]) / name for name in sdk["libcxx_files"]]
    missing = [str(path) for path in wanted if not path.exists()]
    if missing:
        raise ExportError(f"the QAIRT SDK setup left no {', '.join(missing)}:\n{result.stderr[-4000:]}")
    return sdk


def soname(filename: str) -> str:
    """``libc++.so.1.0`` -> ``libc++.so.1``, the name the SDK's libraries ask the loader for."""
    head, _, version = filename.partition(".so.")
    return f"{head}.so.{version.split('.')[0]}" if version else filename


def qnn_env(sdk: dict, links: Path, base: dict[str, str]) -> dict[str, str]:
    """Environment for the Qualcomm script: the SDK and its libc++ on the loader path from the start.

    ExecuTorch's import-time setup only changes os.environ and ctypes-loads libQnnHtp.so and
    libc++ by path. The dynamic loader reads LD_LIBRARY_PATH once, at process start, so what
    QNN later opens by bare name (libQnnSystem.so) is not found, and a process started with
    QNN_SDK_ROOT set skips the libc++ preload. Setting both up front is ExecuTorch's documented
    manual setup; the soname links make the wheel's libc++ copy findable by name.
    """
    links.mkdir(parents=True, exist_ok=True)
    for filename in sdk["libcxx_files"]:
        link = links / soname(filename)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path(sdk["libcxx_dir"]) / filename)
    sdk_lib = Path(sdk["root"]) / "lib" / "x86_64-linux-clang"
    paths = [str(sdk_lib), str(links)] + [p for p in base.get("LD_LIBRARY_PATH", "").split(":") if p]
    return {**base, "QNN_SDK_ROOT": sdk["root"], "LD_LIBRARY_PATH": ":".join(paths), "PYTHONUNBUFFERED": "1"}


def run(
    model_id: str,
    revision: str,
    soc: str,
    out_dir: Path,
    work_dir: Path,
    keep_work: bool = False,
    context: int | None = None,
) -> dict:
    cfg = settings.load()
    recipe = cfg.qnn
    if context is not None:
        if context < recipe.prefill_ar_len:
            raise ExportError(f"--context {context} is below the prefill length {recipe.prefill_ar_len}")
        recipe = dataclasses.replace(recipe, max_context_len=context)
    source = hub.fetch(model_id, revision)
    verdict = eligibility.evaluate(source, cfg)
    if verdict.reasons:
        raise ExportError(f"{model_id} is not eligible: {'; '.join(verdict.reasons)}")
    if verdict.backends[BACKEND] is not None:
        raise ExportError(f"{model_id} cannot be exported to {BACKEND}: {verdict.backends[BACKEND]}")
    decoder = families.qnn_decoder(source.id)
    meta = decoder in families.QNN_META_CHECKPOINT
    if not meta and revision != "main":
        # At "main" the fetch above already resolved what the script will download (bar a
        # push in the seconds between); any other revision has to be checked against it.
        check_script_revision(source, hub.head_sha(model_id))
    window = recipe.max_context_len

    output_repo = naming.output_repo(model_id, cfg.hub_org, cfg.repo_suffix)
    folder = naming.qnn_folder(soc)
    pte_name = naming.qnn_file(model_id, recipe.model_mode, window)
    path_in_repo = f"{folder}/{pte_name}"
    problems = naming.check_app_rules(output_repo, path_in_repo, BACKEND)
    if problems:
        raise ExportError("; ".join(problems))

    backend_dir = out_dir / folder
    backend_dir.mkdir(parents=True, exist_ok=True)
    src_dir = work_dir / "source"
    artifact = work_dir / "artifact"
    print(f"==> {model_id}@{source.sha[:12]}: --decoder_model {decoder}, {soc}, window {window}")

    sdk = qnn_sdk()
    qairt = sdk["version"]
    expected = settings.read_env_file(settings.CONFIG_DIR / "versions.env").get("QAIRT_VERSION")
    if qairt != expected:
        raise ExportError(f"executorch's QAIRT is {qairt}, config/versions.env pins {expected}")
    env = qnn_env(sdk, work_dir / "qnn-libs", dict(os.environ))

    started = time.time()
    # The script fetches the weights itself (repo_id in its registry); only Llama 3.2 needs
    # Meta's original checkpoint handed over.
    hub.download(source, src_dir, weights=False, extra=META_FILES if meta else ())
    tokenizer, licenses = copy_side_files(source, src_dir, out_dir)

    command = llama_command(decoder, soc, recipe, artifact, src_dir / "original" if meta else None)
    print("==> " + " ".join(command))
    export_started = time.time()
    with MemorySampler() as memory:
        run_tool(command, "the Qualcomm llama script", cwd=work_dir, env=env)
    export_seconds = time.time() - export_started

    built = decoder_pte(artifact, recipe.model_mode)
    pte = backend_dir / pte_name
    shutil.move(str(built), pte)
    check = structural_check(pte, recipe.model_mode)
    metadata = check.pop("metadata", {})

    report = {
        "backend": BACKEND,
        "target": soc.lower(),
        "target_name": SOC_NAMES.get(soc, soc),
        "tokenizer": tokenizer,
        "output_repo": output_repo,
        "source": source_report(source, verdict.family, verdict.variant, licenses),
        "toolchain": {**toolchain("transformers", "lm_eval"), "qairt": qairt},
        "recipe": {
            "decoder_model": decoder,
            "params": f"{model_id}: original/params.json" if meta else f"executorch: {families.QNN_PARAMS[decoder]}",
            "model_mode": recipe.model_mode,
            "prefill_ar_len": recipe.prefill_ar_len,
            "max_context_len": window,
            "calibration": {"tasks": list(recipe.calib_tasks), "limit": recipe.calib_limit},
            "label": f"QNN HTP, ExecuTorch {decoder} recipe",
            "description": (
                f"ExecuTorch {toolchain()['executorch']} Qualcomm static LLM "
                f"(`examples/qualcomm/oss_scripts/llama`, `--decoder_model {decoder}`): the "
                f"quantization recipe ExecuTorch registers for this model, calibrated on "
                f"{'/'.join(recipe.calib_tasks)} ({recipe.calib_limit} sample), {recipe.model_mode} "
                f"prefill ({recipe.prefill_ar_len} tokens per step) and decode graphs compiled "
                f"with QAIRT {qairt} for {soc} ({SOC_NAMES.get(soc, soc)})."
            ),
        },
        "window": {
            "context": window,
            "reason": (
                "forced with --context (static NPU graphs)"
                if context is not None
                else "fixed by qnn.max_context_len (static NPU graphs)"
            ),
            "kv_cache_bytes_per_token": None,
        },
        "files": [{"path": path_in_repo, "bytes": pte.stat().st_size, "sha256": sha256(pte)}],
        "host": {
            **host_info(),
            "peak_rss_export_bytes": children_peak_rss(),
            **memory.result(),
            "export_seconds": round(export_seconds, 1),
            "total_seconds": round(time.time() - started, 1),
        },
        "metadata": metadata,
        "smoke": check,
        "run": manifest.run_info(),
    }
    (backend_dir / "config.json").write_text(
        json.dumps(manifest.backend_config(report), indent=2) + "\n", encoding="utf-8"
    )
    (backend_dir / report_file(window)).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)
    if not check["passed"]:
        raise ExportError("structural check failed: " + "; ".join(check["problems"]))
    return report
