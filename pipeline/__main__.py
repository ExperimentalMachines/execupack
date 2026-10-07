"""Command line: ``python -m pipeline <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _plan(args) -> int:
    from pipeline import eligibility, families, hub, settings, sizing
    from pipeline.exporting import host_budget, host_info

    cfg = settings.load()
    source = hub.fetch(args.model, args.revision)
    verdict = eligibility.evaluate(source, cfg)
    result = verdict.to_dict()
    family = families.family_for(source.config) if source.config else None
    if family and source.total_params:
        arch = families.architecture(source.config, source.total_params)
        budget = host_budget(host_info())
        choice = sizing.choose_context(
            arch, cfg.context_tiers, cfg.device_budget_bytes, cfg.runtime_overhead_bytes, budget, qmode=args.qmode
        )
        result["windows"] = {
            "exportable_on_this_host": [row["context"] for row in choice.table if row["fits_host"]],
            "within_phone_budget": [row["context"] for row in choice.table if row["fits_device"]],
            "host_budget_bytes": budget,
            "table": list(choice.table),
        }
    print(json.dumps(result, indent=2, default=str))
    return 0 if verdict.eligible else 3


def _run_export(export) -> int:
    """Exit codes: 0 exported, 2 failed, 4 skipped (this host cannot export this window)."""
    from pipeline.exporting import ExportError, SkipExport

    try:
        report = export()
    except SkipExport as error:
        print(f"export skipped: {error}", file=sys.stderr)
        return 4
    except ExportError as error:
        print(f"export failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"file": report["files"], "window": report["window"]["context"]}, indent=2))
    return 0


def _verify(args) -> int:
    # Its own wrapper rather than _run_export's: the exit codes are the same, but a verify
    # returns how many windows it checked, not an export report with files and a window.
    from pipeline import verify as verify_module
    from pipeline.exporting import ExportError

    try:
        gate = Path(args.gate) if args.gate else None
        summary = verify_module.run(Path(args.out), args.backend, Path(args.work), gate_reference=gate)
    except ExportError as error:
        print(f"verify failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


def _audit(args) -> int:
    """Exit codes: 0 the published file still decides like its fp32 model, 2 it does not."""
    from pipeline import audit as audit_module
    from pipeline.exporting import ExportError

    try:
        return audit_module.main(args.model, Path(args.work), args.backend, args.context)
    except ExportError as error:
        print(f"audit failed: {error}", file=sys.stderr)
        return 2


def _solve(args) -> int:
    from pipeline import solve as solve_module

    report = solve_module.run(
        args.model, args.revision, Path(args.out), Path(args.work), damp=args.damp, reference_only=args.reference_only
    )
    print(json.dumps(report, indent=2))
    return 0


def _export_xnnpack(args) -> int:
    from pipeline import export_xnnpack

    return _run_export(
        lambda: export_xnnpack.run(
            args.model,
            args.revision,
            Path(args.out),
            Path(args.work),
            context=args.context,
            keep_work=args.keep_work,
            skip_smoke=args.skip_smoke,
            codes=Path(args.codes) if args.codes else None,
            qmode=args.qmode or None,
        )
    )


def _export_vulkan(args) -> int:
    from pipeline import export_xnnpack

    return _run_export(
        lambda: export_xnnpack.run(
            args.model,
            args.revision,
            Path(args.out),
            Path(args.work),
            context=args.context,
            keep_work=args.keep_work,
            backend="vulkan",
        )
    )


def _mtk_matrix(args) -> int:
    """One matrix entry per window: the window and the smallest runner that can build it.

    The workflow fans out on this rather than on a bare context list, so the runner size is
    decided by the same calibration_bytes and calibration_disk_bytes the export itself gates
    on. Two copies of that arithmetic, one in Python and one in YAML, would drift.
    """
    import dataclasses

    from pipeline import export_mtk, families, hub, remote, settings

    cfg = settings.load()
    source = hub.fetch(args.model, args.revision)
    if not source.config or not source.total_params:
        print(f"no config or parameter count for {args.model}", file=sys.stderr)
        return 2
    if args.contexts:
        try:
            wanted = json.loads(args.contexts)
        except ValueError:
            wanted = [v for v in args.contexts.strip("[]").split(",") if v.strip()]
        if not isinstance(wanted, list):
            wanted = [wanted]
    else:
        wanted = list(cfg.context_tiers) + [cfg.mtk.cache_size]

    # lfm2.py streams its calibration and sizes differently from MediaTek's Arrow round trip.
    family = families.family_for(source.config)
    plan = families.mtk_plan(family, source.config, cfg.mtk.max_chunks) if family else None
    streaming = plan is not None and plan.script in export_mtk.STREAMING_SCRIPTS

    entries = []
    for window in sorted({int(v) for v in wanted}, reverse=True):
        recipe = dataclasses.replace(cfg.mtk, cache_size=window)
        tier = export_mtk.pick_runner(source.config, recipe, args.prompts, source.total_params, streaming=streaming)
        if tier is None:
            # Left out rather than failed: the other windows are independent and a matrix
            # that refuses to start teaches less than nine jobs that finish.
            print(f"no runner tier can build a {window}-token window", file=sys.stderr)
            continue
        entry = {"context": window, "runner": tier.label, "swap_gib": tier.swap_gib, "max_chunks": ""}
        if window >= export_mtk.ONE_ATTENTION_PER_CHUNK_FROM:
            chunks = families.mtk_chunks_one_attention_each(source.config)
            if chunks is not None and chunks > cfg.mtk.max_chunks:
                entry["max_chunks"] = str(chunks)
        if tier.label == settings.MODAL_TIER:
            need = export_mtk.calibration_bytes(
                source.config, recipe, args.prompts, source.total_params, streaming=streaming
            )
            entry["memory_mib"] = remote.memory_request(max(need, export_mtk.MODAL_FLOOR_BYTES), cfg.modal)
            if entry["memory_mib"] is None:
                print(f"{window} needs more than Modal allows ({need:,} B)", file=sys.stderr)
                continue
            entry["cpu"], entry["timeout_minutes"] = export_mtk.modal_shape(window)
            print(f"{window}: needs {need / 2**30:.1f} GiB, {entry}", file=sys.stderr)
        entries.append(entry)
    print(json.dumps(entries))
    return 0


def _export_matrix(args) -> int:
    """One matrix entry per window for an export_llm backend: the window and the smallest
    runner that can build it (export_xnnpack.pick_runner), from the model's config alone.

    The job re-measures its host and gates on the same sizing.host_need_bytes, so a window
    this sends to a tier is one the job will not refuse, and one no tier can carry is left
    out here instead of being started on a runner it would kill.
    """
    from pipeline import export_xnnpack, families, hub, remote, settings, sizing

    cfg = settings.load()
    recipe = cfg.vulkan if args.backend == "vulkan" else cfg.xnnpack
    if not recipe.runner_tiers:
        print(f"no runner_tiers configured for {args.backend} in config/pipeline.yaml", file=sys.stderr)
        return 2
    source = hub.fetch(args.model, args.revision)
    family = families.family_for(source.config) if source.config else None
    if family is None or not source.total_params:
        print(f"no known family or parameter count for {args.model}", file=sys.stderr)
        return 2
    arch = families.architecture(source.config, source.total_params)
    if args.contexts:
        try:
            wanted = json.loads(args.contexts)
        except ValueError:
            wanted = [v for v in args.contexts.strip("[]").split(",") if v.strip()]
        if not isinstance(wanted, list):
            wanted = [wanted]
    else:
        wanted = list(cfg.context_tiers)

    entries = []
    for window in sorted({int(v) for v in wanted}, reverse=True):
        need = sizing.host_need_bytes(arch, window, args.backend)
        tier = export_xnnpack.pick_runner(arch, window, recipe.runner_tiers, args.backend)
        if tier is None:
            print(f"no runner tier can build a {window}-token window (needs {need:,} B)", file=sys.stderr)
            continue
        entry = {"context": window, "runner": tier.label, "swap_gib": tier.swap_gib}
        if tier.label == settings.MODAL_TIER:
            # The Sandbox is sized to the window, not the tier: what the estimate needs plus
            # the margin, which pick_runner has already checked is within Modal's ceiling.
            entry["memory_mib"] = remote.memory_request(need, cfg.modal)
            if entry["memory_mib"] is None:
                print(f"{window} needs more than Modal allows ({need:,} B)", file=sys.stderr)
                continue
        print(f"{window}: {tier.label}, needs {need / 2**30:.1f} GiB", file=sys.stderr)
        entries.append(entry)
    print(json.dumps(entries))
    return 0


# The GPTQ solve holds the fp32 model and one layer's Hessians: on Blacksmith every solve up to
# LFM2.5-2.6B fit 23.4 GiB and Qwen3-4B took the 48 GB size, about 12 bytes a parameter.
SOLVE_BYTES_PER_PARAM = 12
SOLVE_FLOOR_BYTES = 16 * 2**30


def _solve_memory(args) -> int:
    """MiB for the solve's Modal Sandbox, from the model's parameter count."""
    from pipeline import hub, remote, settings

    cfg = settings.load()
    source = hub.fetch(args.model, args.revision)
    need = max(SOLVE_FLOOR_BYTES, SOLVE_BYTES_PER_PARAM * (source.total_params or 0))
    mib = remote.memory_request(need, cfg.modal)
    if mib is None:
        print(f"the solve needs more than Modal allows ({need:,} B)", file=sys.stderr)
        return 2
    print(mib)
    return 0


def _modal_run(args) -> int:
    from pipeline import remote

    return remote.main(args)


def _export_mtk(args) -> int:
    from pipeline import export_mtk

    if not args.tool_python or not args.examples:
        print("export failed: set --tool-python and --examples (or MTK_PYTHON and MTK_EXAMPLES)", file=sys.stderr)
        return 2
    return _run_export(
        lambda: export_mtk.run(
            args.model,
            args.revision,
            args.soc,
            Path(args.out),
            Path(args.work),
            tool_python=args.tool_python,
            examples_dir=Path(args.examples),
            keep_work=args.keep_work,
            context=args.context,
        )
    )


def _export_qnn(args) -> int:
    from pipeline import export_qnn

    return _run_export(
        lambda: export_qnn.run(
            args.model,
            args.revision,
            args.soc,
            Path(args.out),
            Path(args.work),
            keep_work=args.keep_work,
            context=args.context,
        )
    )


def _publish_hf(args) -> int:
    from pipeline import publish

    print(publish.publish_hf(Path(args.out), args.backend, args.target, args.context))
    return 0


def _correct_reports(args) -> int:
    import subprocess

    from pipeline import correct

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    for repo in args.repos:
        print(json.dumps(correct.correct_repo(repo, commit or "unknown", dry_run=not args.apply), indent=2))
    return 0


def _publish_release(args) -> int:
    from pipeline import publish

    print(publish.publish_release(Path(args.out), args.backend, args.target, args.context))
    return 0


def _summary(args) -> int:
    from pipeline import publish

    sys.stdout.write(publish.summary(Path(args.out), args.backend, args.target, args.context))
    return 0


def _watch(args) -> int:
    from pipeline import settings, watch

    cfg = settings.load()
    state_path = Path(args.state)
    backfill = tuple(m.strip() for m in args.backfill.split(",") if m.strip())
    summary = watch.run(
        state_path,
        cfg,
        dispatch=not args.dry_run,
        backfill=backfill,
        requeue=args.requeue,
        limit_per_org=cfg.limit_per_org,
        max_dispatch=cfg.max_dispatch_per_run,
    )
    if not args.dry_run:
        watch.save_state(state_path, summary["state"])
    text = watch.markdown(summary)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 1 if summary["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="eligibility and window choice for one model, no download")
    plan.add_argument("model")
    plan.add_argument("--revision", default="main")
    plan.add_argument("--qmode", default="8da4w", help="recipe to size the phone estimate for: 8da4w or fp32")
    plan.set_defaults(func=_plan)

    export = commands.add_parser("export-xnnpack", help="download, convert, export and smoke-test")
    export.add_argument("model")
    export.add_argument("--revision", default="main")
    export.add_argument("--out", default="out")
    export.add_argument("--work", default="work")
    export.add_argument(
        "--context", type=int, default=None, help="window in tokens (default: the largest this host can build)"
    )
    export.add_argument(
        "--keep-work", action="store_true", help="keep the work dir after success (a failure always leaves it)"
    )
    export.add_argument("--skip-smoke", action="store_true")
    export.add_argument(
        "--codes",
        default=None,
        help="codes.pt from `solve`; without it the int4 weights are rounded to nearest",
    )
    export.add_argument("--qmode", default="", help="recipe: 8da4w (config default) or fp32 (linears not quantized)")
    export.set_defaults(func=_export_xnnpack)

    verify = commands.add_parser("verify", help="smoke-test an exported .pte on a host whose runner works")
    verify.add_argument("out")
    verify.add_argument("--backend", default="xnnpack")
    verify.add_argument("--work", default="verify-work")
    verify.add_argument(
        "--gate",
        default=None,
        help="gate.json from `solve`: the file must decide like the fp32 model before it publishes",
    )
    verify.set_defaults(func=_verify)

    audit = commands.add_parser("audit", help="gate a file that is already published, without rebuilding it")
    audit.add_argument("model", help="the SOURCE model id, e.g. Qwen/Qwen3-1.7B")
    audit.add_argument("--backend", default="xnnpack")
    audit.add_argument("--work", default="audit-work")
    audit.add_argument("--context", type=int, default=None, help="which window (default: any, they read alike)")
    audit.set_defaults(func=_audit)

    solve = commands.add_parser("solve", help="GPTQ the int4 codes once per model, for every window to reuse")
    solve.add_argument("model")
    solve.add_argument("--revision", default="main")
    solve.add_argument("--out", default="codes.pt")
    solve.add_argument("--work", default="work")
    solve.add_argument("--damp", type=float, default=0.01, help="Hessian damping, as a fraction of its mean diagonal")
    solve.add_argument(
        "--reference-only",
        action="store_true",
        help="write only the gate's fp32 reference (gate.json), no codes: for a recipe that is not solved",
    )
    solve.set_defaults(func=_solve)

    vulkan = commands.add_parser("export-vulkan", help="download, convert, export a Vulkan (GPU) .pte")
    vulkan.add_argument("model")
    vulkan.add_argument("--revision", default="main")
    vulkan.add_argument("--out", default="out")
    vulkan.add_argument("--work", default="work")
    vulkan.add_argument(
        "--context", type=int, default=None, help="window in tokens (default: the largest this host can build)"
    )
    vulkan.add_argument(
        "--keep-work", action="store_true", help="keep the work dir after success (a failure always leaves it)"
    )
    vulkan.set_defaults(func=_export_vulkan)

    qnn = commands.add_parser("export-qnn", help="compile a Qualcomm HTP .pte for one chip")
    qnn.add_argument("model")
    qnn.add_argument("--soc", required=True, help="e.g. SM8650")
    qnn.add_argument("--revision", default="main")
    qnn.add_argument("--out", default="out")
    qnn.add_argument("--work", default="work")
    qnn.add_argument("--context", type=int, default=None, help="window instead of qnn.max_context_len")
    qnn.add_argument(
        "--keep-work", action="store_true", help="keep the work dir after success (a failure always leaves it)"
    )
    qnn.set_defaults(func=_export_qnn)

    mtk = commands.add_parser("export-mtk", help="compile MediaTek NeuroPilot .pte chunks for one chip")
    mtk.add_argument("model")
    mtk.add_argument("--soc", required=True, help="MT6989 or MT6991")
    mtk.add_argument("--revision", default="main")
    mtk.add_argument("--out", default="out")
    mtk.add_argument("--work", default="work")
    mtk.add_argument("--context", type=int, default=None, help="window instead of mtk.cache_size")
    mtk.add_argument(
        "--tool-python",
        default=os.environ.get("MTK_PYTHON"),
        help="Python 3.10 with requirements/mtk-tools.txt and MediaTek's wheels (env MTK_PYTHON)",
    )
    mtk.add_argument(
        "--examples",
        default=os.environ.get("MTK_EXAMPLES"),
        help="ExecuTorch's examples/mediatek at EXECUTORCH_COMMIT (env MTK_EXAMPLES)",
    )
    mtk.add_argument("--keep-work", action="store_true")
    mtk.set_defaults(func=_export_mtk)

    matrix = commands.add_parser("mtk-matrix", help="windows and the runner each needs, as a CI matrix")
    matrix.add_argument("model")
    matrix.add_argument("--revision", default="main")
    matrix.add_argument("--contexts", default="", help="JSON or comma-separated windows; empty for every tier")
    matrix.add_argument(
        "--prompts",
        type=int,
        default=9,
        help="lines in mtk.calibration, which sets the disk the Arrow cache needs; the export "
        "counts the real file and its disk gate catches a mismatch",
    )
    matrix.set_defaults(func=_mtk_matrix)

    export_matrix = commands.add_parser(
        "export-matrix", help="windows and the runner each needs for an export_llm backend, as a CI matrix"
    )
    export_matrix.add_argument("model")
    export_matrix.add_argument("--revision", default="main")
    export_matrix.add_argument("--backend", default="vulkan", choices=["vulkan", "xnnpack"])
    export_matrix.add_argument("--contexts", default="", help="JSON or comma-separated windows; empty for every tier")
    export_matrix.set_defaults(func=_export_matrix)

    solve_memory = commands.add_parser("solve-memory", help="MiB to request for the GPTQ solve on Modal")
    solve_memory.add_argument("model")
    solve_memory.add_argument("--revision", default="main")
    solve_memory.set_defaults(func=_solve_memory)

    modal_run = commands.add_parser(
        "modal-run", help="run a step in a Modal Sandbox sized to it, and bring its outputs back (pipeline/remote.py)"
    )
    modal_run.add_argument("--script", required=True, help="bash run in the repo, with $IN, $OUT and $WORK set")
    modal_run.add_argument("--memory-mib", type=int, required=True)
    modal_run.add_argument("--cpu", type=float, default=None, help="physical cores; default modal.cpu")
    modal_run.add_argument("--requirements", default="requirements/export-xnnpack.txt")
    modal_run.add_argument("--put", action="append", help="a file or directory to upload to $IN (repeatable)")
    modal_run.add_argument("--get", help="local directory that receives $OUT")
    modal_run.add_argument("--env", action="append", help="NAME=VALUE set in the Sandbox (repeatable)")
    modal_run.add_argument("--keep", action="store_true", help="leave the run's files on the Volume")
    modal_run.add_argument("--timeout-minutes", type=int, default=None, help="default modal.timeout_minutes")
    modal_run.add_argument("--mtk-tools", help="requirements for MediaTek's Python 3.10 virtualenv in the image")
    modal_run.set_defaults(func=_modal_run)

    watch = commands.add_parser("watch", help="check the watched orgs, dispatch exports, update state")
    watch.add_argument("--state", required=True, help="state JSON (on the state branch)")
    watch.add_argument("--backfill", default="", help="comma-separated model ids to evaluate and export now")
    watch.add_argument("--dry-run", action="store_true", help="dispatch nothing and leave the state file alone")
    watch.add_argument("--summary", default=None, help="append the markdown summary to this file")
    watch.add_argument(
        "--requeue", action="store_true", help="put every dispatched export back in the queue (after cancelling runs)"
    )
    watch.set_defaults(func=_watch)

    fix = commands.add_parser(
        "correct-reports", help="rewrite fp32 reports' recipe and sizing fields on the Hub; never touches a .pte"
    )
    fix.add_argument(
        "repos", nargs="+", help="output repos, e.g. experimentalmachines/SmolLM2-360M-Instruct-ExecuTorch"
    )
    fix.add_argument("--apply", action="store_true", help="commit the corrections (default: report what would change)")
    fix.set_defaults(func=_correct_reports)

    for name, func, text in (
        ("publish-hf", _publish_hf, "commit one backend folder to the output HF repo"),
        ("publish-release", _publish_release, "attach one backend's files to a GitHub release"),
        ("summary", _summary, "markdown summary of one backend's export"),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument("out")
        command.add_argument("--backend", required=True, choices=["xnnpack", "vulkan", "qnn", "mtk"])
        command.add_argument("--target", default=None)
        command.add_argument("--context", type=int, default=None, help="which window, when the folder holds several")
        command.set_defaults(func=func)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
