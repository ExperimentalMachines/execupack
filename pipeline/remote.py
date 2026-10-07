"""Runs a pipeline step on Modal, in a Sandbox sized to the step, and brings its outputs back.

    python -m pipeline modal-run --memory-mib 12288 --requirements requirements/export-xnnpack.txt \\
        --put "$WORK_DIR/codes.pt" --get "$WORK_DIR/out" \\
        --script 'python -m pipeline export-xnnpack "$MODEL_ID" --context "$CONTEXT" --out "$OUT" --work "$WORK"'

The GitHub job stays small (the Namespace profile) and waits here; the step itself runs in a
Sandbox with the memory its window needs, which a fixed runner cannot offer: an export's peak
grows with the window squared, from a few GiB at 2k to 134 GiB for Qwen3-4B at 32k. Modal
gives a Sandbox up to 344064 MiB and 64 cores (measured 2026-10-07), with no swap.

How a step gets in and out:

- The image is debian slim with the Python of config/versions.env, torch and torchvision from
  the CPU index (as .github/actions/setup-export installs them), then the requirements file,
  then this repository's pipeline, config, third_party and scripts. Modal caches each layer,
  so a run after the first rebuilds only what changed.
- Files given with --put are uploaded to $IN/ on a v2 Volume under a key unique to the run.
- The script runs under bash in /root/execupack, with $IN, $OUT (empty, on the Volume) and
  $WORK (container disk) set, the environment given with --env, and HF_TOKEN passed as a
  Modal secret built from this process's environment, so no token is stored on Modal.
- Afterwards `sync` commits $OUT, the files are downloaded into the --get directory, and the
  key is deleted from the Volume. The script's exit code is this command's, so exit 4 (the
  export's "this window cannot be built") keeps its meaning in the workflow.

Inside a Sandbox /proc/meminfo reports the host's memory, not the request, so a step's own
memory gate always passes there; the matrix that sized the request is the gate that counts.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path

from pipeline import settings

ROOT = settings.ROOT
# What the image carries of this repository, at /root/execupack.
REPO_DIRS = ("pipeline", "config", "third_party", "scripts", "requirements")
REMOTE_ROOT = "/root/execupack"
VOLUME_MOUNT = "/vol"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
# Passed into the Sandbox when set here, besides --env.
PASSTHROUGH = ("HF_HUB_DISABLE_PROGRESS_BARS", "MODEL_ID", "REVISION", "CONTEXT", "QMODE", "GPTQ")


def python_version() -> str:
    return settings.read_env_file(settings.CONFIG_DIR / "versions.env")["PYTHON_VERSION"]


def cpu_pins(requirements: Path) -> list[str]:
    """torch and torchvision, which come from the CPU index: PyPI's Linux wheels pull CUDA."""
    return [
        line.strip()
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if re.match(r"^(torch|torchvision)==", line.strip())
    ]


def memory_request(need_bytes: int, cfg: settings.ModalSettings) -> int | None:
    """MiB to ask Modal for a step that needs `need_bytes`, or None past what Modal allows."""
    mib = -(-int(need_bytes * cfg.memory_margin) // 2**20)
    return mib if mib <= cfg.max_memory_mib else None


def image(requirements: Path):
    import modal

    img = modal.Image.debian_slim(python_version=python_version()).apt_install("git", "build-essential")
    pins = cpu_pins(requirements)
    if pins:
        img = img.pip_install(*pins, index_url=CPU_INDEX)
    img = img.pip_install_from_requirements(str(requirements))
    for name in REPO_DIRS:
        img = img.add_local_dir(str(ROOT / name), f"{REMOTE_ROOT}/{name}", ignore=["**/__pycache__/**"])
    return img


def _stream(stream, out) -> None:
    for chunk in stream:
        out.write(chunk)
        out.flush()


def _download(volume, prefix: str, dest: Path) -> int:
    """Every file under `prefix` on the Volume, into `dest` at the same relative paths."""
    count = 0
    for entry in volume.listdir(prefix, recursive=True):
        if entry.type.name != "FILE":
            continue
        # Listings name paths with or without the leading slash; compare without it.
        relative = entry.path.lstrip("/")[len(prefix.lstrip("/")) :].lstrip("/")
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as handle:
            for chunk in volume.read_file(entry.path):
                handle.write(chunk)
        count += 1
    return count


def run(
    script: str,
    *,
    memory_mib: int,
    requirements: Path,
    puts: list[Path],
    get: Path | None,
    env: dict[str, str],
    cpu: float | None = None,
    keep: bool = False,
) -> int:
    import modal

    cfg = settings.load().modal
    if cfg is None:
        raise SystemExit("config/pipeline.yaml has no modal section")
    if memory_mib > cfg.max_memory_mib:
        raise SystemExit(f"{memory_mib} MiB is more than Modal allows ({cfg.max_memory_mib})")

    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    key = f"{run_id}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}-{uuid.uuid4().hex[:8]}"
    base = f"{VOLUME_MOUNT}/{key}"
    volume = modal.Volume.from_name(cfg.volume, create_if_missing=True, version=2)
    app = modal.App.lookup(cfg.app, create_if_missing=True)

    secrets = []
    if os.environ.get("HF_TOKEN"):
        secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))
    sandbox_env = {name: os.environ[name] for name in PASSTHROUGH if os.environ.get(name)}
    sandbox_env.update(env)
    sandbox_env.update(IN=f"{base}/in", OUT=f"{base}/out", WORK="/work", HF_HOME="/work/hf-home", PYTHONUNBUFFERED="1")
    cores = cpu or cfg.cpu

    sandbox = None
    code = None
    try:
        if puts:
            with volume.batch_upload() as batch:
                for path in puts:
                    if path.is_dir():
                        batch.put_directory(str(path), f"/{key}/in/{path.name}")
                    elif path.exists():
                        batch.put_file(str(path), f"/{key}/in/{path.name}")
            print(f"uploaded {len(puts)} input(s) to {cfg.volume}:/{key}/in", flush=True)

        print(f"Modal Sandbox: {memory_mib} MiB, {cores:g} cores, timeout {cfg.timeout_minutes} min", flush=True)
        started = time.time()
        with modal.enable_output():
            sandbox = modal.Sandbox.create(
                "bash",
                "-c",
                wrap(script),
                app=app,
                image=image(requirements),
                memory=memory_mib,
                cpu=cores,
                timeout=cfg.timeout_minutes * 60,
                volumes={VOLUME_MOUNT: volume},
                secrets=secrets,
                env=sandbox_env,
                workdir=REMOTE_ROOT,
            )
        readers = [
            threading.Thread(target=_stream, args=(sandbox.stdout, sys.stdout), daemon=True),
            threading.Thread(target=_stream, args=(sandbox.stderr, sys.stderr), daemon=True),
        ]
        for reader in readers:
            reader.start()
        try:
            # raise_on_termination=False still raises on the Sandbox's own timeout (client 1.6.1).
            sandbox.wait(raise_on_termination=False)
            code = sandbox.returncode
        except modal.exception.SandboxTimeoutError:
            print(f"::error::the Modal Sandbox ran past its {cfg.timeout_minutes}-minute timeout", flush=True)
        for reader in readers:
            reader.join(timeout=30)
        print(f"Modal Sandbox exited {code} after {(time.time() - started) / 60:.1f} min", flush=True)

        if get is not None:
            got = _download(volume, f"/{key}/out", get)
            print(f"downloaded {got} file(s) to {get}", flush=True)
    finally:
        # Whatever happened, nothing is left running or stored: a Sandbox still alive (an
        # interrupted wait) is stopped, and the run's inputs and outputs leave the Volume even
        # if stopping it fails.
        try:
            if sandbox is not None and sandbox.poll() is None:
                sandbox.terminate()
        finally:
            if not keep:
                try:
                    volume.remove_file(f"/{key}", recursive=True)
                except (FileNotFoundError, modal.exception.NotFoundError):
                    # Nothing was written under the key (the step failed before it did); the
                    # original error is the one to see.
                    pass
    # None means it was killed (the timeout, or Modal's out-of-memory killer).
    return 1 if code is None else code


def wrap(script: str) -> str:
    """The Sandbox's bash: the step, with the Volume committed on every way out.

    An EXIT trap, so a step that ends with `exit` (the export's `|| exit $?`, the gate's
    `exit 1`) still has its reports committed by `sync` before the job downloads them, and its
    status, exit 4 included, is the Sandbox's. A failed sync fails a step that had succeeded.
    """
    return (
        "finish() {\n"
        "  rc=$?\n"
        f"  if ! sync {VOLUME_MOUNT}; then echo 'sync of the Volume failed' >&2; [ \"$rc\" -eq 0 ] && rc=1; fi\n"
        '  exit "$rc"\n'
        "}\n"
        "trap finish EXIT\n"
        'mkdir -p "$IN" "$OUT" "$WORK"\n'
        f"{script}\n"
    )


def main(args) -> int:
    env = {}
    for item in args.env or []:
        name, _, value = item.partition("=")
        env[name] = value
    return run(
        args.script,
        memory_mib=args.memory_mib,
        requirements=Path(args.requirements),
        puts=[Path(p) for p in args.put or []],
        get=Path(args.get) if args.get else None,
        env=env,
        cpu=args.cpu,
        keep=args.keep,
    )
