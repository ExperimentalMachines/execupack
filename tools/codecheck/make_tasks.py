"""Every XNNPACK and Vulkan .pte in the org, split into N shards by repo (a shard keeps one
source model's state in memory at a time), for parallel scan.py workers.

    python make_tasks.py <org> <n_shards> <out_dir> [--only REGEX | --exclude REGEX]
"""

import json
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi

org, n, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
only = re.compile(sys.argv[sys.argv.index("--only") + 1]) if "--only" in sys.argv else None
exclude = re.compile(sys.argv[sys.argv.index("--exclude") + 1]) if "--exclude" in sys.argv else None
out.mkdir(parents=True, exist_ok=True)
api = HfApi()
repos = []
for m in api.list_models(author=org):
    if (only and not only.search(m.id)) or (exclude and exclude.search(m.id)):
        continue
    info = api.model_info(m.id, files_metadata=True)
    ptes = sorted(
        s.rfilename
        for s in info.siblings
        if s.rfilename.endswith(".pte") and s.rfilename.split("/")[0] in ("xnnpack", "vulkan")
    )
    if ptes:
        size = sum(s.size or 0 for s in info.siblings if s.rfilename in ptes)
        repos.append((size, m.id, ptes))
repos.sort(reverse=True)
shards = [[] for _ in range(n)]
load = [0] * n
for size, repo, ptes in repos:  # biggest first onto the lightest shard
    i = load.index(min(load))
    shards[i] += [[repo, p] for p in ptes]
    load[i] += size
for i, s in enumerate(shards):
    (out / f"tasks-{i}.json").write_text(json.dumps(s, indent=0))
print(len(repos), "repos,", sum(len(s) for s in shards), "files,", [f"{x / 1e9:.0f} GB" for x in load])
