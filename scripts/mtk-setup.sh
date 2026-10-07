#!/usr/bin/env bash
# MediaTek's half of an export, set up inside the Modal Sandbox before export-mtk runs:
#
#   bash scripts/mtk-setup.sh && python -m pipeline export-mtk ... \
#       --tool-python /opt/mtk-venv/bin/python --examples "$WORK/executorch/examples/mediatek"
#
# The image already has the Python 3.10 virtualenv with the public half of
# requirements/mtk-tools.txt (pipeline/remote.py, image). This adds what may not be stored:
# the NeuroPilot Express SDK is downloaded from MediaTek on every run, checked against the
# pinned digest, and deleted once its two wheels are installed. It is licensed by MediaTek
# Inc. under its own terms (THIRD_PARTY_NOTICES.md), which every run accepts, and nothing
# from it is cached, uploaded or published.
#
# Then the LLM export scripts, which are in the ExecuTorch source and not the wheel (BSD,
# fetched at the pinned commit), with this repo's patches from third_party/executorch/patches.
set -euo pipefail

eval "$(grep -E '^(NEUROPILOT_SDK_[A-Z0-9_]+|MTK_[A-Z_]+_VERSION|EXECUTORCH_COMMIT)=' config/versions.env | sed 's/^/export /')"
tool_python=/opt/mtk-venv/bin/python
work=${WORK:?}

sdk="$work/neuropilot"
mkdir -p "$sdk"
curl -fsSL --retry 3 --retry-all-errors -o "$sdk.tar.gz" "$NEUROPILOT_SDK_URL"
echo "$NEUROPILOT_SDK_SHA256  $sdk.tar.gz" | sha256sum -c -
tar xzf "$sdk.tar.gz" -C "$sdk" --strip-components=1 --wildcards \
  "*/mtk_neuron-${MTK_NEURON_VERSION}-*.whl" "*/mtk_converter-${MTK_CONVERTER_VERSION}+public-cp310-*.whl"
# The 8.0.8 archive keeps mtk_converter only under mtk_converter-*_public_packages/.
neuron_whl=$(find "$sdk" -name "mtk_neuron-${MTK_NEURON_VERSION}-*.whl" | head -1)
converter_whl=$(find "$sdk" -name "mtk_converter-${MTK_CONVERTER_VERSION}+public-cp310-*.whl" | head -1)
test -n "$neuron_whl" && test -n "$converter_whl"
uv pip install --quiet --python "$tool_python" -r requirements/mtk-tools.txt "$neuron_whl" "$converter_whl"
rm -rf "$sdk" "$sdk.tar.gz"
uv pip list --python "$tool_python" | grep -iE '^(torch|torchao|executorch|transformers|mtk.converter|mtk.neuron) '

src="$work/executorch"
rm -rf "$src"
git init -q "$src"
git -C "$src" remote add origin https://github.com/pytorch/executorch.git
git -C "$src" sparse-checkout set examples/mediatek
git -C "$src" fetch -q --depth 1 origin "$EXECUTORCH_COMMIT"
git -C "$src" checkout -q FETCH_HEAD
for patch in third_party/executorch/patches/mediatek-*.patch; do
  git -C "$src" apply -v "$PWD/$patch"
done
