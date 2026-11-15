# Third-party software used by this pipeline

This repository contains no third-party SDKs. The export workflows download them from
their publishers at run time and keep them only in this repository's GitHub Actions cache.
Published model repos contain the compiled model files, never these SDKs or their runtime
libraries.

## ExecuTorch, PyTorch, torchao

Installed from PyPI (`requirements/*.txt`). ExecuTorch and PyTorch are under BSD-style
licenses; torchao under BSD-3-Clause.

`third_party/executorch/` holds a few model params files copied unchanged from the
ExecuTorch v1.4.0 source (the wheel does not package them), with ExecuTorch's BSD license
beside them in `third_party/executorch/LICENSE`.

## Qualcomm AI Runtime SDK (QAIRT) 2.37.0.250724

Used by `export-qnn.yml` to compile QNN HTP context binaries. The executorch 1.4.0 wheel
downloads it on first import from Qualcomm's public Software Center URL
(`https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.37.0.250724/v2.37.0.250724.zip`).

It is licensed by Qualcomm Technologies, Inc. under the "AI Stack License" (Terms and
Conditions of Use, `LICENSE.pdf` in the SDK). By its own terms, downloading or using the
SDK is acceptance of that agreement, which every run of `export-qnn.yml` does on behalf of
whoever operates this repository. Points that bear on this pipeline, paraphrased from the
agreement (read `LICENSE.pdf` for the binding text):

- Section 1 grants a license to use and copy the SDK to develop software applications, and
  to distribute it only in object code as incorporated in such an application, never
  standalone. This pipeline does not redistribute the SDK.
- Section 2 prohibits reverse engineering, makes the user responsible for inputs and
  outputs, lists prohibited and high-risk use cases, and requires keeping copyright notices.
- Section 10.f subjects Qualcomm products and their direct products to U.S. export
  control and sanctions laws.
- The SDK's third-party notices are in `NOTICE.txt` and `QNN_NOTICE.txt` in the SDK.

Model cards of repos with `qnn/` folders state that the context binaries were compiled with
QAIRT 2.37.0.250724 and that no Qualcomm SDK or runtime library is included.

## MediaTek NeuroPilot Express SDK 8.0.8 (build 20250925)

Used by `export-mtk.yml` to compile MediaTek NeuroPilot model binaries for MT6989
(Dimensity 9300) and MT6991 (Dimensity 9400). Each run downloads
`neuropilot-express-sdk-8.0.8-build20250925.tar.gz` from the URL MediaTek lists on its
NeuroPilot Express SDK page (`NEUROPILOT_SDK_URL` in `config/versions.env`), checks it against
`NEUROPILOT_SDK_SHA256`, installs its `mtk_converter` 8.13.0 and `mtk_neuron` 8.2.23 wheels
into a throwaway environment, and deletes the archive. Nothing from the SDK is committed to
this repository, cached, uploaded as an artifact, or published. (The first runs used the
earlier build 20250327 that ExecuTorch's CI installs; its `mtk_neuron` 8.2.19 lacks a function
ExecuTorch's LLM export needs.)

The SDK is © MediaTek Inc. and licensed under MediaTek's "Terms and Conditions of Use for
NeuroPilot Express SDK License" (`LICENSE AGREEMENT.pdf` in the archive; the document is
marked MediaTek Confidential, so it is not reproduced here). The 8.0.8 archive's agreement is
byte-identical to build 20250327's (SHA-256
`966215c3036abce09af48828be520cff91346775c19141437b9eb5f7069dd911` for both). Accessing or
using the SDK is acceptance of that agreement, which every run of `export-mtk.yml` does on
behalf of whoever operates this repository; ExperimentalMachines accepted it on 2026-09-13.
Points that bear on this pipeline, paraphrased (read the agreement for the binding text):

- The license is non-exclusive, non-transferable and revocable, to use the SDK for developing
  applications used with MediaTek chipsets, and to distribute it only in object code as part
  of such an application, never standalone. This pipeline does not redistribute the SDK.
- No reverse engineering, and MediaTek's copyright and proprietary notices must be kept.
- No action may subject any part of the SDK to open-source license terms (the agreement
  names, among others, the GPL, BSD and Apache licenses). This repository's code only
  downloads and runs the SDK; it contains no part of it.
- The user is responsible for inputs and outputs, and must not use the SDK for the
  applications the agreement lists (such as social scoring or biometric categorisation) where
  applicable law prohibits them.
- The user indemnifies MediaTek, MediaTek's liability is capped, MediaTek may terminate the
  agreement at any time, U.S. and other export-control laws apply, and Singapore law governs.

Model cards of repos with `mtk/` folders state that the binaries were compiled with this SDK
from MediaTek Inc. and that no MediaTek SDK or runtime library is included.

## ExecuTorch examples/mediatek

`export-mtk.yml` also fetches `examples/mediatek` from the ExecuTorch source at
`EXECUTORCH_COMMIT` (v1.5.1) at run time; those scripts are © MediaTek Inc., under the
BSD-style license in ExecuTorch's repository root.
