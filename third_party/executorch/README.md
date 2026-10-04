Files copied unchanged from ExecuTorch v1.4.0 (commit
3dd7ccd1d863fad22639dd2d918ae34a41ce45f0) and byte-identical at v1.5.1 (commit
3b60683923245cf472b7323426920e15623ba361), under its BSD license in `LICENSE`.

`examples/models/*/…config.json` are the params files that ExecuTorch's Qualcomm LLM
registry (`examples/qualcomm/oss_scripts/llama/__init__.py`) points each `--decoder_model`
at. The executorch PyPI wheel packages the example `.py` and `.yaml` files but not these
`.json` files, so `pipeline/export_qnn.py` passes them to the script with `--params`. Keep
them in step with `EXECUTORCH_VERSION` in `config/versions.env`.
