"""Does MediaTek's converter take Gemma 2's 256000 x 2304 head? (docs/research, finding 47)

    /opt/mtk-venv/bin/python mtk-gemma2-head-probe.py <plain|fp16|fold>

Gemma 2 2B's final norm and tied output head alone, quantized A16W8 with NeuropilotQuantizer and
lowered for MT6991 as gemma.py lowers a chunk (ExecuTorch 3b60683, mtk_converter 8.13.0,
mtk_neuron 8.2.23). plain: as gemma.py does it; fp16: the converter's
convert_float32_weights_to_float16; fold: convert_pt2e(fold_quantize=True), int8 weight
constants. All three: "ValueError: Message mtk_converter.ModelProto exceeds maximum protobuf
size of 2GB" (mtk-gemma2-head-probe.log).
"""
import struct
import sys

import mtk_converter
import torch
from executorch import exir
from executorch.backends.mediatek import NeuropilotPartitioner, NeuropilotQuantizer, Precision
from executorch.exir.backend.backend_api import MethodProgramsPartitionerSpec, to_backend
from executorch.exir.backend.backend_details import CompileSpec
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

mode = sys.argv[1]
if mode == "fp16":
    original = mtk_converter.PyTorchV2Converter.from_exported_program

    def from_exported_program(*args, **kwargs):
        converter = original(*args, **kwargs)
        converter.convert_float32_weights_to_float16 = True
        print("converter: convert_float32_weights_to_float16 = True", flush=True)
        return converter

    mtk_converter.PyTorchV2Converter.from_exported_program = staticmethod(from_exported_program)


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(2304)
        self.lm_head = torch.nn.Linear(2304, 256000, bias=False)

    def forward(self, x):
        return self.lm_head(self.norm(x))


torch.manual_seed(0)
model = Head().eval()
x = torch.randn(1, 1, 2304)
graph = torch.export.export(model, (x,), strict=True).module()
quantizer = NeuropilotQuantizer()
quantizer.setup_precision(Precision.A16W8)
prepared = prepare_pt2e(graph, quantizer)
prepared(x)
converted = convert_pt2e(prepared, fold_quantize=(mode == "fold"))
edge = exir.to_edge(torch.export.export(converted, (x,), strict=True)).exported_program()
spec = [CompileSpec("gno", b"LTS"), CompileSpec("gno-exp", b""), CompileSpec("gno-non-4d-tiling", b""),
        CompileSpec("ImportForever", struct.pack("?", True)), CompileSpec("platform-config", b"mt6991")]
delegated = to_backend(MethodProgramsPartitionerSpec({"head": edge}, {"head": NeuropilotPartitioner(spec)}))
program = exir.EdgeProgramManager(delegated, compile_config=exir.EdgeCompileConfig(_check_ir_validity=False))
buffer = program.to_executorch().buffer
print(f"OK {mode}: lowered, {len(buffer):,} bytes", flush=True)
