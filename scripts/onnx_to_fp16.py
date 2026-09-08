#!/usr/bin/env python3
"""Конвертация ONNX-модели в FP16 (strongly typed) для TensorRT 11+.
Использование: scripts/onnx_to_fp16.py model.onnx model-fp16.onnx
TensorRT 11 убрал флаг --fp16: сеть исполняется в заявленной точности ONNX,
поэтому FP16-инференс требует FP16-модели."""
import sys
import onnx
from onnx import numpy_helper, TensorProto

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph

for init in g.initializer:
    if init.data_type == TensorProto.FLOAT:
        arr = numpy_helper.to_array(init).astype('float16')
        init.CopyFrom(numpy_helper.from_array(arr, init.name))

for v in list(g.input) + list(g.output) + list(g.value_info):
    if v.type.HasField('tensor_type') and v.type.tensor_type.elem_type == TensorProto.FLOAT:
        v.type.tensor_type.elem_type = TensorProto.FLOAT16

onnx.save(m, dst)
print(f'{src} -> {dst} (fp16)')
