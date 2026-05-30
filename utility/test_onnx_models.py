import onnxruntime as ort
import numpy as np


onnx_path = "resources/torch/peak/models/mlp.onnx"

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Input name: {input_name}")
print(f"Output name: {output_name}")

X1 = np.random.rand(1, 920).astype(np.float32)
out1 = session.run([output_name], {input_name: X1})[0]
print(f"Output (batch=1): shape={out1.shape}, first batch prediction ={out1[:3]}")

X1024 = np.random.rand(1024, 920).astype(np.float32)
out1024 = session.run([output_name], {input_name: X1024})[0]
print(
    f"Output (batch=1024): shape={out1024.shape}, first 3 batches predictions ={out1024[:3]}"
)

X10 = np.random.rand(10, 920).astype(np.float32)
out10 = session.run([output_name], {input_name: X10})[0]
print(
    f"Output (batch=10): shape={out10.shape}, first 3 batches predictions ={out10[:3]}"
)
