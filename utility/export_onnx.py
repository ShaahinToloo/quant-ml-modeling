import numpy as np

import onnxruntime as ort
import torch


def export_to_onnx_model(input_dim, torch_model, path):
    example_input = torch.randn(1, input_dim, requires_grad=False)

    torch.onnx.export(
        torch_model,
        example_input,
        path,
        export_params=True,
        opset_version=18,
        dynamo=False,
        do_constant_folding=True,
        input_names=["X"],
        output_names=["probs"],
        dynamic_axes={"X": {0: "batch"}, "probs": {0: "batch"}},
        verbose=True,
    )


def export_onnx_and_eval(run_obj, torch_model, onnx_export_path):

    export_to_onnx_model(run_obj.input_dim, torch_model, onnx_export_path)

    # Gen sample
    example_input = torch.randn(5, run_obj.input_dim)
    example_input_np = example_input.numpy()

    # Get names for onnx Infer
    ort_session = ort.InferenceSession(onnx_export_path)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # Infer onnx
    onnx_probs = ort_session.run([output_name], {input_name: example_input_np})[0]

    # Infer torch
    torch_model.eval()
    with torch.no_grad():
        pytorch_probs = torch_model(example_input).numpy()


    print(
        "Max diff between PyTorch and ONNX:", np.max(np.abs(onnx_probs - pytorch_probs))
    )

    print("ONNX probs:\n", onnx_probs)
    print("PyTorch probs:\n", pytorch_probs)
