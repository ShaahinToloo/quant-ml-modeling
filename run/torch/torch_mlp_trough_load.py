import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

from src.data.load_data import read_csv
from src.utils.logging import logger
from torch_models.src.model.isotonic_layer import IsotonicLayer
from torch_models.src.model.mlp_calibrated import MLPWithIsotonic
from torch_models.src.run.run_trough import RunTrough

fractals_period = 7
model_seq_len = 32
model_look_ahead = 7

logger("Loading Data")
df = read_csv(
    "/home/void/ModelCoding/resources/ModelFeatureEngineering.csv",
    timestamp_index="timestamp",
)
main_dir = "resources/torch/trough"

run_obj = RunTrough(fractals_period, model_seq_len, model_look_ahead, df, main_dir)

run_obj.calculate_features(f"df_{model_seq_len}")
run_obj.calculate_labels(f"trough_labels_{model_seq_len}")
run_obj.create_data_loaders(
    batch_size=1024, val_size=0.15, test_size=0.50, do_shuffle=True
)


ckpt_path = "lightning_logs/version_1/checkpoints/best_model.ckpt"
run_obj.load_model_from_lightning_checkpoint(ckpt_path)

X_val = run_obj.X_test
y_val = run_obj.y_test

y_val_probs = run_obj.light_obj.predict_proba(X_val)
y_val_labels = y_val

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(y_val_probs, y_val_labels)
joblib.dump(iso, "resources/torch/trough/isotonic_scaler.pkl")
p_calibrated = iso.predict(y_val_probs[:10])

print(y_val_probs[:10])
print(p_calibrated)

y_probs = y_val_probs
y_true = y_val_labels

prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)

y_cal = iso.predict(y_probs)
prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_cal, n_bins=10)

plt.figure(figsize=(8, 6))

plt.plot(prob_pred, prob_true, marker="o", label="Before Isotonic", linestyle="--")

plt.plot(
    prob_pred_cal, prob_true_cal, marker="o", label="After Isotonic", linestyle="-"
)

plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")

plt.xlabel("Predicted Probability")
plt.ylabel("True Frequency")
plt.title("Calibration Curve")
plt.legend()
plt.grid(True)
plt.show()

x_points = iso.X_thresholds_  # probs مدل
y_points = iso.y_thresholds_  # probs calibrated

iso_layer = IsotonicLayer(x_points=x_points, y_points=y_points)
mlp_iso = MLPWithIsotonic(mlp_model=run_obj.light_obj.model, iso_layer=iso_layer)

# example_input = torch.randn(1, run_obj.input_dim, requires_grad=False)

# torch.onnx.export(
#     mlp_iso,
#     example_input,
#     "/home/void/ModelCoding/resources/torch/trough/models/mlp_iso_32.onnx",
#     export_params=True,
#     opset_version=18,
#     dynamo=False,
#     do_constant_folding=True,
#     input_names=["X"],
#     output_names=["probs_calibrated"],
#     dynamic_axes={"X": {0: "batch"}, "probs_calibrated": {0: "batch"}},
#     verbose=True,
# )

import onnxruntime as ort

example_input = torch.randn(5, run_obj.input_dim)

example_input_np = example_input.numpy()

ort_session = ort.InferenceSession(
    "/home/void/ModelCoding/resources/torch/trough/models/mlp_iso_32.onnx"
)

input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

onnx_probs = ort_session.run([output_name], {input_name: example_input_np})[0]

mlp_iso.eval()
with torch.no_grad():
    pytorch_probs = mlp_iso(example_input).numpy()

print("Max diff between PyTorch and ONNX:", np.max(np.abs(onnx_probs - pytorch_probs)))
print("ONNX probs:\n", onnx_probs)
print("PyTorch probs:\n", pytorch_probs)


# run_obj.load_trainer(learning_rate=2e-2, weight_decay=1e-4, es_patience=10)
# run_obj.fit().test()
# run_obj.export_onnx()
# run_obj.log_metrics(threshold=0.5)
# run_obj.precision_recall_curve(f"resources/torch/trough/precision_recall_curve_{model_seq_len}.png")
