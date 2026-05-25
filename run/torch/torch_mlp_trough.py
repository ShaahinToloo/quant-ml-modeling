from src.data.load_data import read_csv
from torch_models.src.run.run_trough import RunTrough


fractals_period = 7
model_seq_len = 32
model_look_ahead = 7


df = read_csv("resources/ModelData.csv", timestamp_index="timestamp")

main_dir = "resources/torch/trough"

run_obj = RunTrough(fractals_period, model_seq_len, model_look_ahead, df, main_dir)

run_obj.calculate_features(f"df_{model_seq_len}")
run_obj.calculate_labels(f"trough_labels_{model_seq_len}")
run_obj.create_data_loaders(
    batch_size=1024, val_size=0.15, test_size=0.15, do_shuffle=True
)
run_obj.load_trainer(learning_rate=2e-2, weight_decay=1e-4, es_patience=10)
run_obj.fit().test()
run_obj.export_onnx()
run_obj.log_metrics(threshold=0.5)
run_obj.precision_recall_curve(
    f"resources/torch/trough/precision_recall_curve_{model_seq_len}.png"
)
