from scikit_models.src.run.run_trough import RunTrough
from src.data.load_data import read_csv


fractals_period = 7
model_seq_len = 16
model_look_ahead = 7


df = read_csv("resources/data/ModelData.csv", timestamp_index="timestamp")

run_obj = RunTrough(fractals_period, model_seq_len, model_look_ahead, df)

run_obj.calculate_features(f"df_{model_seq_len}")
run_obj.calculate_labels(f"trough_labels_{model_seq_len}")

X, y, X_test, y_test = run_obj.get_model_data(
    test_size=0.3, do_shuffle=True, debug_log_classes=True, balanced_train_classes=False
)

run_obj.create_model()
# run_obj.load_model("resources/sklearn/trough/models/mlp_c_b_trough.pkl")

run_obj.fit_model(
    X, y, X_test, y_test, "resources/sklearn/trough/models/mlp_c_b_trough.pkl"
)
run_obj.log_metrics(X, y, X_test, y_test, threshold=0.5)
run_obj.precision_recall_curve(
    X_test, y_test, "resources/sklearn/trough/pr_model_onTest_fig_trough.png", do_plot=False
)
