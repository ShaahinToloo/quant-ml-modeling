from src.visualization.plot import Plot
from scikit_models.src.run.run_peak import RunPeak
from scikit_models.src.run.run_trough import RunTrough
from scikit_models.src.inference.model_inference import ModelInference


fractals_period = 7
model_seq_len = 16
model_look_ahead = 7



run_obj_peak = RunPeak(fractals_period, model_seq_len, model_look_ahead, None)

run_obj_peak.calculate_features(f"df_{model_seq_len}")

run_obj_peak.load_model("resources/models/mlp_c_b_peak.pkl")


run_obj_trough = RunTrough(fractals_period, model_seq_len, model_look_ahead, None)

run_obj_trough.calculate_features(f"df_{model_seq_len}")

run_obj_trough.load_model("resources/models/mlp_c_b_trough.pkl")


df = run_obj_peak.df.iloc[-2000:].copy(deep=True)

inference = ModelInference(df, model_seq_len, model_look_ahead, 0.5, run_obj_peak.mlp_obj, run_obj_trough.mlp_obj)

inference.peak_inference()
inference.trough_inference()

peak_y = inference.get_peak_y_predictions()
trough_y = inference.get_trough_y_predictions()

plot_obj = Plot(df)
plot_obj.plot_peaks(peak_y)
plot_obj.plot_troughs(trough_y)
plot_obj.plot()