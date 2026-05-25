from src.visualization.plot import Plot
from torch_models.src.inference.model_inference import ModelInference
from torch_models.src.run.run_peak import RunPeak
from torch_models.src.run.run_trough import RunTrough


fractals_period = 7
model_seq_len = 32
model_look_ahead = 7


# get Peak model's Obj (Inferencing Peak Model)
run_obj_peak = RunPeak(fractals_period, model_seq_len, model_look_ahead, df=None)
run_obj_peak.calculate_features(f"df_{model_seq_len}")
run_obj_peak.load_model_from_lightning_checkpoint("lightning_logs/version_1/checkpoints/best_model.ckpt")

# get Trough model's Obj (Inferencing Trough Model)
run_obj_trough = RunTrough(fractals_period, model_seq_len, model_look_ahead, df=None)
run_obj_trough.calculate_features(f"df_{model_seq_len}")
run_obj_trough.load_model_from_lightning_checkpoint("lightning_logs/version_1/checkpoints/best_model.ckpt")

# Infer
# In here we get the dataframe from either RunPeak or RunTrough to do the infer
df = run_obj_peak.df.copy(deep=True)
df = df.iloc[-2000:].copy(deep=True)

inference = ModelInference(
    df, model_seq_len, model_look_ahead, 0.50, run_obj_peak.light_obj, run_obj_trough.light_obj
)

inference.peak_inference()
inference.trough_inference()

peak_y = inference.get_peak_y_predictions()
trough_y = inference.get_trough_y_predictions()


plot_obj = Plot(df)
plot_obj.plot_peaks(peak_y)
plot_obj.plot_troughs(trough_y)
plot_obj.plot()
