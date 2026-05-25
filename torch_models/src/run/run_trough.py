import os

import numpy as np

from torch_models.src.run.run import Run


class RunTrough(Run):
    def __init__(
        self,
        fractals_period,
        model_seq_len,
        model_look_ahead,
        df,
        main_dir="resources/torch/trough"
    ) -> None:
        super().__init__(fractals_period, model_seq_len, model_look_ahead, df, main_dir)

    def calculate_labels(
        self, obj_name, obj_suffix=".npy", obj_path="resources/objects"
    ):
        obj_full_path = os.path.join(obj_path, obj_name + obj_suffix)
        _, self.labels = super().calculate_labels(obj_name, obj_suffix, obj_path)
        np.save(obj_full_path, self.labels)
        return self

    def _get_Xy_arr(self):
        data_preprator_obj = super()._get_Xy_arr()
        X_arr, y_arr = data_preprator_obj.trough_data_prepration()
        return X_arr, y_arr

    def _get_model_data(self):
        return super()._get_model_data()
