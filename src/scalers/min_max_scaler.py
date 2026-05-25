import json
import os

import numpy as np
from src.scalers.base_scaler_class import BaseScaler
from sklearn.preprocessing import MinMaxScaler

from src.utils.type_transformation import to_list, transform_data


class CustomMinMaxScaler(BaseScaler):
    def __init__(self, scaler_dir_path, scaler_range) -> None:
        super().__init__(scaler_dir_path)
        self.scaler_obj = MinMaxScaler()
        self.min = scaler_range[0]
        self.max = scaler_range[1]

    def fit(self, X):
        X = transform_data(X)

        data_min = np.nanmin(X, axis=0)
        data_max = np.nanmax(X, axis=0)

        self.scaler_obj.data_min_ = data_min
        self.scaler_obj.data_max_ = data_max

        scale = (data_max - data_min)
        scale[scale == 0] = 1.0

        self.scaler_obj.scale_ = (self.max - self.min) / scale
        self.scaler_obj.min_ = self.min - data_min * self.scaler_obj.scale_
        self.scaler_obj.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        X = transform_data(X)

        scale = self.scaler_obj.scale_
        offset = self.scaler_obj.min_

        X_scaled = np.where(
            np.isnan(X),
            np.nan,
            X * scale + offset
        )
        return X_scaled

    def _load_scaler_impl(self, data):
        self.scaler_obj = MinMaxScaler(
            with_mean=data.get("withMean", True), with_std=data.get("withStd", True)
        )
        self.scaler_obj.mean_ = np.array(data["mean"], dtype=float)
        self.scaler_obj.scale_ = np.array(data["scale"], dtype=float)
        self.scaler_obj.n_features_in_ = data.get(
            "nFeatures", len(self.scaler_obj.mean_)
        )
        self.scaler_obj.var_ = self.scaler_obj.scale_**2

    def load_scaler_from_json(self, json_filename):
        """
        to get the loaded scaler object from json file, this method must be called
        """
        data = self._load_json(json_filename)
        self._load_scaler_impl(data)
        return self

    def export_to_json(self, json_filename):
        mean = to_list(getattr(self.scaler_obj, "mean_", None))
        scale = None
        if hasattr(self.scaler_obj, "scale_"):
            scale = to_list(getattr(self.scaler_obj, "scale_"))
        # elif hasattr(self.scaler_obj, "std_"):
        #     scale = to_list(getattr(self.scaler_obj, "std_"))
        elif hasattr(self.scaler_obj, "var_"):
            import math

            var = to_list(getattr(self.scaler_obj, "var_"))
            scale = [math.sqrt(v) if v >= 0 else 1.0 for v in var]

        out = {
            "withMean": bool(getattr(self.scaler_obj, "with_mean", True)),
            "withStd": bool(getattr(self.scaler_obj, "with_std", True)),
            "nFeatures": len(mean),
            "mean": mean,
            "scale": scale,
        }

        json_path = os.path.join(self.dir_path, json_filename)
        with open(json_path, "w", encoding="utf8") as f:
            json.dump(out, f, indent=2)

        return self
