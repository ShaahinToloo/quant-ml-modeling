import json
import os
from abc import ABC, abstractmethod
from typing import Any, final

import numpy as np
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator

from src.utils.type_transformation import transform_data


class BaseScaler(ABC):
    def __init__(self, scaler_dir_path) -> None:
        self.dir_path = scaler_dir_path
        self.scaler_obj = None

    @abstractmethod
    def fit(self, X) -> "BaseScaler":
        pass

    @abstractmethod
    def transform(self, X) -> np.ndarray | spmatrix:
        pass

    @final
    def fit_transform(self, X) -> np.ndarray | spmatrix:
        X = transform_data(X)
        self.fit(X)
        return self.transform(X)

    @final
    def get_scaler_obj(self) -> BaseEstimator:
        if self.scaler_obj is not None:
            return self.scaler_obj
        raise ValueError(
            "self.scaler_obj is None, no scaler's initialized.\n  Can not return scaler at get_scaler_obj method."
        )

    @final
    def _load_json(self, json_filename) -> Any:
        json_path = os.path.join(self.dir_path, json_filename)
        with open(json_path, "r", encoding="utf8") as f:
            data = json.load(f)
        return data

    @abstractmethod
    def load_scaler_from_json(self, json_filename) -> "BaseScaler":
        """
        to get the loaded scaler object from json file, this method must be called
        """
        data = self._load_json(json_filename)
        self._load_scaler_impl(data)
        return self

    @abstractmethod
    def _load_scaler_impl(self, data) -> "BaseScaler":
        """Override this in child to build scaler from data (json file)"""
        pass

    @abstractmethod
    def export_to_json(self, json_filename) -> "BaseScaler":
        return self
