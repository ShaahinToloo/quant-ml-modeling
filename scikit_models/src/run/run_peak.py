import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle

from src.data.data_preparator import DataPreparator
from .run import Run
from src.data.label_data import LabelAlgorithms
from src.scalers.standard_scaler import CustomStandardScaler
from src.utils.logging import logger


class RunPeak(Run):
    def __init__(self, fractals_period, model_seq_len, model_look_ahead, df) -> None:
        super().__init__(fractals_period, model_seq_len, model_look_ahead, df)

    def calculate_labels(
        self, obj_name, obj_suffix=".npy", obj_path="resources/objects"
    ):
        obj_full_path = os.path.join(obj_path, obj_name + obj_suffix)

        logger("Applying Labeling Algorithms")
        if not os.path.exists(obj_full_path):
            logger("  Calculating Algorithms")

            labeling_obj = LabelAlgorithms(
                self.df, self.df["fracHigh7"].values, self.df["fracLow7"].values
            )
            labeling_obj.pct_range_eMA3_correction_range(
                look_ahead=20, threshold=0.00075
            )
            labeling_obj.supertrend_direction_change(look_ahead=10)
            labeling_obj.rad_slope_smoothed_sma(look_ahead=12, threshold=0.07)
            self.labels, _ = labeling_obj.get_label_arrays()

            np.save(obj_full_path, self.labels)
        else:
            logger("  Loading Saved Labels")
            self.labels = np.load(obj_full_path, allow_pickle=True)

        return self

    def _get_Xy_arr(self):
        logger("Preparing X and y arrays")
        data_preprator_obj = DataPreparator(
            self.df, self.labels, self.model_seq_len, self.model_look_ahead
        )
        X_arr, y_arr = data_preprator_obj.peak_data_prepration()
        return X_arr, y_arr

    def get_model_data(
        self,
        test_size=0.3,
        do_shuffle=True,
        stratify=None,
        debug_log_classes=False,
        balanced_train_classes=False,
    ):
        X_train_to_return = None
        y_train_to_return = None
        X_test_to_return = None
        y_test_to_return = None

        X, y = self._get_Xy_arr()
        if debug_log_classes:
            self._class_logging(y)

        if stratify is None:
            stratify = y

        logger("Train Test Splitting")
        # We CAN shuffle in here because each sequence is flattened into a single row
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            shuffle=do_shuffle,
            stratify=stratify,
        )

        if balanced_train_classes:
            logger(
                "Balancing (forcing the distribution of classes to be equal) train data"
            )
            # 50 50 the Train, DO NOT touch test!
            X_train_0 = X_train[y_train == 0]
            X_train_1 = X_train[y_train == 1]
            y_train_0 = y_train[y_train == 0]
            y_train_1 = y_train[y_train == 1]

            if len(X_train_0) < len(X_train_1):
                minority_X, minority_y = X_train_0, y_train_0
                majority_X, majority_y = X_train_1, y_train_1
            else:
                minority_X, minority_y = X_train_1, y_train_1
                majority_X, majority_y = X_train_0, y_train_0

            majority_down_X, majority_down_y = resample(
                majority_X,
                majority_y,
                replace=False,
                n_samples=len(minority_X),
                random_state=42,
            )

            X_train_bal = np.vstack((minority_X, majority_down_X))
            y_train_bal = np.hstack((minority_y, majority_down_y))

            X_train_to_return, y_train_to_return = shuffle(
                X_train_bal, y_train_bal, random_state=42
            )

            print(f"\tBalanced class 0: {(y_train_bal == 0).sum()}")
            print(f"\tBalanced class 1: {(y_train_bal == 1).sum()}\n")
        else:
            X_train_to_return, y_train_to_return = X_train, y_train

        logger("Scaling Data")

        std_scaler_obj = CustomStandardScaler("resources/scalers")
        X_train_to_return = std_scaler_obj.fit_transform(X_train_to_return)
        X_test_to_return = std_scaler_obj.transform(X_test)
        y_test_to_return = y_test

        std_scaler_obj.export_to_json("mlp_c_b_peak_std_scaler.json")

        return X_train_to_return, y_train_to_return, X_test_to_return, y_test_to_return
