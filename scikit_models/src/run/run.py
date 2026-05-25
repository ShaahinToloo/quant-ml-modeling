import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.utils import resample, shuffle

from model.mlp import MLPClassifierBinary
from src.data.data_preparator import DataPreparator
from src.data.engineering_data import ApplyFeaturesOnDataFrame
from src.data.label_data import LabelAlgorithms
from src.scalers.standard_scaler import CustomStandardScaler
from src.utils.logging import logger


class Run(ABC):
    def __init__(self, fractals_period, model_seq_len, model_look_ahead, df) -> None:
        super().__init__()
        self.fractals_period = fractals_period
        self.model_seq_len = model_seq_len
        self.model_look_ahead = model_look_ahead
        self.df = df
        self.labels = None
        self.mlp_obj = None

    def calculate_features(
        self, obj_name, obj_suffix=".pkl", obj_path="resources/objects"
    ):
        obj_full_path = os.path.join(obj_path, obj_name + obj_suffix)

        logger("Applying Features")
        if not os.path.exists(obj_full_path):
            logger("  Calculating Features")

            fe_obj = ApplyFeaturesOnDataFrame(self.df)
            fe_obj.william_fractals(self.fractals_period)
            fe_obj.pct_range_correction_phase(self.fractals_period)
            fe_obj.pct_range_impulse_phase(self.fractals_period, self.model_seq_len)
            self.df = fe_obj.get_final_df()
            self.df = self.df.dropna()

            dump(self.df, obj_full_path)
        else:
            logger("  Loading Saved DataFrame")
            self.df = load(obj_full_path)

        return self

    @abstractmethod
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
            self.labels_peak, self.labels_trough = labeling_obj.get_label_arrays()

            np.save(obj_full_path, self.labels)
        else:
            logger("  Loading Saved Labels")
            self.labels = np.load(obj_full_path, allow_pickle=True)

        return self

    @abstractmethod
    def _get_Xy_arr(self):
        logger("Preparing X and y arrays")
        data_preprator_obj = DataPreparator(
            self.df, self.labels, self.model_seq_len, self.model_look_ahead
        )
        X_arr, y_arr = data_preprator_obj.peakortrough_data_prepration()
        return X_arr, y_arr

    def _class_logging(self, y_arr):
        logger("Logging classes")

        # Length and relativation of each class
        label_1_length = np.sum(y_arr == 1.0)
        label_0_length = np.sum(y_arr == 0.0)
        print(f"\tNumber of class 1 labels: {label_1_length}")
        print(f"\tNumber of class 0 labels: {label_0_length}")
        print(f"\tclass 1 / class 0: {(label_1_length / label_0_length):.3f}")
        print(f"\tclass 0 / class 1: {(label_0_length / label_1_length):.3f}")

        # dictionary of each class distribution
        unique, counts = np.unique(y_arr, return_counts=True)
        print("\tClasses distribution dict:")
        for key, val in zip(unique, counts):
            print(f"\t\t{key}: {val}")

        return self

    @abstractmethod
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

        std_scaler_obj.export_to_json("mlp_c_b_peakortrough_std_scaler.json")

        return X_train_to_return, y_train_to_return, X_test_to_return, y_test_to_return

    def get_inference_X(self):
        pass

    def create_model(self):
        logger("Creating The Model")
        self.mlp_obj = MLPClassifierBinary(
            hidden_layers=(32, 16),
            activation="tanh",
            solver="adam",
            alpha=5e-3,
            learning_rate_init=5e-4,
            max_iter=200,
            random_state=42,
            verbose=True,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.15,
        )
        return self

    def fit_model(self, X_train, y_train, X_test, y_test, obj_full_path):
        logger("Fitting Model")
        self.mlp_obj.fit(X_train, y_train)
        self.mlp_obj.save(obj_full_path)
        self.eval_model(X_test, y_test)
        return self

    def load_model(self, obj_full_path):
        logger("Loading Fitted Model")
        self.mlp_obj.load(obj_full_path)
        return self

    def eval_model(self, X, y):
        logger("Fitting Done!, Evaluating the model using Test-set")
        output = self.mlp_obj.evaluate(X, y)
        for key, val in output.items():
            print(f"\t{key}: {val}")
        return self

    def inference_model(self):
        pass

    def get_real_model_obj(self):
        return self.mlp_obj.get_model()

    def cross_validate_model(self, X_train, y_train, n_splits=5):
        logger("K Fold Cross-Validation")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(
            self.mlp_obj.get_model(), X_train, y_train, cv=cv, scoring="roc_auc"
        )
        print("CV ROC AUC scores:")
        for i, score in enumerate(scores):
            print(f"\tRound {i}: {score}")
        print("Mean ROC AUC:", scores.mean())

        return self

    def log_metrics(self, X_train, y_train, X_test, y_test, threshold=0.5):
        logger("Metrics Report")
        y_pred_proba = self.mlp_obj.predict_proba(X_test)
        y_pred = (y_pred_proba >= threshold).astype(float)

        print("TEST classification report:")
        print(classification_report(y_test, y_pred))
        print("\nTEST confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nTEST roc_auc:", roc_auc_score(y_test, y_pred_proba))

        y_train_pred_proba = self.mlp_obj.predict_proba(X_train)
        y_train_pred = (y_train_pred_proba >= threshold).astype(float)
        print("\nTRAIN classification report:")
        print(classification_report(y_train, y_train_pred))
        print("\nTRAIN confusion matrix:")
        print(confusion_matrix(y_train, y_train_pred))
        print("\nTRAIN roc_auc:", roc_auc_score(y_train, y_train_pred_proba))

        return self

    def precision_recall_curve(self, X, y, path, do_plot=False):
        y_proba = self.mlp_obj.predict_proba(X)
        prec, rec, thr = precision_recall_curve(y, y_proba)

        plt.plot(thr, prec[:-1], label="Precision")
        plt.plot(thr, rec[:-1], label="Recall")
        plt.axvline(0.5, color="gray", linestyle="--")
        plt.legend()
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.savefig(path)
        if do_plot:
            plt.show()
        return self
