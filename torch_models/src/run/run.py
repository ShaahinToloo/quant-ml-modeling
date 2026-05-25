import os
from abc import ABC, abstractmethod

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import dump, load
from lightning.pytorch.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ModelSummary)
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from torch.utils.data import DataLoader

from src.data.data_preprator import DataPreparator
from src.data.engineering_data import ApplyFeaturesOnDataFrame
from src.data.label_data import LabelAlgorithms
from src.scalers.standard_scaler import CustomStandardScaler
from src.utils.logging import logger
from torch_models.src.data.dataset import CustomDataset
from torch_models.src.model.lightning import LightningMLPClassifierBinary


class Run(ABC):
    def __init__(
        self, fractals_period, model_seq_len, model_look_ahead, df, main_dir
    ) -> None:
        super().__init__()
        self.main_dir = main_dir
        self.fractals_period = fractals_period
        self.model_seq_len = model_seq_len
        self.model_look_ahead = model_look_ahead
        self.df = df
        self.labels = None
        self.mlp_obj = None

        self.X = None
        self.y = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        self.val_size = 0.2
        self.test_size = 0.3
        self.do_shuffle = True
        self.test_stratify = None
        self.val_stratify = None
        self.debug_log_classes = False
        self.balanced_train_classes = False

        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        self.input_dim = None
        self.light_obj = None
        self.trainer = None

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
            labels_peak, labels_trough = labeling_obj.get_label_arrays()

            return labels_peak, labels_trough
        else:
            logger("  Loading Saved Labels")
            self.labels = np.load(obj_full_path, allow_pickle=True)

        return self.labels, self.labels

    @abstractmethod
    def _get_Xy_arr(self):
        logger("Preparing X and y arrays")
        data_preprator_obj = DataPreparator(
            self.df, self.labels, self.model_seq_len, self.model_look_ahead
        )
        return data_preprator_obj

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
    def _get_model_data(self):
        val_size = self.val_size
        test_size = self.test_size
        do_shuffle = self.do_shuffle
        test_stratify = self.test_stratify
        val_stratify = self.val_stratify
        debug_log_classes = self.debug_log_classes
        balanced_train_classes = self.balanced_train_classes

        X_train_to_return = None
        y_train_to_return = None
        X_val_to_return = None
        y_val_to_return = None
        X_test_to_return = None
        y_test_to_return = None

        X, y = self._get_Xy_arr()
        if debug_log_classes:
            self._class_logging(y)

        self.input_dim = X.shape[1]
        print("X shape:", X.shape)
        print("input_dim:", X.shape[1])

        if test_stratify is None and do_shuffle is True:
            test_stratify = y

        logger("Train Test Splitting")
        # We CAN shuffle in here because each sequence is flattened into a single row
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            shuffle=do_shuffle,
            stratify=test_stratify,
        )

        if val_stratify is None and do_shuffle is True:
            val_stratify = y_train

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            random_state=42,
            shuffle=do_shuffle,
            stratify=val_stratify,
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

        std_scaler_obj = CustomStandardScaler(os.path.join(self.main_dir, "scalers"))
        X_train_to_return = std_scaler_obj.fit_transform(X_train_to_return)
        X_val_to_return = std_scaler_obj.transform(X_val)
        X_test_to_return = std_scaler_obj.transform(X_test)
        y_val_to_return = y_val
        y_test_to_return = y_test

        std_scaler_obj.export_to_json("mlp_std_scaler.json")

        return (
            X_train_to_return,
            y_train_to_return,
            X_val_to_return,
            y_val_to_return,
            X_test_to_return,
            y_test_to_return,
        )

    def _get_datasets(self):
        self.X, self.y, self.X_val, self.y_val, self.X_test, self.y_test = (
            self._get_model_data()
        )
        train_dataset = CustomDataset(self.X, self.y)
        val_dataset = CustomDataset(self.X_val, self.y_val)
        test_dataset = CustomDataset(self.X_test, self.y_test)

        return train_dataset, val_dataset, test_dataset

    def create_data_loaders(
        self,
        batch_size=1024,
        val_size=0.2,
        test_size=0.3,
        do_shuffle=True,
        test_stratify=None,
        val_stratify=None,
        debug_log_classes=True,
        balanced_train_classes=False,
    ):
        logger("Preparing DataLoaders")
        self.val_size = val_size
        self.test_size = test_size
        self.do_shuffle = do_shuffle
        self.test_stratify = test_stratify
        self.val_stratify = val_stratify
        self.debug_log_classes = debug_log_classes
        self.balanced_train_classes = balanced_train_classes

        train_dataset, val_dataset, test_dataset = self._get_datasets()
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=7,
            pin_memory=True,
        )
        return self

    def load_trainer(self, learning_rate=2e-2, weight_decay=1e-4, es_patience=10):
        logger("Creating Trainer")
        self.light_obj = LightningMLPClassifierBinary(
            self.input_dim, learning_rate, weight_decay
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=es_patience, mode="min", verbose=True
        )

        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best_model",
        )

        self.trainer = L.Trainer(
            precision="bf16-mixed",
            max_epochs=100,
            callbacks=[early_stopping, checkpoint],
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=True,
            default_root_dir=self.main_dir,
            log_every_n_steps=5,
        )
        return self

    def continue_training(self, ckpt_relative_path, es_patience=10):
        logger("Preparing Trainer to continue training")
        ckpt_path = os.path.join(self.main_dir, ckpt_relative_path)

        self.light_obj = LightningMLPClassifierBinary.load_from_checkpoint(ckpt_path)

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=es_patience, mode="min", verbose=True
        )

        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best_model-{epoch:02d}-{val_loss:.4f}",
        )

        self.trainer = L.Trainer(
            precision="bf16-mixed",
            max_epochs=100,
            callbacks=[early_stopping, checkpoint],
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=True,
            default_root_dir=self.main_dir,
            log_every_n_steps=5,
        )

        logger(
            "Continuing Fitting Model (DO NOT CALL fit() after calling continue_training(...) if you want the training to be continued, calling fit() will start the training from the beginning!)"
        )
        self.trainer.fit(
            model=self.light_obj,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
            ckpt_path=ckpt_path,
        )
        return self

    def fit(self):
        logger("Fitting Model")
        self.trainer.fit(
            model=self.light_obj,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
        )
        return self

    def test(self):
        logger("Testing Model")
        self.trainer.test(model=self.light_obj, dataloaders=self.test_dataloader)
        return self

    def export_onnx(self, example_input=None):
        path = os.path.join(self.main_dir, "models/mlp.onnx")

        if example_input is None:
            example_input = torch.randn(1, self.input_dim, requires_grad=False)

        self.light_obj.to("cpu")
        self.light_obj.model.eval()
        torch.onnx.export(
            self.light_obj.model,
            example_input,
            path,
            export_params=True,
            opset_version=18,
            dynamo=False,
            do_constant_folding=True,
            input_names=["X"],
            output_names=["out"],
            dynamic_axes={"X": {0: "batch"}, "out": {0: "batch"}},
            verbose=True,
        )
        logger(f"Model exported to ONNX at {path}")
        return self

    def load_model_from_lightning_checkpoint(self, relative_path):
        ckpt_path = os.path.join(self.main_dir, relative_path)
        logger(f"Loading model from checkpoint: {ckpt_path}")

        self.light_obj = LightningMLPClassifierBinary.load_from_checkpoint(ckpt_path)
        self.light_obj.eval()
        self.light_obj.freeze()
        logger("Model loaded and ready for inference.")
        return self

    def log_metrics(self, threshold=0.5):
        logger("Metrics Report")
        y_pred_proba = self.light_obj.predict_proba(self.X_test)
        y_pred = (y_pred_proba >= threshold).astype(float)

        print("TEST classification report:")
        print(classification_report(self.y_test, y_pred))
        print("\nTEST confusion matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        print("\nTEST roc_auc:", roc_auc_score(self.y_test, y_pred_proba))

        y_train_pred_proba = self.light_obj.predict_proba(self.X)
        y_train_pred = (y_train_pred_proba >= threshold).astype(float)
        print("\nTRAIN classification report:")
        print(classification_report(self.y, y_train_pred))
        print("\nTRAIN confusion matrix:")
        print(confusion_matrix(self.y, y_train_pred))
        print("\nTRAIN roc_auc:", roc_auc_score(self.y, y_train_pred_proba))

        return self

    def precision_recall_curve(self, path, do_plot=False):
        y_proba = self.light_obj.predict_proba(self.X)
        prec, rec, thr = precision_recall_curve(self.y, y_proba)

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
