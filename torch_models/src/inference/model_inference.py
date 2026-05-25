import numpy as np
import pandas as pd

from src.scalers.standard_scaler import CustomStandardScaler


class ModelInference:
    def __init__(
        self,
        df,
        seq_len,
        look_ahead,
        threshold=0.5,
        peak_model_custom_obj=None,
        trough_model_custom_obj=None,
    ) -> None:
        self.peak_model_obj = peak_model_custom_obj
        self.trough_model_obj = trough_model_custom_obj
        self.df = df
        self.seq_len = seq_len
        self.look_ahead = look_ahead
        self.threshold = threshold
        self.peak_y = np.zeros(len(df))
        self.trough_y = np.zeros(len(df))

        self.peak_stdscaler = None
        if self.peak_model_obj is not None:
            self.peak_stdscaler = CustomStandardScaler("resources/torch/peak/scalers")
            self.peak_stdscaler.load_scaler_from_json("mlp_std_scaler.json")

        self.trough_stdscaler = None
        if self.trough_model_obj is not None:
            self.trough_stdscaler = CustomStandardScaler(
                "resources/torch/trough/scalers"
            )
            self.trough_stdscaler.load_scaler_from_json("mlp_std_scaler.json")

    def peak_inference(self):
        self.df.drop(
            columns=["pctRange'lastLowestEMA3'-i'"], inplace=True, errors="ignore"
        )

        for i in range(self.seq_len, len(self.df) - self.look_ahead):
            if self.df.iloc[i]["fracHigh7"] == 1.0:
                sequence = self.df.iloc[
                    i - self.seq_len + 1 : i + 1 + self.look_ahead
                ].copy(deep=True)
                sequence = sequence.reset_index(drop=True)
                sequence.loc[
                    len(sequence) - self.look_ahead :, "pctRange'lastHighestEMA3'-i'"
                ] = 0.0

                sequence.drop(
                    columns=[
                        # "fracHigh7",
                        # "fracLow7",
                        "sMA6_soHigh",
                        "sMA6_soLow",
                        "sMA6_soHL2",
                        "sMA6_soHigh_SmoothedSMA4",
                        "sMA6_soLow_SmoothedSMA4",
                        "sMA6_soHL2_SmoothedSMA4",
                        "bBU",
                        "bBM",
                        "bBL",
                        "superTrend",
                        "eMA3_soHigh",
                        "eMA3_soLow",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "sMA2",
                        "sMA2SmoothedSMA3",
                        "hMA25",
                        "hMA25SmoothedHMA25",
                    ],
                    inplace=True,
                    errors="ignore",
                )
                sequence = sequence.values
                sequence = sequence.flatten()
                sequence = np.asarray(sequence).reshape(1, -1)
                sequence = self.peak_stdscaler.transform(sequence)

                self.peak_y[i] = (
                    1
                    if (self.peak_model_obj.predict_proba(sequence) >= self.threshold)
                    else 0
                )

    def trough_inference(self):
        self.df.drop(
            columns=["pctRange'lastHighestEMA3'-i'"], inplace=True, errors="ignore"
        )

        for i in range(self.seq_len, len(self.df) - self.look_ahead):
            if self.df.iloc[i]["fracLow7"] == 1.0:
                sequence = self.df.iloc[
                    i - self.seq_len + 1 : i + 1 + self.look_ahead
                ].copy(deep=True)
                sequence = sequence.reset_index(drop=True)
                sequence.loc[
                    len(sequence) - self.look_ahead :, "pctRange'lastLowestEMA3'-i'"
                ] = 0.0

                sequence.drop(
                    columns=[
                        # "fracHigh7",
                        # "fracLow7",
                        "sMA6_soHigh",
                        "sMA6_soLow",
                        "sMA6_soHL2",
                        "sMA6_soHigh_SmoothedSMA4",
                        "sMA6_soLow_SmoothedSMA4",
                        "sMA6_soHL2_SmoothedSMA4",
                        "bBU",
                        "bBM",
                        "bBL",
                        "superTrend",
                        "eMA3_soHigh",
                        "eMA3_soLow",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "sMA2",
                        "sMA2SmoothedSMA3",
                        "hMA25",
                        "hMA25SmoothedHMA25",
                    ],
                    inplace=True,
                    errors="ignore",
                )
                sequence = sequence.values
                sequence = sequence.flatten()
                sequence = np.asarray(sequence).reshape(1, -1)
                sequence = self.trough_stdscaler.transform(sequence)

                self.trough_y[i] = (
                    1
                    if (self.trough_model_obj.predict_proba(sequence) >= self.threshold)
                    else 0
                )

    def get_peak_y_predictions(self):
        return self.peak_y

    def get_trough_y_predictions(self):
        return self.trough_y
