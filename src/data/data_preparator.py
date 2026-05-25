import numpy as np
import pandas as pd


class DataPreparator:
    def __init__(self, df, labels_arr, seq_len, look_ahead) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be bigger than 0")
        self.df = df.copy(deep=True)
        self.labels_arr = labels_arr
        self.seq_len = seq_len
        self.look_ahead = look_ahead
        self.X_list = []
        self.y_list = []

    def peak_data_prepration(self):
        self.X_list.clear()
        self.y_list.clear()
        self.df.drop(columns=["pctRange'lastLowestEMA3'-i'"], inplace=True)

        for i in range(self.seq_len, len(self.df) - self.look_ahead):
            if self.df.iloc[i]["fracHigh7"] == 1.0:
                sequence = self.df.iloc[
                    i - self.seq_len + 1 : i + 1 + self.look_ahead
                ].copy(deep=True)
                sequence = sequence.reset_index(drop=True)
                sequence.loc[
                    len(sequence) - self.look_ahead :, "pctRange'lastHighestEMA3'-i'"
                ] = 0.0

                label = self.labels_arr[i]
                if label == -1:
                    raise ValueError("a label in data prepration is -1")

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
                sequence= sequence.values
                self.X_list.append(sequence.flatten())
                self.y_list.append(label)

        return np.asarray(self.X_list), np.asarray(self.y_list)

    def trough_data_prepration(self):
        self.X_list.clear()
        self.y_list.clear()
        self.df.drop(columns=["pctRange'lastHighestEMA3'-i'"], inplace=True)

        for i in range(self.seq_len, len(self.df) - self.look_ahead):
            if self.df.iloc[i]["fracLow7"] == 1.0:
                sequence = self.df.iloc[
                    i - self.seq_len + 1 : i + 1 + self.look_ahead
                ].copy(deep=True)
                sequence = sequence.reset_index(drop=True)
                sequence.loc[
                    len(sequence) - self.look_ahead :, "pctRange'lastLowestEMA3'-i'"
                ] = 0.0

                label = self.labels_arr[i]
                if label == -1:
                    raise ValueError("a label in data prepration is -1")

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
                self.X_list.append(sequence.flatten())
                self.y_list.append(label)

        return np.asarray(self.X_list), np.asarray(self.y_list)
