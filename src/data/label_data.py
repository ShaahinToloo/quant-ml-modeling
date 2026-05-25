import numpy as np


class LabelAlgorithms:
    def __init__(self, df, frac_high: np.ndarray, frac_low: np.ndarray):
        self.df = df
        self.peak_labels = frac_high.copy()
        self.trough_labels = frac_low.copy()
        self.peak_labels[frac_high != 1.0] = -1
        self.trough_labels[frac_low != 1.0] = -1

    def pct_range_eMA3_correction_range(self, look_ahead=20, threshold=0.00075):
        """
        threshold is in 0 to 1 range
        """
        for i in range(len(self.peak_labels) - (look_ahead+1)):
            if self.trough_labels[i] == 1:
                flag = -float("inf")
                idx = -1
                for j in range(i + 1, i + look_ahead + 1):
                    if self.df.iloc[j]["eMA3_soHigh"] > flag:
                        idx = j
                        flag = self.df.iloc[j]["eMA3_soHigh"]
                if not self.df.iloc[idx]["pctRange'lastLowestEMA3'i'"] >= threshold:
                    self.trough_labels[i] = 0

            if self.peak_labels[i] == 1:
                flag = float("inf")
                idx = -1
                for j in range(i + 1, i + look_ahead + 1):
                    if self.df.iloc[j]["eMA3_soLow"] < flag:
                        idx = j
                        flag = self.df.iloc[j]["eMA3_soLow"]
                if not self.df.iloc[idx]["pctRange'lastHighestEMA3'i'"] <= -threshold:
                    self.peak_labels[i] = 0
        return self

    def supertrend_direction_change(self, look_ahead=10):
        for i in range(len(self.peak_labels) - (look_ahead+1)):
            if self.trough_labels[i] == 1:
                flag = False
                for j in range(i + 1, i + look_ahead + 1):
                    if self.df.iloc[j - 1]["direction"] == -1.0:
                        if self.df.iloc[j]["direction"] == 1.0:
                            flag = True
                            break
                if not flag:
                    self.trough_labels[i] = 0

            if self.peak_labels[i] == 1:
                flag = False
                for j in range(i + 1, i + look_ahead + 1):
                    if self.df.iloc[j - 1]["direction"] == 1.0:
                        if self.df.iloc[j]["direction"] == -1.0:
                            flag = True
                            break
                if not flag:
                    self.peak_labels[i] = 0
        return self

    def rad_slope_smoothed_sma(self, look_ahead=12, threshold=0.07):
        for i in range(len(self.peak_labels) - (look_ahead + 1)):
            if self.trough_labels[i] == 1:
                counter = 3
                for j in range(i, i + look_ahead + 1):
                    if (
                        self.df.iloc[j]["radSlope_SMA6_soLow_SmoothedSMA4"] >= threshold
                        or self.df.iloc[j]["radSlope_SMA6_soHigh_SmoothedSMA4"]
                        >= threshold
                    ):
                        counter -= 1

                if not counter <= 0:
                    self.trough_labels[i] = 0

            if self.peak_labels[i] == 1:
                counter = 3
                for j in range(i, i + look_ahead + 1):
                    if (
                        self.df.iloc[j]["radSlope_SMA6_soLow_SmoothedSMA4"]
                        <= -threshold
                        or self.df.iloc[j]["radSlope_SMA6_soHigh_SmoothedSMA4"]
                        <= -threshold
                    ):
                        counter -= 1

                if not counter <= 0:
                    self.peak_labels[i] = 0
        return self

    def get_label_arrays(self) -> list[np.ndarray]:
        return self.peak_labels, self.trough_labels


if __name__=="__main__":
    import joblib
    import pandas as pd

    from src.data.engineering_data import ApplyFeaturesOnDataFrame
    from src.visualization.plot import Plot

    df = joblib.load("resources/objects/df_16.pkl")
    df = df.iloc[1000:2000].copy(deep=True)

    labeling_obj = LabelAlgorithms(df, df["fracHigh7"].values, df["fracLow7"].values)

    labeling_obj.pct_range_eMA3_correction_range(look_ahead=20, threshold=0.00075)

    labeling_obj.supertrend_direction_change(look_ahead=10)

    labeling_obj.rad_slope_smoothed_sma(look_ahead=12, threshold=0.07)

    peak_labels, trough_labels = labeling_obj.get_label_arrays()

    plt = Plot(df)
    plt.plot_peaks(peak_labels)
    plt.plot_troughs(trough_labels)
    plt.plot()
