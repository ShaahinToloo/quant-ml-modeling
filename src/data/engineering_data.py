import pandas as pd
from src.data.feature_class import MainFeatures


class ApplyFeaturesOnDataFrame:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy(deep=True)
        self.main_features = MainFeatures()

    def william_fractals(self, period):
        high = self.df["High"].values
        low = self.df["Low"].values
        frac_high, frac_low = self.main_features.william_fractals(high, low, period)

        self.df[f"fracHigh{period}"] = frac_high
        self.df[f"fracLow{period}"] = frac_low
        return self

    def pct_range_correction_phase(self, fractals_period):
        frac_high7 = self.df[f"fracHigh{fractals_period}"].values
        frac_low7 = self.df[f"fracLow{fractals_period}"].values
        ema3_so_high = self.df["eMA3_soHigh"].values
        ema3_so_low = self.df["eMA3_soLow"].values
        pct_range_last_highest_ema3_i, pct_range_last_lowest_ema3_i = (
            self.main_features.pct_range_correction(
                frac_high7, frac_low7, ema3_so_high, ema3_so_low
            )
        )

        self.df["pctRange'lastHighestEMA3'i'"] = pct_range_last_highest_ema3_i
        self.df["pctRange'lastLowestEMA3'i'"] = pct_range_last_lowest_ema3_i
        return self

    def pct_range_impulse_phase(self, fractals_period, seq_len):
        frac_high7 = self.df[f"fracHigh{fractals_period}"].values
        frac_low7 = self.df[f"fracLow{fractals_period}"].values
        ema3_so_high = self.df["eMA3_soHigh"].values
        ema3_so_low = self.df["eMA3_soLow"].values
        pct_range_last_highest_ema3_i, pct_range_last_lowest_ema3_i = (
            self.main_features.pct_range_impulse(
                frac_high7, frac_low7, ema3_so_high, ema3_so_low, seq_len
            )
        )

        self.df["pctRange'lastHighestEMA3'-i'"] = pct_range_last_highest_ema3_i
        self.df["pctRange'lastLowestEMA3'-i'"] = pct_range_last_lowest_ema3_i
        return self

    def get_final_df(self):
        return self.df
