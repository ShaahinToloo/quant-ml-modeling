import numba as nb
import numpy as np
import pandas as pd


class MainFeatures:
    def __init__(self) -> None:
        pass

    def william_fractals(self, high, low, period):
        n = len(high)
        frac_high = np.zeros(n)
        frac_low = np.zeros(n)

        for i in range(period, n - period):
            max_left = np.max(high[i - period : i])
            max_right = np.max(high[i + 1 : i + period + 1])
            if high[i] > max_left and high[i] > max_right:
                frac_high[i] = 1

            min_left = np.min(low[i - period : i])
            min_right = np.min(low[i + 1 : i + period + 1])
            if low[i] < min_left and low[i] < min_right:
                frac_low[i] = 1

        return frac_high, frac_low

    def pct_range_correction(
        self, frac_high7, frac_low7, ema3_so_high, ema3_so_low, close
    ):
        pct_range_last_highest_ema3_i = np.full(frac_high7.shape, np.nan)
        pct_range_last_lowest_ema3_i = np.full(frac_high7.shape, np.nan)

        pct_range_last_highest_close_i = np.full(frac_high7.shape, np.nan)
        pct_range_last_lowest_close_i = np.full(frac_high7.shape, np.nan)

        last_highest_ema3_idx = -1
        last_lowest_ema3_idx = -1

        for i in range(7, len(frac_high7)):
            fracIdx = i - 7
            if frac_high7[fracIdx] == 1.0:
                highest_ema3_idx = 0
                highest_ema3_val = -float("inf")

                for j in range(fracIdx - 1, fracIdx + 2):
                    if ema3_so_high[j] > highest_ema3_val:
                        highest_ema3_val = ema3_so_high[j]
                        highest_ema3_idx = j
                last_highest_ema3_idx = highest_ema3_idx

            elif frac_low7[fracIdx] == 1.0:
                lowest_ema3_idx = 0
                lowest_ema3_val = float("inf")

                for j in range(fracIdx - 1, fracIdx + 2):
                    if ema3_so_low[j] < lowest_ema3_val:
                        lowest_ema3_val = ema3_so_low[j]
                        lowest_ema3_idx = j
                last_lowest_ema3_idx = lowest_ema3_idx

            if last_lowest_ema3_idx != -1:
                pct_range_last_lowest_ema3_i[i] = (
                    ema3_so_high[i] - ema3_so_low[last_lowest_ema3_idx]
                ) / (ema3_so_low[last_lowest_ema3_idx])

                pct_range_last_lowest_close_i[i] = (
                    close[i] - ema3_so_low[last_lowest_ema3_idx]
                ) / (ema3_so_low[last_lowest_ema3_idx])

            if last_highest_ema3_idx != -1:
                pct_range_last_highest_ema3_i[i] = (
                    ema3_so_low[i] - ema3_so_high[last_highest_ema3_idx]
                ) / (ema3_so_high[last_highest_ema3_idx])

                pct_range_last_highest_close_i[i] = (
                    close[i] - ema3_so_high[last_highest_ema3_idx]
                ) / (ema3_so_high[last_highest_ema3_idx])

        return (
            pct_range_last_highest_ema3_i,
            pct_range_last_lowest_ema3_i,
            pct_range_last_highest_close_i,
            pct_range_last_lowest_close_i,
        )

    def trade_features(
        self, entry, sl, tp, frac_high_close, frac_low_close, trade_type
    ):
        entry_sl = np.full(entry.shape, np.nan)
        entry_tp = np.full(entry.shape, np.nan)

        fracLow_sl = np.full(entry.shape, np.nan)
        fracHigh_sl = np.full(entry.shape, np.nan)

        fracLow_tp = np.full(entry.shape, np.nan)
        fracHigh_tp = np.full(entry.shape, np.nan)

        sell = np.full(trade_type.shape, np.nan)
        buy = np.full(trade_type.shape, np.nan)

        for i in range(len(entry)):
            if not np.isnan(entry[i]):
                entry_sl[i] = (sl[i] - entry[i]) / entry[i]
                entry_tp[i] = (tp[i] - entry[i]) / entry[i]
                fracLow_sl[i] = frac_low_close[i] + entry_sl[i]
                fracHigh_sl[i] = frac_high_close[i] + entry_sl[i]
                fracLow_tp[i] = frac_low_close[i] + entry_tp[i]
                fracHigh_tp[i] = frac_high_close[i] + entry_tp[i]

                if trade_type[i] == 1:
                    buy[i] = 1.0
                    sell[i] = 0.0
                else:
                    sell[i] = 1.0
                    buy[i] = 0.0

        return entry_sl, entry_tp, fracLow_sl, fracHigh_sl, fracLow_tp, fracHigh_tp, sell, buy

