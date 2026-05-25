import mplfinance as mpf


class Plot:
    def __init__(self, df) -> None:
        self.df = df
        self.adp = []

    def plot_peaks(self, peak_array):
        self.adp.append(
            mpf.make_addplot(
                self.df["High"].where(peak_array == 1),
                type="scatter",
                marker="^",
                markersize=30,
                color="green",
            )
        )
        return self

    def plot_troughs(self, trough_array):
        self.adp.append(
            mpf.make_addplot(
                self.df["Low"].where(trough_array == 1),
                type="scatter",
                marker="v",
                markersize=30,
                color="red",
            )
        )
        return self

    def plot(self, start=0, end=None):
        if end is None:
            end = len(self.df)
        mpf.plot(
            self.df.iloc[start:end],
            type="candle",
            style="tradingview",
            addplot=self.adp,
            warn_too_much_data=10000,
        )
        return self
