import pandas as pd


def read_csv(file_path, numeric_index=None, timestamp_index=None):
    if numeric_index is not None and timestamp_index is not None:
        raise AttributeError(
            "you passed both numeric_index and timestamp_index, pass either or neither"
        )
    if timestamp_index is not None:
        return pd.read_csv(file_path, parse_dates=True, index_col=timestamp_index)
    if numeric_index is not None:
        return pd.read_csv(file_path, index_col=numeric_index)
    return pd.read_csv(file_path)
