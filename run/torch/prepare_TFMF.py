import pandas as pd


df = pd.read_csv("resources/TradeFilterModelFeatures_3M_4.5M_scaled.csv")
df = df.drop(
    columns=[
        "pctRange'lastHighestEMA3'-i'",
        "pctRange'lastLowestEMA3'-i'",
        "ichiChikouSpan",
        "pctRange'ichiChikouSpan'ichiSenkouSpanB'",
        "radSlope_pctRange'ichiChikouSpan'ichiSenkouSpanB'",
        "pctChange_Ichimoku_ChikouSpan",
        "radSlope_Ichimoku_ChikouSpan",
    ]
)

df.to_csv("resources/TFMF_Fixed_3M_4.5M_scaled.csv")
