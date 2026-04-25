"""
One-time conversion of the raw ICE Coffee C .xls into the tidy
two-column csv that load_coffee_data reads.

Run once after (re-)downloading data/Coffee_Historical_Prices.xls
from ice.com/report/293. The notebooks load data/coffee.csv directly
and never re-run this script.

The .xls has 4 header rows and many columns; we keep:
  - column 0: date
  - column 9: nearby close (the only series we use)
"""
import pandas as pd

df = pd.read_excel("data/Coffee_Historical_Prices.xls", engine="xlrd")
df = df.iloc[4:, [0, 9]]
df = df.rename(columns={df.columns[0]: "ds", df.columns[1]: "y"})
df["ds"] = pd.to_datetime(df["ds"])
df["y"] = pd.to_numeric(df["y"])
df = df.dropna(subset=["y"]).reset_index(drop=True)
df.to_csv("data/coffee.csv", index=False)
