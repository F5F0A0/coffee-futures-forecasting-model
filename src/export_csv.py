import pandas as pd

try:
    df = pd.read_excel("data/Coffee_Historical_Prices.xls", engine="xlrd")

    df = df.iloc[4:, [0, 9]]

    df = df.rename(columns={df.columns[0]: "ds"})
    df = df.rename(columns={df.columns[1]: "y"})

    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"])

    df = df.dropna(subset=["y"])
    df = df.reset_index(drop=True)

    df.to_csv("data/coffee.csv", index=False)

except Exception as e:
    print(f"An error occurred: {e}")
