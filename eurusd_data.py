# ============================================================
# EURUSD 15M DATA DOWNLOADER
# ============================================================

import os
import pandas as pd
import yfinance as yf


# ============================================================
# SETTINGS
# ============================================================

SYMBOL = "EURUSD=X"

OUTPUT_FILE = "data/EURUSD_15m.csv"


# ============================================================
# DOWNLOAD
# ============================================================

def main():

    print("=" * 60)
    print("EURUSD 15M DATA DOWNLOADER")
    print("=" * 60)

    os.makedirs(
        "data",
        exist_ok=True
    )

    print()
    print("Downloading EURUSD data...")

    df = yf.download(
        SYMBOL,
        period="60d",
        interval="15m",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:

        raise RuntimeError(
            "No EURUSD data was downloaded."
        )

    # --------------------------------------------------------
    # Flatten columns if required
    # --------------------------------------------------------

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        df.columns = [
            column[0]
            for column in df.columns
        ]

    # --------------------------------------------------------
    # Rename columns
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        }
    )

    # --------------------------------------------------------
    # Remove timezone if present
    # --------------------------------------------------------

    if df.index.tz is not None:

        df.index = (
            df.index
            .tz_convert("UTC")
            .tz_localize(None)
        )

    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    df = df.reset_index()

    df = df.rename(
        columns={
            "Datetime": "time",
            "Date": "time",
        }
    )

    # --------------------------------------------------------
    # Keep required columns
    # --------------------------------------------------------

    required_columns = [
        "time",
        "Open",
        "High",
        "Low",
        "Close",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise RuntimeError(
                f"Missing column: {column}"
            )

    df = df[
        required_columns
    ]

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    df = df.dropna()

    df = df.sort_values(
        "time"
    )

    df = df.drop_duplicates(
        subset="time"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("EURUSD DATA DOWNLOADED")
    print("=" * 60)

    print(
        "Candles:",
        len(df)
    )

    print(
        "Range:",
        df["time"].min(),
        "->",
        df["time"].max()
    )

    print(
        "File:",
        OUTPUT_FILE
    )

    print("=" * 60)


if __name__ == "__main__":

    main()
