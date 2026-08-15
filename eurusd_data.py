# ============================================================
# EURUSD 15M DATA DOWNLOADER
# SOURCE: DUKASCOPY
# ============================================================

import os
import pandas as pd
from dukascopy_python import fetch


# ============================================================
# SETTINGS
# ============================================================

SYMBOL = "EURUSD"

TIMEFRAME = "15m"

START_DATE = "2020-01-01"

END_DATE = "2026-08-14"

OUTPUT_FILE = "data/EURUSD_15m.csv"


# ============================================================
# DOWNLOAD
# ============================================================

def main():

    print("=" * 60)
    print("EURUSD 15M DATA DOWNLOADER")
    print("=" * 60)

    print()
    print("SOURCE: DUKASCOPY")
    print("SYMBOL:", SYMBOL)
    print("TIMEFRAME:", TIMEFRAME)
    print("START:", START_DATE)
    print("END:", END_DATE)

    os.makedirs(
        "data",
        exist_ok=True
    )

    print()
    print("Downloading EURUSD data...")
    print("This may take several minutes.")

    try:

        df = fetch(
            SYMBOL,
            timeframe=TIMEFRAME,
            start=START_DATE,
            end=END_DATE,
        )

    except Exception as e:

        raise RuntimeError(
            f"EURUSD download failed: {e}"
        )

    if df is None or len(df) == 0:

        raise RuntimeError(
            "Dukascopy returned no EURUSD data."
        )

    # ========================================================
    # NORMALISE COLUMNS
    # ========================================================

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    rename = {}

    for column in df.columns:

        lower = column.lower()

        if lower in [
            "date",
            "datetime",
            "timestamp",
            "time",
        ]:

            rename[column] = "time"

        elif lower == "open":

            rename[column] = "Open"

        elif lower == "high":

            rename[column] = "High"

        elif lower == "low":

            rename[column] = "Low"

        elif lower == "close":

            rename[column] = "Close"

        elif lower == "volume":

            rename[column] = "Volume"

    df = df.rename(
        columns=rename
    )

    # ========================================================
    # HANDLE DATETIME INDEX
    # ========================================================

    if "time" not in df.columns:

        if isinstance(
            df.index,
            pd.DatetimeIndex
        ):

            df = df.reset_index()

            df = df.rename(
                columns={
                    df.columns[0]:
                    "time"
                }
            )

        else:

            raise RuntimeError(
                "Could not find EURUSD time column."
            )

    # ========================================================
    # DATETIME
    # ========================================================

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required = [
        "time",
        "Open",
        "High",
        "Low",
        "Close",
    ]

    for column in required:

        if column not in df.columns:

            raise RuntimeError(
                f"Missing required column: {column}"
            )

    # ========================================================
    # CLEAN
    # ========================================================

    df = df[
        required
        + (
            ["Volume"]
            if "Volume" in df.columns
            else []
        )
    ]

    df = df.dropna()

    df = df.sort_values(
        "time"
    )

    df = df.drop_duplicates(
        subset="time"
    )

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # VERIFY
    # ========================================================

    print()
    print("=" * 60)
    print("EURUSD DOWNLOAD COMPLETE")
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

    # ========================================================
    # BASIC DATA QUALITY CHECK
    # ========================================================

    print()
    print("DATA QUALITY")
    print("-" * 60)

    print(
        "Missing values:",
        int(df.isna().sum().sum())
    )

    print(
        "Duplicate timestamps:",
        int(
            df["time"]
            .duplicated()
            .sum()
        )
    )

    print(
        "Unique timestamps:",
        df["time"].nunique()
    )

    print()
    print("=" * 60)
    print("EURUSD READY FOR BACKTESTING")
    print("=" * 60)


if __name__ == "__main__":

    main()
