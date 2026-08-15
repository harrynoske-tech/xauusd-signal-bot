# ============================================================
# EURUSD 15M DATA DOWNLOADER
# SOURCE: DUKASCOPY
# ============================================================

import os
from datetime import datetime

import pandas as pd
import dukascopy_python

from dukascopy_python.instruments import (
    INSTRUMENT_FX_MAJORS_EUR_USD,
)


# ============================================================
# SETTINGS
# ============================================================

START_DATE = datetime(
    2020,
    1,
    1
)

END_DATE = datetime(
    2026,
    8,
    15
)

OUTPUT_FILE = (
    "data/EURUSD_15m.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DUKASCOPY EURUSD DATA DOWNLOADER")
    print("=" * 60)

    print()
    print("DATA SOURCE: DUKASCOPY")
    print("SYMBOL: EURUSD")
    print("INSTRUMENT: EUR/USD")
    print(
        "START:",
        START_DATE
    )
    print(
        "END:",
        END_DATE
    )

    os.makedirs(
        "data",
        exist_ok=True
    )

    print()
    print("=" * 60)
    print("DOWNLOADING: EURUSD_15m.csv")
    print("=" * 60)

    print(
        "Instrument: EUR/USD"
    )

    print(
        "Start:",
        START_DATE
    )

    print(
        "End:",
        END_DATE
    )

    print(
        "Interval: 15MIN"
    )

    print()
    print(
        "Requesting Dukascopy data..."
    )

    try:

        df = dukascopy_python.fetch(
            instrument=(
                INSTRUMENT_FX_MAJORS_EUR_USD
            ),
            interval=(
                dukascopy_python.INTERVAL_MIN_15
            ),
            offer_side=(
                dukascopy_python.OFFER_SIDE_BID
            ),
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
    # NORMALISE INDEX
    # ========================================================

    if isinstance(
        df.index,
        pd.DatetimeIndex
    ):

        df = df.reset_index()

    # ========================================================
    # NORMALISE COLUMN NAMES
    # ========================================================

    rename = {}

    for column in df.columns:

        lower = str(
            column
        ).lower()

        if lower in [
            "timestamp",
            "datetime",
            "date",
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
    # VERIFY TIME
    # ========================================================

    if "time" not in df.columns:

        raise RuntimeError(
            "Could not find timestamp column "
            "in EURUSD data."
        )

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    )

    # ========================================================
    # VERIFY OHLC
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
                f"Missing required column: "
                f"{column}"
            )

    # ========================================================
    # SELECT COLUMNS
    # ========================================================

    columns = required

    if "Volume" in df.columns:

        columns.append(
            "Volume"
        )

    df = df[
        columns
    ]

    # ========================================================
    # CLEAN
    # ========================================================

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
    # OUTPUT
    # ========================================================

    print()
    print("=" * 60)
    print("EURUSD DOWNLOAD COMPLETE")
    print("=" * 60)

    print(
        "15M CANDLES:",
        len(df)
    )

    print(
        "15M:",
        df["time"].min(),
        "->",
        df["time"].max()
    )

    print()
    print(
        "File created:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    print()
    print("=" * 60)
    print("DATA QUALITY")
    print("=" * 60)

    print(
        "Missing values:",
        int(
            df.isna()
            .sum()
            .sum()
        )
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
    print(
        "EURUSD READY FOR BACKTESTING"
    )
    print("=" * 60)


if __name__ == "__main__":

    main()
