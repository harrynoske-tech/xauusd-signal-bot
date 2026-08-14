import os
from datetime import datetime

import pandas as pd
import dukascopy_python
from dukascopy_python import instruments


# ============================================================
# DUKASCOPY XAUUSD HISTORICAL DATA DOWNLOADER
# ============================================================

OUTPUT_DIR = "data"

START_DATE = datetime(2020, 1, 1)
END_DATE = datetime.now()

OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

XAUUSD = instruments.INSTRUMENT_FX_METALS_XAU_USD


# ============================================================
# DOWNLOAD
# ============================================================

def download_data(interval, filename):

    print()
    print("=" * 60)
    print("DOWNLOADING:", filename)
    print("=" * 60)

    print("Instrument: XAU/USD")
    print("Start:", START_DATE)
    print("End:", END_DATE)
    print("Interval:", interval)
    print()

    print(
        "Requesting Dukascopy data...",
        flush=True
    )

    data = dukascopy_python.fetch(
        XAUUSD,
        interval,
        OFFER_SIDE,
        START_DATE,
        END_DATE,
    )

    if data is None:
        raise RuntimeError(
            "Dukascopy returned None."
        )

    if data.empty:
        raise RuntimeError(
            "Dukascopy returned zero candles."
        )

    data = data.copy()

    data.index = pd.to_datetime(
        data.index
    )

    data = (
        data
        .sort_index()
        .drop_duplicates()
    )

    data.columns = [
        str(column).capitalize()
        for column in data.columns
    ]

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    for column in required:

        if column not in data.columns:

            raise RuntimeError(
                f"Missing column {column}. "
                f"Received: {list(data.columns)}"
            )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    data.to_csv(
        output_path
    )

    print()
    print("CANDLES:", len(data))
    print("FIRST:", data.index.min())
    print("LAST:", data.index.max())
    print("COLUMNS:", list(data.columns))
    print("SAVED:", output_path)

    print()
    print("LATEST 5 CANDLES:")
    print(data.tail(5))

    return data


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("DUKASCOPY XAUUSD DATA DOWNLOADER")
    print("=" * 60)
    print()

    print("DATA SOURCE: DUKASCOPY")
    print("SYMBOL: XAUUSD")
    print("INSTRUMENT:", XAUUSD)
    print("START:", START_DATE)
    print("END:", END_DATE)

    # --------------------------------------------------------
    # 15 MINUTE
    # --------------------------------------------------------

    data_15m = download_data(
        dukascopy_python.INTERVAL_MIN_15,
        "XAUUSD_15m.csv",
    )

    # --------------------------------------------------------
    # DAILY
    # --------------------------------------------------------

    data_daily = download_data(
        dukascopy_python.INTERVAL_DAY_1,
        "XAUUSD_1d.csv",
    )

    # --------------------------------------------------------
    # VERIFY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print()

    print(
        "15M CANDLES:",
        len(data_15m)
    )

    print(
        "DAILY CANDLES:",
        len(data_daily)
    )

    print()

    print(
        "15M:",
        data_15m.index.min(),
        "->",
        data_15m.index.max()
    )

    print(
        "DAILY:",
        data_daily.index.min(),
        "->",
        data_daily.index.max()
    )

    print()
    print("Files created:")
    print("  data/XAUUSD_15m.csv")
    print("  data/XAUUSD_1d.csv")
    print()


if __name__ == "__main__":
    main()
