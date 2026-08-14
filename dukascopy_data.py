import os
from datetime import datetime, timedelta

import pandas as pd
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_XAUUSD


# ============================================================
# DUKASCOPY DATA DOWNLOADER
# ============================================================

SYMBOL = "XAUUSD"

OUTPUT_DIR = "data"

START_DATE = datetime(2020, 1, 1)

END_DATE = datetime.now()

OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID


# ============================================================
# DOWNLOAD
# ============================================================

def download_data():

    print("=" * 60)
    print("DUKASCOPY XAUUSD DATA DOWNLOADER")
    print("=" * 60)

    print()
    print("SOURCE: Dukascopy")
    print("SYMBOL:", SYMBOL)
    print("TIMEFRAME: 15 minutes")
    print("START:", START_DATE)
    print("END:", END_DATE)
    print()

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print(
        "Downloading historical data...",
        flush=True
    )

    data = dukascopy_python.fetch(
        INSTRUMENT_XAUUSD,
        dukascopy_python.INTERVAL_MINUTE_15,
        OFFER_SIDE,
        START_DATE,
        END_DATE,
    )

    if data is None or data.empty:

        raise RuntimeError(
            "Dukascopy returned no data."
        )

    print()
    print(
        "Candles received:",
        len(data)
    )

    print(
        "First candle:",
        data.index.min()
    )

    print(
        "Last candle:",
        data.index.max()
    )

    # --------------------------------------------------------
    # Standardise column names
    # --------------------------------------------------------

    data = data.copy()

    data.columns = [
        str(column).capitalize()
        for column in data.columns
    ]

    rename_map = {
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
    }

    data = data.rename(
        columns=rename_map
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        "XAUUSD_15m.csv"
    )

    data.to_csv(
        output_file
    )

    print()
    print(
        "Saved:",
        output_file
    )

    print()
    print(
        "LATEST 5 CANDLES:"
    )

    print(
        data.tail(5)
    )

    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    download_data()
