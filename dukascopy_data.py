import os
from datetime import datetime

import pandas as pd
import dukascopy_python
from dukascopy_python import instruments


# ============================================================
# MULTI-MARKET DUKASCOPY HISTORICAL DATA DOWNLOADER
# ============================================================

OUTPUT_DIR = "data"

START_DATE = datetime(2020, 1, 1)
END_DATE = datetime.now()

OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID


# ============================================================
# MARKETS
# ============================================================

MARKETS = {
    "XAUUSD": instruments.INSTRUMENT_FX_METALS_XAU_USD,
    "EURUSD": instruments.INSTRUMENT_FX_MAJORS_EUR_USD,
    "GBPUSD": instruments.INSTRUMENT_FX_MAJORS_GBP_USD,
    "USDJPY": instruments.INSTRUMENT_FX_MAJORS_USD_JPY,
    "AUDUSD": instruments.INSTRUMENT_FX_MAJORS_AUD_USD,
    "USDCAD": instruments.INSTRUMENT_FX_MAJORS_USD_CAD,
    "USDCHF": instruments.INSTRUMENT_FX_MAJORS_USD_CHF,
}


# ============================================================
# DOWNLOAD ONE MARKET
# ============================================================

def download_data(symbol, instrument, interval, filename):

    print()
    print("=" * 70)
    print("DOWNLOADING:", filename)
    print("=" * 70)

    print("Symbol:", symbol)
    print("Instrument:", instrument)
    print("Start:", START_DATE)
    print("End:", END_DATE)
    print("Interval:", interval)
    print()

    print(
        "Requesting Dukascopy data...",
        flush=True
    )

    data = dukascopy_python.fetch(
        instrument,
        interval,
        OFFER_SIDE,
        START_DATE,
        END_DATE,
    )

    if data is None:
        raise RuntimeError(
            f"Dukascopy returned None for {symbol}."
        )

    if data.empty:
        raise RuntimeError(
            f"Dukascopy returned zero candles for {symbol}."
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
                f"{symbol}: missing column {column}. "
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
    print("=" * 70)
    print("MULTI-MARKET DUKASCOPY DATA DOWNLOADER")
    print("=" * 70)
    print()

    print("DATA SOURCE: DUKASCOPY")
    print("START:", START_DATE)
    print("END:", END_DATE)
    print()

    print("MARKETS:")
    for symbol in MARKETS:
        print(" ", symbol)

    print()

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results = {}

    # ========================================================
    # DOWNLOAD 15-MINUTE DATA
    # ========================================================

    for symbol, instrument in MARKETS.items():

        filename = f"{symbol}_15m.csv"

        try:

            results[symbol] = download_data(
                symbol,
                instrument,
                dukascopy_python.INTERVAL_MIN_15,
                filename,
            )

        except Exception as error:

            print()
            print("=" * 70)
            print("FAILED:", symbol)
            print("=" * 70)
            print(
                type(error).__name__,
                ":",
                error
            )

            raise

    # ========================================================
    # DOWNLOAD DAILY DATA
    # ========================================================

    print()
    print("=" * 70)
    print("DOWNLOADING DAILY DATA")
    print("=" * 70)

    daily_results = {}

    for symbol, instrument in MARKETS.items():

        filename = f"{symbol}_1d.csv"

        try:

            daily_results[symbol] = download_data(
                symbol,
                instrument,
                dukascopy_python.INTERVAL_DAY_1,
                filename,
            )

        except Exception as error:

            print()
            print("=" * 70)
            print("FAILED DAILY:", symbol)
            print("=" * 70)
            print(
                type(error).__name__,
                ":",
                error
            )

            raise

    # ========================================================
    # FINAL VERIFICATION
    # ========================================================

    print()
    print("=" * 70)
    print("MULTI-MARKET DOWNLOAD COMPLETE")
    print("=" * 70)
    print()

    print("15-MINUTE FILES:")

    for symbol, data in results.items():

        print(
            f"  {symbol}: "
            f"{len(data):,} candles | "
            f"{data.index.min()} -> "
            f"{data.index.max()}"
        )

    print()

    print("DAILY FILES:")

    for symbol, data in daily_results.items():

        print(
            f"  {symbol}: "
            f"{len(data):,} candles | "
            f"{data.index.min()} -> "
            f"{data.index.max()}"
        )

    print()

    print("FILES CREATED:")

    for symbol in MARKETS:

        print(
            f"  data/{symbol}_15m.csv"
        )

        print(
            f"  data/{symbol}_1d.csv"
        )

    print()
    print("ALL MARKETS DOWNLOADED SUCCESSFULLY.")
    print()


if __name__ == "__main__":
    main()
