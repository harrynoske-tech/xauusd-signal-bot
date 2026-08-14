import os
from datetime import datetime

import pandas as pd
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_XAUUSD


# ============================================================
# DUKASCOPY HISTORICAL DATA DOWNLOADER
# ============================================================

OUTPUT_DIR = "data"

START_DATE = datetime(2020, 1, 1)
END_DATE = datetime.now()

OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download(
    instrument,
    interval,
    start_date,
    end_date,
    filename
):

    print()
    print(
        "Downloading:",
        filename,
        flush=True
    )

    data = dukascopy_python.fetch(
        instrument,
        interval,
        OFFER_SIDE,
        start_date,
        end_date,
    )

    if data is None or data.empty:

        raise RuntimeError(
            f"No data returned for {filename}"
        )

    data = data.copy()

    data.columns = [
        str(column).capitalize()
        for column in data.columns
    ]

    data = (
        data
        .sort_index()
        .drop_duplicates()
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    data.to_csv(
        output_path
    )

    print(
        "Candles:",
        len(data)
    )

    print(
        "First:",
        data.index.min()
    )

    print(
        "Last:",
        data.index.max()
    )

    print(
        "Saved:",
        output_path,
        flush=True
    )

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
    print("Source: Dukascopy")
    print("Symbol: XAUUSD")
    print("Start:", START_DATE)
    print("End:", END_DATE)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 15 MINUTE DATA
    # --------------------------------------------------------

    download(
        INSTRUMENT_XAUUSD,
        dukascopy_python.INTERVAL_MINUTE_15,
        START_DATE,
        END_DATE,
        "XAUUSD_15m.csv",
    )

    # --------------------------------------------------------
    # DAILY DATA
    # --------------------------------------------------------

    download(
        INSTRUMENT_XAUUSD,
        dukascopy_python.INTERVAL_DAY_1,
        START_DATE,
        END_DATE,
        "XAUUSD_1d.csv",
    )

    print()
    print("=" * 60)
    print("DUKASCOPY DOWNLOAD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
