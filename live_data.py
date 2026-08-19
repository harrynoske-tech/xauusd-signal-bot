import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests


# ============================================================
# CONTINUOUS MARKET DATA SERVICE
# ============================================================
#
# XAUUSD + EURUSD
#
# Purpose:
#   Keep the latest market data available to the signal engine.
#
# This service does NOT:
#   - place trades
#   - connect to MT5
#   - send Telegram signals
#
# It only maintains fresh 15-minute data.
# ============================================================


DATA_DIR = "data"

MARKETS = {
    "XAUUSD": {
        "file": os.path.join(
            DATA_DIR,
            "XAUUSD_15m.csv"
        )
    },

    "EURUSD": {
        "file": os.path.join(
            DATA_DIR,
            "EURUSD_15m.csv"
        )
    },
}


# ============================================================
# SETTINGS
# ============================================================

CHECK_INTERVAL = 10

DATA_URLS = {
    "XAUUSD": os.getenv(
        "XAUUSD_LIVE_DATA_URL",
        ""
    ),

    "EURUSD": os.getenv(
        "EURUSD_LIVE_DATA_URL",
        ""
    ),
}


# ============================================================
# LOGGING
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    )


def log(message):

    print(
        f"[{utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')}] "
        f"{message}",
        flush=True
    )


# ============================================================
# CSV VALIDATION
# ============================================================

def validate_csv(
    market,
    path
):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{market}: "
            f"{path} does not exist."
        )

    df = pd.read_csv(
        path
    )

    if df.empty:

        raise RuntimeError(
            f"{market}: "
            "CSV is empty."
        )

    df.columns = [

        str(column)
        .strip()
        .lower()
        .replace(
            " ",
            "_"
        )

        for column in df.columns
    ]

    required = [
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [

        column

        for column in required

        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{market}: "
            f"missing columns {missing}"
        )

    time_column = None

    for candidate in (
        "time",
        "datetime",
        "timestamp",
        "date",
    ):

        if candidate in df.columns:

            time_column = candidate

            break

    if time_column is None:

        raise RuntimeError(
            f"{market}: "
            "No timestamp column found."
        )

    df["time"] = pd.to_datetime(

        df[time_column],

        utc=True,

        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "time",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    df = (

        df

        .sort_values(
            "time"
        )

        .drop_duplicates(
            subset="time"
        )
    )

    if df.empty:

        raise RuntimeError(
            f"{market}: "
            "No valid candles."
        )

    return df


# ============================================================
# LATEST CANDLE
# ============================================================

def get_latest_candle(
    market,
    path
):

    df = validate_csv(
        market,
        path
    )

    row = df.iloc[-1]

    return {

        "market":
            market,

        "time":
            row["time"],

        "open":
            float(row["open"]),

        "high":
            float(row["high"]),

        "low":
            float(row["low"]),

        "close":
            float(row["close"]),
    }


# ============================================================
# OPTIONAL HTTP LIVE FEED
# ============================================================
#
# If a live feed URL is configured, it must return JSON in
# this format:
#
# {
#   "time": "...",
#   "open": 1.0,
#   "high": 1.0,
#   "low": 1.0,
#   "close": 1.0
# }
#
# We deliberately do NOT invent a provider URL here.
# ============================================================

def fetch_live_candle(
    market
):

    url = DATA_URLS.get(
        market,
        ""
    )

    if not url:

        return None

    response = requests.get(
        url,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    required = [
        "time",
        "open",
        "high",
        "low",
        "close",
    ]

    for key in required:

        if key not in data:

            raise RuntimeError(
                f"{market}: "
                f"live feed missing {key}"
            )

    return {

        "market":
            market,

        "time":
            pd.to_datetime(
                data["time"],
                utc=True
            ),

        "open":
            float(data["open"]),

        "high":
            float(data["high"]),

        "low":
            float(data["low"]),

        "close":
            float(data["close"]),
    }


# ============================================================
# STATE
# ============================================================

def save_live_state(
    market,
    candle
):

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    path = os.path.join(

        DATA_DIR,

        f"{market}_live.csv"
    )

    pd.DataFrame(
        [candle]
    ).to_csv(
        path,
        index=False
    )


# ============================================================
# MARKET MONITOR
# ============================================================

def monitor_market(
    market,
    config
):

    path = config["file"]

    try:

        live_candle = fetch_live_candle(
            market
        )

        if live_candle is not None:

            candle = live_candle

            source = "LIVE FEED"

        else:

            candle = get_latest_candle(
                market,
                path
            )

            source = "CSV"

        save_live_state(
            market,
            candle
        )

        log(

            f"{market} | "
            f"{source} | "
            f"candle={candle['time']} | "
            f"close={candle['close']}"
        )

        return candle

    except Exception as error:

        log(

            f"{market} ERROR: "
            f"{type(error).__name__}: "
            f"{error}"
        )

        return None


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 60
    )

    print(
        "CONTINUOUS MARKET DATA SERVICE"
    )

    print(
        "=" * 60
    )

    print(
        "MARKETS: XAUUSD, EURUSD"
    )

    print(
        "MODE: CONTINUOUS"
    )

    print(
        "MT5: DISABLED"
    )

    print(
        "TELEGRAM: DISABLED"
    )

    print(
        "=" * 60
    )

    while True:

        loop_start = time.time()

        for market, config in MARKETS.items():

            monitor_market(
                market,
                config
            )

        elapsed = (
            time.time()
            - loop_start
        )

        sleep_time = max(

            1,

            CHECK_INTERVAL
            - elapsed
        )

        time.sleep(
            sleep_time
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "LIVE DATA SERVICE STOPPED."
        )
