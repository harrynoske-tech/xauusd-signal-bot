import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


# ============================================================
# V11.8 MULTI-MARKET TELEGRAM SIGNAL BOT
# ============================================================
#
# SIGNALS ONLY
# MANUAL MT5 EXECUTION
# NO AUTOMATIC TRADING
#
# ENABLED:
#   XAUUSD
#   EURUSD
#
# DISABLED UNTIL VALIDATED:
#   GBPUSD
#   USDJPY
#   AUDUSD
#   USDCAD
#   USDCHF
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN"
)

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID"
)

STATE_FILE = os.path.join(
    DATA_DIR,
    "signal_bot_state.txt"
)


# ============================================================
# V11.8 MARKET PARAMETERS
# ============================================================

MARKETS = {

    "XAUUSD": {
        "file": "data/XAUUSD_15m.csv",
        "enabled": True,

        "rr": 0.35,
        "wick": 0.20,
        "body": 0.15,
        "separation": 0.00040,
        "threshold": -0.25,
        "hours": (3, 4),
    },

    "EURUSD": {
        "file": "data/EURUSD_15m.csv",
        "enabled": True,

        "rr": 0.35,
        "wick": 0.20,
        "body": 0.15,
        "separation": 0.00050,
        "threshold": 0.00,
        "hours": (3, 4, 5),
    },

    "GBPUSD": {
        "enabled": False,
    },

    "USDJPY": {
        "enabled": False,
    },

    "AUDUSD": {
        "enabled": False,
    },

    "USDCAD": {
        "enabled": False,
    },

    "USDCHF": {
        "enabled": False,
    },
}


# ============================================================
# STATE
# ============================================================

def load_state():

    if not os.path.exists(STATE_FILE):
        return {}

    state = {}

    with open(
        STATE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split("|", 1)

            if len(parts) != 2:
                continue

            state[parts[0]] = parts[1]

    return state


def save_state(state):

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    with open(
        STATE_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        for market, signal_id in state.items():

            file.write(
                f"{market}|{signal_id}\n"
            )


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not configured."
        )

    if not TELEGRAM_CHAT_ID:
        raise RuntimeError(
            "TELEGRAM_CHAT_ID is not configured."
        )

    url = (
        "https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    response = requests.post(
        url,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
        },
        timeout=20,
    )

    response.raise_for_status()


# ============================================================
# DATA LOADING
# ============================================================

def load_data(market, config):

    path = config["file"]

    if not os.path.exists(path):

        print(
            f"{market}: missing {path}"
        )

        return None

    df = pd.read_csv(path)

    # --------------------------------------------------------
    # NORMALISE COLUMN NAMES
    # --------------------------------------------------------

    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        for column in df.columns
    ]

    # --------------------------------------------------------
    # FIND TIME COLUMN
    # --------------------------------------------------------

    time_column = None

    for candidate in (
        "time",
        "datetime",
        "date",
        "timestamp",
    ):

        if candidate in df.columns:

            time_column = candidate
            break

    if time_column is None:

        raise RuntimeError(
            f"{market}: no datetime column found."
        )

    df["time"] = pd.to_datetime(
        df[time_column],
        utc=True,
        errors="coerce",
    )

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

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
            f"{market}: missing columns {missing}"
        )

    # --------------------------------------------------------
    # NUMERIC
    # --------------------------------------------------------

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "time",
            *required,
        ]
    )

    df = (
        df
        .sort_values("time")
        .drop_duplicates(
            subset="time"
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# INDICATORS
# ============================================================

def prepare_indicators(df):

    df = df.copy()

    high = df["high"]
    low = df["low"]
    open_price = df["open"]
    close = df["close"]

    # --------------------------------------------------------
    # CANDLE RANGE
    # --------------------------------------------------------

    candle_range = high - low

    body = (
        close - open_price
    ).abs()

    df["body_ratio"] = np.where(
        candle_range > 0,
        body / candle_range,
        np.nan,
    )

    # --------------------------------------------------------
    # WICKS
    # --------------------------------------------------------

    df["upper_wick"] = np.where(
        candle_range > 0,
        (
            high
            - np.maximum(
                open_price,
                close,
            )
        ) / candle_range,
        np.nan,
    )

    df["lower_wick"] = np.where(
        candle_range > 0,
        (
            np.minimum(
                open_price,
                close,
            )
            - low
        ) / candle_range,
        np.nan,
    )

    # --------------------------------------------------------
    # ATR 14
    # --------------------------------------------------------

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,

            (
                high
                - previous_close
            ).abs(),

            (
                low
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr14"] = (
        true_range
        .rolling(
            14,
            min_periods=14,
        )
        .mean()
    )

    # --------------------------------------------------------
    # EMA 20
    # --------------------------------------------------------

    df["ema20"] = (
        close
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    # --------------------------------------------------------
    # EMA 50
    # --------------------------------------------------------

    df["ema50"] = (
        close
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    df["momentum5"] = (
        close
        / close.shift(5)
        - 1.0
    )

    # --------------------------------------------------------
    # RANGE POSITION
    # --------------------------------------------------------

    high20 = (
        high
        .rolling(
            20,
            min_periods=20,
        )
        .max()
    )

    low20 = (
        low
        .rolling(
            20,
            min_periods=20,
        )
        .min()
    )

    range20 = high20 - low20

    df["range_position"] = np.where(
        range20 > 0,
        (
            close - low20
        ) / range20,
        np.nan,
    )

    return df


# ============================================================
# V11.8 SIGNAL
# ============================================================

def calculate_signal(row, config):

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if pd.isna(row["atr14"]):
        return None

    if row["atr14"] <= 0:
        return None

    if pd.isna(row["body_ratio"]):
        return None

    if pd.isna(row["range_position"]):
        return None

    if pd.isna(row["momentum5"]):
        return None

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = 0.0

    bullish = (
        row["close"]
        > row["open"]
    )

    bearish = (
        row["close"]
        < row["open"]
    )

    # --------------------------------------------------------
    # WICK / REJECTION
    # --------------------------------------------------------

    if (
        row["lower_wick"]
        >= config["wick"]
    ):

        score += 1.0

    if (
        row["upper_wick"]
        >= config["wick"]
    ):

        score -= 1.0

    # --------------------------------------------------------
    # BODY
    # --------------------------------------------------------

    if (
        row["body_ratio"]
        <= config["body"]
    ):

        score += 0.50

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    if bullish:

        score += 0.25

    elif bearish:

        score -= 0.25

    # --------------------------------------------------------
    # RANGE LOCATION
    # --------------------------------------------------------

    if (
        bullish
        and
        row["range_position"] <= 0.35
    ):

        score += 0.50

    if (
        bearish
        and
        row["range_position"] >= 0.65
    ):

        score -= 0.50

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    if (
        bullish
        and
        row["momentum5"] > 0
    ):

        score += 0.25

    elif (
        bearish
        and
        row["momentum5"] < 0
    ):

        score -= 0.25

    # --------------------------------------------------------
    # EMA STRUCTURE
    # --------------------------------------------------------

    if (
        row["ema20"]
        > row["ema50"]
    ):

        score += 0.10

    elif (
        row["ema20"]
        < row["ema50"]
    ):

        score -= 0.10

    # --------------------------------------------------------
    # EMA SEPARATION
    # --------------------------------------------------------

    separation = (
        abs(
            row["ema20"]
            - row["ema50"]
        )
        / row["atr14"]
    )

    if (
        separation
        >= config["separation"]
    ):

        if (
            row["ema20"]
            > row["ema50"]
        ):

            score += 0.10

        else:

            score -= 0.10

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    if score < config["threshold"]:
        return None

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    if score >= 0:

        direction = "BUY"

    else:

        direction = "SELL"

    return {
        "direction": direction,
        "score": float(score),
        "atr": float(row["atr14"]),
    }


# ============================================================
# PRICE FORMATTING
# ============================================================

def decimals(market):

    if market == "XAUUSD":
        return 2

    if market == "USDJPY":
        return 3

    return 5


def format_price(market, value):

    places = decimals(market)

    return f"{value:.{places}f}"


# ============================================================
# BUILD TRADE SIGNAL
# ============================================================

def build_signal(
    market,
    df,
    config,
):

    if len(df) < 100:
        return None

    # --------------------------------------------------------
    # LAST COMPLETED CANDLE
    # --------------------------------------------------------

    candle = df.iloc[-2]

    candle_time = candle["time"]

    # --------------------------------------------------------
    # SESSION FILTER
    # --------------------------------------------------------

    if (
        candle_time.hour
        not in config["hours"]
    ):

        return None

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    result = calculate_signal(
        candle,
        config,
    )

    if result is None:
        return None

    # --------------------------------------------------------
    # NEXT CANDLE OPEN
    # --------------------------------------------------------

    entry = float(
        df.iloc[-1]["open"]
    )

    atr = result["atr"]

    rr = config["rr"]

    # --------------------------------------------------------
    # SL / TP
    # --------------------------------------------------------

    if result["direction"] == "BUY":

        sl = entry - atr

        tp = (
            entry
            + atr * rr
        )

    else:

        sl = entry + atr

        tp = (
            entry
            - atr * rr
        )

    return {
        "market": market,
        "direction": result["direction"],
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": rr,
        "score": result["score"],
        "signal_time": candle_time,
    }


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def signal_message(signal):

    market = signal["market"]
    direction = signal["direction"]

    if direction == "BUY":
        emoji = "🟢"
    else:
        emoji = "🔴"

    entry = format_price(
        market,
        signal["entry"],
    )

    sl = format_price(
        market,
        signal["sl"],
    )

    tp = format_price(
        market,
        signal["tp"],
    )

    signal_time = (
        signal["signal_time"]
        .strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    )

    return (
        f"{emoji} V11.8 SIGNAL\n\n"
        f"{market} {direction}\n\n"
        f"Entry: {entry}\n"
        f"SL: {sl}\n"
        f"TP: {tp}\n\n"
        f"RR: {signal['rr']:.2f}\n"
        f"Score: {signal['score']:.2f}\n\n"
        f"Signal candle:\n"
        f"{signal_time}\n\n"
        "MANUAL MT5 EXECUTION"
    )


# ============================================================
# PROCESS MARKET
# ============================================================

def process_market(
    market,
    config,
    state,
):

    if not config.get(
        "enabled",
        False,
    ):

        return state

    print()
    print("=" * 60)
    print(
        f"CHECKING {market}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = load_data(
        market,
        config,
    )

    if df is None:
        return state

    print(
        "Candles:",
        len(df),
    )

    print(
        "Latest:",
        df["time"].iloc[-1],
    )

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

    df = prepare_indicators(df)

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    signal = build_signal(
        market,
        df,
        config,
    )

    if signal is None:

        print(
            "No signal."
        )

        return state

    # --------------------------------------------------------
    # UNIQUE SIGNAL ID
    # --------------------------------------------------------

    signal_id = (
        f"{market}|"
        f"{signal['signal_time']}|"
        f"{signal['direction']}"
    )

    # --------------------------------------------------------
    # DUPLICATE PROTECTION
    # --------------------------------------------------------

    if state.get(market) == signal_id:

        print(
            "Signal already sent."
        )

        return state

    # --------------------------------------------------------
    # MESSAGE
    # --------------------------------------------------------

    message = signal_message(
        signal
    )

    print()
    print(message)

    # --------------------------------------------------------
    # TELEGRAM
    # --------------------------------------------------------

    send_telegram(message)

    state[market] = signal_id

    print()
    print(
        "Telegram signal sent."
    )

    return state


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "V11.8 MULTI-MARKET "
        "TELEGRAM SIGNAL BOT"
    )
    print("=" * 60)

    print(
        "SIGNALS ONLY"
    )

    print(
        "MANUAL MT5 EXECUTION"
    )

    print(
        "NO AUTOMATIC TRADING"
    )

    print("=" * 60)

    state = load_state()

    enabled_markets = [
        market
        for market, config in MARKETS.items()
        if config.get(
            "enabled",
            False,
        )
    ]

    print()
    print(
        "ENABLED MARKETS:",
        ", ".join(enabled_markets),
    )

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    for market, config in MARKETS.items():

        if not config.get(
            "enabled",
            False,
        ):

            continue

        try:

            state = process_market(
                market,
                config,
                state,
            )

        except Exception as error:

            print()
            print(
                f"{market} ERROR:"
            )

            print(
                type(error).__name__,
                error,
            )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_state(state)

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "SIGNAL CHECK COMPLETE"
    )

    print(
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
