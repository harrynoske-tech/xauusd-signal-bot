import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


# ============================================================
# V11.8 CONTINUOUS MULTI-MARKET TELEGRAM SIGNAL BOT
# ============================================================
#
# SIGNALS ONLY
# MANUAL MT5 EXECUTION
# NO AUTOMATIC TRADING
#
# Markets:
#   XAUUSD
#   EURUSD
#
# The bot stays running continuously.
#
# Historical CSV data is loaded once.
# The bot checks for new completed 15-minute candles.
#
# IMPORTANT:
# This version does NOT place MT5 trades.
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

# How often the bot checks whether a new candle is available.
CHECK_INTERVAL_SECONDS = 10

# How often the CSV data is refreshed.
#
# This is deliberately much longer than the checking interval.
# We do NOT redownload the data every 10 seconds.
#
DATA_REFRESH_SECONDS = 60

# Minimum seconds between repeated pre-signal messages.
PRE_SIGNAL_COOLDOWN = 300


# ============================================================
# MARKET CONFIGURATION
# ============================================================

MARKETS = {

    "XAUUSD": {

        "file": "data/XAUUSD_15m.csv",

        "enabled": True,

        # V11.8 XAUUSD baseline
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

        # V11.8 EURUSD baseline
        "rr": 0.35,
        "wick": 0.20,
        "body": 0.15,
        "separation": 0.00050,
        "threshold": 0.00,
        "hours": (3, 4, 5),
    },

    # Not enabled yet.
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

STATE_FILE = os.path.join(
    DATA_DIR,
    "continuous_signal_state.csv"
)


def utc_now():
    return datetime.now(
        timezone.utc
    )


def load_state():

    state = {}

    if not os.path.exists(
        STATE_FILE
    ):
        return state

    try:

        df = pd.read_csv(
            STATE_FILE
        )

        for _, row in df.iterrows():

            market = str(
                row["market"]
            )

            state[market] = {
                "last_candle": str(
                    row["last_candle"]
                ),
                "last_signal": str(
                    row["last_signal"]
                ),
                "last_presignal": str(
                    row["last_presignal"]
                ),
            }

    except Exception as error:

        print(
            "STATE LOAD ERROR:",
            error
        )

    return state


def save_state(state):

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    rows = []

    for market, values in state.items():

        rows.append({
            "market": market,
            "last_candle":
                values.get(
                    "last_candle",
                    ""
                ),
            "last_signal":
                values.get(
                    "last_signal",
                    ""
                ),
            "last_presignal":
                values.get(
                    "last_presignal",
                    ""
                ),
        })

    if rows:

        pd.DataFrame(
            rows
        ).to_csv(
            STATE_FILE,
            index=False
        )


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    if not TELEGRAM_BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN "
            "is not configured."
        )

    if not TELEGRAM_CHAT_ID:

        raise RuntimeError(
            "TELEGRAM_CHAT_ID "
            "is not configured."
        )

    url = (
        "https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    response = requests.post(

        url,

        json={
            "chat_id":
                TELEGRAM_CHAT_ID,

            "text":
                message,
        },

        timeout=20,
    )

    response.raise_for_status()


# ============================================================
# DATA LOADING
# ============================================================

def load_data(
    market,
    config
):

    path = config["file"]

    if not os.path.exists(path):

        raise RuntimeError(
            f"{market}: missing {path}"
        )

    df = pd.read_csv(
        path
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
            f"{market}: "
            "No datetime column."
        )

    df["time"] = pd.to_datetime(

        df[time_column],

        utc=True,

        errors="coerce",
    )

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
            f"{market}: missing "
            f"columns {missing}"
        )

    for column in required:

        df[column] = pd.to_numeric(

            df[column],

            errors="coerce"
        )

    df = df.dropna(

        subset=[
            "time",
            *required,
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

        .reset_index(
            drop=True
        )
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

    candle_range = (
        high - low
    )

    body = (
        close - open_price
    ).abs()

    df["body_ratio"] = np.where(

        candle_range > 0,

        body / candle_range,

        np.nan,
    )

    df["upper_wick"] = np.where(

        candle_range > 0,

        (
            high
            - np.maximum(
                open_price,
                close,
            )
        )
        / candle_range,

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
        )
        / candle_range,

        np.nan,
    )

    previous_close = (
        close.shift(1)
    )

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

    ).max(
        axis=1
    )

    df["atr14"] = (

        true_range

        .rolling(
            14,
            min_periods=14
        )

        .mean()
    )

    df["ema20"] = (

        close

        .ewm(
            span=20,
            adjust=False
        )

        .mean()
    )

    df["ema50"] = (

        close

        .ewm(
            span=50,
            adjust=False
        )

        .mean()
    )

    df["momentum5"] = (

        close
        / close.shift(5)
        - 1.0
    )

    high20 = (

        high

        .rolling(
            20,
            min_periods=20
        )

        .max()
    )

    low20 = (

        low

        .rolling(
            20,
            min_periods=20
        )

        .min()
    )

    range20 = (
        high20 - low20
    )

    df["range_position"] = np.where(

        range20 > 0,

        (
            close - low20
        )
        / range20,

        np.nan,
    )

    return df


# ============================================================
# V11.8 SCORE
# ============================================================

def calculate_score(
    row,
    config
):

    if pd.isna(
        row["atr14"]
    ):

        return None

    if row["atr14"] <= 0:

        return None

    score = 0.0

    bullish = (
        row["close"]
        > row["open"]
    )

    bearish = (
        row["close"]
        < row["open"]
    )

    # Rejection / wick

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

    # Small body

    if (
        row["body_ratio"]
        <= config["body"]
    ):

        score += 0.50

    # Candle direction

    if bullish:

        score += 0.25

    elif bearish:

        score -= 0.25

    # Range location

    if (

        bullish

        and row["range_position"]
        <= 0.35

    ):

        score += 0.50

    if (

        bearish

        and row["range_position"]
        >= 0.65

    ):

        score -= 0.50

    # Momentum

    if (

        bullish

        and row["momentum5"] > 0

    ):

        score += 0.25

    elif (

        bearish

        and row["momentum5"] < 0

    ):

        score -= 0.25

    # EMA structure

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

    # EMA separation

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

    return float(score)


# ============================================================
# PRICE FORMAT
# ============================================================

def decimals(market):

    if market == "XAUUSD":

        return 2

    if market == "USDJPY":

        return 3

    return 5


def format_price(
    market,
    value
):

    return (
        f"{value:.{decimals(market)}f}"
    )


# ============================================================
# CONFIRMED SIGNAL
# ============================================================

def build_confirmed_signal(
    market,
    df,
    config
):

    if len(df) < 100:

        return None

    # Last completed candle.
    candle = df.iloc[-2]

    # Current/new candle.
    current = df.iloc[-1]

    candle_time = candle["time"]

    if (
        candle_time.hour
        not in config["hours"]
    ):

        return None

    score = calculate_score(
        candle,
        config
    )

    if score is None:

        return None

    if (
        score
        < config["threshold"]
    ):

        return None

    direction = (

        "BUY"

        if score >= 0

        else "SELL"
    )

    entry = float(
        current["open"]
    )

    atr = float(
        candle["atr14"]
    )

    rr = config["rr"]

    if direction == "BUY":

        sl = (
            entry - atr
        )

        tp = (
            entry
            + atr * rr
        )

    else:

        sl = (
            entry + atr
        )

        tp = (
            entry
            - atr * rr
        )

    return {

        "market":
            market,

        "direction":
            direction,

        "entry":
            entry,

        "sl":
            sl,

        "tp":
            tp,

        "rr":
            rr,

        "score":
            score,

        "signal_time":
            candle_time,
    }


# ============================================================
# PRE-SIGNAL
# ============================================================

def build_presignal(
    market,
    df,
    config
):

    if len(df) < 100:

        return None

    # Current candle.

    row = df.iloc[-1]

    score = calculate_score(
        row,
        config
    )

    if score is None:

        return None

    # We deliberately require
    # a meaningful amount of
    # directional score before
    # sending a preparation alert.

    if score >= 0.75:

        direction = "BUY"

    elif score <= -0.75:

        direction = "SELL"

    else:

        return None

    current_price = float(
        row["close"]
    )

    return {

        "market":
            market,

        "direction":
            direction,

        "price":
            current_price,

        "score":
            score,

        "candle_time":
            row["time"],
    }


# ============================================================
# TELEGRAM MESSAGES
# ============================================================

def presignal_message(
    signal
):

    market = signal[
        "market"
    ]

    direction = signal[
        "direction"
    ]

    emoji = (
        "🟡"
        if direction == "BUY"
        else "🟠"
    )

    price = format_price(
        market,
        signal["price"]
    )

    candle_time = (

        signal["candle_time"]

        .strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    )

    return (

        f"{emoji} V11.8 SETUP DEVELOPING\n\n"

        f"{market} {direction}\n\n"

        f"Current price: {price}\n"

        f"Score: "
        f"{signal['score']:.2f}\n\n"

        "Conditions are developing "
        "toward a potential V11.8 signal.\n\n"

        "DO NOT ENTER YET.\n"

        "Prepare MT5 and wait for "
        "the confirmed signal.\n\n"

        f"Current candle:\n"
        f"{candle_time}"
    )


def confirmed_message(
    signal
):

    market = signal[
        "market"
    ]

    direction = signal[
        "direction"
    ]

    emoji = (
        "🟢"
        if direction == "BUY"
        else "🔴"
    )

    entry = format_price(
        market,
        signal["entry"]
    )

    sl = format_price(
        market,
        signal["sl"]
    )

    tp = format_price(
        market,
        signal["tp"]
    )

    signal_time = (

        signal["signal_time"]

        .strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    )

    return (

        f"{emoji} V11.8 SIGNAL CONFIRMED\n\n"

        f"{market} {direction}\n\n"

        f"Entry: {entry}\n"
        f"SL: {sl}\n"
        f"TP: {tp}\n\n"

        f"RR: "
        f"{signal['rr']:.2f}\n"

        f"Score: "
        f"{signal['score']:.2f}\n\n"

        f"Signal candle:\n"
        f"{signal_time}\n\n"

        "MANUAL MT5 EXECUTION"
    )


# ============================================================
# MARKET PROCESSING
# ============================================================

def process_market(
    market,
    config,
    state
):

    if not config.get(
        "enabled",
        False
    ):

        return

    print()
    print(
        "=" * 60
    )

    print(
        f"MONITORING {market}"
    )

    print(
        "=" * 60
    )

    df = load_data(
        market,
        config
    )

    df = prepare_indicators(
        df
    )

    latest_time = str(
        df["time"].iloc[-1]
    )

    previous_latest = state[
        market
    ].get(
        "last_candle",
        ""
    )

    # --------------------------------------------------------
    # New candle detected
    # --------------------------------------------------------

    new_candle = (
        latest_time
        != previous_latest
    )

    if new_candle:

        print(
            f"{market}: "
            f"NEW CANDLE {latest_time}"
        )

        state[
            market
        ]["last_candle"] = (
            latest_time
        )

        # ----------------------------------------------------
        # Confirmed signal
        # ----------------------------------------------------

        signal = build_confirmed_signal(
            market,
            df,
            config
        )

        if signal is not None:

            signal_id = (

                f"{market}|"
                f"{signal['signal_time']}|"
                f"{signal['direction']}"
            )

            last_signal = state[
                market
            ].get(
                "last_signal",
                ""
            )

            if signal_id != last_signal:

                message = confirmed_message(
                    signal
                )

                print()
                print(message)

                send_telegram(
                    message
                )

                state[
                    market
                ]["last_signal"] = (
                    signal_id
                )

                print(
                    "CONFIRMED SIGNAL SENT."
                )

        else:

            print(
                f"{market}: "
                "No confirmed signal."
            )

    # --------------------------------------------------------
    # Developing setup
    # --------------------------------------------------------

    presignal = build_presignal(
        market,
        df,
        config
    )

    if presignal is not None:

        presignal_id = (

            f"{market}|"
            f"{presignal['candle_time']}|"
            f"{presignal['direction']}"
        )

        last_presignal = state[
            market
        ].get(
            "last_presignal",
            ""
        )

        if (
            presignal_id
            != last_presignal
        ):

            message = presignal_message(
                presignal
            )

            print()
            print(message)

            send_telegram(
                message
            )

            state[
                market
            ]["last_presignal"] = (
                presignal_id
            )

            print(
                "PRE-SIGNAL SENT."
            )


# ============================================================
# INITIAL STATE
# ============================================================

def initialise_state():

    state = load_state()

    for market in MARKETS:

        if market not in state:

            state[market] = {

                "last_candle": "",

                "last_signal": "",

                "last_presignal": "",
            }

    return state


# ============================================================
# MAIN CONTINUOUS LOOP
# ============================================================

def main():

    print()
    print(
        "=" * 60
    )

    print(
        "V11.8 CONTINUOUS "
        "MULTI-MARKET SIGNAL BOT"
    )

    print(
        "=" * 60
    )

    print(
        "SIGNALS ONLY"
    )

    print(
        "MANUAL MT5 EXECUTION"
    )

    print(
        "NO AUTOMATIC TRADING"
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
        "CHECK INTERVAL:",
        CHECK_INTERVAL_SECONDS,
        "seconds"
    )

    print(
        "=" * 60
    )

    state = initialise_state()

    last_refresh = {}

    # --------------------------------------------------------
    # Continuous operation
    # --------------------------------------------------------

    while True:

        loop_start = time.time()

        print()
        print(
            "HEARTBEAT:",
            utc_now().strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
        )

        for market, config in MARKETS.items():

            if not config.get(
                "enabled",
                False
            ):

                continue

            try:

                # ------------------------------------------------
                # Refresh data only when required.
                # ------------------------------------------------

                now = time.time()

                last_market_refresh = (
                    last_refresh.get(
                        market,
                        0
                    )
                )

                if (
                    now
                    - last_market_refresh
                    >= DATA_REFRESH_SECONDS
                ):

                    process_market(
                        market,
                        config,
                        state
                    )

                    last_refresh[
                        market
                    ] = now

                else:

                    print(
                        f"{market}: "
                        "waiting for data refresh..."
                    )

            except Exception as error:

                print()
                print(
                    f"{market} ERROR:"
                )

                print(
                    type(error).__name__,
                    error
                )

        save_state(
            state
        )

        elapsed = (
            time.time()
            - loop_start
        )

        sleep_for = max(

            1,

            CHECK_INTERVAL_SECONDS
            - elapsed
        )

        print()
        print(
            "NEXT CHECK IN:",
            round(
                sleep_for,
                1
            ),
            "seconds"
        )

        time.sleep(
            sleep_for
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
            "SIGNAL BOT STOPPED."
        )

    except Exception as error:

        print()
        print(
            "FATAL ERROR:"
        )

        print(
            type(error).__name__,
            error
        )

        raise
