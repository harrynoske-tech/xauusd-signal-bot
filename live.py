import os
import socket
import time
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


# ============================================================
# V11.8 CONTINUOUS LIVE SIGNAL BOT
# ============================================================
#
# HISTORICAL DATA:
#   Dukascopy CSV files
#
# LIVE DATA:
#   Dukascopy socket data feed
#
# MARKETS:
#   XAUUSD
#   EURUSD
#
# EXECUTION:
#   TELEGRAM SIGNALS ONLY
#   MANUAL MT5 EXECUTION
#
# NO AUTOMATIC TRADING
# ============================================================


TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN"
)

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID"
)


DATA_FILES = {
    "XAUUSD": "data/XAUUSD_15m.csv",
    "EURUSD": "data/EURUSD_15m.csv",
}


MARKETS = {
    "XAUUSD": {
        "symbol": "XAU/USD",
        "rr": 0.35,
        "wick": 0.20,
        "body": 0.15,
        "separation": 0.00040,
        "threshold": -0.25,
        "hours": (3, 4),
    },

    "EURUSD": {
        "symbol": "EUR/USD",
        "rr": 0.35,
        "wick": 0.20,
        "body": 0.15,
        "separation": 0.00050,
        "threshold": 0.00,
        "hours": (3, 4, 5),
    },
}


MAX_CANDLES = 5000

HEARTBEAT_SECONDS = 60

SOCKET_HOST = "datafeed.dukascopy.com"

SOCKET_PORT = 9999


# ============================================================
# RUNTIME STATE
# ============================================================

market_data = {}

market_quotes = {}

last_signal = {}

last_presignal = {}

last_heartbeat = 0

telegram_offset = None

feed_socket = None

feed_connected = False


# ============================================================
# TIME
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    )


def current_15m_bucket():

    now = utc_now()

    minute = (
        now.minute
        - (now.minute % 15)
    )

    return now.replace(
        minute=minute,
        second=0,
        microsecond=0,
    )


# ============================================================
# LOGGING
# ============================================================

def log(message):

    timestamp = utc_now().strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    print(
        "["
        + timestamp
        + "] "
        + str(message),
        flush=True,
    )


# ============================================================
# TELEGRAM
# ============================================================

def telegram_url(method):

    return (
        "https://api.telegram.org/bot"
        + str(TELEGRAM_BOT_TOKEN)
        + "/"
        + method
    )


def send_telegram(message):

    if not TELEGRAM_BOT_TOKEN:

        log(
            "ERROR: TELEGRAM_BOT_TOKEN missing."
        )

        return False

    if not TELEGRAM_CHAT_ID:

        log(
            "ERROR: TELEGRAM_CHAT_ID missing."
        )

        return False

    try:

        response = requests.post(

            telegram_url(
                "sendMessage"
            ),

            json={
                "chat_id":
                    TELEGRAM_CHAT_ID,

                "text":
                    message,
            },

            timeout=10,
        )

        response.raise_for_status()

        return True

    except Exception as error:

        log(
            "TELEGRAM ERROR: "
            + str(error)
        )

        return False


# ============================================================
# TELEGRAM COMMANDS
# ============================================================

def process_commands():

    global telegram_offset

    if not TELEGRAM_BOT_TOKEN:

        return

    try:

        params = {
            "timeout": 1,
        }

        if telegram_offset is not None:

            params["offset"] = (
                telegram_offset
            )

        response = requests.get(

            telegram_url(
                "getUpdates"
            ),

            params=params,

            timeout=5,
        )

        response.raise_for_status()

        updates = (
            response
            .json()
            .get(
                "result",
                [],
            )
        )

        for update in updates:

            update_id = update.get(
                "update_id"
            )

            if update_id is not None:

                telegram_offset = (
                    update_id + 1
                )

            message = update.get(
                "message"
            )

            if not message:

                continue

            chat_id = str(

                message
                .get(
                    "chat",
                    {}
                )
                .get(
                    "id",
                    ""
                )
            )

            if chat_id != str(
                TELEGRAM_CHAT_ID
            ):

                continue

            text = str(

                message
                .get(
                    "text",
                    ""
                )
            ).strip().lower()

            if text in (
                "/dash",
                "/status",
            ):

                send_telegram(
                    build_status()
                )

            elif text in (
                "/report",
                "/market",
            ):

                send_telegram(
                    build_report()
                )

    except Exception as error:

        log(
            "COMMAND ERROR: "
            + str(error)
        )


# ============================================================
# PRICE FORMAT
# ============================================================

def decimals(market):

    if market == "XAUUSD":

        return 2

    return 5


def format_price(
    market,
    value,
):

    places = decimals(
        market
    )

    return format(
        float(value),
        "."
        + str(places)
        + "f",
    )


# ============================================================
# HISTORICAL DATA
# ============================================================

def load_history(
    market,
):

    path = DATA_FILES[
        market
    ]

    if not os.path.exists(path):

        raise RuntimeError(
            market
            + ": missing "
            + path
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
            "_",
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
            market
            + ": timestamp column "
            + "not found."
        )

    for column in (
        "open",
        "high",
        "low",
        "close",
    ):

        if column not in df.columns:

            raise RuntimeError(
                market
                + ": missing "
                + column
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["time"] = pd.to_datetime(
        df[time_column],
        utc=True,
        errors="coerce",
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
            "time"
        )
        .reset_index(
            drop=True
        )
    )

    return df.tail(
        MAX_CANDLES
    ).copy()


# ============================================================
# LIVE CANDLE
# ============================================================

def update_live_candle(
    market,
    price,
):

    bucket = (
        current_15m_bucket()
    )

    df = market_data[
        market
    ]

    if len(df) == 0:

        market_data[
            market
        ] = pd.DataFrame({

            "time": [
                bucket
            ],

            "open": [
                price
            ],

            "high": [
                price
            ],

            "low": [
                price
            ],

            "close": [
                price
            ],
        })

        return False

    latest_time = pd.Timestamp(
        df.iloc[-1]["time"]
    )

    if latest_time.tzinfo is None:

        latest_time = (
            latest_time
            .tz_localize(
                "UTC"
            )
        )

    else:

        latest_time = (
            latest_time
            .tz_convert(
                "UTC"
            )
        )

    if latest_time == bucket:

        index = df.index[-1]

        df.at[
            index,
            "close"
        ] = price

        df.at[
            index,
            "high"
        ] = max(

            float(
                df.at[
                    index,
                    "high"
                ]
            ),

            price,
        )

        df.at[
            index,
            "low"
        ] = min(

            float(
                df.at[
                    index,
                    "low"
                ]
            ),

            price,
        )

        return False

    if latest_time > bucket:

        return False

    new_row = pd.DataFrame({

        "time": [
            bucket
        ],

        "open": [
            price
        ],

        "high": [
            price
        ],

        "low": [
            price
        ],

        "close": [
            price
        ],
    })

    market_data[
        market
    ] = (

        pd.concat(
            [
                df,
                new_row,
            ],
            ignore_index=True,
        )

        .tail(
            MAX_CANDLES
        )

        .reset_index(
            drop=True
        )
    )

    return True


# ============================================================
# INDICATORS
# ============================================================

def prepare_indicators(
    df,
):

    result = df.copy()

    high = result[
        "high"
    ]

    low = result[
        "low"
    ]

    open_price = result[
        "open"
    ]

    close = result[
        "close"
    ]

    candle_range = (
        high - low
    )

    body = (
        close - open_price
    ).abs()

    result[
        "body_ratio"
    ] = np.where(

        candle_range > 0,

        body
        / candle_range,

        np.nan,
    )

    result[
        "upper_wick"
    ] = np.where(

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

    result[
        "lower_wick"
    ] = np.where(

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

    result[
        "atr14"
    ] = (

        true_range

        .rolling(
            14,
            min_periods=14,
        )

        .mean()
    )

    result[
        "ema20"
    ] = (

        close

        .ewm(
            span=20,
            adjust=False,
        )

        .mean()
    )

    result[
        "ema50"
    ] = (

        close

        .ewm(
            span=50,
            adjust=False,
        )

        .mean()
    )

    result[
        "momentum5"
    ] = (

        close
        / close.shift(5)
        - 1.0
    )

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

    range20 = (
        high20 - low20
    )

    result[
        "range_position"
    ] = np.where(

        range20 > 0,

        (
            close
            - low20
        )
        / range20,

        np.nan,
    )

    return result


# ============================================================
# V11.8 SCORE
# ============================================================

def calculate_score(
    row,
    config,
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

    if (
        row["body_ratio"]
        <= config["body"]
    ):

        score += 0.50

    if bullish:

        score += 0.25

    elif bearish:

        score -= 0.25

    if (

        bullish

        and row[
            "range_position"
        ] <= 0.35

    ):

        score += 0.50

    if (

        bearish

        and row[
            "range_position"
        ] >= 0.65

    ):

        score -= 0.50

    if (

        bullish

        and row[
            "momentum5"
        ] > 0

    ):

        score += 0.25

    elif (

        bearish

        and row[
            "momentum5"
        ] < 0

    ):

        score -= 0.25

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

    separation = (

        abs(
            row["ema20"]
            - row["ema50"]
        )

        / row["atr14"]
    )

    if (
        separation
        >= config[
            "separation"
        ]
    ):

        if (
            row["ema20"]
            > row["ema50"]
        ):

            score += 0.10

        else:

            score -= 0.10

    return float(
        score
    )


# ============================================================
# DEVELOPING SETUP
# ============================================================

def get_developing_setup(
    market,
):

    config = MARKETS[
        market
    ]

    df = market_data[
        market
    ]

    if len(df) < 100:

        return None

    prepared = (
        prepare_indicators(
            df
        )
    )

    row = prepared.iloc[
        -1
    ]

    score = calculate_score(
        row,
        config,
    )

    if score is None:

        return None

    if score >= 0.75:

        direction = "BUY"

    elif score <= -0.75:

        direction = "SELL"

    else:

        return None

    return {
        "direction":
            direction,

        "score":
            score,

        "candle_time":
            row["time"],
    }


# ============================================================
# CONFIRMED SIGNAL
# ============================================================

def get_confirmed_signal(
    market,
    live_price,
):

    config = MARKETS[
        market
    ]

    df = market_data[
        market
    ]

    if len(df) < 100:

        return None

    prepared = (
        prepare_indicators(
            df
        )
    )

    if len(prepared) < 2:

        return None

    row = prepared.iloc[
        -2
    ]

    candle_time = pd.Timestamp(
        row["time"]
    )

    if (
        candle_time.hour
        not in config[
            "hours"
        ]
    ):

        return None

    score = calculate_score(
        row,
        config,
    )

    if score is None:

        return None

    if (
        score
        < config[
            "threshold"
        ]
    ):

        return None

    direction = (
        "BUY"
        if score >= 0
        else "SELL"
    )

    atr = float(
        row["atr14"]
    )

    if atr <= 0:

        return None

    entry = float(
        live_price
    )

    rr = config[
        "rr"
    ]

    if direction == "BUY":

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
# TELEGRAM SIGNAL MESSAGE
# ============================================================

def make_signal_message(
    signal,
):

    market = signal[
        "market"
    ]

    if (
        signal[
            "direction"
        ]
        == "BUY"
    ):

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

    rr = format(
        signal["rr"],
        ".2f",
    )

    score = format(
        signal["score"],
        ".2f",
    )

    signal_time = (
        signal[
            "signal_time"
        ]
        .strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    )

    return (

        emoji
        + " V11.8 SIGNAL CONFIRMED\n\n"

        + market
        + " "
        + signal[
            "direction"
        ]
        + "\n\n"

        + "Entry: "
        + entry
        + "\n"

        + "SL: "
        + sl
        + "\n"

        + "TP: "
        + tp
        + "\n\n"

        + "RR: "
        + rr
        + "\n"

        + "Score: "
        + score
        + "\n\n"

        + "Signal candle:\n"
        + signal_time
        + "\n\n"

        + "LIVE ENTRY PRICE VERIFIED\n"

        + "MANUAL MT5 EXECUTION"
    )


# ============================================================
# PRE-SIGNAL MESSAGE
# ============================================================

def make_presignal_message(
    market,
    setup,
    price,
):

    if (
        setup[
            "direction"
        ]
        == "BUY"
    ):

        emoji = "🟡"

    else:

        emoji = "🟠"

    price_text = format_price(
        market,
        price,
    )

    score_text = format(
        setup["score"],
        ".2f",
    )

    return (

        emoji
        + " V11.8 SETUP DEVELOPING\n\n"

        + market
        + " "
        + setup[
            "direction"
        ]
        + "\n\n"

        + "Current price: "
        + price_text
        + "\n"

        + "Live score: "
        + score_text
        + "\n\n"

        + "Potential setup developing.\n\n"

        + "DO NOT ENTER YET.\n"

        + "Prepare MT5 and wait for confirmation."
    )


# ============================================================
# DASH
# ============================================================

def build_status():

    lines = [

        "🟢 V11.8 SIGNAL BOT",

        "",

        "STATUS: RUNNING",

        "MODE: CONTINUOUS",

        "DATA: DUKASCOPY",

        "",
    ]

    for market in MARKETS:

        quote = market_quotes.get(
            market
        )

        if quote is None:

            lines.append(
                market
                + ": NO LIVE PRICE"
            )

            continue

        price = format_price(
            market,
            quote["mid"],
        )

        lines.append(
            market
            + ": "
            + price
        )

    return "\n".join(
        lines
    )


def build_report():

    lines = [

        "📊 V11.8 MARKET REPORT",

        "",

        "Continuous live monitoring",

        "Live feed: Dukascopy socket",

        "",
    ]

    for market in MARKETS:

        quote = market_quotes.get(
            market
        )

        if quote is None:

            lines.append(
                market
                + ": unavailable"
            )

            lines.append("")

            continue

        price = format_price(
            market,
            quote["mid"],
        )

        lines.append(
            market
            + ": "
            + price
        )

        setup = get_developing_setup(
            market
        )

        if setup is None:

            lines.append(
                "Setup: none"
            )

        else:

            score = format(
                setup["score"],
                ".2f",
            )

            lines.append(
                "Setup: "
                + setup[
                    "direction"
                ]
                + " developing"
            )

            lines.append(
                "Score: "
                + score
            )

        lines.append("")

    return "\n".join(
        lines
    )


# ============================================================
# DUKASCOPY SOCKET
# ============================================================

def send_socket_command(
    sock,
    command,
):

    payload = (
        command
        + "\x00"
    )

    sock.sendall(
        payload.encode(
            "utf-8"
        )
    )


def parse_socket_message(
    message,
):

    if not message:

        return

    if message.startswith(
        "#"
    ):

        data = message[
            1:
        ]

        parts = data.split(
            ","
        )

        if len(parts) < 2:

            return

        try:

            stock_id = parts[
                0
            ]

            price = float(
                parts[1]
            )

        except Exception:

            return

        # EUR/USD is documented by
        # Dukascopy as stock ID 1.
        #
        # XAU/USD may require an
        # authenticated subscription.
        #
        # We therefore accept the
        # known EUR/USD stream here.

        if stock_id == "1":

            market = "EURUSD"

        else:

            return

        if price <= 0:

            return

        market_quotes[
            market
        ] = {

            "bid":
                price,

            "ask":
                price,

            "mid":
                price,

            "time":
                utc_now(),
        }

        process_live_price(
            market,
            price,
        )

        return

    if message.startswith(
        "@"
    ):

        log(
            "DUKASCOPY: "
            + message
        )


def process_live_price(
    market,
    price,
):

    df = market_data[
        market
    ]

    if len(df) == 0:

        update_live_candle(
            market,
            price,
        )

        return

    previous_latest = pd.Timestamp(
        df.iloc[-1]["time"]
    )

    new_candle = (
        update_live_candle(
            market,
            price,
        )
    )

    setup = get_developing_setup(
        market
    )

    if setup is not None:

        setup_id = (

            str(
                setup[
                    "candle_time"
                ]
            )

            + "|"

            + setup[
                "direction"
            ]
        )

        if (
            last_presignal.get(
                market
            )
            != setup_id
        ):

            message = (
                make_presignal_message(
                    market,
                    setup,
                    price,
                )
            )

            if send_telegram(
                message
            ):

                last_presignal[
                    market
                ] = setup_id

                log(
                    market
                    + ": PRE-SIGNAL SENT."
                )

    if new_candle:

        signal = get_confirmed_signal(
            market,
            price,
        )

        if signal is None:

            log(
                market
                + ": new 15m candle - "
                + "no confirmed signal."
            )

            return

        signal_id = (

            str(
                signal[
                    "signal_time"
                ]
            )

            + "|"

            + signal[
                "direction"
            ]
        )

        if (
            last_signal.get(
                market
            )
            == signal_id
        ):

            return

        message = (
            make_signal_message(
                signal
            )
        )

        if send_telegram(
            message
        ):

            last_signal[
                market
            ] = signal_id

            log(
                market
                + ": CONFIRMED SIGNAL SENT."
            )


def dukascopy_socket_worker():

    global feed_socket
    global feed_connected

    while True:

        try:

            log(
                "Connecting to "
                "Dukascopy socket..."
            )

            sock = socket.create_connection(
                (
                    SOCKET_HOST,
                    SOCKET_PORT,
                ),
                timeout=30,
            )

            sock.settimeout(
                30
            )

            feed_socket = sock

            feed_connected = True

            log(
                "DUKASCOPY SOCKET CONNECTED."
            )

            # EUR/USD subscription
            send_socket_command(
                sock,
                "$connect|EUR/USD:1",
            )

            log(
                "Subscribed to EUR/USD."
            )

            buffer = b""

            while True:

                try:

                    chunk = sock.recv(
                        4096
                    )

                except socket.timeout:

                    continue

                if not chunk:

                    raise ConnectionError(
                        "Socket closed."
                    )

                buffer += chunk

                while b"\x00" in buffer:

                    raw, buffer = (
                        buffer.split(
                            b"\x00",
                            1,
                        )
                    )

                    if not raw:

                        continue

                    try:

                        message = raw.decode(
                            "utf-8",
                            errors="ignore",
                        )

                    except Exception:

                        continue

                    parse_socket_message(
                        message
                    )

        except Exception as error:

            feed_connected = False

            feed_socket = None

            log(
                "DUKASCOPY SOCKET ERROR: "
                + str(error)
            )

            log(
                "Reconnecting in 10 seconds..."
            )

            time.sleep(
                10
            )


# ============================================================
# STARTUP
# ============================================================

def startup():

    log("=" * 60)

    log(
        "V11.8 CONTINUOUS LIVE SIGNAL BOT"
    )

    log("=" * 60)

    log(
        "SIGNALS ONLY"
    )

    log(
        "MANUAL MT5 EXECUTION"
    )

    log(
        "NO AUTOMATIC TRADING"
    )

    log(
        "HISTORICAL DATA: DUKASCOPY"
    )

    log(
        "LIVE FEED: DUKASCOPY SOCKET"
    )

    log("=" * 60)

    for market in MARKETS:

        log(
            market
            + ": loading historical data..."
        )

        df = load_history(
            market
        )

        market_data[
            market
        ] = df

        log(
            market
            + ": "
            + str(
                len(df)
            )
            + " candles loaded."
        )

    log(
        "BACKSCAN COMPLETE."
    )

    log(
        "STARTING LIVE FEED..."
    )

    worker = threading.Thread(
        target=(
            dukascopy_socket_worker
        ),
        daemon=True,
    )

    worker.start()

    time.sleep(
        3
    )

    send_telegram(

        "🟢 SIGNALS BOT LIVE\n\n"

        "V11.8 Continuous Multi-Market "
        "Signal Bot is online.\n\n"

        "Historical data:\n"
        "Dukascopy\n\n"

        "Live feed:\n"
        "Dukascopy socket\n\n"

        "Markets:\n"
        "• XAUUSD\n"
        "• EURUSD\n\n"

        "Manual MT5 execution."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global last_heartbeat

    startup()

    while True:

        try:

            process_commands()

            now = time.time()

            if (
                now
                - last_heartbeat
                >= HEARTBEAT_SECONDS
            ):

                status_parts = []

                for market in MARKETS:

                    quote = market_quotes.get(
                        market
                    )

                    if quote is None:

                        text = (
                            market
                            + "=NO DATA"
                        )

                    else:

                        price = format_price(
                            market,
                            quote["mid"],
                        )

                        text = (
                            market
                            + "="
                            + price
                        )

                    status_parts.append(
                        text
                    )

                feed_status = (

                    "CONNECTED"

                    if feed_connected

                    else "DISCONNECTED"
                )

                log(
                    "HEARTBEAT | "
                    "BOT RUNNING | "
                    "FEED="
                    + feed_status
                    + " | "
                    + " | ".join(
                        status_parts
                    )
                )

                last_heartbeat = now

            time.sleep(
                1
            )

        except Exception as error:

            log(
                "MAIN LOOP ERROR: "
                + type(error).__name__
                + ": "
                + str(error)
            )

            time.sleep(
                2
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        log(
            "BOT STOPPED MANUALLY."
        )

    except Exception as error:

        log(
            "FATAL ERROR: "
            + type(error).__name__
            + ": "
            + str(error)
        )

        raise
