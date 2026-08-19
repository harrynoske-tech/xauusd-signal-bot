import os
import time
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


# ============================================================
# V11.8 CONTINUOUS DUKASCOPY LIVE SIGNAL BOT
# ============================================================
#
# XAUUSD + EURUSD
#
# STARTUP:
#   1. Load historical data for context/backscan
#   2. Build indicators
#   3. Connect to Dukascopy current prices
#
# LIVE:
#   - Poll Dukascopy current prices every second
#   - Maintain the current 15-minute candle in memory
#   - Continuously analyse the developing candle
#   - Send pre-signals when a setup is developing
#   - Confirm signals when the 15m candle closes
#   - Use the CURRENT live price for the actual entry
#
# TELEGRAM:
#   /dash
#   /status
#   /report
#
# TRADING:
#   NO MT5 CONNECTION
#   NO AUTOMATIC TRADING
#   SIGNALS ONLY
#
# The process intentionally does not terminate.
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

DUKASCOPY_BASE = (
    "https://freeserv.dukascopy.com/2.0/"
)

PRICE_INTERVAL_SECONDS = 1

HEARTBEAT_SECONDS = 60

BACKSCAN_CANDLES = 5000

MAX_CANDLES_IN_MEMORY = 5000

PRE_SIGNAL_COOLDOWN_SECONDS = 300


# ============================================================
# MARKETS
# ============================================================

MARKETS = {

    "XAUUSD": {

        "instrument": "xauusd",

        "display": "XAUUSD",

        "file": (
            "data/XAUUSD_15m.csv"
        ),

        "rr": 0.35,

        "wick": 0.20,

        "body": 0.15,

        "separation": 0.00040,

        "threshold": -0.25,

        "hours": (3, 4),

    },

    "EURUSD": {

        "instrument": "eurusd",

        "display": "EURUSD",

        "file": (
            "data/EURUSD_15m.csv"
        ),

        "rr": 0.35,

        "wick": 0.20,

        "body": 0.15,

        "separation": 0.00050,

        "threshold": 0.00,

        "hours": (3, 4, 5),

    },
}


# ============================================================
# RUNTIME STATE
# ============================================================

market_data = {}

market_quotes = {}

market_state = {}

telegram_offset = None

state_lock = threading.Lock()

running = True


# ============================================================
# TIME
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    )


def current_bucket():

    now = utc_now()

    minute = (
        now.minute
        - (now.minute % 15)
    )

    return now.replace(
        minute=minute,
        second=0,
        microsecond=0
    )


# ============================================================
# LOGGING
# ============================================================

def log(message):

    print(
        "["
        + utc_now().strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        + "] "
        + str(message),
        flush=True
    )


# ============================================================
# TELEGRAM
# ============================================================

def telegram_url(method):

    return (
        "https://api.telegram.org/bot"
        + TELEGRAM_BOT_TOKEN
        + "/"
        + method
    )


def send_telegram(message):

    if not TELEGRAM_BOT_TOKEN:

        log(
            "ERROR: "
            "TELEGRAM_BOT_TOKEN missing."
        )

        return False

    if not TELEGRAM_CHAT_ID:

        log(
            "ERROR: "
            "TELEGRAM_CHAT_ID missing."
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

def get_telegram_updates():

    global telegram_offset

    if not TELEGRAM_BOT_TOKEN:

        return []

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

        return response.json().get(
            "result",
            []
        )

    except Exception as error:

        log(
            "TELEGRAM UPDATE ERROR: "
            + str(error)
        )

        return []


# ============================================================
# PRICE FORMAT
# ============================================================

def decimals(market):

    if market == "XAUUSD":

        return 2

    return 5


def fmt_price(
    market,
    value
):

    return (
        f"{float(value):."
        f"{decimals(market)}f}"
    )


# ============================================================
# DUKASCOPY CURRENT PRICE
# ============================================================

def get_current_prices():

    response = requests.get(

        DUKASCOPY_BASE,

        params={
            "path": "api/currentPrices",

            "instruments":
                "xauusd,eurusd",
        },

        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    if isinstance(
        data,
        dict
    ):

        if "data" in data:

            data = data["data"]

        elif "result" in data:

            data = data["result"]

    if not isinstance(
        data,
        list
    ):

        raise RuntimeError(
            "Unexpected Dukascopy "
            "currentPrices response."
        )

    prices = {}

    for item in data:

        if not isinstance(
            item,
            dict
        ):

            continue

        raw_symbol = str(

            item.get(
                "instrument",
                item.get(
                    "symbol",
                    item.get(
                        "name",
                        ""
                    )
                )
            )
        ).upper()

        symbol = (
            raw_symbol
            .replace(
                "/",
                ""
            )
            .replace(
                "_",
                ""
            )
            .replace(
                "-",
                ""
            )
        )

        if symbol == "XAUUSD":

            market = "XAUUSD"

        elif symbol == "EURUSD":

            market = "EURUSD"

        else:

            continue

        bid = (
            item.get("bid")
            if item.get("bid")
            is not None

            else item.get(
                "bidPrice"
            )
        )

        ask = (
            item.get("ask")
            if item.get("ask")
            is not None

            else item.get(
                "askPrice"
            )
        )

        if bid is None:

            bid = item.get(
                "Bid"
            )

        if ask is None:

            ask = item.get(
                "Ask"
            )

        if bid is None:

            continue

        if ask is None:

            ask = bid

        bid = float(bid)

        ask = float(ask)

        if bid <= 0:

            continue

        if ask <= 0:

            continue

        prices[market] = {

            "bid": bid,

            "ask": ask,

            "mid": (
                bid + ask
            ) / 2.0,

            "time": utc_now(),

        }

    missing = [

        market

        for market in MARKETS

        if market not in prices
    ]

    if missing:

        raise RuntimeError(
            "Dukascopy did not return "
            + ", ".join(missing)
        )

    return prices


# ============================================================
# HISTORICAL DATA
# ============================================================

def load_csv_history(
    market,
    config
):

    path = config["file"]

    if not os.path.exists(path):

        raise RuntimeError(
            f"{market}: "
            f"{path} not found."
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
            "no timestamp column."
        )

    required = [
        "open",
        "high",
        "low",
        "close",
    ]

    for column in required:

        if column not in df.columns:

            raise RuntimeError(
                f"{market}: "
                f"missing {column}."
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["time"] = pd.to_datetime(

        df[time_column],

        utc=True,

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


def try_dukascopy_history(
    market,
    config
):

    try:

        response = requests.get(

            DUKASCOPY_BASE,

            params={

                "path":
                    "api/historicalPrices",

                "instrument":
                    config["instrument"],

                "timeFrame":
                    "15m",

                "count":
                    BACKSCAN_CANDLES,

                "offerSide":
                    "B",

            },

            timeout=20,
        )

        response.raise_for_status()

        payload = response.json()

        if isinstance(
            payload,
            dict
        ):

            if "data" in payload:

                payload = payload["data"]

            elif "result" in payload:

                payload = payload["result"]

        if not isinstance(
            payload,
            list
        ):

            return None

        rows = []

        for item in payload:

            if not isinstance(
                item,
                dict
            ):

                continue

            timestamp = (

                item.get("time")

                if item.get("time")
                is not None

                else item.get(
                    "timestamp"
                )
            )

            if timestamp is None:

                continue

            if isinstance(
                timestamp,
                str
            ):

                parsed_time = pd.to_datetime(
                    timestamp,
                    utc=True,
                    errors="coerce"
                )

            else:

                timestamp = float(
                    timestamp
                )

                if timestamp < 10_000_000_000:

                    timestamp *= 1000

                parsed_time = pd.to_datetime(

                    timestamp,

                    unit="ms",

                    utc=True,

                    errors="coerce"
                )

            if pd.isna(
                parsed_time
            ):

                continue

            open_price = item.get(
                "open"
            )

            high = item.get(
                "high"
            )

            low = item.get(
                "low"
            )

            close = item.get(
                "close"
            )

            if None in (
                open_price,
                high,
                low,
                close,
            ):

                continue

            rows.append({

                "time":
                    parsed_time,

                "open":
                    float(open_price),

                "high":
                    float(high),

                "low":
                    float(low),

                "close":
                    float(close),

            })

        if not rows:

            return None

        df = pd.DataFrame(
            rows
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

    except Exception as error:

        log(
            f"{market}: "
            "Dukascopy historical API "
            "refresh failed: "
            + str(error)
        )

        return None


def build_startup_history(
    market,
    config
):

    log(
        f"{market}: loading startup "
        "historical context..."
    )

    local_df = load_csv_history(
        market,
        config
    )

    live_history = (
        try_dukascopy_history(
            market,
            config
        )
    )

    if (
        live_history is not None
        and len(live_history) >= 100
    ):

        log(
            f"{market}: using "
            "fresh Dukascopy historical "
            f"data ({len(live_history)} candles)."
        )

        df = live_history

    else:

        log(
            f"{market}: using local "
            f"historical data ({len(local_df)} candles)."
        )

        df = local_df

    return df.tail(
        MAX_CANDLES_IN_MEMORY
    ).reset_index(
        drop=True
    )


# ============================================================
# LIVE CANDLE MANAGEMENT
# ============================================================

def ensure_current_candle(
    market,
    price
):

    df = market_data[
        market
    ]

    bucket = current_bucket()

    if len(df) == 0:

        candle = pd.DataFrame({

            "time":
                [bucket],

            "open":
                [price],

            "high":
                [price],

            "low":
                [price],

            "close":
                [price],

        })

        market_data[
            market
        ] = candle

        return

    latest_time = pd.Timestamp(
        df.iloc[-1]["time"]
    )

    latest_time = latest_time.tz_convert(
        "UTC"
    )

    if latest_time > bucket:

        return

    if latest_time == bucket:

        idx = df.index[-1]

        df.at[
            idx,
            "close"
        ] = price

        df.at[
            idx,
            "high"
        ] = max(

            float(
                df.at[
                    idx,
                    "high"
                ]
            ),

            price
        )

        df.at[
            idx,
            "low"
        ] = min(

            float(
                df.at[
                    idx,
                    "low"
                ]
            ),

            price
        )

        return

    candle = pd.DataFrame({

        "time":
            [bucket],

        "open":
            [price],

        "high":
            [price],

        "low":
            [price],

        "close":
            [price],

    })

    market_data[
        market
    ] = pd.concat(

        [
            df,
            candle,
        ],

        ignore_index=True
    ).tail(
        MAX_CANDLES_IN_MEMORY
    ).reset_index(
        drop=True
    )


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

        np.nan
    )

    df["upper_wick"] = np.where(

        candle_range > 0,

        (
            high
            - np.maximum(
                open_price,
                close
            )
        )
        / candle_range,

        np.nan
    )

    df["lower_wick"] = np.where(

        candle_range > 0,

        (
            np.minimum(
                open_price,
                close
            )
            - low
        )
        / candle_range,

        np.nan
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

        axis=1

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

        np.nan
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
# DEVELOPING SETUP
# ============================================================

def get_developing_setup(
    market
):

    config = MARKETS[
        market
    ]

    df = market_data[
        market
    ]

    if len(df) < 100:

        return None

    live_df = prepare_indicators(
        df
    )

    row = live_df.iloc[-1]

    score = calculate_score(
        row,
        config
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

        "atr":
            float(
                row["atr14"]
            ),

    }


# ============================================================
# CONFIRMED SIGNAL
# ============================================================

def get_confirmed_signal(
    market,
    live_price
):

    config = MARKETS[
        market
    ]

    df = market_data[
        market
    ]

    if len(df) < 100:

        return None

    prepared = prepare_indicators(
        df
    )

    if len(prepared) < 2:

        return None

    candle = prepared.iloc[-2]

    candle_time = (
        pd.Timestamp(
            candle["time"]
        )
        .to_pydatetime()
    )

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

    atr = float(
        candle["atr14"]
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
# PRE-SIGNAL MESSAGE
# ============================================================

def make_presignal_message(
    market,
    setup,
    price
):

    direction = setup[
        "direction"
    ]

    emoji = (
        "🟡"
        if direction == "BUY"
        else "🟠"
    )

    return (

        f"{emoji} V11.8 SETUP DEVELOPING\n\n"

        f"{market} {direction}\n\n"

        f"Current price: "
        f"{fmt_price(market, price)}\n"

        f"Live score: "
        f"{setup['score']:.2f}\n\n"

        "Conditions are developing "
        "toward a potential V11.8 signal.\n\n"

        "DO NOT ENTER YET.\n"

        "Prepare MT5 and wait for "
        "confirmation."
    )


# ============================================================
# CONFIRMED MESSAGE
# ============================================================

def make_signal_message(
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

    return (

        f"{emoji} V11.8 SIGNAL CONFIRMED\n\n"

        f"{market} {direction}\n\n"

        f"Entry: "
        f"{fmt_price(market, signal['entry'])}\n"

        f"SL: "
        f"{fmt_price(market, signal['sl'])}\n"

        f"TP: "
        f"{fmt_price(market, signal['tp'])}\n\n"

        f"RR: "
        f"{signal['rr']:.2f}\n"

        f"Score: "
        f"{signal['score']:.2f}\n\n"

        f"Signal candle: "
        f"{signal['signal_time'].strftime('%Y-%m-%d %H:%M UTC')}\n\n"

        "LIVE ENTRY PRICE VERIFIED\n"

        "MANUAL MT5 EXECUTION"
    )


# ============================================================
# STATUS / DASH
# ============================================================

def make_status():

    lines = [

        "🟢 V11.8 SIGNALS BOT",

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

        df = market_data.get(
            market
        )

        if quote is None:

            lines.append(
                f"{market}: NO LIVE PRICE"
            )

            continue

        price = quote[
            "mid"
        ]

        if df is not None:

            candle = df.iloc[-1]

            candle_time = (
                pd.Timestamp(
                    candle["time"]
                )
                .strftime(
                    "%H:%M UTC"
                )
            )

        else:

            candle_time = "N/A"

        lines.append(

            f"{market}: "
            f"{fmt_price(market, price)}"
        )

        lines.append(

            f"  15m candle: "
            f"{candle_time}"
        )

    return "\n".join(
        lines
    )


def make_report():

    report = [

        "📊 V11.8 MARKET REPORT",

        "",

        "Continuous live monitoring",

        "Data source: Dukascopy",

        "",

    ]

    for market in MARKETS:

        quote = market_quotes.get(
            market
        )

        df = market_data.get(
            market
        )

        report.append(
            f"{market}"
        )

        if quote is None:

            report.append(
                "Price: unavailable"
            )

            report.append("")

            continue

        report.append(

            "Price: "
            + fmt_price(
                market,
                quote["mid"]
            )
        )

        if df is not None:

            prepared = (
                prepare_indicators(
                    df
                )
            )

            if len(prepared) >= 100:

                row = prepared.iloc[-1]

                score = calculate_score(

                    row,

                    MARKETS[market]
                )

                if score is not None:

                    if score >= 0.75:

                        setup = "BUY developing"

                    elif score <= -0.75:

                        setup = "SELL developing"

                    else:

                        setup = "No active setup"

                    report.append(
                        "Setup: "
                        + setup
                    )

                    report.append(

                        "Score: "
                        + f"{score:.2f}"
                    )

        report.append("")

    return "\n".join(
        report
    )


# ============================================================
# TELEGRAM COMMAND PROCESSOR
# ============================================================

def process_telegram_commands():

    global telegram_offset

    updates = (
        get_telegram_updates()
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

        chat = message.get(
            "chat",
            {}
        )

        chat_id = str(
            chat.get(
                "id",
                ""
            )
        )

        if (
            chat_id
            != str(
                TELEGRAM_CHAT_ID
            )
        ):

            continue

        text = str(
            message.get(
                "text",
                ""
            )
        ).strip().lower()

        if text in (
            "/dash",
            "/status",
        ):

            send_telegram(
                make_status()
            )

        elif text in (
            "/report",
            "/market",
        ):

            send_telegram(
                make_report()
            )

        elif text == "/start":

            send_telegram(

                "🟢 V11.8 SIGNAL BOT\n\n"

                "Continuous monitoring active.\n\n"

                "Commands:\n"
                "/dash\n"
                "/report\n"
                "/status"
            )


# ============================================================
# STARTUP
# ============================================================

def startup():

    log(
        "=" * 60
    )

    log(
        "V11.8 CONTINUOUS "
        "DUKASCOPY SIGNAL BOT"
    )

    log(
        "=" * 60
    )

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
        "LIVE DATA: DUKASCOPY"
    )

    log(
        "CHECK: EVERY 1 SECOND"
    )

    log(
        "=" * 60
    )

    for market, config in (
        MARKETS.items()
    ):

        market_data[
            market
        ] = build_startup_history(
            market,
            config
        )

        market_state[
            market
        ] = {

            "last_candle":
                None,

            "last_signal":
                None,

            "last_presignal":
                None,

            "last_presignal_time":
                0,

        }

    prices = (
        get_current_prices()
    )

    market_quotes.update(
        prices
    )

    for market in MARKETS:

        price = prices[
            market
        ]["mid"]

        ensure_current_candle(
            market,
            price
        )

        log(

            f"{market}: "
            f"live price "
            f"{fmt_price(market, price)}"
        )

    log(
        "BACKSCAN COMPLETE."
    )

    log(
        "LIVE MONITORING ACTIVE."
    )

    send_telegram(

        "🟢 SIGNALS BOT LIVE\n\n"

        "V11.8 Continuous Multi-Market "
        "Signal Bot is online.\n\n"

        "Markets:\n"
        "• XAUUSD\n"
        "• EURUSD\n\n"

        "Live Dukascopy monitoring:\n"
        "Every second\n\n"

        "Manual MT5 execution."
    )


# ============================================================
# LIVE LOOP
# ============================================================

def live_loop():

    last_heartbeat = 0

    while True:

        loop_start = time.time()

        try:

            prices = (
                get_current_prices()
            )

            with state_lock:

                market_quotes.update(
                    prices
                )

            for market in MARKETS:

                price = prices[
                    market
                ]["mid"]

                previous_bucket = (
                    current_bucket()
                )

                ensure_current_candle(
                    market,
                    price
                )

                # --------------------------------------------
                # DEVELOPING SETUP
                # --------------------------------------------

                setup = (
                    get_developing_setup(
                        market
                    )
                )

                if setup is not None:

                    state = market_state[
                        market
                    ]

                    candle_id = str(
                        setup[
                            "candle_time"
                        ]
                    )

                    setup_id = (

                        candle_id
                        + "|"
                        + setup["direction"]
                    )

                    now = time.time()

                    if (

                        state[
                            "last_presignal"
                        ]
                        != setup_id

                        and (

                            now
                            - state[
                                "last_presignal_time"
                            ]

                            >=
                            PRE_SIGNAL_COOLDOWN_SECONDS
                        )

                    ):

                        message = (
                            make_presignal_message(
                                market,
                                setup,
                                price
                            )
                        )

                        if send_telegram(
                            message
                        ):

                            state[
                                "last_presignal"
                            ] = setup_id

                            state[
                                "last_presignal_time"
                            ] = now

                            log(
                                f"{market}: "
                                "PRE-SIGNAL SENT"
                            )

                # --------------------------------------------
                # CHECK FOR NEW COMPLETED CANDLE
                # --------------------------------------------

                df = market_data[
                    market
                ]

                if len(df) < 2:

                    continue

                completed_time = (
                    pd.Timestamp(
                        df.iloc[-2]["time"]
                    )
                )

                state = market_state[
                    market
                ]

                completed_id = str(
                    completed_time
                )

                if (
                    state[
                        "last_candle"
                    ]
                    == completed_id
                ):

                    continue

                state[
                    "last_candle"
                ] = completed_id

                # --------------------------------------------
                # CONFIRMED SIGNAL
                # --------------------------------------------

                signal = (
                    get_confirmed_signal(
                        market,
                        price
                    )
                )

                if signal is None:

                    log(
                        f"{market}: "
                        "15m candle closed - "
                        "no confirmed signal."
                    )

                    continue

                signal_id = (

                    completed_id
                    + "|"
                    + signal["direction"]
                )

                if (
                    state[
                        "last_signal"
                    ]
                    == signal_id
                ):

                    continue

                message = (
                    make_signal_message(
                        signal
                    )
                )

                if send_telegram(
                    message
                ):

                    state[
                        "last_signal"
                    ] = signal_id

                    log(
                        f"{market}: "
                        "CONFIRMED SIGNAL SENT."
                    )

        except Exception as error:

            log(
                "LIVE LOOP ERROR: "
                + type(error).__name__
                + ": "
                + str(error)
            )

        # --------------------------------------------
        # TELEGRAM COMMANDS
        # --------------------------------------------

        try:

            process_telegram_commands()

        except Exception as error:

            log(
                "COMMAND ERROR: "
                + str(error)
            )

        # --------------------------------------------
        # HEARTBEAT
        # --------------------------------------------

        now = time.time()

        if (
            now
            - last_heartbeat
            >= HEARTBEAT_SECONDS
        ):

            log(
                "HEARTBEAT | "
                "BOT RUNNING | "
                "XAUUSD="
                + (
                    fmt_price(
                        "XAUUSD",
                        market_quotes[
                            "XAUUSD"
                        ]["mid"]
                    )
                    if "XAUUSD"
                    in market_quotes
                    else "N/A"
                )
                + " | EURUSD="
                + (
                    fmt_price(
                        "EURUSD",
                        market_quotes[
                            "EURUSD"
                        ]["mid"]
                    )
                    if "EURUSD"
                    in market_quotes
                    else "N/A"
                )
            )

            last_heartbeat = now

        elapsed = (
            time.time()
            - loop_start
        )

        sleep_time = max(

            0.1,

            PRICE_INTERVAL_SECONDS
            - elapsed
        )

        time.sleep(
            sleep_time
        )


# ============================================================
# MAIN
# ============================================================

def main():

    startup()

    live_loop()


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
