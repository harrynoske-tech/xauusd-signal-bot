import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


# ============================================================
# V11.8 CONTINUOUS LIVE SIGNAL BOT
# ============================================================

DATA_DIR = "data"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

PRICE_INTERVAL = 1
HEARTBEAT_INTERVAL = 60

MAX_CANDLES = 5000

MARKETS = {
    "XAUUSD": {
        "instrument": "xauusd",
        "file": "data/XAUUSD_15m.csv",
        "rr": 0.35,
        "wick": 0.20,
        "body": 0.15,
        "separation": 0.00040,
        "threshold": -0.25,
        "hours": (3, 4),
    },

    "EURUSD": {
        "instrument": "eurusd",
        "file": "data/EURUSD_15m.csv",
        "rr": 0.35,
        "wick": 0.20,
        "body": 0.15,
        "separation": 0.00050,
        "threshold": 0.00,
        "hours": (3, 4, 5),
    },
}


market_data = {}
market_quotes = {}

last_signal = {}
last_presignal = {}
last_heartbeat = 0


# ============================================================
# TIME
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


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

    print(
        f"[{utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')}] "
        f"{message}",
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
            "ERROR: TELEGRAM_BOT_TOKEN is missing."
        )

        return False

    if not TELEGRAM_CHAT_ID:

        log(
            "ERROR: TELEGRAM_CHAT_ID is missing."
        )

        return False

    try:

        response = requests.post(

            telegram_url("sendMessage"),

            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
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

telegram_offset = None


def process_commands():

    global telegram_offset

    if not TELEGRAM_BOT_TOKEN:

        return

    try:

        params = {
            "timeout": 1,
        }

        if telegram_offset is not None:

            params["offset"] = telegram_offset

        response = requests.get(

            telegram_url("getUpdates"),

            params=params,

            timeout=5,
        )

        response.raise_for_status()

        updates = response.json().get(
            "result",
            [],
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
                message.get(
                    "chat",
                    {},
                ).get(
                    "id",
                    "",
                )
            )

            if chat_id != str(
                TELEGRAM_CHAT_ID
            ):

                continue

            text = str(
                message.get(
                    "text",
                    "",
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

    return format(
        float(value),
        f".{decimals(market)}f",
    )


# ============================================================
# DUKASCOPY LIVE PRICE
# ============================================================

def get_current_prices():

    prices = {}

    for market, config in MARKETS.items():

        try:

            response = requests.get(

                "https://freeserv.dukascopy.com/2.0/",

                params={
                    "path": "api/currentPrices",
                    "instruments":
                        config["instrument"],
                },

                timeout=10,
            )

            response.raise_for_status()

            payload = response.json()

            if isinstance(
                payload,
                dict,
            ):

                if "data" in payload:

                    payload = payload["data"]

                elif "result" in payload:

                    payload = payload["result"]

            if not isinstance(
                payload,
                list,
            ):

                continue

            for item in payload:

                if not isinstance(
                    item,
                    dict,
                ):

                    continue

                bid = item.get(
                    "bid"
                )

                ask = item.get(
                    "ask"
                )

                if bid is None:

                    bid = item.get(
                        "bidPrice"
                    )

                if ask is None:

                    ask = item.get(
                        "askPrice"
                    )

                if bid is None:

                    continue

                if ask is None:

                    ask = bid

                bid = float(bid)
                ask = float(ask)

                if bid <= 0 or ask <= 0:

                    continue

                prices[market] = {

                    "bid": bid,

                    "ask": ask,

                    "mid": (
                        bid + ask
                    ) / 2.0,

                    "time": utc_now(),
                }

                break

        except Exception as error:

            log(
                f"{market} PRICE ERROR: "
                + str(error)
            )

    if len(prices) != len(MARKETS):

        missing = [

            market

            for market in MARKETS

            if market not in prices
        ]

        if missing:

            log(
                "Missing live prices: "
                + ", ".join(missing)
            )

    return prices


# ============================================================
# HISTORICAL DATA
# ============================================================

def load_history(
    market,
    config,
):

    path = config["file"]

    if not os.path.exists(path):

        raise RuntimeError(
            f"{market}: "
            f"{path} does not exist."
        )

    df = pd.read_csv(path)

    df.columns = [

        str(column)
        .strip()
        .lower()
        .replace(" ", "_")

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

    for column in (
        "open",
        "high",
        "low",
        "close",
    ):

        if column not in df.columns:

            raise RuntimeError(
                f"{market}: "
                f"missing {column}."
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
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
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

    bucket = current_15m_bucket()

    df = market_data[market]

    if len(df) == 0:

        market_data[market] = pd.DataFrame({

            "time": [bucket],

            "open": [price],

            "high": [price],

            "low": [price],

            "close": [price],
        })

        return

    latest_time = pd.Timestamp(
        df.iloc[-1]["time"]
    )

    if latest_time.tzinfo is None:

        latest_time = latest_time.tz_localize(
            "UTC"
        )

    else:

        latest_time = latest_time.tz_convert(
            "UTC"
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

        return

    if latest_time > bucket:

        return

    new_row = pd.DataFrame({

        "time": [bucket],

        "open": [price],

        "high": [price],

        "low": [price],

        "close": [price],
    })

    market_data[market] = (

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


# ============================================================
# INDICATORS
# ============================================================

def prepare_indicators(df):

    result = df.copy()

    high = result["high"]

    low = result["low"]

    open_price = result["open"]

    close = result["close"]

    candle_range = (
        high - low
    )

    body = (
        close - open_price
    ).abs()

    result["body_ratio"] = np.where(

        candle_range > 0,

        body / candle_range,

        np.nan,
    )

    result["upper_wick"] = np.where(

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

    result["lower_wick"] = np.where(

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
    ).max(axis=1)

    result["atr14"] = (

        true_range

        .rolling(
            14,
            min_periods=14,
        )

        .mean()
    )

    result["ema20"] = (

        close

        .ewm(
            span=20,
            adjust=False,
        )

        .mean()
    )

    result["ema50"] = (

        close

        .ewm(
            span=50,
            adjust=False,
        )

        .mean()
    )

    result["momentum5"] = (

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

    result["range_position"] = np.where(

        range20 > 0,

        (
            close - low20
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

def developing_setup(
    market,
):

    config = MARKETS[market]

    df = market_data[market]

    if len(df) < 100:

        return None

    prepared = prepare_indicators(
        df
    )

    row = prepared.iloc[-1]

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

        "direction": direction,

        "score": score,

        "candle_time": row["time"],

    }


# ============================================================
# CONFIRMED SIGNAL
# ============================================================

def confirmed_signal(
    market,
    live_price,
):

    config = MARKETS[market]

    df = market_data[market]

    if len(df) < 100:

        return None

    prepared = prepare_indicators(
        df
    )

    if len(prepared) < 2:

        return None

    row = prepared.iloc[-2]

    candle_time = pd.Timestamp(
        row["time"]
    )

    if candle_time.hour not in (
        config["hours"]
    ):

        return None

    score = calculate_score(
        row,
        config,
    )

    if score is None:

        return None

    if score < config["threshold"]:

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

    rr = config["rr"]

    if direction == "BUY":

        sl = entry - atr

        tp = entry + (
            atr * rr
        )

    else:

        sl = entry + atr

        tp = entry - (
            atr * rr
        )

    return {

        "market": market,

        "direction": direction,

        "entry": entry,

        "sl": sl,

        "tp": tp,

        "rr": rr,

        "score": score,

        "signal_time": candle_time,

    }


# ============================================================
# MESSAGES
# ============================================================

def presignal_message(
    market,
    setup,
    price,
):

    emoji = (
        "🟡"
        if setup["direction"] == "BUY"
        else "🟠"
    )

    return (

        f"{emoji} V11.8 SETUP DEVELOPING\n\n"

        f"{market} "
        f"{setup['direction']}\n\n"

        f"Current price: "
        f"{format_price(market, price)}\n"

        f"Live score: "
        f"{setup['score']:.2f}\n\n"

        "Potential V11.8 setup developing.\n\n"

        "DO NOT ENTER YET.\n"

        "Prepare MT5 and wait for confirmation."
    )


def signal_message(
    signal,
):

    market = signal["market"]

    emoji = (
        "🟢"
        if signal["direction"] == "BUY"
        else "🔴"
    )

    return (

        f"{emoji} V11.8 SIGNAL CONFIRMED\n\n"

        f"{market} "
        f"{signal['direction']}\n\n"

        f"Entry: "
        f"{format_price(market, signal['entry'])}\n"

        f"SL: "
        f"{format_price(market, signal['sl'])}\n"

        f"TP: "
        f"{format_price(market, signal['tp'])}\n\n"

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
                f"{market}: NO LIVE PRICE"
            )

            continue

        lines.append(

            f"{market}: "
            f"{format_price(market, quote['mid'])}"
        )

    return "\n".join(
        lines
    )


def build_report():

    lines = [

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

        if quote is None:

            lines.append(
                f"{market}: unavailable"
            )

            lines.append("")

            continue

        lines.append(
            f"{market}: "
            f"{format_price(market, quote['mid'])}"
        )

        setup = developing_setup(
            market
        )

        if setup:

            lines.append(

                "Setup: "
                + setup["direction"]
                + " developing"
            )

            lines.append(

                f"Score: "
                f"{setup['score']:.2f}"
            )

        else:

            lines.append(
                "Setup: none"
            )

        lines.append("")

    return "\n".join(
        lines
    )


# ============================================================
# STARTUP
# ============================================================

def startup():

    log("=" * 60)

    log(
        "V11.8 CONTINUOUS "
        "LIVE SIGNAL BOT"
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
        "DUKASCOPY LIVE PRICE"
    )

    log(
        "1 SECOND MONITORING"
    )

    log("=" * 60)

    for market, config in MARKETS.items():

        log(
            f"{market}: loading "
            "historical context..."
        )

        df = load_history(
            market,
            config,
        )

        market_data[
            market
        ] = df

        log(

            f"{market}: "
            f"{len(df)} candles loaded."
        )

    prices = get_current_prices()

    market_quotes.update(
        prices
    )

    for market in MARKETS:

        quote = prices.get(
            market
        )

        if quote is None:

            continue

        update_live_candle(

            market,

            quote["mid"],
        )

        log(

            f"{market}: LIVE PRICE "
            f"{format_price(market, quote['mid'])}"
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

        "Live Dukascopy monitoring active.\n"

        "Manual MT5 execution."
    )


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    global last_heartbeat

    startup()

    while True:

        loop_start = time.time()

        try:

            prices = get_current_prices()

            if prices:

                market_quotes.update(
                    prices
                )

            for market in MARKETS:

                quote = prices.get(
                    market
                )

                if quote is None:

                    continue

                price = quote["mid"]

                previous_bucket = (
                    current_15m_bucket()
                )

                previous_last = (
                    market_data[market].iloc[-1]["time"]
                )

                update_live_candle(
                    market,
                    price,
                )

                current_last = (
                    market_data[market].iloc[-1]["time"]
                )

                # --------------------------------------------
                # PRE-SIGNAL
                # --------------------------------------------

                setup = developing_setup(
                    market
                )

                if setup is not None:

                    setup_id = (

                        str(
                            setup["candle_time"]
                        )

                        + "|"

                        + setup["direction"]
                    )

                    if (
                        last_presignal.get(
                            market
                        )
                        != setup_id
                    ):

                        if send_telegram(

                            presignal_message(
                                market,
                                setup,
                                price,
                            )
                        ):

                            last_presignal[
                                market
                            ] = setup_id

                            log(
                                f"{market}: "
                                "PRE-SIGNAL SENT"
                            )

                # --------------------------------------------
                # NEW 15M CANDLE
                # --------------------------------------------

                if (
                    current_last
                    != previous_last
                ):

                    signal = confirmed_signal(
                        market,
                        price,
                    )

                    if signal is not None:

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
                            != signal_id
                        ):

                            if send_telegram(

                                signal_message(
                                    signal
                                )
                            ):

                                last_signal[
                                    market
                                ] = signal_id

                                log(
                                    f"{market}: "
                                    "CONFIRMED SIGNAL SENT"
                                )

                    else:

                        log(

                            f"{market}: "
                            "new 15m candle - "
                            "no confirmed signal."
                        )

            # --------------------------------------------
            # TELEGRAM COMMANDS
            # --------------------------------------------

            process_commands()

            # --------------------------------------------
            # HEARTBEAT
            # --------------------------------------------

            now = time.time()

            if (
                now
                - last_heartbeat
                >= HEARTBEAT_INTERVAL
            ):

                prices_text = []

                for market in MARKETS:

                    quote = market_quotes.get(
                        market
                    )

                    if quote:

                        prices_text.append(

                            f"{market}="
                            f"{format_price("
                                market,
                                quote["mid"]
                            )}"
                        )

                log(

                    "HEARTBEAT | BOT RUNNING | "
                    + " | ".join(
                        prices_text
                    )
                )

                last_heartbeat = now

            elapsed = (
                time.time()
                - loop_start
            )

            time.sleep(
                max(
                    0.1,
                    PRICE_INTERVAL
                    - elapsed,
                )
            )

        except Exception as error:

            log(
                "LIVE LOOP ERROR: "
                + type(error).__name__
                + ": "
                + str(error)
            )

            time.sleep(2)


# ============================================================
# RUN
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
