import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


# ============================================================
# V11.8 CONTINUOUS LIVE SIGNAL BOT
# ============================================================
#
# DATA:
#   Dukascopy
#
# MARKETS:
#   XAUUSD
#   EURUSD
#
# MODE:
#   Continuous live monitoring
#
# EXECUTION:
#   Telegram signals only
#   Manual MT5 execution
#
# NO AUTOMATIC TRADING
# ============================================================


DATA_DIR = "data"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

PRICE_INTERVAL_SECONDS = 1
HEARTBEAT_SECONDS = 60
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


# ============================================================
# RUNTIME STATE
# ============================================================

market_data = {}
market_quotes = {}

last_signal = {}
last_presignal = {}

telegram_offset = None
last_heartbeat = 0


# ============================================================
# TIME
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def current_15m_bucket():

    now = utc_now()

    minute = now.minute - (
        now.minute % 15
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
        "[" + timestamp + "] " + str(message),
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
            "TELEGRAM ERROR: " + str(error)
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

            if chat_id != str(
                TELEGRAM_CHAT_ID
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
                    build_status()
                )

            elif text in (
                "/report",
                "/market",
            ):

                send_telegram(
                    build_report()
                )

            elif text == "/start":

                send_telegram(
                    "🟢 V11.8 SIGNAL BOT\n\n"
                    "Continuous monitoring active.\n\n"
                    "/dash\n"
                    "/report\n"
                    "/status"
                )

    except Exception as error:

        log(
            "COMMAND ERROR: " + str(error)
        )


# ============================================================
# PRICE FORMATTING
# ============================================================

def decimals(market):

    if market == "XAUUSD":
        return 2

    return 5


def format_price(
    market,
    value,
):

    decimal_places = decimals(
        market
    )

    return format(
        float(value),
        "." + str(decimal_places) + "f",
    )


# ============================================================
# DUKASCOPY CURRENT PRICE
# ============================================================

def get_current_prices():

    prices = {}

    for market, config in MARKETS.items():

        try:

            response = requests.get(
                "https://freeserv.dukascopy.com/2.0/",
                params={
                    "path": "api/currentPrices",
                    "instruments": config["instrument"],
                },
                timeout=10,
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

                continue

            for item in payload:

                if not isinstance(
                    item,
                    dict
                ):

                    continue

                bid = item.get("bid")
                ask = item.get("ask")

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

                break

        except Exception as error:

            log(
                market
                + " PRICE ERROR: "
                + str(error)
            )

    return prices


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

def load_history(
    market,
    config,
):

    path = config["file"]

    if not os.path.exists(path):

        raise RuntimeError(
            market
            + ": "
            + path
            + " does not exist."
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
            market
            + ": no timestamp column."
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
                market
                + ": missing "
                + column
                + "."
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
# UPDATE CURRENT LIVE CANDLE
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
        .tail(MAX_CANDLES)
        .reset_index(drop=True)
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

    candle_range = high - low

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
        ) / candle_range,
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
        ) / candle_range,
        np.nan,
    )

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

    range20 = high20 - low20

    result["range_position"] = np.where(
        range20 > 0,
        (
            close - low20
        ) / range20,
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
        and row["range_position"] <= 0.35
    ):

        score += 0.50

    if (
        bearish
        and row["range_position"] >= 0.65
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

def get_confirmed_signal(
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

    if (
        candle_time.hour
        not in config["hours"]
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
# TELEGRAM MESSAGES
# ============================================================

def make_presignal_message(
    market,
    setup,
    price,
):

    if setup["direction"] == "BUY":

        emoji = "🟡"

    else:

        emoji = "🟠"

    current_price = format_price(
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
        + setup["direction"]
        + "\n\n"
        + "Current price: "
        + current_price
        + "\n"
        + "Live score: "
        + score_text
        + "\n\n"
        + "Potential V11.8 setup developing.\n\n"
        + "DO NOT ENTER YET.\n"
        + "Prepare MT5 and wait for confirmation."
    )


def make_signal_message(
    signal,
):

    market = signal["market"]

    if signal["direction"] == "BUY":

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
        signal["signal_time"]
        .strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    )

    return (
        emoji
        + " V11.8 SIGNAL CONFIRMED\n\n"
        + market
        + " "
        + signal["direction"]
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
# DASH / REPORT
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
        "Data source: Dukascopy",
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
                + setup["direction"]
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
        "DATA: DUKASCOPY"
    )

    log(
        "LIVE CHECK: EVERY SECOND"
    )

    log("=" * 60)

    for market, config in MARKETS.items():

        log(
            market
            + ": loading historical context..."
        )

        df = load_history(
            market,
            config,
        )

        market_data[market] = df

        log(
            market
            + ": "
            + str(len(df))
            + " candles loaded."
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

            log(
                market
                + ": live price unavailable."
            )

            continue

        price = quote["mid"]

        update_live_candle(
            market,
            price,
        )

        formatted = format_price(
            market,
            price,
        )

        log(
            market
            + ": LIVE PRICE "
            + formatted
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
# MAIN CONTINUOUS LOOP
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

                df_before = (
                    market_data[market]
                )

                if len(df_before) == 0:

                    old_latest = None

                else:

                    old_latest = pd.Timestamp(
                        df_before.iloc[-1]["time"]
                    )

                update_live_candle(
                    market,
                    price,
                )

                df_after = (
                    market_data[market]
                )

                new_latest = pd.Timestamp(
                    df_after.iloc[-1]["time"]
                )

                # ------------------------------------------------
                # LIVE DEVELOPING SETUP
                # ------------------------------------------------

                setup = get_developing_setup(
                    market
                )

                if setup is not None:

                    setup_time = str(
                        setup["candle_time"]
                    )

                    setup_id = (
                        setup_time
                        + "|"
                        + setup["direction"]
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

                        sent = send_telegram(
                            message
                        )

                        if sent:

                            last_presignal[
                                market
                            ] = setup_id

                            log(
                                market
                                + ": PRE-SIGNAL SENT."
                            )

                # ------------------------------------------------
                # NEW 15-MINUTE CANDLE
                # ------------------------------------------------

                new_candle_started = False

                if old_latest is None:

                    new_candle_started = True

                elif new_latest > old_latest:

                    new_candle_started = True

                if new_candle_started:

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

                    else:

                        signal_time = str(
                            signal["signal_time"]
                        )

                        signal_id = (
                            signal_time
                            + "|"
                            + signal["direction"]
                        )

                        if (
                            last_signal.get(
                                market
                            )
                            != signal_id
                        ):

                            message = (
                                make_signal_message(
                                    signal
                                )
                            )

                            sent = send_telegram(
                                message
                            )

                            if sent:

                                last_signal[
                                    market
                                ] = signal_id

                                log(
                                    market
                                    + ": CONFIRMED "
                                    + "SIGNAL SENT."
                                )

            # ----------------------------------------------------
            # TELEGRAM COMMANDS
            # ----------------------------------------------------

            process_commands()

            # ----------------------------------------------------
            # HEARTBEAT
            # ----------------------------------------------------

            now = time.time()

            if (
                now - last_heartbeat
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
                            + "=N/A"
                        )

                    else:

                        price_text = format_price(
                            market,
                            quote["mid"],
                        )

                        text = (
                            market
                            + "="
                            + price_text
                        )

                    status_parts.append(
                        text
                    )

                heartbeat = (
                    "HEARTBEAT | BOT RUNNING | "
                    + " | ".join(
                        status_parts
                    )
                )

                log(
                    heartbeat
                )

                last_heartbeat = now

            elapsed = (
                time.time()
                - loop_start
            )

            sleep_time = max(
                0.1,
                PRICE_INTERVAL_SECONDS
                - elapsed,
            )

            time.sleep(
                sleep_time
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
