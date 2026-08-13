import os
import time
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf

from strategy import generate_signal


PRIMARY_PRICE_URL = "https://api.gold-api.com/price/XAU"

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = "6371468101"

PRICE_INTERVAL = 1
DATA_REFRESH_INTERVAL = 60
HEARTBEAT_INTERVAL = 30

# How close price must be to an AOI before a WATCH alert is sent.
AOI_APPROACH_DISTANCE = 5.0


def get_live_price():
    try:
        response = requests.get(
            PRIMARY_PRICE_URL,
            timeout=10
        )
        response.raise_for_status()

        return float(response.json()["price"]), "Gold API"

    except Exception as primary_error:
        print(
            "PRIMARY PRICE ERROR:",
            primary_error,
            flush=True
        )

    try:
        gold = yf.Ticker("GC=F")

        fallback_data = gold.history(
            period="1d",
            interval="1m"
        )

        if fallback_data.empty:
            raise RuntimeError(
                "Fallback returned no data."
            )

        price = float(
            fallback_data["Close"].iloc[-1]
        )

        return price, "Yahoo Finance"

    except Exception as fallback_error:
        raise RuntimeError(
            "ALL PRICE SOURCES FAILED. "
            + str(fallback_error)
        )


def send_telegram(message):
    url = (
        "https://api.telegram.org/bot"
        + TELEGRAM_TOKEN
        + "/sendMessage"
    )

    response = requests.post(
        url,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message
        },
        timeout=10
    )

    response.raise_for_status()


def get_market_data():
    gold = yf.Ticker("GC=F")

    data_daily = gold.history(
        period="5y",
        interval="1d"
    )

    data_15m = gold.history(
        period="5d",
        interval="15m"
    )

    if data_daily.empty or data_15m.empty:
        raise RuntimeError(
            "No historical market data received."
        )

    return data_15m, data_daily


def get_current_15m_timestamp(index):
    now = pd.Timestamp.now(
        tz=index.tz
    )

    return now.floor("15min")


def update_live_candle(data_15m, price):
    data_15m = data_15m.copy()

    timestamp = get_current_15m_timestamp(
        data_15m.index
    )

    if timestamp in data_15m.index:

        data_15m.loc[
            timestamp,
            "Close"
        ] = price

        data_15m.loc[
            timestamp,
            "High"
        ] = max(
            float(
                data_15m.loc[
                    timestamp,
                    "High"
                ]
            ),
            price
        )

        data_15m.loc[
            timestamp,
            "Low"
        ] = min(
            float(
                data_15m.loc[
                    timestamp,
                    "Low"
                ]
            ),
            price
        )

    else:

        candle = pd.DataFrame(
            {
                "Open": [price],
                "High": [price],
                "Low": [price],
                "Close": [price],
                "Volume": [0],
            },
            index=[timestamp]
        )

        data_15m = pd.concat(
            [
                data_15m,
                candle
            ]
        )

    return data_15m


def find_approaching_aoi(signal, price):
    """
    Looks at the AOIs already identified by the strategy.

    Only produces a WATCH alert when:
    - overall bias is BEARISH and price is approaching resistance
    - overall bias is BULLISH and price is approaching support
    """

    bias = signal["bias"]["overall"]
    aois = signal.get("aoi")

    if not aois:
        return None

    if isinstance(aois, dict):
        aois = [aois]

    for zone in aois:

        zone_type = zone.get("type")

        low = zone.get("low")
        high = zone.get("high")

        if low is None or high is None:
            continue

        low = float(low)
        high = float(high)

        # Bearish setup: approaching resistance from below.
        if (
            bias == "BEARISH"
            and zone_type == "resistance"
            and price < low
            and low - price <= AOI_APPROACH_DISTANCE
        ):

            return {
                "direction": "SELL",
                "zone": zone,
                "distance": low - price
            }

        # Bullish setup: approaching support from above.
        if (
            bias == "BULLISH"
            and zone_type == "support"
            and price > high
            and price - high <= AOI_APPROACH_DISTANCE
        ):

            return {
                "direction": "BUY",
                "zone": zone,
                "distance": price - high
            }

    return None


def format_watch_message(watch, price, signal):

    direction = watch["direction"]
    zone = watch["zone"]
    distance = watch["distance"]

    message = (
        "XAUUSD APPROACHING SETUP\n\n"
        "Direction: "
        + direction
        + "\n"
        "Current Price: "
        + str(round(price, 2))
        + "\n"
        "Bias: "
        + signal["bias"]["overall"]
        + "\n"
        "AOI: "
        + str(zone["low"])
        + " - "
        + str(zone["high"])
        + "\n"
        "Distance to AOI: "
        + str(round(distance, 2))
        + "\n\n"
        "This is an EARLY WARNING only. "
        "No trade signal has been confirmed."
    )

    return message


def format_signal_message(signal, price):

    message = (
        "XAUUSD SIGNAL\n\n"
        "Signal: "
        + signal["signal"]
        + "\n"
        "Price: "
        + str(round(price, 2))
        + "\n"
        "Reason: "
        + signal["reason"]
        + "\n"
        "Overall Bias: "
        + signal["bias"]["overall"]
    )

    if signal.get("aoi"):
        message += (
            "\n\nAOI: "
            + str(signal["aoi"])
        )

    if signal.get("entry") is not None:
        message += (
            "\n\nEntry: "
            + str(
                round(
                    signal["entry"],
                    2
                )
            )
        )

    if signal.get("stop_loss") is not None:
        message += (
            "\nStop Loss: "
            + str(
                round(
                    signal["stop_loss"],
                    2
                )
            )
        )

    if signal.get("take_profit") is not None:
        message += (
            "\nTake Profit: "
            + str(
                round(
                    signal["take_profit"],
                    2
                )
            )
        )

    return message


print()
print("=" * 60)
print("LIVE XAUUSD SIGNAL BOT")
print("=" * 60)
print()

print("Loading historical market data...")

data_15m, data_daily = get_market_data()

print(
    "Historical data loaded:",
    len(data_15m),
    "15m candles |",
    len(data_daily),
    "daily candles"
)

print()
print("Live price: EVERY 1 SECOND")
print("Historical refresh: EVERY 60 SECONDS")
print("Telegram alerts: ENABLED")
print(
    "AOI approach distance:",
    AOI_APPROACH_DISTANCE
)
print("Waiting for BUY or SELL...")
print("Press Ctrl+C to stop.")
print()

last_data_refresh = time.time()
last_heartbeat = time.time()

last_signal_alert = None
last_watch_alert = None

while True:

    try:

        if (
            time.time()
            - last_data_refresh
            >= DATA_REFRESH_INTERVAL
        ):

            print()
            print(
                "Refreshing market data..."
            )

            data_15m, data_daily = (
                get_market_data()
            )

            last_data_refresh = time.time()

            print(
                "Data refreshed:",
                len(data_15m),
                "15m candles |",
                len(data_daily),
                "daily candles"
            )

        price, price_source = get_live_price()

        data_15m = update_live_candle(
            data_15m,
            price
        )

        signal = generate_signal(
            data_15m,
            data_daily,
            price
        )

        current_signal = signal["signal"]

        # -------------------------------------------------
        # REAL BUY / SELL SIGNAL
        # -------------------------------------------------

        if current_signal in (
            "BUY",
            "SELL"
        ):

            entry = signal.get("entry")
            stop_loss = signal.get("stop_loss")
            take_profit = signal.get("take_profit")

            alert_key = (
                current_signal,
                entry,
                stop_loss,
                take_profit
            )

            if alert_key != last_signal_alert:

                send_telegram(
                    format_signal_message(
                        signal,
                        price
                    )
                )

                print()
                print("=" * 60)
                print(
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )
                print(
                    "SIGNAL:",
                    current_signal
                )
                print(
                    "PRICE:",
                    round(price, 2)
                )
                print(
                    "SOURCE:",
                    price_source
                )
                print(
                    "TELEGRAM: SIGNAL SENT"
                )
                print("=" * 60)

                last_signal_alert = alert_key

                # Reset the watch alert so a future setup
                # can generate a new warning.
                last_watch_alert = None

        # -------------------------------------------------
        # APPROACHING BUY / SELL WATCH
        # -------------------------------------------------

        else:

            watch = find_approaching_aoi(
                signal,
                price
            )

            if watch is not None:

                watch_key = (
                    watch["direction"],
                    round(
                        watch["zone"]["low"],
                        2
                    ),
                    round(
                        watch["zone"]["high"],
                        2
                    )
                )

                if watch_key != last_watch_alert:

                    send_telegram(
                        format_watch_message(
                            watch,
                            price,
                            signal
                        )
                    )

                    print()
                    print(
                        "TELEGRAM: APPROACHING "
                        + watch["direction"]
                        + " ALERT SENT"
                    )

                    last_watch_alert = watch_key

            else:

                # Once price moves away from the AOI,
                # allow a future approach alert.
                last_watch_alert = None

        # -------------------------------------------------
        # HEARTBEAT
        # -------------------------------------------------

        if (
            time.time()
            - last_heartbeat
            >= HEARTBEAT_INTERVAL
        ):

            print(
                datetime.now().strftime(
                    "%H:%M:%S"
                ),
                "| BOT ALIVE",
                "| PRICE:",
                round(price, 2),
                "| SOURCE:",
                price_source,
                "| SIGNAL:",
                current_signal,
                "| REASON:",
                signal["reason"],
                flush=True
            )

            last_heartbeat = time.time()

        time.sleep(
            PRICE_INTERVAL
        )

    except KeyboardInterrupt:

        print()
        print("BOT STOPPED")
        break

    except Exception as error:

        print()
        print(
            "ERROR:",
            error,
            flush=True
        )

        time.sleep(1)
