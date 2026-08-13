import os
import time
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf

from strategy import generate_signal


PRICE_URL = "https://api.gold-api.com/price/XAU"

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = "6371468101"

PRICE_INTERVAL = 1
DATA_REFRESH_INTERVAL = 60
HEARTBEAT_INTERVAL = 30


def get_live_price():
    response = requests.get(
        PRICE_URL,
        timeout=10
    )
    response.raise_for_status()

    return float(response.json()["price"])


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
    now = pd.Timestamp.now(tz=index.tz)
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
            float(data_15m.loc[
                timestamp,
                "High"
            ]),
            price
        )

        data_15m.loc[
            timestamp,
            "Low"
        ] = min(
            float(data_15m.loc[
                timestamp,
                "Low"
            ]),
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
            [data_15m, candle]
        )

    return data_15m


def format_signal_message(signal, price):

    message = (
        "XAUUSD SIGNAL\n\n"
        "Signal: " + signal["signal"] + "\n"
        "Price: " + str(round(price, 2)) + "\n"
        "Reason: " + signal["reason"] + "\n"
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
            + str(round(signal["entry"], 2))
        )

    if signal.get("stop_loss") is not None:
        message += (
            "\nStop Loss: "
            + str(round(signal["stop_loss"], 2))
        )

    if signal.get("take_profit") is not None:
        message += (
            "\nTake Profit: "
            + str(round(signal["take_profit"], 2))
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
print("Waiting for BUY or SELL...")
print("Press Ctrl+C to stop.")
print()

last_data_refresh = time.time()
last_heartbeat = time.time()

# Prevent duplicate alerts for the same signal.
last_alert_key = None

while True:

    try:

        # Refresh historical market data.
        if (
            time.time() - last_data_refresh
            >= DATA_REFRESH_INTERVAL
        ):

            print()
            print("Refreshing market data...")

            data_15m, data_daily = get_market_data()

            last_data_refresh = time.time()

            print(
                "Data refreshed:",
                len(data_15m),
                "15m candles |",
                len(data_daily),
                "daily candles"
            )

        # Get live XAU price.
        price = get_live_price()

        # Update the current 15-minute candle.
        data_15m = update_live_candle(
            data_15m,
            price
        )

        # Run the existing strategy.
        signal = generate_signal(
            data_15m,
            data_daily,
            price
        )

        current_signal = signal["signal"]

        # -------------------------------------------------
        # TELEGRAM ALERT
        # -------------------------------------------------

        if current_signal in ("BUY", "SELL"):

            entry = signal.get("entry")
            stop_loss = signal.get("stop_loss")
            take_profit = signal.get("take_profit")

            alert_key = (
                current_signal,
                round(price, 2),
                entry,
                stop_loss,
                take_profit
            )

            if alert_key != last_alert_key:

                message = format_signal_message(
                    signal,
                    price
                )

                send_telegram(message)

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
                    "TELEGRAM: SIGNAL SENT"
                )
                print("=" * 60)

                last_alert_key = alert_key

        # -------------------------------------------------
        # HEARTBEAT
        # -------------------------------------------------

        if (
            time.time() - last_heartbeat
            >= HEARTBEAT_INTERVAL
        ):

            print(
                datetime.now().strftime("%H:%M:%S"),
                "| BOT ALIVE",
                "| PRICE:",
                round(price, 2),
                "| SIGNAL:",
                current_signal,
                "| REASON:",
                signal["reason"],
                flush=True
            )

            last_heartbeat = time.time()

        time.sleep(PRICE_INTERVAL)

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