import os
import time
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf

from strategy import (
    generate_signal,
    get_weekly_daily_areas,
)


PRIMARY_PRICE_URL = "https://api.gold-api.com/price/XAU"

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = "6371468101"

PRICE_INTERVAL = 1
DATA_REFRESH_INTERVAL = 60
HEARTBEAT_INTERVAL = 30

# Maximum acceptable difference between the fallback
# price and the last trusted Gold API price.
MAX_PRICE_DISCREPANCY = 5.0

# Price distance from an AOI required for an early warning.
AOI_APPROACH_DISTANCE = 5.0


def get_gold_api_price():

    response = requests.get(
        PRIMARY_PRICE_URL,
        timeout=10
    )

    response.raise_for_status()

    return float(
        response.json()["price"]
    )


def get_yahoo_price():

    gold = yf.Ticker("GC=F")

    data = gold.history(
        period="1d",
        interval="1m"
    )

    if data.empty:
        raise RuntimeError(
            "Yahoo Finance returned no price data."
        )

    return float(
        data["Close"].iloc[-1]
    )


def get_live_price(last_trusted_price):

    # -------------------------------------------------
    # PRIMARY PRICE SOURCE
    # -------------------------------------------------

    try:

        price = get_gold_api_price()

        return {
            "price": price,
            "source": "Gold API",
            "trusted": True,
        }

    except Exception as error:

        print(
            "PRIMARY PRICE ERROR:",
            error,
            flush=True
        )

    # -------------------------------------------------
    # FALLBACK PRICE SOURCE
    # -------------------------------------------------

    try:

        fallback_price = get_yahoo_price()

        # If we have a previously trusted Gold API price,
        # make sure Yahoo is reasonably close to it.
        if last_trusted_price is not None:

            difference = abs(
                fallback_price
                - last_trusted_price
            )

            if difference > MAX_PRICE_DISCREPANCY:

                print(
                    "UNSAFE FALLBACK PRICE:",
                    round(fallback_price, 2),
                    "| LAST TRUSTED:",
                    round(last_trusted_price, 2),
                    "| DIFFERENCE:",
                    round(difference, 2),
                    flush=True
                )

                return {
                    "price": last_trusted_price,
                    "source": "LAST TRUSTED PRICE",
                    "trusted": False,
                }

        return {
            "price": fallback_price,
            "source": "Yahoo Finance",
            "trusted": False,
        }

    except Exception as error:

        print(
            "FALLBACK PRICE ERROR:",
            error,
            flush=True
        )

        if last_trusted_price is not None:

            return {
                "price": last_trusted_price,
                "source": "LAST TRUSTED PRICE",
                "trusted": False,
            }

        raise RuntimeError(
            "ALL PRICE SOURCES FAILED."
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
            "text": message,
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


def update_live_candle(
    data_15m,
    price
):

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


def find_approaching_aoi(
    data_daily,
    bias,
    price
):

    areas = get_weekly_daily_areas(
        data_daily
    )

    if not areas:
        return None

    overall_bias = bias["overall"]

    for zone in areas:

        zone_type = zone.get("type")

        low = zone.get("low")
        high = zone.get("high")

        if low is None or high is None:
            continue

        low = float(low)
        high = float(high)

        # -------------------------------------------------
        # BEARISH -> APPROACHING RESISTANCE
        # -------------------------------------------------

        if (
            overall_bias == "BEARISH"
            and zone_type == "resistance"
            and price < low
        ):

            distance = low - price

            if distance <= AOI_APPROACH_DISTANCE:

                return {
                    "direction": "SELL",
                    "zone": zone,
                    "distance": distance,
                }

        # -------------------------------------------------
        # BULLISH -> APPROACHING SUPPORT
        # -------------------------------------------------

        if (
            overall_bias == "BULLISH"
            and zone_type == "support"
            and price > high
        ):

            distance = price - high

            if distance <= AOI_APPROACH_DISTANCE:

                return {
                    "direction": "BUY",
                    "zone": zone,
                    "distance": distance,
                }

    return None


def format_watch_message(
    watch,
    price,
    bias
):

    direction = watch["direction"]
    zone = watch["zone"]
    distance = watch["distance"]

    return (
        "XAUUSD APPROACHING SETUP\n\n"
        "Direction: "
        + direction
        + "\n"
        "Current Price: "
        + str(round(price, 2))
        + "\n"
        "Bias: "
        + bias["overall"]
        + "\n"
        "AOI: "
        + str(zone["low"])
        + " - "
        + str(zone["high"])
        + "\n"
        "Distance: "
        + str(round(distance, 2))
        + "\n\n"
        "EARLY WARNING ONLY"
        "\n"
        "Confirmation is still required."
    )


def format_signal_message(
    signal,
    price
):

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


def get_telegram_updates(offset=None):

    url = (
        "https://api.telegram.org/bot"
        + TELEGRAM_TOKEN
        + "/getUpdates"
    )

    params = {
        "timeout": 1
    }

    if offset is not None:
        params["offset"] = offset

    response = requests.get(
        url,
        params=params,
        timeout=5
    )

    response.raise_for_status()

    return response.json()["result"]

def send_report(
    price,
    price_source,
    price_trusted,
    signal,
    data_daily,
    last_data_refresh,
    last_heartbeat
):

    bias = signal["bias"]

    areas = get_weekly_daily_areas(
        data_daily
    )

    report = (
        "XAUUSD MARKET REPORT\n\n"
        "PRICE\n"
        "Current: "
        + str(round(price, 2))
        + "\n"
        "Source: "
        + price_source
        + "\n"
        "Trusted: "
        + ("YES" if price_trusted else "NO")
        + "\n\n"
        "BIAS\n"
        "Weekly: "
        + bias["weekly"]
        + "\n"
        "Daily: "
        + bias["daily"]
        + "\n"
        "4H: "
        + bias["4h"]
        + "\n"
        "Overall: "
        + bias["overall"]
        + "\n\n"
        "SIGNAL\n"
        "Status: "
        + signal["signal"]
        + "\n"
        "Reason: "
        + signal["reason"]
    )

    if areas:

        report += "\n\nAOIs\n"

        for zone in areas[:5]:

            report += (
                "\n"
                + zone["type"].upper()
                + ": "
                + str(zone["low"])
                + " - "
                + str(zone["high"])
                + "\nTouches: "
                + str(zone.get("touches", "N/A"))
                + "\nBias: "
                + str(
                    zone.get(
                        "structure_bias",
                        "N/A"
                    )
                )
            )

    if signal.get("entry") is not None:

        report += (
            "\n\nTRADE LEVELS\n"
            "Entry: "
            + str(
                round(
                    signal["entry"],
                    2
                )
            )
        )

    if signal.get("stop_loss") is not None:

        report += (
            "\nStop Loss: "
            + str(
                round(
                    signal["stop_loss"],
                    2
                )
            )
        )

    if signal.get("take_profit") is not None:

        report += (
            "\nTake Profit: "
            + str(
                round(
                    signal["take_profit"],
                    2
                )
            )
        )

    report += (
        "\n\nBOT STATUS\n"
        "Status: ONLINE\n"
        "Last data refresh: "
        + datetime.fromtimestamp(
            last_data_refresh
        ).strftime(
            "%H:%M:%S"
        )
        + "\n"
        "Last heartbeat: "
        + datetime.fromtimestamp(
            last_heartbeat
        ).strftime(
            "%H:%M:%S"
        )
    )

    send_telegram(report)

print()
print("=" * 60)
print("LIVE XAUUSD SIGNAL BOT")
print("=" * 60)
print()

print(
    "Loading historical market data..."
)

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
print("Primary price source: Gold API")
print("Fallback price source: Yahoo Finance")
print(
    "Maximum fallback discrepancy:",
    MAX_PRICE_DISCREPANCY
)
print(
    "AOI approach distance:",
    AOI_APPROACH_DISTANCE
)
print("Waiting for BUY or SELL...")
print("Press Ctrl+C to stop.")
print()

last_data_refresh = time.time()
last_heartbeat = time.time()

last_trusted_price = None

last_signal_alert = None
last_watch_alert = None

last_price = None
last_price_source = "NONE"
last_price_trusted = False
last_signal = {
    "signal": "NONE",
    "reason": "BOT_STARTING",
    "bias": {
        "weekly": "UNKNOWN",
        "daily": "UNKNOWN",
        "4h": "UNKNOWN",
        "overall": "UNKNOWN"
    },
    "aoi": None
}

while True:

    try:

                updates = get_telegram_updates()

        for update in updates:

            message = update.get(
                "message"
            )

            if not message:
                continue

            chat_id = str(
                message["chat"]["id"]
            )

            if chat_id != TELEGRAM_CHAT_ID:
                continue

            text = message.get(
                "text",
                ""
            ).strip()

            if text == "/report":

                send_report(
                    price,
                    price_source,
                    price_trusted,
                    signal,
                    data_daily,
                    last_data_refresh,
                    last_heartbeat
                )

        # -------------------------------------------------
        # REFRESH HISTORICAL DATA
        # -------------------------------------------------

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

        # -------------------------------------------------
        # LIVE PRICE
        # -------------------------------------------------

        price_data = get_live_price(
            last_trusted_price
        )

        price = price_data["price"]
        price_source = price_data["source"]
        price_trusted = price_data["trusted"]

        if price_trusted:

            last_trusted_price = price

        # -------------------------------------------------
        # UPDATE LIVE CANDLE
        # -------------------------------------------------

        data_15m = update_live_candle(
            data_15m,
            price
        )

        # -------------------------------------------------
        # RUN STRATEGY
        # -------------------------------------------------

        signal = generate_signal(
            data_15m,
            data_daily,
            price
        )

        current_signal = signal["signal"]

        # -------------------------------------------------
        # ONLY ALLOW TRADING SIGNALS FROM A TRUSTED
        # LIVE PRICE
        # -------------------------------------------------

        if (
            current_signal in (
                "BUY",
                "SELL"
            )
            and price_trusted
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

                last_watch_alert = None

        # -------------------------------------------------
        # APPROACHING AOI
        # -------------------------------------------------

        elif price_trusted:

            watch = find_approaching_aoi(
                data_daily,
                signal["bias"],
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
                            signal["bias"]
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
                "| TRUSTED:",
                price_trusted,
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
