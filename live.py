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


# ============================================================
# CONFIGURATION
# ============================================================

PRIMARY_PRICE_URL = "https://api.gold-api.com/price/XAU"

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = "6371468101"

PRICE_INTERVAL = 1
DATA_REFRESH_INTERVAL = 60
HEARTBEAT_INTERVAL = 30

MAX_PRICE_DISCREPANCY = 5.0
AOI_APPROACH_DISTANCE = 5.0


# ============================================================
# PRICE
# ============================================================

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

    try:

        price = get_gold_api_price()

        return {
            "price": price,
            "source": "Gold API",
            "trusted": True
        }

    except Exception as error:

        print(
            "PRIMARY PRICE ERROR:",
            error,
            flush=True
        )

    try:

        fallback_price = get_yahoo_price()

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
                    "trusted": False
                }

        return {
            "price": fallback_price,
            "source": "Yahoo Finance",
            "trusted": False
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
                "trusted": False
            }

        raise RuntimeError(
            "ALL PRICE SOURCES FAILED."
        )


# ============================================================
# TELEGRAM
# ============================================================

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

    return response.json().get(
        "result",
        []
    )


# ============================================================
# MARKET DATA
# ============================================================

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

    if data_daily.empty:

        raise RuntimeError(
            "No daily market data received."
        )

    if data_15m.empty:

        raise RuntimeError(
            "No 15m market data received."
        )

    return data_15m, data_daily


# ============================================================
# LIVE 15M CANDLE
# ============================================================

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
                "Volume": [0]
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


# ============================================================
# AOI HELPERS
# ============================================================

def get_all_aois(
    data_daily,
    current_price
):

    areas = get_weekly_daily_areas(
        data_daily,
        current_price=current_price
    )

    if not isinstance(
        areas,
        dict
    ):

        return []

    all_areas = []

    for timeframe in (
        "weekly",
        "daily"
    ):

        zones = areas.get(
            timeframe,
            []
        )

        if not isinstance(
            zones,
            list
        ):

            continue

        for zone in zones:

            if not isinstance(
                zone,
                dict
            ):

                continue

            zone_copy = dict(
                zone
            )

            zone_copy["timeframe"] = (
                timeframe
            )

            # Calculate distance from current price.
            low = zone_copy.get("low")
            high = zone_copy.get("high")

            if (
                low is not None
                and high is not None
            ):

                low = float(low)
                high = float(high)

                if (
                    low
                    <= current_price
                    <= high
                ):

                    distance = 0.0

                elif current_price < low:

                    distance = (
                        low
                        - current_price
                    )

                else:

                    distance = (
                        current_price
                        - high
                    )

                zone_copy["distance"] = (
                    distance
                )

            all_areas.append(
                zone_copy
            )

    all_areas.sort(
        key=lambda zone: (
            zone.get(
                "distance",
                float("inf")
            ),
            -zone.get(
                "touches",
                0
            )
        )
    )

    return all_areas


def find_approaching_aoi(
    data_daily,
    bias,
    price
):

    areas = get_all_aois(
        data_daily,
        price
    )

    overall_bias = bias.get(
        "overall",
        "NEUTRAL"
    )

    for zone in areas:

        zone_type = zone.get(
            "type"
        )

        low = zone.get(
            "low"
        )

        high = zone.get(
            "high"
        )

        if (
            low is None
            or high is None
        ):

            continue

        low = float(low)
        high = float(high)

        # ----------------------------------------------------
        # BEARISH -> RESISTANCE ABOVE PRICE
        # ----------------------------------------------------

        if (
            overall_bias == "BEARISH"
            and zone_type == "resistance"
            and price < low
        ):

            distance = (
                low
                - price
            )

            if (
                distance
                <= AOI_APPROACH_DISTANCE
            ):

                return {
                    "direction": "SELL",
                    "zone": zone,
                    "distance": distance
                }

        # ----------------------------------------------------
        # BULLISH -> SUPPORT BELOW PRICE
        # ----------------------------------------------------

        if (
            overall_bias == "BULLISH"
            and zone_type == "support"
            and price > high
        ):

            distance = (
                price
                - high
            )

            if (
                distance
                <= AOI_APPROACH_DISTANCE
            ):

                return {
                    "direction": "BUY",
                    "zone": zone,
                    "distance": distance
                }

    return None


# ============================================================
# TELEGRAM MESSAGE FORMATTING
# ============================================================

def format_watch_message(
    watch,
    price,
    bias
):

    zone = watch["zone"]

    return (
        "XAUUSD APPROACHING SETUP\n\n"

        "Direction: "
        + watch["direction"]
        + "\n"

        "Current Price: "
        + str(round(price, 2))
        + "\n"

        "Bias: "
        + str(
            bias.get(
                "overall",
                "UNKNOWN"
            )
        )
        + "\n"

        "AOI Timeframe: "
        + str(
            zone.get(
                "timeframe",
                "UNKNOWN"
            )
        )
        + "\n"

        "AOI Type: "
        + str(
            zone.get(
                "type",
                "UNKNOWN"
            )
        )
        + "\n"

        "AOI: "
        + str(
            zone.get("low")
        )
        + " - "
        + str(
            zone.get("high")
        )
        + "\n"

        "Distance: "
        + str(
            round(
                watch["distance"],
                2
            )
        )
        + "\n\n"

        "EARLY WARNING ONLY\n"
        "Confirmation is still required."
    )


def format_signal_message(
    signal,
    price
):

    bias = signal.get(
        "bias",
        {}
    )

    message = (
        "XAUUSD SIGNAL\n\n"

        "Signal: "
        + str(
            signal.get(
                "signal",
                "UNKNOWN"
            )
        )
        + "\n"

        "Price: "
        + str(
            round(price, 2)
        )
        + "\n"

        "Reason: "
        + str(
            signal.get(
                "reason",
                "UNKNOWN"
            )
        )
        + "\n"

        "Weekly Bias: "
        + str(
            bias.get(
                "weekly",
                "UNKNOWN"
            )
        )
        + "\n"

        "Daily Bias: "
        + str(
            bias.get(
                "daily",
                "UNKNOWN"
            )
        )
        + "\n"

        "4H Bias: "
        + str(
            bias.get(
                "4h",
                "UNKNOWN"
            )
        )
        + "\n"

        "Overall Bias: "
        + str(
            bias.get(
                "overall",
                "UNKNOWN"
            )
        )
    )

    aoi = signal.get(
        "aoi"
    )

    if isinstance(
        aoi,
        dict
    ):

        message += (
            "\n\nAOI: "
            + str(
                aoi.get("low")
            )
            + " - "
            + str(
                aoi.get("high")
            )
        )

    if signal.get(
        "entry"
    ) is not None:

        message += (
            "\n\nEntry: "
            + str(
                round(
                    signal["entry"],
                    2
                )
            )
        )

    if signal.get(
        "stop_loss"
    ) is not None:

        message += (
            "\nStop Loss: "
            + str(
                round(
                    signal["stop_loss"],
                    2
                )
            )
        )

    if signal.get(
        "take_profit"
    ) is not None:

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


# ============================================================
# REPORT
# ============================================================

def send_report(
    price,
    price_source,
    price_trusted,
    signal,
    data_daily,
    last_data_refresh,
    last_heartbeat
):

    bias = signal.get(
        "bias",
        {}
    )

    # IMPORTANT:
    # Pass the CURRENT PRICE into the AOI engine.
    areas = get_all_aois(
        data_daily,
        price
    )

    report = (
        "XAUUSD MARKET REPORT\n\n"

        "PRICE\n"
        "Current: "
        + str(
            round(price, 2)
        )
        + "\n"

        "Source: "
        + price_source
        + "\n"

        "Trusted: "
        + (
            "YES"
            if price_trusted
            else "NO"
        )

        + "\n\n"

        "BIAS\n"

        "Weekly: "
        + str(
            bias.get(
                "weekly",
                "UNKNOWN"
            )
        )
        + "\n"

        "Daily: "
        + str(
            bias.get(
                "daily",
                "UNKNOWN"
            )
        )
        + "\n"

        "4H: "
        + str(
            bias.get(
                "4h",
                "UNKNOWN"
            )
        )
        + "\n"

        "Overall: "
        + str(
            bias.get(
                "overall",
                "UNKNOWN"
            )
        )

        + "\n\n"

        "SIGNAL\n"

        "Status: "
        + str(
            signal.get(
                "signal",
                "NONE"
            )
        )
        + "\n"

        "Reason: "
        + str(
            signal.get(
                "reason",
                "UNKNOWN"
            )
        )
    )

    # ========================================================
    # NEAREST AOIs
    # ========================================================

    if areas:

        report += (
            "\n\nNEAREST RELEVANT AOIs"
        )

        # Only show the closest 6.
        for zone in areas[:6]:

            zone_type = str(
                zone.get(
                    "type",
                    "UNKNOWN"
                )
            ).upper()

            timeframe = str(
                zone.get(
                    "timeframe",
                    "UNKNOWN"
                )
            ).upper()

            low = zone.get(
                "low",
                "?"
            )

            high = zone.get(
                "high",
                "?"
            )

            touches = zone.get(
                "touches",
                "N/A"
            )

            distance = zone.get(
                "distance"
            )

            if distance is None:

                distance_text = "N/A"

            else:

                distance_text = str(
                    round(
                        float(distance),
                        2
                    )
                )

            report += (
                "\n\n"
                + timeframe
                + " "
                + zone_type
                + "\n"

                "Zone: "
                + str(low)
                + " - "
                + str(high)
                + "\n"

                "Distance: "
                + distance_text
                + "\n"

                "Recent Touches: "
                + str(touches)
                + "\n"

                "Structure Bias: "
                + str(
                    zone.get(
                        "structure_bias",
                        "N/A"
                    )
                )
            )

    else:

        report += (
            "\n\nNEAREST RELEVANT AOIs\n"
            "No relevant AOIs within "
            + str(
                300
            )
            + " points of current price."
        )

    # ========================================================
    # TRADE LEVELS
    # ========================================================

    if signal.get(
        "entry"
    ) is not None:

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

    if signal.get(
        "stop_loss"
    ) is not None:

        report += (
            "\nStop Loss: "
            + str(
                round(
                    signal["stop_loss"],
                    2
                )
            )
        )

    if signal.get(
        "take_profit"
    ) is not None:

        report += (
            "\nTake Profit: "
            + str(
                round(
                    signal["take_profit"],
                    2
                )
            )
        )

    # ========================================================
    # APPROACHING SETUP
    # ========================================================

    watch = find_approaching_aoi(
        data_daily,
        bias,
        price
    )

    if watch:

        zone = watch["zone"]

        report += (
            "\n\nAPPROACHING SETUP\n"

            "Direction: "
            + watch["direction"]
            + "\n"

            "Timeframe: "
            + str(
                zone.get(
                    "timeframe",
                    "UNKNOWN"
                )
            )
            + "\n"

            "AOI: "
            + str(
                zone.get("low")
            )
            + " - "
            + str(
                zone.get("high")
            )
            + "\n"

            "Distance: "
            + str(
                round(
                    watch["distance"],
                    2
                )
            )
        )

    # ========================================================
    # BOT STATUS
    # ========================================================

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

    send_telegram(
        report
    )


# ============================================================
# STARTUP
# ============================================================

print()
print("=" * 60)
print("LIVE XAUUSD SIGNAL BOT")
print("=" * 60)
print()

print(
    "Loading historical market data..."
)

data_15m, data_daily = (
    get_market_data()
)

print(
    "Historical data loaded:",
    len(data_15m),
    "15m candles |",
    len(data_daily),
    "daily candles"
)

print()
print(
    "Live price: EVERY 1 SECOND"
)
print(
    "Historical refresh: EVERY 60 SECONDS"
)
print(
    "Telegram alerts: ENABLED"
)
print(
    "Primary price source: Gold API"
)
print(
    "Fallback price source: Yahoo Finance"
)
print(
    "Maximum fallback discrepancy:",
    MAX_PRICE_DISCREPANCY
)
print(
    "AOI approach distance:",
    AOI_APPROACH_DISTANCE
)
print(
    "Telegram command: /report"
)
print(
    "Waiting for BUY or SELL..."
)
print(
    "Press Ctrl+C to stop."
)
print()


# ============================================================
# STATE
# ============================================================

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

telegram_offset = None


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    try:

        # ----------------------------------------------------
        # TELEGRAM COMMANDS
        # ----------------------------------------------------

        updates = get_telegram_updates(
            telegram_offset
        )

        for update in updates:

            telegram_offset = (
                update["update_id"]
                + 1
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

            if chat_id != TELEGRAM_CHAT_ID:
                continue

            text = message.get(
                "text",
                ""
            ).strip().lower()

            if text == "/report":

                if last_price is None:

                    send_telegram(
                        "XAUUSD REPORT\n\n"
                        "Bot is still loading "
                        "market data. Try again "
                        "in a few seconds."
                    )

                else:

                    send_report(
                        last_price,
                        last_price_source,
                        last_price_trusted,
                        last_signal,
                        data_daily,
                        last_data_refresh,
                        last_heartbeat
                    )

        # ----------------------------------------------------
        # REFRESH DATA
        # ----------------------------------------------------

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

            last_data_refresh = (
                time.time()
            )

            print(
                "Data refreshed:",
                len(data_15m),
                "15m candles |",
                len(data_daily),
                "daily candles"
            )

        # ----------------------------------------------------
        # LIVE PRICE
        # ----------------------------------------------------

        price_data = get_live_price(
            last_trusted_price
        )

        price = price_data[
            "price"
        ]

        price_source = price_data[
            "source"
        ]

        price_trusted = price_data[
            "trusted"
        ]

        last_price = price
        last_price_source = (
            price_source
        )
        last_price_trusted = (
            price_trusted
        )

        if price_trusted:

            last_trusted_price = (
                price
            )

        # ----------------------------------------------------
        # UPDATE LIVE CANDLE
        # ----------------------------------------------------

        data_15m = update_live_candle(
            data_15m,
            price
        )

        # ----------------------------------------------------
        # STRATEGY
        # ----------------------------------------------------

        signal = generate_signal(
            data_15m,
            data_daily,
            price
        )

        current_signal = signal.get(
            "signal",
            "NONE"
        )

        last_signal = signal

        # ----------------------------------------------------
        # CONFIRMED BUY / SELL
        # ----------------------------------------------------

        if (
            current_signal
            in (
                "BUY",
                "SELL"
            )
            and price_trusted
        ):

            alert_key = (
                current_signal,
                signal.get(
                    "entry"
                ),
                signal.get(
                    "stop_loss"
                ),
                signal.get(
                    "take_profit"
                )
            )

            if (
                alert_key
                != last_signal_alert
            ):

                send_telegram(
                    format_signal_message(
                        signal,
                        price
                    )
                )

                print()
                print(
                    "=" * 60
                )

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
                    round(
                        price,
                        2
                    )
                )

                print(
                    "SOURCE:",
                    price_source
                )

                print(
                    "TELEGRAM: SIGNAL SENT"
                )

                print(
                    "=" * 60
                )

                last_signal_alert = (
                    alert_key
                )

                last_watch_alert = None

        # ----------------------------------------------------
        # APPROACHING AOI
        # ----------------------------------------------------

        elif price_trusted:

            watch = (
                find_approaching_aoi(
                    data_daily,
                    signal.get(
                        "bias",
                        {}
                    ),
                    price
                )
            )

            if watch:

                zone = watch[
                    "zone"
                ]

                watch_key = (
                    watch[
                        "direction"
                    ],
                    zone.get(
                        "timeframe"
                    ),
                    zone.get(
                        "type"
                    ),
                    round(
                        float(
                            zone["low"]
                        ),
                        2
                    ),
                    round(
                        float(
                            zone["high"]
                        ),
                        2
                    )
                )

                if (
                    watch_key
                    != last_watch_alert
                ):

                    send_telegram(
                        format_watch_message(
                            watch,
                            price,
                            signal.get(
                                "bias",
                                {}
                            )
                        )
                    )

                    print()
                    print(
                        "TELEGRAM: "
                        "APPROACHING "
                        + watch[
                            "direction"
                        ]
                        + " ALERT SENT"
                    )

                    last_watch_alert = (
                        watch_key
                    )

            else:

                last_watch_alert = None

        # ----------------------------------------------------
        # HEARTBEAT
        # ----------------------------------------------------

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
                round(
                    price,
                    2
                ),
                "| SOURCE:",
                price_source,
                "| TRUSTED:",
                price_trusted,
                "| SIGNAL:",
                current_signal,
                "| REASON:",
                signal.get(
                    "reason",
                    "UNKNOWN"
                ),
                flush=True
            )

            last_heartbeat = (
                time.time()
            )

        time.sleep(
            PRICE_INTERVAL
        )

    except KeyboardInterrupt:

        print()
        print(
            "BOT STOPPED"
        )

        break

    except Exception as error:

        print()
        print(
            "ERROR:",
            error,
            flush=True
        )

        time.sleep(1)
