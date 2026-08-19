import os
import time
import json
import threading
from datetime import datetime, timezone

import requests


# ============================================================
# V11.8 LIVE MARKET FEED
# ============================================================
#
# XAUUSD + EURUSD
#
# PURPOSE:
# Continuous live-price polling layer.
#
# IMPORTANT:
# This file does NOT:
# - place trades
# - connect to MT5
# - send Telegram signals
#
# It provides the current market price to the signal engine.
# ============================================================


CHECK_INTERVAL_SECONDS = 2

MARKETS = {
    "XAUUSD": {
        "symbol": "XAUUSD",
    },
    "EURUSD": {
        "symbol": "EURUSD",
    },
}


# ============================================================
# SHARED STATE
# ============================================================

market_state = {}

state_lock = threading.Lock()


# ============================================================
# TIME
# ============================================================

def utc_now():
    return datetime.now(
        timezone.utc
    )


# ============================================================
# LOGGING
# ============================================================

def log(message):

    print(
        f"[{utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')}] "
        f"{message}",
        flush=True
    )


# ============================================================
# ENVIRONMENT
# ============================================================

def get_required_env(name):

    value = os.getenv(name)

    if not value:

        raise RuntimeError(
            f"Missing environment variable: {name}"
        )

    return value


# ============================================================
# LIVE PRICE PROVIDER
# ============================================================
#
# The provider URL and credentials are supplied through
# environment variables.
#
# Expected response:
#
# {
#   "symbol": "EURUSD",
#   "bid": 1.16500,
#   "ask": 1.16502
# }
#
# OR:
#
# {
#   "symbol": "XAUUSD",
#   "bid": 3340.10,
#   "ask": 3340.30
# }
#
# ============================================================

def fetch_price(
    market
):

    config = MARKETS[market]

    symbol = config["symbol"]

    base_url = os.getenv(
        "LIVE_PRICE_API_URL"
    )

    api_key = os.getenv(
        "LIVE_PRICE_API_KEY"
    )

    if not base_url:

        raise RuntimeError(
            "LIVE_PRICE_API_URL "
            "is not configured."
        )

    params = {
        "symbol": symbol,
    }

    headers = {}

    if api_key:

        headers[
            "Authorization"
        ] = f"Bearer {api_key}"

    response = requests.get(

        base_url,

        params=params,

        headers=headers,

        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(
        data,
        dict
    ):

        raise RuntimeError(
            f"{market}: "
            "invalid live-price response."
        )

    if "bid" not in data:

        raise RuntimeError(
            f"{market}: "
            "live response missing bid."
        )

    if "ask" not in data:

        raise RuntimeError(
            f"{market}: "
            "live response missing ask."
        )

    bid = float(
        data["bid"]
    )

    ask = float(
        data["ask"]
    )

    if bid <= 0:

        raise RuntimeError(
            f"{market}: "
            "invalid bid."
        )

    if ask <= 0:

        raise RuntimeError(
            f"{market}: "
            "invalid ask."
        )

    if ask < bid:

        raise RuntimeError(
            f"{market}: "
            "ask below bid."
        )

    mid = (
        bid + ask
    ) / 2.0

    return {

        "market":
            market,

        "symbol":
            symbol,

        "bid":
            bid,

        "ask":
            ask,

        "mid":
            mid,

        "spread":
            ask - bid,

        "timestamp":
            utc_now().isoformat(),
    }


# ============================================================
# UPDATE SHARED MARKET STATE
# ============================================================

def update_state(
    market,
    quote
):

    with state_lock:

        market_state[
            market
        ] = quote


# ============================================================
# GET CURRENT PRICE
# ============================================================

def get_market_state(
    market
):

    with state_lock:

        quote = market_state.get(
            market
        )

        if quote is None:

            return None

        return dict(
            quote
        )


# ============================================================
# MARKET LOOP
# ============================================================

def monitor_market(
    market
):

    while True:

        try:

            quote = fetch_price(
                market
            )

            update_state(
                market,
                quote
            )

            log(

                f"{market} | "
                f"Bid={quote['bid']} | "
                f"Ask={quote['ask']} | "
                f"Spread={quote['spread']}"
            )

        except Exception as error:

            log(

                f"{market} LIVE FEED ERROR: "
                f"{type(error).__name__}: "
                f"{error}"
            )

        time.sleep(
            CHECK_INTERVAL_SECONDS
        )


# ============================================================
# HEARTBEAT
# ============================================================

def heartbeat():

    while True:

        time.sleep(
            30
        )

        with state_lock:

            markets = list(
                market_state.keys()
            )

        log(
            "LIVE FEED HEARTBEAT | "
            f"Markets receiving data: "
            f"{', '.join(markets) if markets else 'NONE'}"
        )


# ============================================================
# START
# ============================================================

def main():

    print()
    print(
        "=" * 60
    )

    print(
        "V11.8 LIVE MARKET FEED"
    )

    print(
        "=" * 60
    )

    print(
        "XAUUSD + EURUSD"
    )

    print(
        "CONTINUOUS MONITORING"
    )

    print(
        "MT5: DISABLED"
    )

    print(
        "TELEGRAM: DISABLED"
    )

    print(
        "=" * 60
    )

    # Start heartbeat.

    heartbeat_thread = threading.Thread(
        target=heartbeat,
        daemon=True,
    )

    heartbeat_thread.start()

    # Start one continuous feed
    # thread per market.

    threads = []

    for market in MARKETS:

        thread = threading.Thread(

            target=monitor_market,

            args=(market,),

            daemon=True,
        )

        thread.start()

        threads.append(
            thread
        )

    # Keep process alive.

    while True:

        time.sleep(
            60
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "LIVE MARKET FEED STOPPED."
        )
