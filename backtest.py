import pandas as pd
import yfinance as yf

from strategy import generate_signal


# ============================================================
# CONFIG
# ============================================================

PERIOD = "60d"
INTERVAL = "15m"

STARTING_BALANCE = 1000.0


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 60)
print("XAUUSD STRATEGY BACKTEST")
print("=" * 60)
print()

print(
    "Loading XAUUSD historical data..."
)

gold = yf.Ticker("GC=F")

data_15m = gold.history(
    period=PERIOD,
    interval=INTERVAL
)

data_daily = gold.history(
    period="5y",
    interval="1d"
)

if data_15m.empty:

    raise RuntimeError(
        "No 15m historical data received."
    )

if data_daily.empty:

    raise RuntimeError(
        "No daily historical data received."
    )


print(
    "15m candles:",
    len(data_15m)
)

print(
    "Daily candles:",
    len(data_daily)
)

print()


# ============================================================
# PREPARE DATA
# ============================================================

data_15m = data_15m.copy()
data_daily = data_daily.copy()

# Remove timezone complications.
if data_15m.index.tz is not None:

    data_15m.index = (
        data_15m.index
        .tz_convert(None)
    )

if data_daily.index.tz is not None:

    data_daily.index = (
        data_daily.index
        .tz_convert(None)
    )


# ============================================================
# BACKTEST STATE
# ============================================================

signals = []

current_trade = None

processed_signal_keys = set()


# ============================================================
# HELPER
# ============================================================

def check_trade_result(
    trade,
    future_data
):

    direction = trade["signal"]

    entry = trade["entry"]
    stop_loss = trade["stop_loss"]
    take_profit = trade["take_profit"]

    for timestamp, candle in future_data.iterrows():

        high = float(
            candle["High"]
        )

        low = float(
            candle["Low"]
        )

        # ----------------------------------------------------
        # SELL
        # ----------------------------------------------------

        if direction == "SELL":

            stop_hit = (
                high
                >= stop_loss
            )

            target_hit = (
                low
                <= take_profit
            )

            if stop_hit and target_hit:

                return {
                    "result": "AMBIGUOUS",
                    "exit_price": None,
                    "exit_time": timestamp
                }

            if stop_hit:

                return {
                    "result": "SL",
                    "exit_price": stop_loss,
                    "exit_time": timestamp
                }

            if target_hit:

                return {
                    "result": "TP",
                    "exit_price": take_profit,
                    "exit_time": timestamp
                }

        # ----------------------------------------------------
        # BUY
        # ----------------------------------------------------

        if direction == "BUY":

            stop_hit = (
                low
                <= stop_loss
            )

            target_hit = (
                high
                >= take_profit
            )

            if stop_hit and target_hit:

                return {
                    "result": "AMBIGUOUS",
                    "exit_price": None,
                    "exit_time": timestamp
                }

            if stop_hit:

                return {
                    "result": "SL",
                    "exit_price": stop_loss,
                    "exit_time": timestamp
                }

            if target_hit:

                return {
                    "result": "TP",
                    "exit_price": take_profit,
                    "exit_time": timestamp
                }

    return {
        "result": "OPEN",
        "exit_price": None,
        "exit_time": None
    }


# ============================================================
# RUN STRATEGY THROUGH HISTORY
# ============================================================

print(
    "Running strategy through historical data..."
)

# Start far enough into the dataset to give the strategy
# enough candles to calculate structure.

MINIMUM_CANDLES = 300

for i in range(
    MINIMUM_CANDLES,
    len(data_15m)
):

    timestamp = data_15m.index[i]

    historical_15m = data_15m.iloc[
        :i + 1
    ].copy()

    current_price = float(
        historical_15m["Close"].iloc[-1]
    )

    # Only use daily candles that existed at this point.
    historical_daily = data_daily[
        data_daily.index
        <= timestamp
    ].copy()

    if len(historical_daily) < 100:

        continue

    try:

        signal = generate_signal(
            historical_15m,
            historical_daily,
            current_price
        )

    except Exception as error:

        print(
            "BACKTEST ERROR:",
            timestamp,
            error
        )

        continue

    signal_type = signal.get(
        "signal",
        "NONE"
    )

    if signal_type not in (
        "BUY",
        "SELL"
    ):

        continue

    entry = signal.get(
        "entry"
    )

    stop_loss = signal.get(
        "stop_loss"
    )

    take_profit = signal.get(
        "take_profit"
    )

    if (
        entry is None
        or stop_loss is None
        or take_profit is None
    ):

        continue

    # --------------------------------------------------------
    # Prevent duplicate signals from the same setup.
    # --------------------------------------------------------

    signal_key = (
        timestamp,
        signal_type,
        round(
            float(entry),
            2
        ),
        round(
            float(stop_loss),
            2
        ),
        round(
            float(take_profit),
            2
        )
    )

    if signal_key in processed_signal_keys:

        continue

    processed_signal_keys.add(
        signal_key
    )

    # --------------------------------------------------------
    # Evaluate what happened after the signal.
    # --------------------------------------------------------

    future_data = data_15m.iloc[
        i + 1:
    ]

    result = check_trade_result(
        {
            "signal": signal_type,
            "entry": float(entry),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit)
        },
        future_data
    )

    trade = {
        "time": timestamp,
        "signal": signal_type,
        "entry": float(entry),
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "reason": signal.get(
            "reason",
            "UNKNOWN"
        ),
        "weekly_bias": signal.get(
            "bias",
            {}
        ).get(
            "weekly",
            "UNKNOWN"
        ),
        "daily_bias": signal.get(
            "bias",
            {}
        ).get(
            "daily",
            "UNKNOWN"
        ),
        "four_hour_bias": signal.get(
            "bias",
            {}
        ).get(
            "4h",
            "UNKNOWN"
        ),
        "overall_bias": signal.get(
            "bias",
            {}
        ).get(
            "overall",
            "UNKNOWN"
        ),
        "result": result["result"],
        "exit_price": result["exit_price"],
        "exit_time": result["exit_time"]
    }

    signals.append(
        trade
    )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("BACKTEST COMPLETE")
print("=" * 60)
print()


total_signals = len(
    signals
)

buy_signals = sum(
    trade["signal"] == "BUY"
    for trade in signals
)

sell_signals = sum(
    trade["signal"] == "SELL"
    for trade in signals
)

tp_count = sum(
    trade["result"] == "TP"
    for trade in signals
)

sl_count = sum(
    trade["result"] == "SL"
    for trade in signals
)

ambiguous_count = sum(
    trade["result"] == "AMBIGUOUS"
    for trade in signals
)

open_count = sum(
    trade["result"] == "OPEN"
    for trade in signals
)


# ============================================================
# PERIOD
# ============================================================

start_date = data_15m.index[0]
end_date = data_15m.index[-1]

days_tested = (
    end_date
    - start_date
).total_seconds() / 86400

weeks_tested = (
    days_tested / 7
)


if weeks_tested > 0:

    signals_per_week = (
        total_signals
        / weeks_tested
    )

else:

    signals_per_week = 0


# ============================================================
# WIN RATE
# ============================================================

resolved_trades = (
    tp_count
    + sl_count
)

if resolved_trades > 0:

    win_rate = (
        tp_count
        / resolved_trades
        * 100
    )

else:

    win_rate = 0


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "TEST PERIOD"
)

print(
    start_date,
    "→",
    end_date
)

print()

print(
    "DAYS TESTED:",
    round(
        days_tested,
        1
    )
)

print(
    "WEEKS TESTED:",
    round(
        weeks_tested,
        1
    )
)

print()

print(
    "TOTAL SIGNALS:",
    total_signals
)

print(
    "BUY SIGNALS:",
    buy_signals
)

print(
    "SELL SIGNALS:",
    sell_signals
)

print()

print(
    "SIGNALS PER WEEK:",
    round(
        signals_per_week,
        2
    )
)

print()

print(
    "TP:",
    tp_count
)

print(
    "SL:",
    sl_count
)

print(
    "AMBIGUOUS:",
    ambiguous_count
)

print(
    "OPEN:",
    open_count
)

print()

print(
    "RESOLVED TRADES:",
    resolved_trades
)

print(
    "WIN RATE:",
    round(
        win_rate,
        2
    ),
    "%"
)

print()


# ============================================================
# LONGEST GAP BETWEEN SIGNALS
# ============================================================

if len(signals) >= 2:

    signal_times = [
        trade["time"]
        for trade in signals
    ]

    gaps = []

    for j in range(
        1,
        len(signal_times)
    ):

        gap = (
            signal_times[j]
            - signal_times[j - 1]
        ).total_seconds() / 86400

        gaps.append(
            gap
        )

    longest_gap = max(
        gaps
    )

else:

    longest_gap = None


if longest_gap is not None:

    print(
        "LONGEST GAP BETWEEN SIGNALS:",
        round(
            longest_gap,
            2
        ),
        "days"
    )

else:

    print(
        "LONGEST GAP BETWEEN SIGNALS: N/A"
    )


print()


# ============================================================
# INDIVIDUAL SIGNALS
# ============================================================

if signals:

    print(
        "=" * 60
    )

    print(
        "INDIVIDUAL SIGNALS"
    )

    print(
        "=" * 60
    )

    for number, trade in enumerate(
        signals,
        start=1
    ):

        print()

        print(
            number,
            "|",
            trade["time"]
        )

        print(
            "Signal:",
            trade["signal"]
        )

        print(
            "Entry:",
            round(
                trade["entry"],
                2
            )
        )

        print(
            "SL:",
            round(
                trade["stop_loss"],
                2
            )
        )

        print(
            "TP:",
            round(
                trade["take_profit"],
                2
            )
        )

        print(
            "Result:",
            trade["result"]
        )

        print(
            "Reason:",
            trade["reason"]
        )

        print(
            "Bias:",
            trade["overall_bias"]
        )

        if trade["exit_time"] is not None:

            print(
                "Exit:",
                trade["exit_time"]
            )


else:

    print(
        "NO SIGNALS FOUND."
    )


print()
print(
    "=" * 60
)
print(
    "END OF BACKTEST"
)
print(
    "=" * 60
)
