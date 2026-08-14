import pandas as pd
import yfinance as yf

from strategy import generate_signal


# ============================================================
# XAUUSD STRATEGY BACKTEST
# V6.1 - PROPER SETUP ACCOUNTING
#
# IMPORTANT:
# - Evaluates every 15-minute candle.
# - Only one position can be open at a time.
# - One market setup cannot be re-entered while price remains
#   around the same AOI.
# - Each trade receives a setup ID.
# - Results are measured in R as well as TP/SL.
#
# This file deliberately does NOT change strategy.py.
# It fixes the measurement of the strategy.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

PERIOD = "60d"
INTERVAL = "15m"

# V6 is a 15-minute execution strategy.
# Therefore every completed 15m candle is evaluated.
CHECK_EVERY = 1

MAX_15M_CANDLES = 1000
MAX_DAILY_CANDLES = 500

MINIMUM_15M_CANDLES = 300
MINIMUM_DAILY_CANDLES = 100

# Prevent immediate re-entry into the same AOI.
# The strategy must move away from the AOI before another
# independent setup at that same zone can be traded.
SETUP_RESET_DISTANCE = 10.0

# Used when identifying an AOI setup.
AOI_PRICE_ROUNDING = 1

# Maximum number of bars we are willing to scan after an entry.
# None means scan until TP/SL or the end of the dataset.
MAX_TRADE_BARS = None


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 60)
print("XAUUSD STRATEGY BACKTEST V6.1")
print("=" * 60)
print()

print("Loading XAUUSD historical data...", flush=True)

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


# ============================================================
# REMOVE TIMEZONE
# ============================================================

data_15m = data_15m.copy()
data_daily = data_daily.copy()

if data_15m.index.tz is not None:
    data_15m.index = (
        data_15m.index.tz_convert(None)
    )

if data_daily.index.tz is not None:
    data_daily.index = (
        data_daily.index.tz_convert(None)
    )


print(
    "15m candles:",
    len(data_15m),
    flush=True
)

print(
    "Daily candles:",
    len(data_daily),
    flush=True
)

print(
    "Checking every 15 minutes...",
    flush=True
)

print(
    "One open trade at a time: YES",
    flush=True
)

print(
    "Same-AOI re-entry lock: YES",
    flush=True
)

print()


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value):
    try:
        if value is None:
            return None

        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def _get_aoi(signal):
    aoi = signal.get("aoi")

    if not isinstance(aoi, dict):
        return None

    return aoi


def _setup_key(signal):
    """
    Identifies the market area responsible for a setup.

    We intentionally use the AOI rather than entry price.
    Entry price changes as subsequent candles form, but the
    underlying setup should still be treated as the same setup.
    """

    aoi = _get_aoi(signal)

    if aoi is None:
        return None

    zone_type = aoi.get(
        "type",
        "UNKNOWN"
    )

    timeframe = aoi.get(
        "timeframe",
        "UNKNOWN"
    )

    low = _safe_float(
        aoi.get("low")
    )

    high = _safe_float(
        aoi.get("high")
    )

    if (
        low is None
        or high is None
    ):
        return None

    return (
        timeframe,
        zone_type,
        round(
            low,
            AOI_PRICE_ROUNDING
        ),
        round(
            high,
            AOI_PRICE_ROUNDING
        )
    )


def _setup_description(signal):
    key = _setup_key(signal)

    if key is None:
        return "UNKNOWN_SETUP"

    timeframe, zone_type, low, high = key

    return (
        f"{timeframe.upper()} "
        f"{zone_type.upper()} "
        f"{low}-{high}"
    )


# ============================================================
# TRADE RESULT
# ============================================================

def check_trade_result(
    trade,
    future_data
):
    direction = trade["signal"]

    entry = trade["entry"]
    stop_loss = trade["stop_loss"]
    take_profit = trade["take_profit"]

    if MAX_TRADE_BARS is not None:
        future_data = future_data.iloc[
            :MAX_TRADE_BARS
        ]

    for timestamp, candle in future_data.iterrows():

        high = float(candle["High"])
        low = float(candle["Low"])

        # ----------------------------------------------------
        # SELL
        # ----------------------------------------------------

        if direction == "SELL":

            stop_hit = (
                high >= stop_loss
            )

            target_hit = (
                low <= take_profit
            )

            # If both are touched by the same candle,
            # the OHLC data cannot tell us which came first.
            if stop_hit and target_hit:

                return {
                    "result": "AMBIGUOUS",
                    "exit_price": None,
                    "exit_time": timestamp,
                    "r_multiple": 0.0,
                }

            if stop_hit:

                return {
                    "result": "SL",
                    "exit_price": stop_loss,
                    "exit_time": timestamp,
                    "r_multiple": -1.0,
                }

            if target_hit:

                return {
                    "result": "TP",
                    "exit_price": take_profit,
                    "exit_time": timestamp,
                    "r_multiple": (
                        abs(
                            take_profit - entry
                        )
                        / abs(
                            entry - stop_loss
                        )
                    ),
                }

        # ----------------------------------------------------
        # BUY
        # ----------------------------------------------------

        if direction == "BUY":

            stop_hit = (
                low <= stop_loss
            )

            target_hit = (
                high >= take_profit
            )

            if stop_hit and target_hit:

                return {
                    "result": "AMBIGUOUS",
                    "exit_price": None,
                    "exit_time": timestamp,
                    "r_multiple": 0.0,
                }

            if stop_hit:

                return {
                    "result": "SL",
                    "exit_price": stop_loss,
                    "exit_time": timestamp,
                    "r_multiple": -1.0,
                }

            if target_hit:

                return {
                    "result": "TP",
                    "exit_price": take_profit,
                    "exit_time": timestamp,
                    "r_multiple": (
                        abs(
                            take_profit - entry
                        )
                        / abs(
                            entry - stop_loss
                        )
                    ),
                }

    return {
        "result": "OPEN",
        "exit_price": None,
        "exit_time": None,
        "r_multiple": 0.0,
    }


# ============================================================
# SAME-AOI RESET LOGIC
# ============================================================

def setup_has_reset(
    current_price,
    aoi
):
    """
    A setup becomes available again only after price has moved
    sufficiently away from the AOI.

    This prevents:
        SELL
        SELL
        SELL
        SELL

    from the same resistance/support interaction being counted
    as separate opportunities.
    """

    if not isinstance(aoi, dict):
        return True

    low = _safe_float(
        aoi.get("low")
    )

    high = _safe_float(
        aoi.get("high")
    )

    if (
        low is None
        or high is None
    ):
        return True

    price = float(current_price)

    if low <= price <= high:
        return False

    if price > high:
        return (
            price - high
            >= SETUP_RESET_DISTANCE
        )

    return (
        low - price
        >= SETUP_RESET_DISTANCE
    )


# ============================================================
# BACKTEST
# ============================================================

signals = []

evaluation_count = 0

total_points = (
    len(data_15m)
    - MINIMUM_15M_CANDLES
)

# The currently active trade prevents every subsequent
# historical candle from opening another position until that
# trade has resolved.
active_trade = None

# Once a trade is taken from an AOI, the AOI remains locked until
# price has genuinely moved away from it.
locked_setup_key = None
locked_setup_aoi = None

# Diagnostics.
reason_counts = {}

duplicate_block_count = 0
open_trade_block_count = 0
aoi_lock_block_count = 0

print(
    "Starting historical simulation...",
    flush=True
)

print()


# ============================================================
# MAIN LOOP
# ============================================================

for i in range(
    MINIMUM_15M_CANDLES,
    len(data_15m),
    CHECK_EVERY
):

    evaluation_count += 1

    timestamp = data_15m.index[i]

    # --------------------------------------------------------
    # If a trade is currently open, we do not allow another
    # trade to open before that trade has resolved.
    #
    # The historical result is known immediately in the
    # backtest, so we can jump the simulation forward to the
    # exit candle after recording the trade.
    # --------------------------------------------------------

    if active_trade is not None:

        open_trade_block_count += 1

        exit_time = active_trade.get(
            "exit_time"
        )

        if exit_time is not None:

            if timestamp <= exit_time:
                continue

            # The trade has already resolved.
            active_trade = None

    # --------------------------------------------------------
    # Historical data available at this exact point.
    # --------------------------------------------------------

    historical_15m = data_15m.iloc[
        max(
            0,
            i - MAX_15M_CANDLES
        ):
        i + 1
    ].copy()

    historical_daily = data_daily[
        data_daily.index <= timestamp
    ].tail(
        MAX_DAILY_CANDLES
    ).copy()

    if len(historical_daily) < MINIMUM_DAILY_CANDLES:
        continue

    current_price = float(
        historical_15m["Close"].iloc[-1]
    )

    # --------------------------------------------------------
    # Unlock a previous AOI only after price has moved away.
    # --------------------------------------------------------

    if (
        locked_setup_key is not None
        and locked_setup_aoi is not None
    ):

        if setup_has_reset(
            current_price,
            locked_setup_aoi
        ):

            locked_setup_key = None
            locked_setup_aoi = None

        else:
            aoi_lock_block_count += 1

    # --------------------------------------------------------
    # Run strategy.
    # --------------------------------------------------------

    try:

        signal = generate_signal(
            historical_15m,
            historical_daily,
            current_price
        )

    except Exception as error:

        print(
            "ERROR at",
            timestamp,
            ":",
            error,
            flush=True
        )

        reason_counts[
            "STRATEGY_ERROR"
        ] = (
            reason_counts.get(
                "STRATEGY_ERROR",
                0
            )
            + 1
        )

        continue

    if not isinstance(signal, dict):

        reason_counts[
            "INVALID_SIGNAL_OBJECT"
        ] = (
            reason_counts.get(
                "INVALID_SIGNAL_OBJECT",
                0
            )
            + 1
        )

        continue

    signal_type = signal.get(
        "signal",
        "NONE"
    )

    reason = signal.get(
        "reason",
        "UNKNOWN"
    )

    if signal_type not in (
        "BUY",
        "SELL"
    ):

        reason_counts[reason] = (
            reason_counts.get(
                reason,
                0
            )
            + 1
        )

    # --------------------------------------------------------
    # Progress indicator.
    # --------------------------------------------------------

    if evaluation_count % 100 == 0:

        progress = (
            evaluation_count
            / max(
                1,
                total_points / CHECK_EVERY
            )
            * 100
        )

        print(
            "Progress:",
            round(
                min(
                    progress,
                    100
                ),
                1
            ),
            "%",
            "|",
            "Checked:",
            evaluation_count,
            "|",
            "Signals:",
            len(signals),
            flush=True
        )

    # --------------------------------------------------------
    # Ignore NONE.
    # --------------------------------------------------------

    if signal_type not in (
        "BUY",
        "SELL"
    ):
        continue

    entry = _safe_float(
        signal.get("entry")
    )

    stop_loss = _safe_float(
        signal.get("stop_loss")
    )

    take_profit = _safe_float(
        signal.get("take_profit")
    )

    if (
        entry is None
        or stop_loss is None
        or take_profit is None
    ):
        reason_counts[
            "INVALID_TRADE_LEVELS"
        ] = (
            reason_counts.get(
                "INVALID_TRADE_LEVELS",
                0
            )
            + 1
        )

        continue

    # --------------------------------------------------------
    # Identify the underlying setup.
    # --------------------------------------------------------

    setup_key = _setup_key(
        signal
    )

    setup_aoi = _get_aoi(
        signal
    )

    # --------------------------------------------------------
    # Same setup still locked?
    # --------------------------------------------------------

    if (
        locked_setup_key is not None
        and setup_key == locked_setup_key
        and not setup_has_reset(
            current_price,
            locked_setup_aoi
        )
    ):

        duplicate_block_count += 1
        continue

    # --------------------------------------------------------
    # Trade result.
    # --------------------------------------------------------

    future_data = data_15m.iloc[
        i + 1:
    ]

    result = check_trade_result(
        {
            "signal": signal_type,
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        },
        future_data
    )

    # --------------------------------------------------------
    # Trade record.
    # --------------------------------------------------------

    setup_id = (
        f"{timestamp.strftime('%Y%m%d-%H%M')}"
        f"-{signal_type}-"
        f"{abs(hash(str(setup_key))) % 100000:05d}"
    )

    trade = {
        "setup_id": setup_id,
        "time": timestamp,
        "signal": signal_type,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk": abs(
            entry - stop_loss
        ),
        "reason": reason,
        "setup": _setup_description(
            signal
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
        "exit_price": result[
            "exit_price"
        ],
        "exit_time": result[
            "exit_time"
        ],
        "r_multiple": result[
            "r_multiple"
        ],
    }

    signals.append(
        trade
    )

    # --------------------------------------------------------
    # Lock the setup.
    # --------------------------------------------------------

    locked_setup_key = setup_key
    locked_setup_aoi = setup_aoi

    # --------------------------------------------------------
    # Mark the trade active until its result.
    # --------------------------------------------------------

    active_trade = trade

    print(
        "SIGNAL FOUND:",
        timestamp,
        "|",
        signal_type,
        "| Entry:",
        round(
            entry,
            2
        ),
        "| Result:",
        result["result"],
        "| R:",
        round(
            result["r_multiple"],
            2
        ),
        "| Setup:",
        _setup_description(
            signal
        ),
        flush=True
    )


# ============================================================
# SUMMARY
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
# TEST PERIOD
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
# R STATISTICS
# ============================================================

total_r = sum(
    trade["r_multiple"]
    for trade in signals
)

winning_r = sum(
    trade["r_multiple"]
    for trade in signals
    if trade["r_multiple"] > 0
)

losing_r = sum(
    trade["r_multiple"]
    for trade in signals
    if trade["r_multiple"] < 0
)

if losing_r < 0:
    profit_factor = (
        winning_r
        / abs(losing_r)
    )
else:
    profit_factor = None

if resolved_trades > 0:
    expectancy_r = (
        total_r
        / resolved_trades
    )
else:
    expectancy_r = 0


# ============================================================
# MAX DRAWDOWN IN R
# ============================================================

equity_r = 0.0
peak_r = 0.0
max_drawdown_r = 0.0

for trade in signals:

    equity_r += trade[
        "r_multiple"
    ]

    peak_r = max(
        peak_r,
        equity_r
    )

    drawdown = (
        peak_r
        - equity_r
    )

    max_drawdown_r = max(
        max_drawdown_r,
        drawdown
    )


# ============================================================
# LONGEST SIGNAL GAP
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


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "TEST PERIOD:"
)

print(
    start_date,
    "->",
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
    "15M EVALUATIONS:",
    evaluation_count
)

print(
    "CHECK INTERVAL: 15 minutes"
)

print(
    "ONE OPEN TRADE AT A TIME: YES"
)

print(
    "SAME-AOI RESET LOCK: YES"
)

print()

print(
    "TOTAL INDEPENDENT SETUPS:",
    total_signals
)

print(
    "BUY SETUPS:",
    buy_signals
)

print(
    "SELL SETUPS:",
    sell_signals
)

print()

print(
    "SETUPS PER WEEK:",
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

print(
    "TOTAL R:",
    round(
        total_r,
        2
    )
)

print(
    "EXPECTANCY:",
    round(
        expectancy_r,
        3
    ),
    "R/trade"
)

if profit_factor is not None:

    print(
        "PROFIT FACTOR:",
        round(
            profit_factor,
            2
        )
    )

else:

    print(
        "PROFIT FACTOR: N/A"
    )

print(
    "MAX DRAWDOWN:",
    round(
        max_drawdown_r,
        2
    ),
    "R"
)

print()

if longest_gap is not None:

    print(
        "LONGEST GAP BETWEEN INDEPENDENT SETUPS:",
        round(
            longest_gap,
            2
        ),
        "days"
    )

else:

    print(
        "LONGEST GAP BETWEEN INDEPENDENT SETUPS: N/A"
    )


# ============================================================
# DATA QUALITY / ACCOUNTING DIAGNOSTICS
# ============================================================

print()
print("=" * 60)
print("BACKTEST ACCOUNTING")
print("=" * 60)

print(
    "Evaluations blocked while trade open:",
    open_trade_block_count
)

print(
    "Repeated same-AOI signals blocked:",
    duplicate_block_count
)

print(
    "Evaluations inside locked AOI:",
    aoi_lock_block_count
)

print()


# ============================================================
# STRATEGY REJECTION DIAGNOSTICS
# ============================================================

if reason_counts:

    print("=" * 60)
    print("TOP SIGNAL REJECTION REASONS")
    print("=" * 60)

    sorted_reasons = sorted(
        reason_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )

    for reason, count in sorted_reasons[:20]:

        print(
            reason,
            ":",
            count
        )

    print()


# ============================================================
# INDIVIDUAL SETUPS
# ============================================================

if signals:

    print("=" * 60)
    print("INDEPENDENT SETUPS")
    print("=" * 60)

    for number, trade in enumerate(
        signals,
        start=1
    ):

        print()

        print(
            number,
            "|",
            trade["setup_id"]
        )

        print(
            "Time:",
            trade["time"]
        )

        print(
            "Setup:",
            trade["setup"]
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
            "R:",
            round(
                trade["r_multiple"],
                2
            )
        )

        print(
            "Reason:",
            trade["reason"]
        )

        print(
            "Weekly Bias:",
            trade["weekly_bias"]
        )

        print(
            "Daily Bias:",
            trade["daily_bias"]
        )

        print(
            "4H Bias:",
            trade["four_hour_bias"]
        )

        print(
            "Overall Bias:",
            trade["overall_bias"]
        )

        if trade["exit_time"] is not None:

            print(
                "Exit:",
                trade["exit_time"]
            )

else:

    print(
        "NO INDEPENDENT SETUPS FOUND."
    )


print()
print("=" * 60)
print("END OF BACKTEST")
print("=" * 60)
