import os
import numpy as np
import pandas as pd

from strategy import generate_signal


# ============================================================
# XAUUSD STRATEGY BACKTEST V7.1
# OPTIMISED DUKASCOPY ENGINE
# ============================================================

DATA_15M = "data/XAUUSD_15m.csv"
DATA_DAILY = "data/XAUUSD_1d.csv"

MIN_15M_CANDLES = 300
MAX_15M_CANDLES = 1000
MAX_DAILY_CANDLES = 500
MIN_DAILY_CANDLES = 100

RESET_DISTANCE = 10.0

# Evaluate every completed 15-minute candle.
CHECK_EVERY = 1


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("=" * 60)
    print("XAUUSD STRATEGY BACKTEST V7.1")
    print("=" * 60)
    print()

    print("DATA SOURCE: DUKASCOPY")
    print("SYMBOL: XAUUSD")
    print("EXECUTION TIMEFRAME: 15M")
    print()

    if not os.path.exists(DATA_15M):
        raise RuntimeError(
            f"Missing file: {DATA_15M}"
        )

    if not os.path.exists(DATA_DAILY):
        raise RuntimeError(
            f"Missing file: {DATA_DAILY}"
        )

    data_15m = pd.read_csv(
        DATA_15M,
        index_col=0,
        parse_dates=True
    )

    data_daily = pd.read_csv(
        DATA_DAILY,
        index_col=0,
        parse_dates=True
    )

    data_15m = (
        data_15m
        .sort_index()
        .loc[~data_15m.index.duplicated(keep="last")]
    )

    data_daily = (
        data_daily
        .sort_index()
        .loc[~data_daily.index.duplicated(keep="last")]
    )

    # Standardise columns.
    data_15m.columns = [
        str(c).capitalize()
        for c in data_15m.columns
    ]

    data_daily.columns = [
        str(c).capitalize()
        for c in data_daily.columns
    ]

    required = [
        "Open",
        "High",
        "Low",
        "Close"
    ]

    for column in required:

        if column not in data_15m.columns:
            raise RuntimeError(
                f"15M data missing column: {column}"
            )

        if column not in data_daily.columns:
            raise RuntimeError(
                f"Daily data missing column: {column}"
            )

    print(
        "15m candles:",
        len(data_15m)
    )

    print(
        "Daily candles:",
        len(data_daily)
    )

    print(
        "15M range:",
        data_15m.index.min(),
        "->",
        data_15m.index.max()
    )

    print(
        "Daily range:",
        data_daily.index.min(),
        "->",
        data_daily.index.max()
    )

    if len(data_15m) < MIN_15M_CANDLES:
        raise RuntimeError(
            "Not enough 15M candles."
        )

    if len(data_daily) < MIN_DAILY_CANDLES:
        raise RuntimeError(
            "Not enough daily candles."
        )

    return data_15m, data_daily


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):

    try:
        return float(value)

    except (
        TypeError,
        ValueError
    ):
        return None


def get_aoi(signal):

    aoi = signal.get("aoi")

    if isinstance(aoi, dict):
        return aoi

    return None


def setup_key(signal):

    aoi = get_aoi(signal)

    if not aoi:
        return None

    low = safe_float(
        aoi.get("low")
    )

    high = safe_float(
        aoi.get("high")
    )

    if low is None or high is None:
        return None

    return (
        aoi.get(
            "timeframe",
            "UNKNOWN"
        ),
        aoi.get(
            "type",
            "UNKNOWN"
        ),
        round(low, 1),
        round(high, 1)
    )


def setup_text(signal):

    key = setup_key(signal)

    if not key:
        return "UNKNOWN_SETUP"

    tf, typ, low, high = key

    return (
        f"{str(tf).upper()} "
        f"{str(typ).upper()} "
        f"{low}-{high}"
    )


def aoi_reset(price, aoi):

    if not isinstance(aoi, dict):
        return True

    low = safe_float(
        aoi.get("low")
    )

    high = safe_float(
        aoi.get("high")
    )

    if low is None or high is None:
        return True

    if low <= price <= high:
        return False

    if price > high:
        return (
            price - high
            >= RESET_DISTANCE
        )

    return (
        low - price
        >= RESET_DISTANCE
    )


# ============================================================
# FAST TRADE RESULT
# ============================================================

def fast_trade_result(
    direction,
    entry,
    stop_loss,
    take_profit,
    start_index,
    highs,
    lows,
    timestamps
):

    """
    Same TP/SL logic as before, but uses NumPy arrays
    instead of iterating through pandas rows.
    """

    if start_index >= len(highs):

        return (
            "OPEN",
            None,
            None,
            0.0
        )

    future_highs = highs[
        start_index:
    ]

    future_lows = lows[
        start_index:
    ]

    if direction == "SELL":

        stop_hits = (
            future_highs
            >= stop_loss
        )

        target_hits = (
            future_lows
            <= take_profit
        )

    else:

        stop_hits = (
            future_lows
            <= stop_loss
        )

        target_hits = (
            future_highs
            >= take_profit
        )

    stop_positions = np.flatnonzero(
        stop_hits
    )

    target_positions = np.flatnonzero(
        target_hits
    )

    first_stop = (
        int(stop_positions[0])
        if len(stop_positions)
        else None
    )

    first_target = (
        int(target_positions[0])
        if len(target_positions)
        else None
    )

    # Nothing hit.
    if (
        first_stop is None
        and first_target is None
    ):

        return (
            "OPEN",
            None,
            None,
            0.0
        )

    # Both happen on the same candle.
    if (
        first_stop is not None
        and first_target is not None
        and first_stop == first_target
    ):

        position = first_stop

        return (
            "AMBIGUOUS",
            None,
            timestamps[
                start_index + position
            ],
            0.0
        )

    # Stop first.
    if (
        first_target is None
        or (
            first_stop is not None
            and first_stop < first_target
        )
    ):

        position = first_stop

        return (
            "SL",
            stop_loss,
            timestamps[
                start_index + position
            ],
            -1.0
        )

    # Target first.
    position = first_target

    risk = abs(
        entry - stop_loss
    )

    reward = abs(
        take_profit - entry
    )

    r_multiple = (
        reward / risk
        if risk > 0
        else 0.0
    )

    return (
        "TP",
        take_profit,
        timestamps[
            start_index + position
        ],
        r_multiple
    )


# ============================================================
# LOAD
# ============================================================

data_15m, data_daily = load_data()


# ============================================================
# PRE-COMPUTE ARRAYS
# ============================================================

timestamps = data_15m.index.to_numpy()

highs = (
    data_15m["High"]
    .to_numpy(
        dtype=float
    )
)

lows = (
    data_15m["Low"]
    .to_numpy(
        dtype=float
    )
)

closes = (
    data_15m["Close"]
    .to_numpy(
        dtype=float
    )
)

daily_index = (
    data_daily.index
)

# Pre-compute the daily candle position corresponding
# to every 15M timestamp.
daily_positions = (
    daily_index
    .searchsorted(
        data_15m.index,
        side="right"
    )
)


# ============================================================
# START
# ============================================================

print()

print(
    "Checking every 15 minutes..."
)

print(
    "One open trade at a time: YES"
)

print(
    "Same-AOI re-entry lock: YES"
)

print()

print(
    "Starting historical simulation...",
    flush=True
)


# ============================================================
# STATE
# ============================================================

signals = []

reasons = {}

active_trade = None

locked_key = None
locked_aoi = None

blocked_open = 0
blocked_duplicate = 0
blocked_lock = 0

evaluations = 0

total_iterations = (
    len(data_15m)
    - MIN_15M_CANDLES
)


# ============================================================
# MAIN BACKTEST
# ============================================================

for i in range(
    MIN_15M_CANDLES,
    len(data_15m),
    CHECK_EVERY
):

    evaluations += 1

    ts = timestamps[i]

    # --------------------------------------------------------
    # Existing trade
    # --------------------------------------------------------

    if active_trade is not None:

        exit_time = active_trade[
            "exit_time"
        ]

        if (
            exit_time is not None
            and ts <= exit_time
        ):

            blocked_open += 1

            continue

        active_trade = None

    # --------------------------------------------------------
    # Fast daily position lookup
    # --------------------------------------------------------

    daily_end = (
        daily_positions[i]
    )

    if daily_end < MIN_DAILY_CANDLES:

        continue

    daily_start = max(
        0,
        daily_end
        - MAX_DAILY_CANDLES
    )

    histdaily = data_daily.iloc[
        daily_start:daily_end
    ]

    # --------------------------------------------------------
    # 15M historical window
    #
    # IMPORTANT:
    # No .copy()
    # This avoids creating a new 1000-row DataFrame
    # unnecessarily on every iteration.
    # --------------------------------------------------------

    start_15m = max(
        0,
        i - MAX_15M_CANDLES + 1
    )

    hist15 = data_15m.iloc[
        start_15m:i + 1
    ]

    price = closes[i]

    # --------------------------------------------------------
    # AOI LOCK
    # --------------------------------------------------------

    if (
        locked_key is not None
        and locked_aoi is not None
    ):

        if aoi_reset(
            price,
            locked_aoi
        ):

            locked_key = None
            locked_aoi = None

        else:

            blocked_lock += 1

    # --------------------------------------------------------
    # STRATEGY
    # --------------------------------------------------------

    try:

        signal = generate_signal(
            hist15,
            histdaily,
            price
        )

    except Exception as error:

        reasons[
            "STRATEGY_ERROR"
        ] = (
            reasons.get(
                "STRATEGY_ERROR",
                0
            ) + 1
        )

        print(
            "STRATEGY ERROR:",
            ts,
            "|",
            error,
            flush=True
        )

        continue

    if not isinstance(
        signal,
        dict
    ):

        reasons[
            "INVALID_SIGNAL_OBJECT"
        ] = (
            reasons.get(
                "INVALID_SIGNAL_OBJECT",
                0
            ) + 1
        )

        continue

    direction = signal.get(
        "signal",
        "NONE"
    )

    reason = signal.get(
        "reason",
        "UNKNOWN"
    )

    if direction not in (
        "BUY",
        "SELL"
    ):

        reasons[
            reason
        ] = (
            reasons.get(
                reason,
                0
            ) + 1
        )

        continue

    entry = safe_float(
        signal.get("entry")
    )

    stop_loss = safe_float(
        signal.get("stop_loss")
    )

    take_profit = safe_float(
        signal.get("take_profit")
    )

    if (
        entry is None
        or stop_loss is None
        or take_profit is None
    ):

        reasons[
            "INVALID_TRADE_LEVELS"
        ] = (
            reasons.get(
                "INVALID_TRADE_LEVELS",
                0
            ) + 1
        )

        continue

    # --------------------------------------------------------
    # AOI
    # --------------------------------------------------------

    key = setup_key(
        signal
    )

    aoi = get_aoi(
        signal
    )

    # --------------------------------------------------------
    # Same-AOI re-entry lock
    # --------------------------------------------------------

    if (
        locked_key is not None
        and key == locked_key
        and not aoi_reset(
            price,
            locked_aoi
        )
    ):

        blocked_duplicate += 1

        continue

    # --------------------------------------------------------
    # TRADE RESULT
    # --------------------------------------------------------

    (
        result,
        exit_price,
        exit_time,
        r_multiple
    ) = fast_trade_result(
        direction,
        entry,
        stop_loss,
        take_profit,
        i + 1,
        highs,
        lows,
        timestamps
    )

    # --------------------------------------------------------
    # Bias
    # --------------------------------------------------------

    bias = signal.get(
        "bias",
        {}
    )

    if not isinstance(
        bias,
        dict
    ):

        bias = {}

    setup_id = (
        f"{pd.Timestamp(ts).strftime('%Y%m%d-%H%M')}-"
        f"{direction}-"
        f"{abs(hash(str(key))) % 100000:05d}"
    )

    trade = {

        "setup_id": setup_id,

        "time": pd.Timestamp(ts),

        "signal": direction,

        "entry": entry,

        "sl": stop_loss,

        "tp": take_profit,

        "result": result,

        "r": r_multiple,

        "reason": reason,

        "setup": setup_text(
            signal
        ),

        "weekly": bias.get(
            "weekly",
            "UNKNOWN"
        ),

        "daily": bias.get(
            "daily",
            "UNKNOWN"
        ),

        "4h": bias.get(
            "4h",
            "UNKNOWN"
        ),

        "overall": bias.get(
            "overall",
            "UNKNOWN"
        ),

        "exit_time": exit_time,

        "exit_price": exit_price
    }

    signals.append(
        trade
    )

    locked_key = key
    locked_aoi = aoi

    active_trade = trade

    print(
        "SIGNAL FOUND:",
        ts,
        "|",
        direction,
        "| Entry:",
        round(
            entry,
            2
        ),
        "| Result:",
        result,
        "| R:",
        round(
            r_multiple,
            2
        ),
        "| Setup:",
        trade[
            "setup"
        ],
        flush=True
    )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        evaluations % 2500 == 0
    ):

        progress = (
            evaluations
            / total_iterations
            * 100
        )

        print(
            "Progress:",
            f"{progress:.1f}%",
            "| Checked:",
            evaluations,
            "| Signals:",
            len(signals),
            flush=True
        )


# ============================================================
# RESULTS
# ============================================================

print()

print(
    "=" * 60
)

print(
    "BACKTEST COMPLETE"
)

print(
    "=" * 60
)

print()

start = pd.Timestamp(
    data_15m.index[0]
)

end = pd.Timestamp(
    data_15m.index[-1]
)

days = (
    end - start
).total_seconds() / 86400

weeks = days / 7

total = len(
    signals
)

buys = sum(
    x["signal"] == "BUY"
    for x in signals
)

sells = sum(
    x["signal"] == "SELL"
    for x in signals
)

tp_count = sum(
    x["result"] == "TP"
    for x in signals
)

sl_count = sum(
    x["result"] == "SL"
    for x in signals
)

ambiguous = sum(
    x["result"] == "AMBIGUOUS"
    for x in signals
)

open_count = sum(
    x["result"] == "OPEN"
    for x in signals
)

resolved = (
    tp_count
    + sl_count
)

win_rate = (
    tp_count
    / resolved
    * 100
    if resolved
    else 0
)

total_r = sum(
    x["r"]
    for x in signals
)

winning_r = sum(
    x["r"]
    for x in signals
    if x["r"] > 0
)

losing_r = sum(
    x["r"]
    for x in signals
    if x["r"] < 0
)

profit_factor = (
    winning_r
    / abs(losing_r)
    if losing_r < 0
    else None
)

expectancy = (
    total_r
    / resolved
    if resolved
    else 0
)


# ============================================================
# MAX DRAWDOWN
# ============================================================

equity = 0.0
peak = 0.0
max_drawdown = 0.0

for trade in signals:

    equity += trade["r"]

    peak = max(
        peak,
        equity
    )

    drawdown = (
        peak - equity
    )

    max_drawdown = max(
        max_drawdown,
        drawdown
    )


# ============================================================
# LONGEST GAP
# ============================================================

if len(signals) >= 2:

    signal_times = [
        x["time"]
        for x in signals
    ]

    longest_gap = max(
        (
            signal_times[i]
            - signal_times[i - 1]
        ).total_seconds()
        / 86400

        for i in range(
            1,
            len(signal_times)
        )
    )

else:

    longest_gap = None


# ============================================================
# SUMMARY
# ============================================================

print(
    "DATA SOURCE: DUKASCOPY"
)

print(
    "SYMBOL: XAUUSD"
)

print(
    "INSTRUMENT: GOLD SPOT"
)

print()

print(
    "TEST PERIOD:"
)

print(
    start,
    "->",
    end
)

print(
    "DAYS TESTED:",
    round(
        days,
        1
    )
)

print(
    "WEEKS TESTED:",
    round(
        weeks,
        1
    )
)

print()

print(
    "15M CANDLES:",
    len(data_15m)
)

print(
    "DAILY CANDLES:",
    len(data_daily)
)

print(
    "15M EVALUATIONS:",
    evaluations
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
    total
)

print(
    "BUY SETUPS:",
    buys
)

print(
    "SELL SETUPS:",
    sells
)

print(
    "SETUPS PER WEEK:",
    round(
        total / weeks,
        2
    )
    if weeks
    else 0
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
    ambiguous
)

print(
    "OPEN:",
    open_count
)

print(
    "RESOLVED TRADES:",
    resolved
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
        expectancy,
        3
    ),
    "R/trade"
)

print(
    "PROFIT FACTOR:",
    (
        round(
            profit_factor,
            2
        )
        if profit_factor is not None
        else "N/A"
    )
)

print(
    "MAX DRAWDOWN:",
    round(
        max_drawdown,
        2
    ),
    "R"
)

print(
    "LONGEST GAP BETWEEN "
    "INDEPENDENT SETUPS:",
    (
        round(
            longest_gap,
            2
        )
        if longest_gap is not None
        else "N/A"
    ),
    "days"
    if longest_gap is not None
    else ""
)


# ============================================================
# ACCOUNTING
# ============================================================

print()

print(
    "=" * 60
)

print(
    "BACKTEST ACCOUNTING"
)

print(
    "=" * 60
)

print(
    "Evaluations blocked while trade open:",
    blocked_open
)

print(
    "Repeated same-AOI signals blocked:",
    blocked_duplicate
)

print(
    "Evaluations inside locked AOI:",
    blocked_lock
)


# ============================================================
# REJECTION REASONS
# ============================================================

if reasons:

    print()

    print(
        "=" * 60
    )

    print(
        "TOP SIGNAL REJECTION REASONS"
    )

    print(
        "=" * 60
    )

    for reason, count in sorted(
        reasons.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]:

        print(
            reason,
            ":",
            count
        )


# ============================================================
# INDIVIDUAL SIGNALS
# ============================================================

print()

print(
    "=" * 60
)

print(
    "INDEPENDENT SETUPS"
)

print(
    "=" * 60
)

if not signals:

    print(
        "NO INDEPENDENT SETUPS FOUND."
    )

for number, trade in enumerate(
    signals,
    1
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
            trade["sl"],
            2
        )
    )

    print(
        "TP:",
        round(
            trade["tp"],
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
            trade["r"],
            2
        )
    )

    print(
        "Reason:",
        trade["reason"]
    )

    print(
        "Weekly Bias:",
        trade["weekly"]
    )

    print(
        "Daily Bias:",
        trade["daily"]
    )

    print(
        "4H Bias:",
        trade["4h"]
    )

    print(
        "Overall Bias:",
        trade["overall"]
    )

    if trade["exit_time"] is not None:

        print(
            "Exit:",
            trade["exit_time"]
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
