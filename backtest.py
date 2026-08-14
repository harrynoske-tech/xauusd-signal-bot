import os
import numpy as np
import pandas as pd

from strategy_v8 import generate_signal


# ============================================================
# XAUUSD STRATEGY BACKTEST V8.0
# MULTI-FACTOR RESEARCH ENGINE
# ============================================================

DATA_15M = "data/XAUUSD_15m.csv"
DATA_DAILY = "data/XAUUSD_1d.csv"

MIN_15M_CANDLES = 300
MIN_DAILY_CANDLES = 100

# Evaluate every completed 15-minute candle.
CHECK_EVERY = 1

# Maximum historical windows passed into strategy_v8.
MAX_15M_CANDLES = 1000
MAX_DAILY_CANDLES = 500


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("=" * 60)
    print("XAUUSD STRATEGY BACKTEST V8.0")
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
        .loc[
            ~data_15m.index.duplicated(
                keep="last"
            )
        ]
    )

    data_daily = (
        data_daily
        .sort_index()
        .loc[
            ~data_daily.index.duplicated(
                keep="last"
            )
        ]
    )

    # Standardise OHLC names.
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

    # Ensure numeric OHLC.
    for column in required:

        data_15m[column] = pd.to_numeric(
            data_15m[column],
            errors="coerce"
        )

        data_daily[column] = pd.to_numeric(
            data_daily[column],
            errors="coerce"
        )

    data_15m = data_15m.dropna(
        subset=required
    )

    data_daily = data_daily.dropna(
        subset=required
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


def get_score(signal):

    score = safe_float(
        signal.get("score")
    )

    if score is None:
        return 0.0

    return score


def get_components(signal):

    components = signal.get(
        "components",
        {}
    )

    if not isinstance(
        components,
        dict
    ):
        return {}

    return components


# ============================================================
# TRADE RESULT
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

    # Neither level hit.
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

    # Both hit during the same candle.
    # Conservative assumption:
    # treat as ambiguous.
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

    # Stop hit first.
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

    # Target hit first.
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
# RESULT STATISTICS
# ============================================================

def print_score_analysis(signals):

    print()
    print("=" * 60)
    print("V8 SCORE ANALYSIS")
    print("=" * 60)

    if not signals:

        print("No signals.")
        return

    thresholds = [
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95
    ]

    print()
    print(
        "THRESHOLD | TRADES | WINS | LOSSES | WIN RATE | TOTAL R"
    )
    print("-" * 60)

    for threshold in thresholds:

        trades = [
            x
            for x in signals
            if x["score"] >= threshold
            and x["result"] in (
                "TP",
                "SL"
            )
        ]

        total = len(trades)

        wins = sum(
            x["result"] == "TP"
            for x in trades
        )

        losses = sum(
            x["result"] == "SL"
            for x in trades
        )

        total_r = sum(
            x["r"]
            for x in trades
        )

        win_rate = (
            wins / total * 100
            if total
            else 0
        )

        print(
            f"{threshold:9.0f} | "
            f"{total:6d} | "
            f"{wins:4d} | "
            f"{losses:6d} | "
            f"{win_rate:8.2f}% | "
            f"{total_r:7.2f}"
        )

    print()
    print(
        "IMPORTANT: Higher thresholds reduce trade frequency."
    )


def print_component_analysis(signals):

    print()
    print("=" * 60)
    print("V8 COMPONENT ANALYSIS")
    print("=" * 60)

    if not signals:
        return

    components = [
        "trend",
        "momentum",
        "breakout",
        "daily",
        "htf",
        "volatility",
        "session"
    ]

    for component_name in components:

        component_trades = []

        for trade in signals:

            component = trade[
                "components"
            ].get(
                component_name
            )

            if not isinstance(
                component,
                dict
            ):
                continue

            component_trades.append(
                (
                    trade,
                    component
                )
            )

        if not component_trades:
            continue

        print()
        print(
            component_name.upper()
        )

        directional = {
            "BUY": [],
            "SELL": [],
            "NONE": []
        }

        for trade, component in component_trades:

            direction = component.get(
                "direction",
                "NONE"
            )

            if direction not in directional:
                direction = "NONE"

            directional[
                direction
            ].append(trade)

        for direction, trades in directional.items():

            if not trades:
                continue

            resolved = [
                x
                for x in trades
                if x["result"] in (
                    "TP",
                    "SL"
                )
            ]

            if not resolved:
                continue

            wins = sum(
                x["result"] == "TP"
                for x in resolved
            )

            win_rate = (
                wins
                / len(resolved)
                * 100
            )

            total_r = sum(
                x["r"]
                for x in resolved
            )

            print(
                f"  {direction:5s} | "
                f"Trades: {len(resolved):4d} | "
                f"Win: {win_rate:6.2f}% | "
                f"R: {total_r:7.2f}"
            )


# ============================================================
# LOAD
# ============================================================

data_15m, data_daily = load_data()


# ============================================================
# PRE-COMPUTE ARRAYS
# ============================================================

timestamps = (
    data_15m.index.to_numpy()
)

highs = (
    data_15m["High"]
    .to_numpy(dtype=float)
)

lows = (
    data_15m["Low"]
    .to_numpy(dtype=float)
)

closes = (
    data_15m["Close"]
    .to_numpy(dtype=float)
)


# ============================================================
# DAILY POSITION LOOKUP
# ============================================================

daily_index = (
    data_daily.index
)

daily_positions = (
    daily_index.searchsorted(
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
    "V8 MULTI-FACTOR ENGINE: ACTIVE"
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

            continue

        active_trade = None

    # --------------------------------------------------------
    # Daily history
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
    # 15M history
    # --------------------------------------------------------

    start_15m = max(
        0,
        i
        - MAX_15M_CANDLES
        + 1
    )

    hist15 = data_15m.iloc[
        start_15m:i + 1
    ]

    price = closes[i]

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
            )
            + 1
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
            )
            + 1
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

    # --------------------------------------------------------
    # No signal
    # --------------------------------------------------------

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
            )
            + 1
        )

        continue

    # --------------------------------------------------------
    # Trade levels
    # --------------------------------------------------------

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
            )
            + 1
        )

        continue

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    score = get_score(
        signal
    )

    components = get_components(
        signal
    )

    # --------------------------------------------------------
    # Trade result
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
    # Setup ID
    # --------------------------------------------------------

    setup_id = (
        f"{pd.Timestamp(ts).strftime('%Y%m%d-%H%M')}-"
        f"{direction}-"
        f"{i}"
    )

    trade = {

        "setup_id":
            setup_id,

        "time":
            pd.Timestamp(ts),

        "signal":
            direction,

        "score":
            score,

        "entry":
            entry,

        "sl":
            stop_loss,

        "tp":
            take_profit,

        "result":
            result,

        "r":
            r_multiple,

        "reason":
            reason,

        "components":
            components,

        "exit_time":
            exit_time,

        "exit_price":
            exit_price
    }

    signals.append(
        trade
    )

    active_trade = trade

    print(
        "SIGNAL FOUND:",
        ts,
        "|",
        direction,
        "| Score:",
        round(score, 1),
        "| Entry:",
        round(entry, 2),
        "| Result:",
        result,
        "| R:",
        round(r_multiple, 2),
        flush=True
    )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        evaluations % 5000 == 0
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
    round(days, 1)
)

print(
    "WEEKS TESTED:",
    round(weeks, 1)
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

print()

print(
    "TOTAL V8 SIGNALS:",
    total
)

print(
    "BUY SIGNALS:",
    buys
)

print(
    "SELL SIGNALS:",
    sells
)

print(
    "SIGNALS PER WEEK:",
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
    "LONGEST GAP BETWEEN SIGNALS:",
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
# SCORE ANALYSIS
# ============================================================

print_score_analysis(
    signals
)


# ============================================================
# COMPONENT ANALYSIS
# ============================================================

print_component_analysis(
    signals
)


# ============================================================
# SIGNAL DETAILS
# ============================================================

print()

print(
    "=" * 60
)

print(
    "V8 SIGNAL DETAILS"
)

print(
    "=" * 60
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
        "Signal:",
        trade["signal"]
    )

    print(
        "Score:",
        round(
            trade["score"],
            2
        )
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
        "Components:",
        trade["components"]
    )

    if trade["exit_time"] is not None:

        print(
            "Exit:",
            trade["exit_time"]
        )


# ============================================================
# END
# ============================================================

print()

print(
    "=" * 60
)

print(
    "END OF V8 BACKTEST"
)

print(
    "=" * 60
)
