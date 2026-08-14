import os
import numpy as np
import pandas as pd

from strategy_v8 import generate_signal


# ============================================================
# XAUUSD STRATEGY BACKTEST V8.2
# ============================================================

DATA_15M = "data/XAUUSD_15m.csv"
DATA_DAILY = "data/XAUUSD_1d.csv"

MIN_15M_CANDLES = 300
MIN_DAILY_CANDLES = 100

MAX_15M_CANDLES = 1000
MAX_DAILY_CANDLES = 500

REARM_BARS = 8


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("=" * 60)
    print("XAUUSD STRATEGY BACKTEST V8.2")
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

        data_15m[column] = pd.to_numeric(
            data_15m[column],
            errors="coerce"
        )

        data_daily[column] = pd.to_numeric(
            data_daily[column],
            errors="coerce"
        )

    data_15m = (
        data_15m
        .dropna(subset=required)
        .sort_index()
    )

    data_daily = (
        data_daily
        .dropna(subset=required)
        .sort_index()
    )

    data_15m = data_15m.loc[
        ~data_15m.index.duplicated(
            keep="last"
        )
    ]

    data_daily = data_daily.loc[
        ~data_daily.index.duplicated(
            keep="last"
        )
    ]

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

    return data_15m, data_daily


# ============================================================
# TRADE RESULT
# ============================================================

def result_for_trade(
    direction,
    entry,
    stop_loss,
    take_profit,
    start_index,
    highs,
    lows,
    timestamps
):

    future_highs = highs[start_index:]
    future_lows = lows[start_index:]

    if len(future_highs) == 0:

        return {
            "result": "OPEN",
            "exit_index": None,
            "exit_time": None,
            "r": 0.0
        }

    if direction == "BUY":

        stop_hits = (
            future_lows <= stop_loss
        )

        target_hits = (
            future_highs >= take_profit
        )

    else:

        stop_hits = (
            future_highs >= stop_loss
        )

        target_hits = (
            future_lows <= take_profit
        )

    stop_positions = np.flatnonzero(
        stop_hits
    )

    target_positions = np.flatnonzero(
        target_hits
    )

    stop_pos = (
        int(stop_positions[0])
        if len(stop_positions)
        else None
    )

    target_pos = (
        int(target_positions[0])
        if len(target_positions)
        else None
    )

    if (
        stop_pos is None
        and target_pos is None
    ):

        return {
            "result": "OPEN",
            "exit_index": None,
            "exit_time": None,
            "r": 0.0
        }

    # Conservative treatment when TP and SL
    # are both hit on the same candle.
    if (
        stop_pos is not None
        and target_pos is not None
        and stop_pos == target_pos
    ):

        position = stop_pos

        return {
            "result": "AMBIGUOUS",
            "exit_index":
                start_index + position,
            "exit_time":
                timestamps[
                    start_index + position
                ],
            "r": -1.0
        }

    # SL first
    if (
        target_pos is None
        or (
            stop_pos is not None
            and stop_pos < target_pos
        )
    ):

        position = stop_pos

        return {
            "result": "SL",
            "exit_index":
                start_index + position,
            "exit_time":
                timestamps[
                    start_index + position
                ],
            "r": -1.0
        }

    # TP first
    position = target_pos

    risk = abs(
        entry - stop_loss
    )

    reward = abs(
        take_profit - entry
    )

    r = (
        reward / risk
        if risk > 0
        else 0.0
    )

    return {
        "result": "TP",
        "exit_index":
            start_index + position,
        "exit_time":
            timestamps[
                start_index + position
            ],
        "r": float(r)
    }


# ============================================================
# LOAD
# ============================================================

data_15m, data_daily = load_data()

timestamps = data_15m.index.to_numpy()

highs = data_15m[
    "High"
].to_numpy(
    dtype=float
)

lows = data_15m[
    "Low"
].to_numpy(
    dtype=float
)

closes = data_15m[
    "Close"
].to_numpy(
    dtype=float
)


# ============================================================
# MAIN BACKTEST
# ============================================================

print()
print("Checking every 15 minutes...")
print("ONE OPEN TRADE AT A TIME: YES")
print("SAME-SETUP RE-ENTRY LOCK: YES")
print("DAILY LOOK-AHEAD PROTECTION: YES")
print()
print(
    "Starting historical simulation...",
    flush=True
)


trades = []

evaluations = 0

i = MIN_15M_CANDLES

total_iterations = (
    len(data_15m)
    - MIN_15M_CANDLES
)


# ============================================================
# CHRONOLOGICAL SIMULATION
# ============================================================

while i < len(data_15m):

    evaluations += 1

    timestamp = timestamps[i]

    # --------------------------------------------------------
    # DAILY DATA
    # --------------------------------------------------------

    daily_end = data_daily.index.searchsorted(
        timestamp,
        side="left"
    )

    if daily_end < MIN_DAILY_CANDLES:

        i += 1
        continue

    daily_start = max(
        0,
        daily_end - MAX_DAILY_CANDLES
    )

    historical_daily = data_daily.iloc[
        daily_start:daily_end
    ]

    # --------------------------------------------------------
    # 15M DATA
    # --------------------------------------------------------

    start_15m = max(
        0,
        i - MAX_15M_CANDLES + 1
    )

    historical_15m = data_15m.iloc[
        start_15m:i + 1
    ]

    price = closes[i]

    # --------------------------------------------------------
    # STRATEGY
    # --------------------------------------------------------

    try:

        signal = generate_signal(
            historical_15m,
            historical_daily,
            price
        )

    except Exception as error:

        print(
            "STRATEGY ERROR:",
            timestamp,
            "|",
            error,
            flush=True
        )

        i += 1
        continue

    if not isinstance(
        signal,
        dict
    ):

        i += 1
        continue

    direction = signal.get(
        "signal",
        "NONE"
    )

    if direction not in (
        "BUY",
        "SELL"
    ):

        i += 1
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

        i += 1
        continue

    entry = float(entry)
    stop_loss = float(stop_loss)
    take_profit = float(take_profit)

    score = float(
        signal.get(
            "score",
            0
        )
    )

    components = signal.get(
        "components",
        {}
    )

    # --------------------------------------------------------
    # DETERMINE FUTURE RESULT
    # --------------------------------------------------------

    outcome = result_for_trade(
        direction,
        entry,
        stop_loss,
        take_profit,
        i + 1,
        highs,
        lows,
        timestamps
    )

    trade = {

        "time":
            pd.Timestamp(timestamp),

        "direction":
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
            outcome["result"],

        "r":
            outcome["r"],

        "exit_time":
            outcome["exit_time"],

        "components":
            components
    }

    trades.append(
        trade
    )

    print(
        "SIGNAL FOUND:",
        timestamp,
        "|",
        direction,
        "| Score:",
        round(score, 1),
        "| Entry:",
        round(entry, 2),
        "| Result:",
        outcome["result"],
        "| R:",
        round(
            outcome["r"],
            2
        ),
        flush=True
    )

    # --------------------------------------------------------
    # JUMP PAST THE ENTIRE TRADE
    # --------------------------------------------------------

    exit_index = outcome[
        "exit_index"
    ]

    if exit_index is None:
        break

    i = exit_index + 1

    # --------------------------------------------------------
    # RE-ENTRY LOCK
    # --------------------------------------------------------

    i = min(
        len(data_15m),
        i + REARM_BARS
    )

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if evaluations % 5000 == 0:

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
            "| Trades:",
            len(trades),
            flush=True
        )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("BACKTEST COMPLETE")
print("=" * 60)
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
    trades
)

wins = sum(
    x["result"] == "TP"
    for x in trades
)

losses = sum(
    x["result"] == "SL"
    for x in trades
)

ambiguous = sum(
    x["result"] == "AMBIGUOUS"
    for x in trades
)

resolved = (
    wins
    + losses
    + ambiguous
)

win_rate = (
    wins
    / resolved
    * 100
    if resolved
    else 0
)

total_r = sum(
    x["r"]
    for x in trades
)

winning_r = sum(
    x["r"]
    for x in trades
    if x["r"] > 0
)

losing_r = sum(
    x["r"]
    for x in trades
    if x["r"] < 0
)

profit_factor = (
    winning_r
    / abs(losing_r)
    if losing_r < 0
    else 0
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

for trade in trades:

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
# SUMMARY
# ============================================================

print(
    "TEST PERIOD:",
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

print()

print(
    "TOTAL INDEPENDENT TRADES:",
    total
)

print(
    "WINS:",
    wins
)

print(
    "LOSSES:",
    losses
)

print(
    "AMBIGUOUS:",
    ambiguous
)

print(
    "WIN RATE:",
    round(
        win_rate,
        2
    ),
    "%"
)

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
    round(
        profit_factor,
        2
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
    "TRADES PER WEEK:",
    round(
        total / weeks,
        2
    )
    if weeks
    else 0
)


# ============================================================
# SCORE ANALYSIS
# ============================================================

print()
print("=" * 60)
print("SCORE THRESHOLD ANALYSIS")
print("=" * 60)

print()
print(
    "THRESHOLD | TRADES | WINS | LOSSES | WIN RATE | TOTAL R"
)
print("-" * 60)

for threshold in [
    70,
    75,
    80,
    85,
    87,
    90,
    92,
    95
]:

    selected = [
        trade
        for trade in trades
        if trade["score"] >= threshold
        and trade["result"] in (
            "TP",
            "SL"
        )
    ]

    count = len(
        selected
    )

    threshold_wins = sum(
        x["result"] == "TP"
        for x in selected
    )

    threshold_losses = sum(
        x["result"] == "SL"
        for x in selected
    )

    threshold_r = sum(
        x["r"]
        for x in selected
    )

    threshold_win_rate = (
        threshold_wins
        / count
        * 100
        if count
        else 0
    )

    print(
        f"{threshold:9d} | "
        f"{count:6d} | "
        f"{threshold_wins:4d} | "
        f"{threshold_losses:6d} | "
        f"{threshold_win_rate:8.2f}% | "
        f"{threshold_r:7.2f}"
    )


# ============================================================
# SCORE BUCKETS
# ============================================================

print()
print("=" * 60)
print("SCORE BUCKETS")
print("=" * 60)

for low, high in [
    (70, 74),
    (75, 79),
    (80, 84),
    (85, 89),
    (90, 94),
    (95, 100)
]:

    selected = [
        trade
        for trade in trades
        if low <= trade["score"] <= high
        and trade["result"] in (
            "TP",
            "SL"
        )
    ]

    if not selected:
        continue

    bucket_wins = sum(
        x["result"] == "TP"
        for x in selected
    )

    bucket_win_rate = (
        bucket_wins
        / len(selected)
        * 100
    )

    bucket_r = sum(
        x["r"]
        for x in selected
    )

    print(
        f"{low}-{high}: "
        f"{len(selected)} trades | "
        f"{bucket_win_rate:.2f}% win rate | "
        f"{bucket_r:.2f}R"
    )


# ============================================================
# EXPORT FULL TRADE DATA
# ============================================================

os.makedirs(
    "data",
    exist_ok=True
)

export_rows = []

for trade in trades:

    row = {

        "time":
            trade["time"],

        "direction":
            trade["direction"],

        "score":
            trade["score"],

        "entry":
            trade["entry"],

        "sl":
            trade["sl"],

        "tp":
            trade["tp"],

        "result":
            trade["result"],

        "r":
            trade["r"],

        "exit_time":
            trade["exit_time"]
    }

    components = trade.get(
        "components",
        {}
    )

    for name in [
        "trend",
        "momentum",
        "breakout",
        "daily",
        "htf",
        "volatility"
    ]:

        component = components.get(
            name,
            {}
        )

        if isinstance(
            component,
            dict
        ):

            row[
                f"{name}_direction"
            ] = component.get(
                "direction",
                ""
            )

            row[
                f"{name}_score"
            ] = component.get(
                "score",
                0
            )

            if name == "volatility":

                row[
                    "volatility_regime"
                ] = component.get(
                    "regime",
                    ""
                )

        else:

            row[
                f"{name}_direction"
            ] = ""

            row[
                f"{name}_score"
            ] = 0

    export_rows.append(
        row
    )


trades_df = pd.DataFrame(
    export_rows
)

trades_df.to_csv(
    "data/v8_trades.csv",
    index=False
)

print()
print("=" * 60)
print("TRADE DATA EXPORTED")
print("=" * 60)

print(
    "File: data/v8_trades.csv"
)

print(
    "Trades exported:",
    len(trades_df)
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 60)
print("END OF V8.2 BACKTEST")
print("=" * 60)
