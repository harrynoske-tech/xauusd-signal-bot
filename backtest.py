import os
import numpy as np
import pandas as pd

from strategy_v8 import generate_signal


# ============================================================
# XAUUSD STRATEGY BACKTEST V8.1
# CORRECT CHRONOLOGICAL SIMULATION
# ============================================================

DATA_15M = "data/XAUUSD_15m.csv"
DATA_DAILY = "data/XAUUSD_1d.csv"

MIN_15M_CANDLES = 300
MIN_DAILY_CANDLES = 100

MAX_15M_CANDLES = 1000
MAX_DAILY_CANDLES = 500

# After a trade closes, the system must see this many
# consecutive candles with NO signal before another trade
# using the same market condition can be considered.
REARM_BARS = 8


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("=" * 60)
    print("XAUUSD STRATEGY BACKTEST V8.1")
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
                f"15M data missing {column}"
            )

        if column not in data_daily.columns:
            raise RuntimeError(
                f"Daily data missing {column}"
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
# HELPERS
# ============================================================

def number(value):

    try:
        return float(value)
    except Exception:
        return None


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

    future_highs = highs[
        start_index:
    ]

    future_lows = lows[
        start_index:
    ]

    if len(future_highs) == 0:

        return {
            "result": "OPEN",
            "exit_index": None,
            "exit_time": None,
            "r": 0.0
        }

    if direction == "BUY":

        stop_hits = (
            future_lows
            <= stop_loss
        )

        target_hits = (
            future_highs
            >= take_profit
        )

    else:

        stop_hits = (
            future_highs
            >= stop_loss
        )

        target_hits = (
            future_lows
            <= take_profit
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

    # Both TP and SL occur on the same candle.
    # Conservative assumption = SL.
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
        else 0
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
# MAIN SIMULATION
# ============================================================

print()
print(
    "Checking every 15 minutes..."
)
print(
    "ONE OPEN TRADE AT A TIME: YES"
)
print(
    "SAME-SETUP RE-ENTRY LOCK: YES"
)
print(
    "DAILY LOOK-AHEAD PROTECTION: YES"
)
print()
print(
    "Starting historical simulation...",
    flush=True
)


trades = []

evaluations = 0

i = MIN_15M_CANDLES

# Number of consecutive no-signal candles required
# before another setup can be taken.
no_signal_bars = REARM_BARS

total_iterations = (
    len(data_15m)
    - MIN_15M_CANDLES
)


# ============================================================
# CHRONOLOGICAL LOOP
# ============================================================

while i < len(data_15m):

    evaluations += 1

    timestamp = timestamps[i]

    # --------------------------------------------------------
    # ONLY USE COMPLETED DAILY CANDLES
    #
    # side="left" means today's incomplete daily candle
    # is excluded from the strategy.
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
        daily_end
        - MAX_DAILY_CANDLES
    )

    historical_daily = (
        data_daily.iloc[
            daily_start:daily_end
        ]
    )

    # --------------------------------------------------------
    # 15M HISTORY
    # --------------------------------------------------------

    start_15m = max(
        0,
        i
        - MAX_15M_CANDLES
        + 1
    )

    historical_15m = (
        data_15m.iloc[
            start_15m:i + 1
        ]
    )

    price = closes[i]

    # --------------------------------------------------------
    # STRATEGY DECISION
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

    # --------------------------------------------------------
    # NO SIGNAL
    # --------------------------------------------------------

    if direction not in (
        "BUY",
        "SELL"
    ):

        no_signal_bars = min(
            REARM_BARS,
            no_signal_bars + 1
        )

        i += 1
        continue

    # --------------------------------------------------------
    # SAME-SETUP RE-ENTRY LOCK
    # --------------------------------------------------------

    if no_signal_bars < REARM_BARS:

        i += 1
        continue

    # --------------------------------------------------------
    # VALID TRADE LEVELS
    # --------------------------------------------------------

    entry = number(
        signal.get("entry")
    )

    stop_loss = number(
        signal.get("stop_loss")
    )

    take_profit = number(
        signal.get("take_profit")
    )

    if (
        entry is None
        or stop_loss is None
        or take_profit is None
    ):

        i += 1
        continue

    score = number(
        signal.get("score")
    )

    if score is None:
        score = 0.0

    # --------------------------------------------------------
    # FIND THE ACTUAL FUTURE EXIT
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
            float(score),

        "entry":
            float(entry),

        "sl":
            float(stop_loss),

        "tp":
            float(take_profit),

        "result":
            outcome["result"],

        "r":
            outcome["r"],

        "exit_time":
            outcome["exit_time"],

        "components":
            signal.get(
                "components",
                {}
            )
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
    # CRITICAL:
    #
    # Jump directly to the candle AFTER the trade closes.
    #
    # This guarantees we cannot open another trade while
    # the previous trade is still active.
    # --------------------------------------------------------

    exit_index = outcome[
        "exit_index"
    ]

    if exit_index is None:

        break

    i = (
        exit_index
        + 1
    )

    # --------------------------------------------------------
    # RE-ARM LOCK
    #
    # Require a genuine period with no signal before another
    # trade can be opened.
    # --------------------------------------------------------

    no_signal_bars = 0

    # We don't count the exit candle as a fresh setup.
    # The next REARM_BARS candles must pass without a signal.
    rearm_until = min(
        len(data_15m),
        i + REARM_BARS
    )

    i = rearm_until


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
# DRAWDOWN
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
# SCORE THRESHOLD ANALYSIS
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

print()

print("=" * 60)
print("SCORE THRESHOLD ANALYSIS")
print("=" * 60)

print()

print(
    "THRESHOLD | TRADES | WINS | LOSSES | WIN RATE | TOTAL R"
)

print("-" * 60)

for threshold in (
    70,
    75,
    80,
    85,
    87,
    90,
    92,
    95
):

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

for low, high in (
    (70, 74),
    (75, 79),
    (80, 84),
    (85, 89),
    (90, 94),
    (95, 100)
):

    selected = [
        trade
        for trade in trades
        if low
        <= trade["score"]
        <= high
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
# TRADE LOG
# ============================================================

print()
print("=" * 60)
print("INDEPENDENT TRADE LOG")
print("=" * 60)

for number, trade in enumerate(
    trades,
    1
):

    print()

    print(
        number,
        "|",
        trade["time"],
        "|",
        trade["direction"]
    )

    print(
        "Score:",
        round(
            trade["score"],
            1
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
        "Exit:",
        trade["exit_time"]
    )


print()
print("=" * 60)
print("END OF V8.1 BACKTEST")
print("=" * 60)
