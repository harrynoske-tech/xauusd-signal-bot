import os
import time
import pandas as pd

from strategy import generate_signal


# ============================================================
# XAUUSD STRATEGY BACKTEST V7.0
# DUKASCOPY DATA
# ============================================================

DATA_FILE = "data/XAUUSD_15m.csv"

DAILY_FILE = "data/XAUUSD_1d.csv"

MIN_15M_CANDLES = 300
MAX_15M_CANDLES = 1000
MAX_DAILY_CANDLES = 500

RESET_DISTANCE = 10.0


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("=" * 60)
    print("XAUUSD STRATEGY BACKTEST V7.0")
    print("=" * 60)
    print()

    print("DATA SOURCE: DUKASCOPY")
    print("SYMBOL: XAUUSD")
    print("EXECUTION TIMEFRAME: 15M")
    print()

    if not os.path.exists(DATA_FILE):

        raise RuntimeError(
            f"Missing data file: {DATA_FILE}"
        )

    data_15m = pd.read_csv(
        DATA_FILE,
        index_col=0,
        parse_dates=True
    )

    data_15m.columns = [
        str(c).capitalize()
        for c in data_15m.columns
    ]

    # --------------------------------------------------------
    # Daily data
    # --------------------------------------------------------

    if os.path.exists(DAILY_FILE):

        data_daily = pd.read_csv(
            DAILY_FILE,
            index_col=0,
            parse_dates=True
        )

        data_daily.columns = [
            str(c).capitalize()
            for c in data_daily.columns
        ]

    else:

        print(
            "Daily Dukascopy file not found."
        )

        print(
            "Creating daily candles from 15M data..."
        )

        data_daily = (
            data_15m
            .resample("1D")
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum"
                }
            )
            .dropna()
        )

    data_15m = (
        data_15m
        .sort_index()
        .drop_duplicates()
    )

    data_daily = (
        data_daily
        .sort_index()
        .drop_duplicates()
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

    print()

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

    aoi = signal.get(
        "aoi"
    )

    if isinstance(
        aoi,
        dict
    ):
        return aoi

    return None


def setup_key(signal):

    aoi = get_aoi(
        signal
    )

    if not aoi:
        return None

    low = safe_float(
        aoi.get("low")
    )

    high = safe_float(
        aoi.get("high")
    )

    if (
        low is None
        or high is None
    ):
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
        round(
            low,
            1
        ),
        round(
            high,
            1
        )
    )


def setup_text(signal):

    key = setup_key(
        signal
    )

    if not key:
        return "UNKNOWN_SETUP"

    timeframe, zone_type, low, high = key

    return (
        f"{str(timeframe).upper()} "
        f"{str(zone_type).upper()} "
        f"{low}-{high}"
    )


def aoi_reset(
    price,
    aoi
):

    if not isinstance(
        aoi,
        dict
    ):
        return True

    low = safe_float(
        aoi.get("low")
    )

    high = safe_float(
        aoi.get("high")
    )

    if (
        low is None
        or high is None
    ):
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
# TRADE RESULT
# ============================================================

def result_for_trade(
    direction,
    entry,
    stop_loss,
    take_profit,
    future
):

    for timestamp, candle in future.iterrows():

        high = float(
            candle["High"]
        )

        low = float(
            candle["Low"]
        )

        if direction == "SELL":

            stop_hit = (
                high >= stop_loss
            )

            target_hit = (
                low <= take_profit
            )

        else:

            stop_hit = (
                low <= stop_loss
            )

            target_hit = (
                high >= take_profit
            )

        if (
            stop_hit
            and target_hit
        ):

            return (
                "AMBIGUOUS",
                None,
                timestamp,
                0.0
            )

        if stop_hit:

            return (
                "SL",
                stop_loss,
                timestamp,
                -1.0
            )

        if target_hit:

            risk = abs(
                entry - stop_loss
            )

            reward = abs(
                take_profit - entry
            )

            r_multiple = (
                reward / risk
                if risk > 0
                else 0
            )

            return (
                "TP",
                take_profit,
                timestamp,
                r_multiple
            )

    return (
        "OPEN",
        None,
        None,
        0.0
    )


# ============================================================
# LOAD
# ============================================================

data_15m, data_daily = load_data()


# ============================================================
# BACKTEST
# ============================================================

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


signals = []

reason_counts = {}

active_trade = None

locked_setup_key = None
locked_setup_aoi = None

blocked_open = 0
blocked_duplicate = 0
blocked_lock = 0

evaluations = 0


# ============================================================
# MAIN LOOP
# ============================================================

for i in range(
    MIN_15M_CANDLES,
    len(data_15m)
):

    evaluations += 1

    timestamp = data_15m.index[i]

    # --------------------------------------------------------
    # Only one open trade
    # --------------------------------------------------------

    if active_trade is not None:

        exit_time = active_trade.get(
            "exit_time"
        )

        if (
            exit_time is not None
            and timestamp <= exit_time
        ):

            blocked_open += 1

            continue

        active_trade = None

    # --------------------------------------------------------
    # Historical windows
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

    if len(
        historical_daily
    ) < 100:

        continue

    current_price = float(
        historical_15m[
            "Close"
        ].iloc[-1]
    )

    # --------------------------------------------------------
    # AOI lock
    # --------------------------------------------------------

    if (
        locked_setup_key is not None
        and locked_setup_aoi is not None
    ):

        if aoi_reset(
            current_price,
            locked_setup_aoi
        ):

            locked_setup_key = None
            locked_setup_aoi = None

        else:

            blocked_lock += 1

    # --------------------------------------------------------
    # Generate strategy signal
    # --------------------------------------------------------

    try:

        signal = generate_signal(
            historical_15m,
            historical_daily,
            current_price
        )

    except Exception as error:

        reason_counts[
            "STRATEGY_ERROR"
        ] = (
            reason_counts.get(
                "STRATEGY_ERROR",
                0
            ) + 1
        )

        continue

    if not isinstance(
        signal,
        dict
    ):

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

        reason_counts[
            reason
        ] = (
            reason_counts.get(
                reason,
                0
            ) + 1
        )

        continue

    entry = safe_float(
        signal.get(
            "entry"
        )
    )

    stop_loss = safe_float(
        signal.get(
            "stop_loss"
        )
    )

    take_profit = safe_float(
        signal.get(
            "take_profit"
        )
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
            ) + 1
        )

        continue

    current_setup_key = setup_key(
        signal
    )

    current_setup_aoi = get_aoi(
        signal
    )

    # --------------------------------------------------------
    # Same AOI lock
    # --------------------------------------------------------

    if (
        locked_setup_key is not None
        and current_setup_key == locked_setup_key
        and not aoi_reset(
            current_price,
            locked_setup_aoi
        )
    ):

        blocked_duplicate += 1

        continue

    # --------------------------------------------------------
    # Resolve
    # --------------------------------------------------------

    result, exit_price, exit_time, r_multiple = (
        result_for_trade(
            direction,
            entry,
            stop_loss,
            take_profit,
            data_15m.iloc[
                i + 1:
            ]
        )
    )

    bias = signal.get(
        "bias",
        {}
    )

    if not isinstance(
        bias,
        dict
    ):

        bias = {}

    trade = {

        "time": timestamp,

        "signal": direction,

        "entry": entry,

        "stop_loss": stop_loss,

        "take_profit": take_profit,

        "result": result,

        "r": r_multiple,

        "reason": reason,

        "setup": setup_text(
            signal
        ),

        "weekly_bias": bias.get(
            "weekly",
            "UNKNOWN"
        ),

        "daily_bias": bias.get(
            "daily",
            "UNKNOWN"
        ),

        "4h_bias": bias.get(
            "4h",
            "UNKNOWN"
        ),

        "overall_bias": bias.get(
            "overall",
            "UNKNOWN"
        ),

        "exit_time": exit_time,

        "exit_price": exit_price
    }

    signals.append(
        trade
    )

    locked_setup_key = (
        current_setup_key
    )

    locked_setup_aoi = (
        current_setup_aoi
    )

    active_trade = trade

    print(
        "SIGNAL FOUND:",
        timestamp,
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

    if evaluations % 1000 == 0:

        progress = (
            evaluations
            / max(
                1,
                len(data_15m)
                - MIN_15M_CANDLES
            )
            * 100
        )

        print(
            "Progress:",
            round(
                progress,
                1
            ),
            "%",
            "| Checked:",
            evaluations,
            "| Signals:",
            len(signals),
            flush=True
        )


# ============================================================
# SUMMARY
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

start = data_15m.index[0]
end = data_15m.index[-1]

days = (
    end - start
).total_seconds() / 86400

weeks = days / 7

total = len(
    signals
)

buy_count = sum(
    x["signal"] == "BUY"
    for x in signals
)

sell_count = sum(
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

ambiguous_count = sum(
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

equity = 0
peak = 0
max_drawdown = 0

for trade in signals:

    equity += trade[
        "r"
    ]

    peak = max(
        peak,
        equity
    )

    max_drawdown = max(
        max_drawdown,
        peak - equity
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "DATA SOURCE: DUKASCOPY"
)

print(
    "SYMBOL: XAUUSD"
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
    "TOTAL SETUPS:",
    total
)

print(
    "BUY SETUPS:",
    buy_count
)

print(
    "SELL SETUPS:",
    sell_count
)

print(
    "SETUPS PER WEEK:",
    round(
        total / weeks,
        2
    )
    if weeks > 0
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
    ambiguous_count
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

print()

print(
    "=" * 60
)

print(
    "INDIVIDUAL SIGNALS"
)

print(
    "=" * 60
)

if not signals:

    print(
        "NO SIGNALS FOUND."
    )

for number, trade in enumerate(
    signals,
    1
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
        trade["entry"]
    )

    print(
        "SL:",
        trade["stop_loss"]
    )

    print(
        "TP:",
        trade["take_profit"]
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
        "Setup:",
        trade["setup"]
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
