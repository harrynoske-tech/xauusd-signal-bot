import os
import pandas as pd
import numpy as np
from itertools import product


# ============================================================
# MULTI-MARKET OPTIMIZER V4
#
# XAUUSD + EURUSD
#
# PURPOSE:
#   1. Test the existing strategy across multiple markets.
#   2. Use walk-forward / out-of-sample testing.
#   3. Give greater importance to recent market regimes.
#   4. Keep older periods as robustness checks.
#
# IMPORTANT:
#   NO LIVE TRADING.
#   NO FUTURE DATA USED DURING TRAINING.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

MARKETS = {
    "XAUUSD": {
        "data_file": "data/XAUUSD_15m.csv",
        "output_file": "data/xauusd_optimizer_v4_results.csv",
    },
    "EURUSD": {
        "data_file": "data/EURUSD_15m.csv",
        "output_file": "data/eurusd_optimizer_v4_results.csv",
    },
}

COMBINED_OUTPUT_FILE = "data/multi_market_optimizer_v4_results.csv"


# ============================================================
# SEARCH SPACE
# ============================================================

RR_VALUES = [
    0.50,
    0.60,
    0.75,
    0.90,
    1.00,
    1.25,
    1.50,
]

WICK_VALUES = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
]

BODY_VALUES = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
]

SEPARATION_VALUES = [
    0.0005,
    0.0008,
    0.0010,
    0.0012,
    0.0015,
    0.0020,
]

MAX_CROSS_VALUES = [
    15,
    20,
    25,
    30,
    40,
]

HOUR_SETS = [
    (2, 3),
    (3, 4),
    (4, 5),
    (2, 3, 4),
    (3, 4, 5),
    (2, 3, 4, 5),
    (2, 3, 4, 5, 12, 13),
    (3, 4, 5, 12, 13),
]


# ============================================================
# WALK-FORWARD PERIODS
#
# The final two periods deliberately concentrate on the
# modern market regime.
# ============================================================

PERIODS = [
    {
        "name": "2020-2023 -> 2024",
        "train_start": "2020-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
        "era_weight": 1.00,
    },
    {
        "name": "2020-2024 -> 2025",
        "train_start": "2020-01-01",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
        "era_weight": 1.50,
    },
    {
        "name": "2020-2025 -> 2026",
        "train_start": "2020-01-01",
        "train_end": "2025-12-31",
        "test_start": "2026-01-01",
        "test_end": "2026-08-14",
        "era_weight": 2.00,
    },
]


# ============================================================
# DATA LOADING
# ============================================================

def load_data(market, config):

    print()
    print("=" * 60)
    print(f"{market} OPTIMIZER V4")
    print("=" * 60)

    data_file = config["data_file"]

    if not os.path.exists(data_file):
        raise RuntimeError(
            f"Missing file: {data_file}"
        )

    df = pd.read_csv(data_file)

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    time_column = None

    for column in [
        "time",
        "Time",
        "timestamp",
        "Timestamp",
        "date",
        "Date",
    ]:

        if column in df.columns:
            time_column = column
            break

    if time_column is None:
        raise RuntimeError(
            f"{market}: Could not find time column."
        )

    df[time_column] = pd.to_datetime(
        df[time_column],
        utc=True
    )

    df = df.set_index(
        time_column
    )

    rename = {}

    for column in df.columns:

        lower = column.lower()

        if lower == "open":
            rename[column] = "Open"

        elif lower == "high":
            rename[column] = "High"

        elif lower == "low":
            rename[column] = "Low"

        elif lower == "close":
            rename[column] = "Close"

    df = df.rename(
        columns=rename
    )

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
    ]:

        if column not in df.columns:
            raise RuntimeError(
                f"{market}: Missing column: {column}"
            )

    df = df.sort_index()

    print()
    print("Market:", market)
    print("Candles:", len(df))
    print(
        "Range:",
        df.index.min(),
        "->",
        df.index.max()
    )

    return df


# ============================================================
# PREPARE INDICATORS
# ============================================================

def prepare_data(df):

    print(
        "Preparing indicators..."
    )

    close = df["Close"].astype(float)
    open_price = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    close_np = close.to_numpy()
    open_np = open_price.to_numpy()
    high_np = high.to_numpy()
    low_np = low.to_numpy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    ema20 = (
        close
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
        .to_numpy()
    )

    ema50 = (
        close
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
        .to_numpy()
    )

    ema20_slope = (
        ema20
        -
        np.roll(
            ema20,
            4
        )
    )

    ema50_slope = (
        ema50
        -
        np.roll(
            ema50,
            4
        )
    )

    separation = (
        np.abs(
            ema20 - ema50
        )
        /
        close_np
    )

    # --------------------------------------------------------
    # CANDLE STRUCTURE
    # --------------------------------------------------------

    candle_range = (
        high_np - low_np
    )

    body = np.abs(
        close_np - open_np
    )

    body_ratio = np.divide(
        body,
        candle_range,
        out=np.zeros_like(body),
        where=candle_range > 0
    )

    upper_wick = (
        high_np
        -
        np.maximum(
            open_np,
            close_np
        )
    )

    upper_wick_ratio = np.divide(
        upper_wick,
        candle_range,
        out=np.zeros_like(
            upper_wick
        ),
        where=candle_range > 0
    )

    hours = (
        df.index.hour.to_numpy()
    )

    # --------------------------------------------------------
    # BEARISH CROSS AGE
    # --------------------------------------------------------

    bearish_cross = (
        (ema20[:-1] >= ema50[:-1])
        &
        (ema20[1:] < ema50[1:])
    )

    cross_age = np.full(
        len(df),
        9999,
        dtype=np.int32
    )

    last_cross = -9999

    for i in range(
        1,
        len(df)
    ):

        if bearish_cross[i - 1]:
            last_cross = i

        if last_cross >= 0:

            cross_age[i] = (
                i - last_cross
            )

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------

    previous_close = np.roll(
        close_np,
        1
    )

    previous_ema20 = np.roll(
        ema20,
        1
    )

    previous_distance = np.divide(
        np.abs(
            previous_close
            -
            previous_ema20
        ),
        previous_close,
        out=np.ones_like(
            previous_close
        ),
        where=previous_close != 0
    )

    current_distance = np.divide(
        np.abs(
            close_np
            -
            ema20
        ),
        close_np,
        out=np.ones_like(
            close_np
        ),
        where=close_np != 0
    )

    pullback = (
        (previous_distance <= 0.0020)
        |
        (current_distance <= 0.0020)
    )

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    recent_high = (
        pd.Series(high_np)
        .rolling(
            8,
            min_periods=1
        )
        .max()
        .to_numpy()
    )

    return {
        "close": close_np,
        "open": open_np,
        "high": high_np,
        "low": low_np,
        "ema20": ema20,
        "ema50": ema50,
        "ema20_slope": ema20_slope,
        "ema50_slope": ema50_slope,
        "separation": separation,
        "body_ratio": body_ratio,
        "upper_wick_ratio": upper_wick_ratio,
        "hours": hours,
        "cross_age": cross_age,
        "pullback": pullback,
        "recent_high": recent_high,
    }


# ============================================================
# BUILD SIGNALS
# ============================================================

def build_signals(
    data,
    wick,
    body,
    separation,
    max_cross,
    hours,
):

    mask = (
        np.isin(
            data["hours"],
            np.array(hours)
        )
        &
        (
            data["ema20"]
            <
            data["ema50"]
        )
        &
        (
            data["ema20_slope"]
            <
            0
        )
        &
        (
            data["ema50_slope"]
            <
            0
        )
        &
        (
            data["separation"]
            >=
            separation
        )
        &
        (
            data["cross_age"]
            <=
            max_cross
        )
        &
        data["pullback"]
        &
        (
            data["close"]
            <
            data["open"]
        )
        &
        (
            data["upper_wick_ratio"]
            >=
            wick
        )
        &
        (
            data["body_ratio"]
            >=
            body
        )
        &
        (
            data["close"]
            <
            data["ema20"]
        )
    )

    return np.flatnonzero(
        mask
    )


# ============================================================
# SIMULATE TRADES
# ============================================================

def simulate(
    data,
    indices,
    rr,
    start_index,
    end_index,
):

    close = data["close"]
    high = data["high"]
    low = data["low"]
    recent_high = data["recent_high"]

    trades = []

    next_available = start_index

    for index in indices:

        if index < start_index:
            continue

        if index > end_index:
            break

        if index < next_available:
            continue

        entry = close[index]

        stop = recent_high[index]

        risk = (
            stop - entry
        )

        if risk <= 0:
            continue

        target = (
            entry
            -
            risk * rr
        )

        result = None
        exit_index = None

        search_end = min(
            end_index,
            len(close) - 1
        )

        for j in range(
            index + 1,
            search_end + 1
        ):

            # Conservative:
            # if both levels occur inside
            # the same candle, SL wins.

            if high[j] >= stop:

                result = -1.0
                exit_index = j

                break

            if low[j] <= target:

                result = rr
                exit_index = j

                break

        if result is None:

            # Do not manufacture an outcome
            # for trades still open at the end
            # of a testing period.

            continue

        trades.append(
            result
        )

        next_available = (
            exit_index + 1
        )

    return trades


# ============================================================
# METRICS
# ============================================================

def metrics(
    trades,
    days
):

    if not trades:
        return None

    trades = np.asarray(
        trades,
        dtype=float
    )

    wins = (
        trades > 0
    )

    losses = (
        trades < 0
    )

    total = len(
        trades
    )

    win_count = int(
        wins.sum()
    )

    loss_count = int(
        losses.sum()
    )

    win_rate = (
        win_count
        /
        total
        *
        100
    )

    total_r = float(
        trades.sum()
    )

    gross_profit = float(
        trades[wins].sum()
    )

    gross_loss = abs(
        float(
            trades[losses].sum()
        )
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            /
            gross_loss
        )

    else:

        profit_factor = 999.0

    equity = np.cumsum(
        trades
    )

    peak = np.maximum.accumulate(
        equity
    )

    drawdown = (
        peak - equity
    )

    max_drawdown = float(
        drawdown.max()
    )

    weeks = (
        days / 7
    )

    if weeks > 0:

        trades_per_week = (
            total / weeks
        )

    else:

        trades_per_week = 0

    return {
        "trades": total,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": win_rate,
        "total_r": total_r,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "trades_per_week": trades_per_week,
    }


# ============================================================
# RECENCY WEIGHT
#
# Older trades still matter.
# Recent trades matter more.
# ============================================================

def calculate_recency_weight(
    timestamp,
    train_start,
    train_end
):

    total_seconds = (
        train_end - train_start
    ).total_seconds()

    elapsed_seconds = (
        timestamp - train_start
    ).total_seconds()

    if total_seconds <= 0:
        return 1.0

    progress = (
        elapsed_seconds
        /
        total_seconds
    )

    progress = max(
        0.0,
        min(
            1.0,
            progress
        )
    )

    # Older history:
    # approximately 1.0x
    #
    # Most recent history:
    # approximately 2.0x

    return (
        1.0
        +
        progress
    )


# ============================================================
# TRAINING SCORE V4
#
# Unlike V3, this score:
#
#   - rewards positive R
#   - rewards PF
#   - penalises drawdown
#   - requires enough trades
#   - rewards useful frequency
#   - favours recent-era performance
# ============================================================

def training_score(
    metrics_full,
    metrics_recent,
    era_weight
):

    if metrics_full is None:
        return -999999999

    if metrics_recent is None:
        return -999999999

    if metrics_full["trades"] < 30:
        return -999999999

    if metrics_recent["trades"] < 8:
        return -999999999

    score = 0.0

    # --------------------------------------------------------
    # FULL HISTORY
    # --------------------------------------------------------

    score += (
        metrics_full["win_rate"]
        * 5.0
    )

    score += (
        metrics_full["total_r"]
        * 2.0
    )

    score += (
        min(
            metrics_full["profit_factor"],
            3.0
        )
        * 10.0
    )

    score -= (
        metrics_full["max_drawdown"]
        * 2.0
    )

    # --------------------------------------------------------
    # RECENT ERA
    #
    # This is deliberately stronger.
    # --------------------------------------------------------

    score += (
        metrics_recent["win_rate"]
        *
        8.0
        *
        era_weight
    )

    score += (
        metrics_recent["total_r"]
        *
        3.0
        *
        era_weight
    )

    score += (
        min(
            metrics_recent["profit_factor"],
            3.0
        )
        *
        15.0
        *
        era_weight
    )

    score -= (
        metrics_recent["max_drawdown"]
        *
        3.0
        *
        era_weight
    )

    # --------------------------------------------------------
    # FREQUENCY
    # --------------------------------------------------------

    frequency = (
        metrics_recent["trades_per_week"]
    )

    if (
        0.20
        <= frequency
        <= 1.50
    ):

        score += 30

    elif (
        0.10
        <= frequency
        <= 2.00
    ):

        score += 10

    return score


# ============================================================
# FIND INDEX RANGE
# ============================================================

def get_index_range(
    timestamps,
    start_timestamp,
    end_timestamp
):

    start_mask = (
        timestamps
        >=
        start_timestamp
    )

    end_mask = (
        timestamps
        <=
        end_timestamp
    )

    start_positions = np.flatnonzero(
        start_mask
    )

    end_positions = np.flatnonzero(
        end_mask
    )

    if (
        len(start_positions) == 0
        or
        len(end_positions) == 0
    ):
        return None, None

    return (
        int(start_positions[0]),
        int(end_positions[-1])
    )


# ============================================================
# OPTIMISE TRAINING
# ============================================================

def optimise_training(
    data,
    train_start,
    train_end,
    timestamps,
):

    train_start_index, train_end_index = (
        get_index_range(
            timestamps,
            train_start,
            train_end
        )
    )

    if train_start_index is None:
        return None

    train_days = (
        timestamps[train_end_index]
        -
        timestamps[train_start_index]
    ).total_seconds() / 86400

    # --------------------------------------------------------
    # RECENT ERA
    #
    # Last 2 years of the training window.
    # --------------------------------------------------------

    recent_start = max(
        train_start,
        train_end
        -
        pd.Timedelta(days=730)
    )

    recent_start_index, recent_end_index = (
        get_index_range(
            timestamps,
            recent_start,
            train_end
        )
    )

    if recent_start_index is None:
        return None

    recent_days = (
        timestamps[recent_end_index]
        -
        timestamps[recent_start_index]
    ).total_seconds() / 86400

    combinations = product(
        RR_VALUES,
        WICK_VALUES,
        BODY_VALUES,
        SEPARATION_VALUES,
        MAX_CROSS_VALUES,
        HOUR_SETS,
    )

    best = None

    for params in combinations:

        (
            rr,
            wick,
            body,
            separation,
            max_cross,
            hours,
        ) = params

        signals = build_signals(
            data,
            wick,
            body,
            separation,
            max_cross,
            hours,
        )

        # ----------------------------------------------------
        # FULL TRAINING
        # ----------------------------------------------------

        full_trades = simulate(
            data,
            signals,
            rr,
            train_start_index,
            train_end_index,
        )

        full_metrics = metrics(
            full_trades,
            train_days
        )

        if full_metrics is None:
            continue

        # ----------------------------------------------------
        # RECENT TRAINING
        # ----------------------------------------------------

        recent_trades = simulate(
            data,
            signals,
            rr,
            recent_start_index,
            recent_end_index,
        )

        recent_metrics = metrics(
            recent_trades,
            recent_days
        )

        if recent_metrics is None:
            continue

        score = training_score(
            full_metrics,
            recent_metrics,
            2.0
        )

        if (
            best is None
            or
            score > best["score"]
        ):

            best = {
                "score": score,

                "rr": rr,
                "wick": wick,
                "body": body,
                "separation": separation,
                "max_cross": max_cross,
                "hours": hours,

                "metrics": full_metrics,
                "recent_metrics": recent_metrics,
            }

    return best


# ============================================================
# RUN ONE WALK-FORWARD PERIOD
# ============================================================

def run_period(
    market,
    period,
    data,
    timestamps,
):

    print()
    print("=" * 60)
    print(
        "WALK-FORWARD:",
        period["name"]
    )
    print("=" * 60)

    train_start = pd.Timestamp(
        period["train_start"],
        tz="UTC"
    )

    train_end = pd.Timestamp(
        period["train_end"],
        tz="UTC"
    )

    test_start = pd.Timestamp(
        period["test_start"],
        tz="UTC"
    )

    test_end = pd.Timestamp(
        period["test_end"],
        tz="UTC"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print(
        "Optimising training period..."
    )

    best = optimise_training(
        data,
        train_start,
        train_end,
        timestamps,
    )

    if best is None:

        print(
            "No valid training strategy."
        )

        return None

    print()
    print(
        "BEST TRAINING STRATEGY"
    )
    print("-" * 60)

    print(
        "Training win rate:",
        f"{best['metrics']['win_rate']:.2f}%"
    )

    print(
        "Training trades:",
        best["metrics"]["trades"]
    )

    print(
        "Training total R:",
        f"{best['metrics']['total_r']:.2f}"
    )

    print(
        "Recent-era win rate:",
        f"{best['recent_metrics']['win_rate']:.2f}%"
    )

    print(
        "Recent-era trades:",
        best["recent_metrics"]["trades"]
    )

    print(
        "Recent-era total R:",
        f"{best['recent_metrics']['total_r']:.2f}"
    )

    print(
        "Recent-era PF:",
        f"{best['recent_metrics']['profit_factor']:.2f}"
    )

    print()
    print("Parameters:")

    print(
        "RR:",
        best["rr"]
    )

    print(
        "Wick:",
        best["wick"]
    )

    print(
        "Body:",
        best["body"]
    )

    print(
        "Separation:",
        best["separation"]
    )

    print(
        "Max cross:",
        best["max_cross"]
    )

    print(
        "Hours:",
        ",".join(
            str(x)
            for x in best["hours"]
        )
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_start_index, test_end_index = (
        get_index_range(
            timestamps,
            test_start,
            test_end
        )
    )

    if test_start_index is None:

        print(
            "No test data available."
        )

        return None

    test_days = (
        timestamps[test_end_index]
        -
        timestamps[test_start_index]
    ).total_seconds() / 86400

    signals = build_signals(
        data,
        best["wick"],
        best["body"],
        best["separation"],
        best["max_cross"],
        best["hours"],
    )

    test_trades = simulate(
        data,
        signals,
        best["rr"],
        test_start_index,
        test_end_index,
    )

    test_metrics = metrics(
        test_trades,
        test_days
    )

    if test_metrics is None:

        print()
        print(
            "NO OUT-OF-SAMPLE TRADES"
        )

        return {
            "market": market,
            "period": period["name"],
            "train_trades":
                best["metrics"]["trades"],
            "train_win_rate":
                best["metrics"]["win_rate"],
            "train_total_r":
                best["metrics"]["total_r"],
            "recent_trades":
                best["recent_metrics"]["trades"],
            "recent_win_rate":
                best["recent_metrics"]["win_rate"],
            "recent_total_r":
                best["recent_metrics"]["total_r"],
            "recent_profit_factor":
                best["recent_metrics"]["profit_factor"],
            "test_trades": 0,
            "test_wins": 0,
            "test_losses": 0,
            "test_win_rate": 0,
            "test_total_r": 0,
            "test_profit_factor": 0,
            "test_drawdown": 0,
            "test_trades_per_week": 0,
            "rr": best["rr"],
            "wick": best["wick"],
            "body": best["body"],
            "separation": best["separation"],
            "max_cross": best["max_cross"],
            "hours": ",".join(
                str(x)
                for x in best["hours"]
            ),
        }

    print()
    print(
        "OUT-OF-SAMPLE RESULT"
    )
    print("-" * 60)

    print(
        "Win rate:",
        f"{test_metrics['win_rate']:.2f}%"
    )

    print(
        "Trades:",
        test_metrics["trades"]
    )

    print(
        "Total R:",
        f"{test_metrics['total_r']:.2f}"
    )

    print(
        "Profit factor:",
        f"{test_metrics['profit_factor']:.2f}"
    )

    print(
        "Max drawdown:",
        f"{test_metrics['max_drawdown']:.2f}R"
    )

    print(
        "Trades/week:",
        f"{test_metrics['trades_per_week']:.2f}"
    )

    return {
        "market": market,
        "period": period["name"],

        "train_trades":
            best["metrics"]["trades"],

        "train_win_rate":
            best["metrics"]["win_rate"],

        "train_total_r":
            best["metrics"]["total_r"],

        "recent_trades":
            best["recent_metrics"]["trades"],

        "recent_win_rate":
            best["recent_metrics"]["win_rate"],

        "recent_total_r":
            best["recent_metrics"]["total_r"],

        "recent_profit_factor":
            best["recent_metrics"]["profit_factor"],

        "test_trades":
            test_metrics["trades"],

        "test_wins":
            test_metrics["wins"],

        "test_losses":
            test_metrics["losses"],

        "test_win_rate":
            test_metrics["win_rate"],

        "test_total_r":
            test_metrics["total_r"],

        "test_profit_factor":
            test_metrics["profit_factor"],

        "test_drawdown":
            test_metrics["max_drawdown"],

        "test_trades_per_week":
            test_metrics["trades_per_week"],

        "rr":
            best["rr"],

        "wick":
            best["wick"],

        "body":
            best["body"],

        "separation":
            best["separation"],

        "max_cross":
            best["max_cross"],

        "hours":
            ",".join(
                str(x)
                for x in best["hours"]
            ),
    }


# ============================================================
# MARKET VERDICT
# ============================================================

def market_verdict(
    results_df
):

    if results_df.empty:
        return "NO DATA"

    profitable_periods = int(
        (
            results_df["test_total_r"]
            > 0
        ).sum()
    )

    total_periods = len(
        results_df
    )

    total_trades = int(
        results_df["test_trades"]
        .sum()
    )

    total_wins = int(
        results_df["test_wins"]
        .sum()
    )

    total_r = float(
        results_df["test_total_r"]
        .sum()
    )

    if total_trades > 0:

        win_rate = (
            total_wins
            /
            total_trades
            *
            100
        )

    else:

        win_rate = 0

    # Stronger than V3:
    #
    # We want:
    # - positive total R
    # - majority profitable periods
    # - at least 55% OOS win rate
    # - enough trades to have meaning

    if (
        total_r > 0
        and
        profitable_periods
        >=
        max(
            2,
            int(
                np.ceil(
                    total_periods
                    *
                    0.66
                )
            )
        )
        and
        win_rate >= 55
        and
        total_trades >= 20
    ):

        return "ROBUST"

    if (
        total_r > 0
        and
        win_rate >= 50
    ):

        return "PROMISING"

    return "NOT ROBUST"


# ============================================================
# RUN ONE MARKET
# ============================================================

def run_market(
    market,
    config
):

    df = load_data(
        market,
        config
    )

    data = prepare_data(
        df
    )

    timestamps = (
        df.index.to_numpy()
    )

    print()
    print("=" * 60)
    print(
        f"{market} WALK-FORWARD TESTING"
    )
    print("=" * 60)

    print(
        "Periods:",
        len(PERIODS)
    )

    results = []

    for period in PERIODS:

        result = run_period(
            market,
            period,
            data,
            timestamps,
        )

        if result is not None:

            results.append(
                result
            )

    if not results:

        raise RuntimeError(
            f"{market}: No walk-forward results."
        )

    results_df = pd.DataFrame(
        results
    )

    os.makedirs(
        "data",
        exist_ok=True
    )

    results_df.to_csv(
        config["output_file"],
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        f"{market} WALK-FORWARD SUMMARY"
    )
    print("=" * 60)

    summary_columns = [
        "period",
        "train_trades",
        "train_win_rate",
        "recent_trades",
        "recent_win_rate",
        "recent_total_r",
        "test_trades",
        "test_win_rate",
        "test_total_r",
        "test_profit_factor",
        "test_drawdown",
        "test_trades_per_week",
    ]

    print(
        results_df[
            summary_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # COMBINED OOS
    # --------------------------------------------------------

    total_test_trades = int(
        results_df[
            "test_trades"
        ].sum()
    )

    total_test_wins = int(
        results_df[
            "test_wins"
        ].sum()
    )

    total_test_losses = int(
        results_df[
            "test_losses"
        ].sum()
    )

    total_test_r = float(
        results_df[
            "test_total_r"
        ].sum()
    )

    if total_test_trades > 0:

        combined_win_rate = (
            total_test_wins
            /
            total_test_trades
            *
            100
        )

    else:

        combined_win_rate = 0

    gross_profit = 0
    gross_loss = 0

    for _, row in results_df.iterrows():

        wins = row[
            "test_wins"
        ]

        losses = row[
            "test_losses"
        ]

        rr = row[
            "rr"
        ]

        gross_profit += (
            wins * rr
        )

        gross_loss += (
            losses * 1.0
        )

    if gross_loss > 0:

        combined_pf = (
            gross_profit
            /
            gross_loss
        )

    else:

        combined_pf = 999

    profitable_periods = int(
        (
            results_df[
                "test_total_r"
            ]
            > 0
        ).sum()
    )

    print()
    print("=" * 60)
    print(
        f"{market} COMBINED OUT-OF-SAMPLE"
    )
    print("=" * 60)

    print(
        "Total test trades:",
        total_test_trades
    )

    print(
        "Wins:",
        total_test_wins
    )

    print(
        "Losses:",
        total_test_losses
    )

    print(
        "Combined win rate:",
        f"{combined_win_rate:.2f}%"
    )

    print(
        "Combined total R:",
        f"{total_test_r:.2f}"
    )

    print(
        "Approx profit factor:",
        f"{combined_pf:.2f}"
    )

    print(
        "Profitable periods:",
        f"{profitable_periods}/{len(results_df)}"
    )

    verdict = market_verdict(
        results_df
    )

    print()
    print("=" * 60)
    print(
        f"{market} VERDICT"
    )
    print("=" * 60)

    print(
        "VERDICT:",
        verdict
    )

    if verdict == "ROBUST":

        print(
            "This market has passed the "
            "current preliminary robustness filter."
        )

    elif verdict == "PROMISING":

        print(
            "This market shows potential "
            "but requires further testing."
        )

    else:

        print(
            "Do not implement this version live."
        )

    print()
    print(
        "Results saved to:",
        config["output_file"]
    )

    return results_df


# ============================================================
# COMPARE MARKETS
# ============================================================

def compare_markets(
    market_results
):

    rows = []

    for market, df in market_results.items():

        total_trades = int(
            df["test_trades"].sum()
        )

        total_wins = int(
            df["test_wins"].sum()
        )

        total_r = float(
            df["test_total_r"].sum()
        )

        profitable_periods = int(
            (
                df["test_total_r"]
                > 0
            ).sum()
        )

        if total_trades > 0:

            win_rate = (
                total_wins
                /
                total_trades
                *
                100
            )

        else:

            win_rate = 0

        rows.append(
            {
                "market": market,
                "test_trades": total_trades,
                "test_win_rate": win_rate,
                "test_total_r": total_r,
                "profitable_periods":
                    profitable_periods,
                "periods":
                    len(df),
                "verdict":
                    market_verdict(df),
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    comparison.to_csv(
        COMBINED_OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 60)
    print(
        "MULTI-MARKET COMPARISON"
    )
    print("=" * 60)

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print("=" * 60)
    print(
        "IMPORTANT"
    )
    print("=" * 60)

    print(
        "A strategy is NOT considered validated "
        "just because one market performs well."
    )

    print(
        "We want an edge that survives different "
        "market regimes and preferably different instruments."
    )

    print()
    print(
        "Comparison saved to:",
        COMBINED_OUTPUT_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "MULTI-MARKET STRATEGY OPTIMIZER V4"
    )
    print("=" * 60)

    print()
    print(
        "Current-era weighting: ENABLED"
    )

    print(
        "Markets:",
        ", ".join(
            MARKETS.keys()
        )
    )

    print()
    print(
        "NO LIVE TRADING"
    )

    market_results = {}

    for market, config in MARKETS.items():

        try:

            results_df = run_market(
                market,
                config
            )

            market_results[
                market
            ] = results_df

        except Exception as error:

            print()
            print("=" * 60)
            print(
                f"{market} FAILED"
            )
            print("=" * 60)

            print(
                type(error).__name__,
                ":",
                error
            )

    if not market_results:

        raise RuntimeError(
            "No markets completed successfully."
        )

    compare_markets(
        market_results
    )

    print()
    print("=" * 60)
    print(
        "OPTIMIZER V4 COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
