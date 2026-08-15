import os
import itertools
import numpy as np
import pandas as pd


# ============================================================
# MULTI-MARKET SIGNAL QUALITY OPTIMIZER V7
# ============================================================
#
# GOAL:
# Find highly selective, repeatable signals.
#
# TARGET:
# ~82% win rate over 200+ genuinely OOS trades
#
# IMPORTANT:
# - Signals only
# - No live trading
# - Walk-forward testing
# - XAUUSD + EURUSD
# - Winner/loser feature analysis
# - Quality scoring
# - Threshold optimisation
# ============================================================


MARKETS = {
    "XAUUSD": "data/XAUUSD_15m.csv",
    "EURUSD": "data/EURUSD_15m.csv",
}


# ============================================================
# WALK-FORWARD PERIODS
# ============================================================

PERIODS = [
    (
        "2021-2023 -> 2024",
        "2021-01-01",
        "2023-12-31",
        "2024-01-01",
        "2024-12-31",
    ),
    (
        "2022-2024 -> 2025",
        "2022-01-01",
        "2024-12-31",
        "2025-01-01",
        "2025-12-31",
    ),
    (
        "2023-2025 -> 2026",
        "2023-01-01",
        "2025-12-31",
        "2026-01-01",
        "2026-08-14",
    ),
]


# ============================================================
# BASE ENTRY PARAMETERS
# ============================================================

RR_VALUES = [
    0.50,
    0.60,
    0.75,
    0.90,
    1.00,
    1.25,
]


WICK_VALUES = [
    0.20,
    0.25,
    0.30,
    0.35,
]


BODY_VALUES = [
    0.20,
    0.25,
    0.30,
    0.35,
]


SEPARATION_VALUES = [
    0.0005,
    0.0008,
    0.0010,
]


MAX_CROSS_VALUES = [
    20,
    30,
    40,
]


HOUR_SETS = [
    (2, 3, 4),
    (3, 4, 5),
    (2, 3, 4, 5),
    (3, 4, 5, 12, 13),
    (2, 3, 4, 5, 12, 13),
]


# ============================================================
# QUALITY SCORE
# ============================================================

SCORE_THRESHOLDS = [
    4,
    5,
    6,
    7,
    8,
    9,
    10,
]


# ============================================================
# REQUIREMENTS
# ============================================================

MIN_TRAIN_TRADES = 30
MIN_TEST_TRADES = 5

# We don't want a strategy getting rewarded
# simply because it takes almost no trades.
MIN_TOTAL_OOS_TARGET = 200

# Recent data receives additional weight.
RECENT_DAYS = 365


# ============================================================
# TIME
# ============================================================

def utc(value):

    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")


def get_bounds(index, start, end):

    start = utc(start)
    end = utc(end)

    left = np.searchsorted(
        index,
        start,
        side="left",
    )

    right = np.searchsorted(
        index,
        end,
        side="right",
    ) - 1

    if left >= len(index):
        return None

    if right < left:
        return None

    return int(left), int(right)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path):

    if not os.path.exists(path):

        raise RuntimeError(
            f"Missing data file: {path}"
        )

    df = pd.read_csv(path)

    df.columns = [
        str(x).strip()
        for x in df.columns
    ]

    time_col = None

    for col in [
        "time",
        "Time",
        "timestamp",
        "Timestamp",
        "date",
        "Date",
    ]:

        if col in df.columns:

            time_col = col
            break

    if time_col is None:

        raise RuntimeError(
            "Could not find timestamp column."
        )

    df[time_col] = pd.to_datetime(
        df[time_col],
        utc=True,
    )

    df = df.set_index(
        time_col
    )

    rename = {}

    for col in df.columns:

        name = str(col).lower()

        if name == "open":
            rename[col] = "Open"

        elif name == "high":
            rename[col] = "High"

        elif name == "low":
            rename[col] = "Low"

        elif name == "close":
            rename[col] = "Close"

    df = df.rename(
        columns=rename
    )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    for col in required:

        if col not in df.columns:

            raise RuntimeError(
                f"Missing column: {col}"
            )

    df = (
        df[required]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .dropna()
    )

    df = (
        df[
            ~df.index.duplicated(
                keep="first"
            )
        ]
        .sort_index()
    )

    return df


# ============================================================
# INDICATORS
# ============================================================

def prepare_indicators(df):

    o = df["Open"].to_numpy(
        dtype=float
    )

    h = df["High"].to_numpy(
        dtype=float
    )

    l = df["Low"].to_numpy(
        dtype=float
    )

    c = df["Close"].to_numpy(
        dtype=float
    )

    index = df.index

    # --------------------------------------------------------
    # EMAs
    # --------------------------------------------------------

    ema20 = (
        pd.Series(c)
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
        .to_numpy()
    )

    ema50 = (
        pd.Series(c)
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
        .to_numpy()
    )

    ema100 = (
        pd.Series(c)
        .ewm(
            span=100,
            adjust=False,
        )
        .mean()
        .to_numpy()
    )

    ema200 = (
        pd.Series(c)
        .ewm(
            span=200,
            adjust=False,
        )
        .mean()
        .to_numpy()
    )

    # --------------------------------------------------------
    # EMA slopes
    # --------------------------------------------------------

    ema20_slope = (
        ema20
        - np.roll(ema20, 4)
    ) / np.where(
        ema20 == 0,
        1,
        ema20,
    )

    ema50_slope = (
        ema50
        - np.roll(ema50, 4)
    ) / np.where(
        ema50 == 0,
        1,
        ema50,
    )

    ema100_slope = (
        ema100
        - np.roll(ema100, 8)
    ) / np.where(
        ema100 == 0,
        1,
        ema100,
    )

    # --------------------------------------------------------
    # Candle structure
    # --------------------------------------------------------

    candle_range = h - l

    body = np.abs(
        c - o
    )

    body_ratio = np.divide(
        body,
        candle_range,
        out=np.zeros_like(c),
        where=candle_range > 0,
    )

    upper_wick = (
        h
        - np.maximum(o, c)
    )

    lower_wick = (
        np.minimum(o, c)
        - l
    )

    upper_wick_ratio = np.divide(
        upper_wick,
        candle_range,
        out=np.zeros_like(c),
        where=candle_range > 0,
    )

    lower_wick_ratio = np.divide(
        lower_wick,
        candle_range,
        out=np.zeros_like(c),
        where=candle_range > 0,
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = np.roll(
        c,
        1,
    )

    true_range = np.maximum.reduce(
        [
            h - l,
            np.abs(
                h - previous_close
            ),
            np.abs(
                l - previous_close
            ),
        ]
    )

    atr = (
        pd.Series(
            true_range
        )
        .rolling(
            14,
            min_periods=14,
        )
        .mean()
        .to_numpy()
    )

    atr_average = (
        pd.Series(atr)
        .rolling(
            100,
            min_periods=30,
        )
        .mean()
        .to_numpy()
    )

    atr_ratio = np.divide(
        atr,
        atr_average,
        out=np.ones_like(c),
        where=atr_average > 0,
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    momentum_4 = (
        c
        - np.roll(c, 4)
    ) / np.where(
        np.roll(c, 4) == 0,
        1,
        np.roll(c, 4),
    )

    momentum_8 = (
        c
        - np.roll(c, 8)
    ) / np.where(
        np.roll(c, 8) == 0,
        1,
        np.roll(c, 8),
    )

    momentum_16 = (
        c
        - np.roll(c, 16)
    ) / np.where(
        np.roll(c, 16) == 0,
        1,
        np.roll(c, 16),
    )

    # --------------------------------------------------------
    # EMA distances
    # --------------------------------------------------------

    distance_20 = (
        c - ema20
    ) / np.where(
        c == 0,
        1,
        c,
    )

    distance_50 = (
        c - ema50
    ) / np.where(
        c == 0,
        1,
        c,
    )

    separation = np.abs(
        ema20 - ema50
    ) / np.where(
        c == 0,
        1,
        c,
    )

    separation_50_100 = np.abs(
        ema50 - ema100
    ) / np.where(
        c == 0,
        1,
        c,
    )

    # --------------------------------------------------------
    # Recent highs / lows
    # --------------------------------------------------------

    recent_high_8 = (
        pd.Series(h)
        .rolling(
            8,
            min_periods=1,
        )
        .max()
        .to_numpy()
    )

    recent_low_8 = (
        pd.Series(l)
        .rolling(
            8,
            min_periods=1,
        )
        .min()
        .to_numpy()
    )

    recent_high_16 = (
        pd.Series(h)
        .rolling(
            16,
            min_periods=1,
        )
        .max()
        .to_numpy()
    )

    recent_low_16 = (
        pd.Series(l)
        .rolling(
            16,
            min_periods=1,
        )
        .min()
        .to_numpy()
    )

    distance_recent_high = (
        recent_high_8 - c
    ) / np.where(
        c == 0,
        1,
        c,
    )

    distance_recent_low = (
        c - recent_low_8
    ) / np.where(
        c == 0,
        1,
        c,
    )

    # --------------------------------------------------------
    # Cross age
    # --------------------------------------------------------

    cross_age = np.full(
        len(c),
        9999,
        dtype=np.int32,
    )

    last_cross = -9999

    for i in range(
        1,
        len(c),
    ):

        crossed = (
            ema20[i - 1]
            >= ema50[i - 1]
            and
            ema20[i]
            < ema50[i]
        )

        if crossed:

            last_cross = i

        if last_cross >= 0:

            cross_age[i] = (
                i - last_cross
            )

    # --------------------------------------------------------
    # Candle direction history
    # --------------------------------------------------------

    bearish = (
        c < o
    )

    bullish = (
        c > o
    )

    bearish_count_3 = (
        pd.Series(
            bearish.astype(int)
        )
        .rolling(
            3,
            min_periods=1,
        )
        .sum()
        .to_numpy()
    )

    bearish_count_5 = (
        pd.Series(
            bearish.astype(int)
        )
        .rolling(
            5,
            min_periods=1,
        )
        .sum()
        .to_numpy()
    )

    bullish_count_3 = (
        pd.Series(
            bullish.astype(int)
        )
        .rolling(
            3,
            min_periods=1,
        )
        .sum()
        .to_numpy()
    )

    # --------------------------------------------------------
    # Volatility change
    # --------------------------------------------------------

    atr_previous = np.roll(
        atr,
        20,
    )

    volatility_change = np.divide(
        atr - atr_previous,
        atr_previous,
        out=np.zeros_like(c),
        where=atr_previous != 0,
    )

    # --------------------------------------------------------
    # Hour
    # --------------------------------------------------------

    hours = (
        index.hour.to_numpy()
    )

    return {
        "open": o,
        "high": h,
        "low": l,
        "close": c,

        "ema20": ema20,
        "ema50": ema50,
        "ema100": ema100,
        "ema200": ema200,

        "ema20_slope":
            ema20_slope,

        "ema50_slope":
            ema50_slope,

        "ema100_slope":
            ema100_slope,

        "body_ratio":
            body_ratio,

        "upper_wick_ratio":
            upper_wick_ratio,

        "lower_wick_ratio":
            lower_wick_ratio,

        "atr": atr,

        "atr_ratio":
            atr_ratio,

        "momentum_4":
            momentum_4,

        "momentum_8":
            momentum_8,

        "momentum_16":
            momentum_16,

        "distance_20":
            distance_20,

        "distance_50":
            distance_50,

        "separation":
            separation,

        "separation_50_100":
            separation_50_100,

        "recent_high_8":
            recent_high_8,

        "recent_low_8":
            recent_low_8,

        "recent_high_16":
            recent_high_16,

        "recent_low_16":
            recent_low_16,

        "distance_recent_high":
            distance_recent_high,

        "distance_recent_low":
            distance_recent_low,

        "cross_age":
            cross_age,

        "bearish_count_3":
            bearish_count_3,

        "bearish_count_5":
            bearish_count_5,

        "bullish_count_3":
            bullish_count_3,

        "volatility_change":
            volatility_change,

        "hours":
            hours,
    }


# ============================================================
# BASE SETUPS
# ============================================================

def generate_setups(
    d,
    params,
):

    (
        rr,
        wick,
        body,
        separation,
        max_cross,
        hours,
    ) = params

    mask = (

        np.isin(
            d["hours"],
            hours,
        )

        & (
            d["ema20"]
            < d["ema50"]
        )

        & (
            d["ema50"]
            < d["ema100"]
        )

        & (
            d["ema100"]
            < d["ema200"]
        )

        & (
            d["ema20_slope"]
            < 0
        )

        & (
            d["ema50_slope"]
            < 0
        )

        & (
            d["separation"]
            >= separation
        )

        & (
            d["cross_age"]
            <= max_cross
        )

        & (
            d["close"]
            < d["open"]
        )

        & (
            d["upper_wick_ratio"]
            >= wick
        )

        & (
            d["body_ratio"]
            >= body
        )

        & (
            d["close"]
            < d["ema20"]
        )
    )

    return np.flatnonzero(
        mask
    )


# ============================================================
# FEATURE SCORE
# ============================================================

def calculate_quality_score(
    d,
    indices,
):

    scores = np.zeros(
        len(indices),
        dtype=float,
    )

    if len(indices) == 0:
        return scores

    # --------------------------------------------------------
    # 1. Strong EMA alignment
    # --------------------------------------------------------

    scores += (
        d["ema20"][indices]
        < d["ema50"][indices]
    ) * 1

    scores += (
        d["ema50"][indices]
        < d["ema100"][indices]
    ) * 1

    scores += (
        d["ema100"][indices]
        < d["ema200"][indices]
    ) * 1

    # --------------------------------------------------------
    # 2. Stronger EMA separation
    # --------------------------------------------------------

    scores += (
        d["separation"][indices]
        >= 0.0005
    ) * 1

    scores += (
        d["separation"][indices]
        >= 0.0010
    ) * 1

    # --------------------------------------------------------
    # 3. Trend slope
    # --------------------------------------------------------

    scores += (
        d["ema20_slope"][indices]
        < -0.0003
    ) * 1

    scores += (
        d["ema50_slope"][indices]
        < -0.0002
    ) * 1

    # --------------------------------------------------------
    # 4. Candle quality
    # --------------------------------------------------------

    scores += (
        d["body_ratio"][indices]
        >= 0.25
    ) * 1

    scores += (
        d["body_ratio"][indices]
        >= 0.35
    ) * 1

    scores += (
        d["upper_wick_ratio"][indices]
        >= 0.30
    ) * 1

    # --------------------------------------------------------
    # 5. Momentum
    # --------------------------------------------------------

    scores += (
        d["momentum_4"][indices]
        < 0
    ) * 1

    scores += (
        d["momentum_8"][indices]
        < -0.0005
    ) * 1

    # --------------------------------------------------------
    # 6. Price relative to EMA
    # --------------------------------------------------------

    scores += (
        d["distance_20"][indices]
        < -0.0005
    ) * 1

    scores += (
        d["distance_50"][indices]
        < -0.0010
    ) * 1

    # --------------------------------------------------------
    # 7. ATR regime
    # --------------------------------------------------------

    scores += (
        d["atr_ratio"][indices]
        >= 0.90
    ) * 1

    scores += (
        d["atr_ratio"][indices]
        <= 1.80
    ) * 1

    # --------------------------------------------------------
    # 8. Avoid excessive volatility expansion
    # --------------------------------------------------------

    scores += (
        d["volatility_change"][indices]
        < 0.50
    ) * 1

    # --------------------------------------------------------
    # 9. Avoid being too close to recent low
    # --------------------------------------------------------

    scores += (
        d["distance_recent_low"][indices]
        > 0.0005
    ) * 1

    # --------------------------------------------------------
    # 10. Fresh trend
    # --------------------------------------------------------

    scores += (
        d["cross_age"][indices]
        <= 40
    ) * 1

    # --------------------------------------------------------
    # 11. Candle sequence
    # --------------------------------------------------------

    scores += (
        d["bearish_count_3"][indices]
        >= 1
    ) * 1

    scores += (
        d["bearish_count_5"][indices]
        >= 2
    ) * 1

    return scores


# ============================================================
# SIMULATE TRADES
# ============================================================

def simulate(
    d,
    signal_indices,
    rr,
    start_idx,
    end_idx,
):

    trades = []

    next_available = start_idx

    for i in signal_indices:

        if i < start_idx:
            continue

        if i > end_idx:
            break

        if i < next_available:
            continue

        entry = d["close"][i]

        atr = d["atr"][i]

        if not np.isfinite(atr):
            continue

        risk = atr

        if risk <= 0:
            continue

        stop = (
            entry + risk
        )

        target = (
            entry
            - risk * rr
        )

        result = None
        exit_index = None

        for j in range(
            i + 1,
            end_idx + 1,
        ):

            stop_hit = (
                d["high"][j]
                >= stop
            )

            target_hit = (
                d["low"][j]
                <= target
            )

            # Conservative assumption:
            # if both occur in the same candle,
            # assume the stop was hit first.
            if (
                stop_hit
                and target_hit
            ):

                result = -1.0
                exit_index = j
                break

            if stop_hit:

                result = -1.0
                exit_index = j
                break

            if target_hit:

                result = rr
                exit_index = j
                break

        if result is not None:

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
    start_time,
    end_time,
):

    if not trades:
        return None

    values = np.asarray(
        trades,
        dtype=float,
    )

    wins = values > 0
    losses = values < 0

    count = len(
        values
    )

    win_count = int(
        wins.sum()
    )

    gross_profit = float(
        values[wins].sum()
    )

    gross_loss = abs(
        float(
            values[losses].sum()
        )
    )

    profit_factor = (
        gross_profit
        / gross_loss
        if gross_loss > 0
        else 999.0
    )

    equity = np.cumsum(
        values
    )

    peak = np.maximum.accumulate(
        equity
    )

    drawdown = float(
        np.max(
            peak - equity
        )
    )

    losing_streak = 0
    longest_losing_streak = 0

    for value in values:

        if value < 0:

            losing_streak += 1

            longest_losing_streak = max(
                longest_losing_streak,
                losing_streak,
            )

        else:

            losing_streak = 0

    start = pd.Timestamp(
        start_time
    )

    end = pd.Timestamp(
        end_time
    )

    days = max(
        (
            end - start
        ).total_seconds()
        / 86400.0,
        1,
    )

    trades_per_week = (
        count
        / (days / 7)
    )

    return {
        "trades":
            count,

        "wins":
            win_count,

        "losses":
            int(
                losses.sum()
            ),

        "win_rate":
            win_count
            / count
            * 100,

        "total_r":
            float(
                values.sum()
            ),

        "profit_factor":
            profit_factor,

        "drawdown":
            drawdown,

        "longest_losing_streak":
            longest_losing_streak,

        "trades_per_week":
            trades_per_week,
    }


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    d,
    timestamps,
    candidates,
    rr,
    start,
    end,
):

    bounds = get_bounds(
        timestamps,
        start,
        end,
    )

    if bounds is None:
        return None

    start_idx, end_idx = bounds

    valid = candidates[
        (candidates >= start_idx)
        & (candidates <= end_idx)
    ]

    trades = simulate(
        d,
        valid,
        rr,
        start_idx,
        end_idx,
    )

    return metrics(
        trades,
        timestamps[start_idx],
        timestamps[end_idx],
    )


# ============================================================
# BASE OPTIMIZATION
# ============================================================

def optimize_base(
    d,
    timestamps,
    train_start,
    train_end,
):

    combinations = list(
        itertools.product(
            RR_VALUES,
            WICK_VALUES,
            BODY_VALUES,
            SEPARATION_VALUES,
            MAX_CROSS_VALUES,
            HOUR_SETS,
        )
    )

    print(
        f"BASE COMBINATIONS: "
        f"{len(combinations)}"
    )

    candidates = []

    recent_end = utc(
        train_end
    )

    recent_start = (
        recent_end
        - pd.Timedelta(
            days=RECENT_DAYS
        )
    )

    for number, params in enumerate(
        combinations,
        start=1,
    ):

        if (
            number == 1
            or number % 250 == 0
            or number == len(combinations)
        ):

            print(
                f"Base progress: "
                f"{number}/"
                f"{len(combinations)} "
                f"("
                f"{number / len(combinations) * 100:.1f}%"
                f")",
                flush=True,
            )

        setups = generate_setups(
            d,
            params,
        )

        if len(setups) == 0:
            continue

        train = evaluate(
            d,
            timestamps,
            setups,
            params[0],
            train_start,
            train_end,
        )

        if (
            train is None
            or train["trades"]
            < MIN_TRAIN_TRADES
        ):
            continue

        recent = evaluate(
            d,
            timestamps,
            setups,
            params[0],
            recent_start,
            train_end,
        )

        if recent is None:
            continue

        if recent["trades"] < 8:
            continue

        # Score is intentionally balanced.
        score = (
            train["total_r"]
            + (
                recent["total_r"]
                * 3
            )
            + (
                train["win_rate"]
                * 0.10
            )
            + (
                recent["win_rate"]
                * 0.40
            )
            + (
                min(
                    recent[
                        "profit_factor"
                    ],
                    3,
                )
                * 3
            )
            - (
                recent["drawdown"]
                * 1.5
            )
        )

        candidates.append(
            {
                "params":
                    params,

                "setups":
                    setups,

                "train":
                    train,

                "recent":
                    recent,

                "score":
                    score,
            }
        )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return candidates[:20]


# ============================================================
# WINNER / LOSER FEATURE ANALYSIS
# ============================================================

def winner_loser_analysis(
    d,
    timestamps,
    candidate,
    train_start,
    train_end,
):

    params = candidate[
        "params"
    ]

    setups = candidate[
        "setups"
    ]

    bounds = get_bounds(
        timestamps,
        train_start,
        train_end,
    )

    if bounds is None:
        return None

    start_idx, end_idx = bounds

    setups = setups[
        (setups >= start_idx)
        & (setups <= end_idx)
    ]

    if len(setups) < 20:
        return None

    wins = []
    losses = []

    # Evaluate each setup individually.
    for i in setups:

        trade = simulate(
            d,
            np.array([i]),
            params[0],
            i,
            min(
                i + 100,
                end_idx,
            ),
        )

        if not trade:
            continue

        if trade[0] > 0:
            wins.append(i)
        else:
            losses.append(i)

    if len(wins) < 5:
        return None

    if len(losses) < 5:
        return None

    features = [
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "atr_ratio",
        "momentum_4",
        "momentum_8",
        "momentum_16",
        "distance_20",
        "distance_50",
        "separation",
        "separation_50_100",
        "distance_recent_high",
        "distance_recent_low",
        "cross_age",
        "bearish_count_3",
        "bearish_count_5",
        "volatility_change",
    ]

    rows = []

    for feature in features:

        winner_values = d[
            feature
        ][wins]

        loser_values = d[
            feature
        ][losses]

        winner_median = float(
            np.nanmedian(
                winner_values
            )
        )

        loser_median = float(
            np.nanmedian(
                loser_values
            )
        )

        difference = (
            winner_median
            - loser_median
        )

        rows.append(
            {
                "feature":
                    feature,

                "winner_median":
                    winner_median,

                "loser_median":
                    loser_median,

                "difference":
                    difference,

                "absolute_difference":
                    abs(difference),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result = result.sort_values(
        "absolute_difference",
        ascending=False,
    )

    return result


# ============================================================
# QUALITY THRESHOLD TEST
# ============================================================

def threshold_test(
    d,
    timestamps,
    setups,
    scores,
    rr,
    train_start,
    train_end,
):

    results = []

    for threshold in SCORE_THRESHOLDS:

        filtered = setups[
            scores >= threshold
        ]

        train = evaluate(
            d,
            timestamps,
            filtered,
            rr,
            train_start,
            train_end,
        )

        if train is None:
            continue

        if train["trades"] < 15:
            continue

        recent_end = utc(
            train_end
        )

        recent_start = (
            recent_end
            - pd.Timedelta(
                days=RECENT_DAYS
            )
        )

        recent = evaluate(
            d,
            timestamps,
            filtered,
            rr,
            recent_start,
            train_end,
        )

        if recent is None:
            continue

        # Quality score.
        quality_score = (
            recent["win_rate"]
            * 1.5
            + recent["total_r"]
            * 5
            + min(
                recent[
                    "profit_factor"
                ],
                3,
            )
            * 10
            - recent["drawdown"]
            * 3
        )

        # Penalise tiny samples.
        if recent["trades"] < 10:
            quality_score -= 20

        results.append(
            {
                "threshold":
                    threshold,

                "train":
                    train,

                "recent":
                    recent,

                "score":
                    quality_score,
            }
        )

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return results


# ============================================================
# RUN PERIOD
# ============================================================

def run_period(
    market,
    d,
    timestamps,
    period,
    output_rows,
):

    (
        period_name,
        train_start,
        train_end,
        test_start,
        test_end,
    ) = period

    print()
    print("=" * 60)

    print(
        f"{market} V7: "
        f"{period_name}"
    )

    print("=" * 60)

    print(
        "PHASE 1: BASE SIGNAL SEARCH"
    )

    bases = optimize_base(
        d,
        timestamps,
        train_start,
        train_end,
    )

    if not bases:

        print(
            "No valid base strategies."
        )

        return None

    print(
        f"Base candidates: "
        f"{len(bases)}"
    )

    best_base = bases[0]

    params = best_base[
        "params"
    ]

    setups = best_base[
        "setups"
    ]

    print()
    print(
        "PHASE 2: WINNER / LOSER ANALYSIS"
    )

    analysis = winner_loser_analysis(
        d,
        timestamps,
        best_base,
        train_start,
        train_end,
    )

    if analysis is not None:

        print()
        print(
            "TOP WINNER / LOSER FEATURES"
        )

        print("-" * 60)

        for _, row in analysis.head(
            10
        ).iterrows():

            print(
                f"{row['feature']:25s} "
                f"W:{row['winner_median']:.6f} "
                f"L:{row['loser_median']:.6f} "
                f"D:{row['difference']:.6f}"
            )

    # --------------------------------------------------------
    # Calculate quality score.
    # --------------------------------------------------------

    scores = calculate_quality_score(
        d,
        setups,
    )

    print()
    print(
        "PHASE 3: QUALITY THRESHOLD SEARCH"
    )

    threshold_results = threshold_test(
        d,
        timestamps,
        setups,
        scores,
        params[0],
        train_start,
        train_end,
    )

    if not threshold_results:

        print(
            "No valid quality thresholds."
        )

        return None

    print()
    print(
        "QUALITY THRESHOLDS"
    )

    print("-" * 60)

    for result in threshold_results:

        print(
            f"Threshold "
            f"{result['threshold']:2d} | "
            f"Recent trades "
            f"{result['recent']['trades']:3d} | "
            f"WR "
            f"{result['recent']['win_rate']:6.2f}% | "
            f"R "
            f"{result['recent']['total_r']:7.2f} | "
            f"PF "
            f"{result['recent']['profit_factor']:5.2f}"
        )

    best_threshold = threshold_results[0]

    threshold = best_threshold[
        "threshold"
    ]

    final_setups = setups[
        scores >= threshold
    ]

    print()
    print(
        "SELECTED SIGNAL QUALITY"
    )

    print("-" * 60)

    print(
        f"Threshold: "
        f"{threshold}"
    )

    print(
        f"Training setups: "
        f"{len(final_setups)}"
    )

    # --------------------------------------------------------
    # OOS
    # --------------------------------------------------------

    print()
    print(
        "PHASE 4: COMPLETELY "
        "OUT-OF-SAMPLE TEST"
    )

    oos = evaluate(
        d,
        timestamps,
        final_setups,
        params[0],
        test_start,
        test_end,
    )

    if oos is None:

        print(
            "No OOS trades."
        )

        return None

    print("-" * 60)

    print(
        f"Trades: "
        f"{oos['trades']}"
    )

    print(
        f"Wins: "
        f"{oos['wins']}"
    )

    print(
        f"Losses: "
        f"{oos['losses']}"
    )

    print(
        f"Win rate: "
        f"{oos['win_rate']:.2f}%"
    )

    print(
        f"Total R: "
        f"{oos['total_r']:.2f}"
    )

    print(
        f"Profit factor: "
        f"{oos['profit_factor']:.2f}"
    )

    print(
        f"Max drawdown: "
        f"{oos['drawdown']:.2f}R"
    )

    print(
        f"Longest losing streak: "
        f"{oos['longest_losing_streak']}"
    )

    print(
        f"Trades/week: "
        f"{oos['trades_per_week']:.2f}"
    )

    output_rows.append(
        {
            "market":
                market,

            "period":
                period_name,

            "rr":
                params[0],

            "wick":
                params[1],

            "body":
                params[2],

            "separation":
                params[3],

            "max_cross":
                params[4],

            "hours":
                ",".join(
                    map(
                        str,
                        params[5],
                    )
                ),

            "quality_threshold":
                threshold,

            "oos_trades":
                oos["trades"],

            "oos_wins":
                oos["wins"],

            "oos_losses":
                oos["losses"],

            "oos_win_rate":
                oos["win_rate"],

            "oos_total_r":
                oos["total_r"],

            "oos_profit_factor":
                oos["profit_factor"],

            "oos_drawdown":
                oos["drawdown"],

            "oos_losing_streak":
                oos[
                    "longest_losing_streak"
                ],

            "oos_trades_per_week":
                oos[
                    "trades_per_week"
                ],
        }
    )

    return oos


# ============================================================
# RUN MARKET
# ============================================================

def run_market(
    market,
    path,
):

    print()
    print("=" * 60)

    print(
        f"{market} V7 "
        "SIGNAL QUALITY OPTIMIZER"
    )

    print("=" * 60)

    df = load_data(
        path
    )

    timestamps = (
        df.index.to_numpy()
    )

    print(
        f"Candles: "
        f"{len(df)}"
    )

    print(
        f"Range: "
        f"{df.index.min()} -> "
        f"{df.index.max()}"
    )

    print(
        "Preparing indicators..."
    )

    d = prepare_indicators(
        df
    )

    rows = []

    completed = 0

    for period in PERIODS:

        try:

            result = run_period(
                market,
                d,
                timestamps,
                period,
                rows,
            )

            if result is not None:

                completed += 1

        except Exception as error:

            print()
            print(
                f"{market} PERIOD FAILED"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

    result_df = pd.DataFrame(
        rows
    )

    output_path = (
        f"data/"
        f"{market.lower()}_"
        f"optimizer_v7_results.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    return result_df


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    market,
    df,
):

    if df.empty:
        return None

    trades = int(
        df[
            "oos_trades"
        ].sum()
    )

    wins = int(
        df[
            "oos_wins"
        ].sum()
    )

    total_r = float(
        df[
            "oos_total_r"
        ].sum()
    )

    profitable_periods = int(
        (
            df[
                "oos_total_r"
            ]
            > 0
        ).sum()
    )

    periods = len(
        df
    )

    win_rate = (
        wins
        / trades
        * 100
        if trades
        else 0
    )

    if (
        trades >= 200
        and win_rate >= 82
        and total_r > 0
        and profitable_periods
        >= max(
            2,
            int(
                periods
                * 0.67
            ),
        )
    ):

        verdict = (
            "TARGET ACHIEVED"
        )

    elif (
        trades >= 100
        and win_rate >= 75
        and total_r > 0
    ):

        verdict = (
            "VERY PROMISING"
        )

    elif (
        trades >= 50
        and win_rate >= 65
        and total_r > 0
    ):

        verdict = (
            "PROMISING"
        )

    else:

        verdict = (
            "NOT THERE YET"
        )

    return {
        "market":
            market,

        "oos_trades":
            trades,

        "oos_win_rate":
            round(
                win_rate,
                2,
            ),

        "oos_total_r":
            round(
                total_r,
                2,
            ),

        "profitable_periods":
            f"{profitable_periods}/"
            f"{periods}",

        "verdict":
            verdict,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "MULTI-MARKET SIGNAL "
        "QUALITY OPTIMIZER V7"
    )

    print("=" * 60)

    print(
        "WINNER / LOSER ANALYSIS: ENABLED"
    )

    print(
        "QUALITY SCORING: ENABLED"
    )

    print(
        "WALK-FORWARD TESTING: ENABLED"
    )

    print(
        "CURRENT-ERA DATA: ENABLED"
    )

    print(
        "XAUUSD + EURUSD"
    )

    print(
        "NO LIVE TRADING"
    )

    print("=" * 60)

    results = {}

    for market, path in (
        MARKETS.items()
    ):

        try:

            results[
                market
            ] = run_market(
                market,
                path,
            )

        except Exception as error:

            print()
            print("=" * 60)

            print(
                f"{market} FAILED"
            )

            print("=" * 60)

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            results[
                market
            ] = pd.DataFrame()

    summaries = []

    for market, df in (
        results.items()
    ):

        summary = build_summary(
            market,
            df,
        )

        if summary is not None:

            summaries.append(
                summary
            )

    summary_df = pd.DataFrame(
        summaries
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)

    print(
        "V7 MULTI-MARKET SUMMARY"
    )

    print("=" * 60)

    if summary_df.empty:

        print(
            "No completed results."
        )

    else:

        print(
            summary_df.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Combined OOS
    # --------------------------------------------------------

    total_trades = 0
    total_wins = 0
    total_r = 0.0

    for df in results.values():

        if df.empty:
            continue

        total_trades += int(
            df[
                "oos_trades"
            ].sum()
        )

        total_wins += int(
            df[
                "oos_wins"
            ].sum()
        )

        total_r += float(
            df[
                "oos_total_r"
            ].sum()
        )

    combined_win_rate = (
        total_wins
        / total_trades
        * 100
        if total_trades
        else 0
    )

    print()
    print("=" * 60)

    print(
        "COMBINED CROSS-MARKET OOS"
    )

    print("=" * 60)

    print(
        f"Trades: "
        f"{total_trades}"
    )

    print(
        f"Wins: "
        f"{total_wins}"
    )

    print(
        f"Win rate: "
        f"{combined_win_rate:.2f}%"
    )

    print(
        f"Total R: "
        f"{total_r:.2f}"
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    print()
    print("=" * 60)

    print(
        "V7 TARGET CHECK"
    )

    print("=" * 60)

    print(
        "TARGET:"
    )

    print(
        "~82% WIN RATE"
    )

    print(
        "200+ GENUINELY OOS TRADES"
    )

    print()

    if (
        total_trades >= 200
        and combined_win_rate >= 82
        and total_r > 0
    ):

        print(
            "TARGET STATUS: ACHIEVED"
        )

    elif (
        total_trades >= 200
        and combined_win_rate >= 75
        and total_r > 0
    ):

        print(
            "TARGET STATUS: "
            "VERY CLOSE"
        )

    elif (
        total_trades >= 100
        and combined_win_rate >= 65
        and total_r > 0
    ):

        print(
            "TARGET STATUS: "
            "PROMISING"
        )

    else:

        print(
            "TARGET STATUS: "
            "NOT ACHIEVED YET"
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "A high win rate from a "
        "small sample is NOT considered "
        "success."
    )

    print(
        "OOS sample size and robustness "
        "remain mandatory."
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    summary_df.to_csv(
        "data/"
        "multi_market_optimizer_v7_summary.csv",
        index=False,
    )

    print()
    print(
        "Results saved:"
    )

    print(
        "data/"
        "xauusd_optimizer_v7_results.csv"
    )

    print(
        "data/"
        "eurusd_optimizer_v7_results.csv"
    )

    print(
        "data/"
        "multi_market_optimizer_v7_summary.csv"
    )

    print()
    print("=" * 60)

    print(
        "OPTIMIZER V7 COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
