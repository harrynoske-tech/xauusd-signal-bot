import os
import itertools
import numpy as np
import pandas as pd


# ============================================================
# MULTI-MARKET SIGNAL QUALITY OPTIMIZER V8
# ============================================================
#
# ADAPTIVE SIGNAL SCORING
#
# V8 fixes the main V7 problem:
# V7's score thresholds were not actually separating trades.
#
# V8:
# - Builds individual setup outcomes during training
# - Learns feature -> win/loss relationships from TRAINING ONLY
# - Uses quantile bins instead of crude yes/no points
# - Gives every setup a continuous adaptive score
# - Tests score buckets
# - Tests multiple score thresholds
# - Walk-forward validates on completely unseen data
# - XAUUSD + EURUSD
# - No live trading
#
# TARGET:
# ~82% win rate
# 200+ genuinely OOS trades
# Positive R
# Robust across periods/markets
# ============================================================


MARKETS = {
    "XAUUSD": "data/XAUUSD_15m.csv",
    "EURUSD": "data/EURUSD_15m.csv",
}


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


# ------------------------------------------------------------
# Base strategy search
# ------------------------------------------------------------

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


MIN_TRAIN_TRADES = 30
MIN_RECENT_TRADES = 8
MIN_OOS_TRADES = 5

RECENT_DAYS = 365

# V8 scoring parameters
N_BINS = 5

# We test these after the adaptive score is learned.
SCORE_THRESHOLDS = [
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
]


# ============================================================
# TIME HELPERS
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

    df = df.set_index(time_col)

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

    ema20_slope = (
        ema20 - np.roll(ema20, 4)
    ) / np.where(
        ema20 == 0,
        1,
        ema20,
    )

    ema50_slope = (
        ema50 - np.roll(ema50, 4)
    ) / np.where(
        ema50 == 0,
        1,
        ema50,
    )

    ema100_slope = (
        ema100 - np.roll(ema100, 8)
    ) / np.where(
        ema100 == 0,
        1,
        ema100,
    )

    candle_range = h - l

    body = np.abs(c - o)

    body_ratio = np.divide(
        body,
        candle_range,
        out=np.zeros_like(c),
        where=candle_range > 0,
    )

    upper_wick = (
        h - np.maximum(o, c)
    )

    lower_wick = (
        np.minimum(o, c) - l
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

    previous_close = np.roll(c, 1)

    true_range = np.maximum.reduce(
        [
            h - l,
            np.abs(h - previous_close),
            np.abs(l - previous_close),
        ]
    )

    atr = (
        pd.Series(true_range)
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

    momentum_4 = (
        c - np.roll(c, 4)
    ) / np.where(
        np.roll(c, 4) == 0,
        1,
        np.roll(c, 4),
    )

    momentum_8 = (
        c - np.roll(c, 8)
    ) / np.where(
        np.roll(c, 8) == 0,
        1,
        np.roll(c, 8),
    )

    momentum_16 = (
        c - np.roll(c, 16)
    ) / np.where(
        np.roll(c, 16) == 0,
        1,
        np.roll(c, 16),
    )

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

    separation = (
        np.abs(ema20 - ema50)
    ) / np.where(
        c == 0,
        1,
        c,
    )

    separation_50_100 = (
        np.abs(ema50 - ema100)
    ) / np.where(
        c == 0,
        1,
        c,
    )

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

    bearish = c < o
    bullish = c > o

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

    hours = index.hour.to_numpy()

    return {
        "open": o,
        "high": h,
        "low": l,
        "close": c,

        "ema20": ema20,
        "ema50": ema50,
        "ema100": ema100,
        "ema200": ema200,

        "ema20_slope": ema20_slope,
        "ema50_slope": ema50_slope,
        "ema100_slope": ema100_slope,

        "body_ratio": body_ratio,
        "upper_wick_ratio": upper_wick_ratio,
        "lower_wick_ratio": lower_wick_ratio,

        "atr": atr,
        "atr_ratio": atr_ratio,

        "momentum_4": momentum_4,
        "momentum_8": momentum_8,
        "momentum_16": momentum_16,

        "distance_20": distance_20,
        "distance_50": distance_50,

        "separation": separation,
        "separation_50_100":
            separation_50_100,

        "distance_recent_high":
            distance_recent_high,

        "distance_recent_low":
            distance_recent_low,

        "cross_age": cross_age,

        "bearish_count_3":
            bearish_count_3,

        "bearish_count_5":
            bearish_count_5,

        "bullish_count_3":
            bullish_count_3,

        "volatility_change":
            volatility_change,

        "hours": hours,
    }


# ============================================================
# BASE SETUPS
# ============================================================

def generate_setups(d, params):

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

    return np.flatnonzero(mask)


# ============================================================
# SINGLE TRADE OUTCOME
# ============================================================

def single_trade(
    d,
    index,
    rr,
    end_idx,
):

    entry = d["close"][index]

    atr = d["atr"][index]

    if not np.isfinite(atr):
        return None

    if atr <= 0:
        return None

    stop = entry + atr

    target = entry - (
        atr * rr
    )

    for j in range(
        index + 1,
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

        if (
            stop_hit
            and target_hit
        ):
            return -1.0, j

        if stop_hit:
            return -1.0, j

        if target_hit:
            return rr, j

    return None


# ============================================================
# GENERATE INDIVIDUAL TRAINING TRADES
# ============================================================

def build_training_examples(
    d,
    timestamps,
    setups,
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
        return []

    start_idx, end_idx = bounds

    selected = setups[
        (setups >= start_idx)
        & (setups <= end_idx)
    ]

    examples = []

    next_available = start_idx

    for index in selected:

        if index < next_available:
            continue

        result = single_trade(
            d,
            index,
            rr,
            end_idx,
        )

        if result is None:
            continue

        r_value, exit_index = result

        examples.append(
            {
                "index": int(index),
                "r": float(r_value),
                "win": int(r_value > 0),
            }
        )

        next_available = (
            exit_index + 1
        )

    return examples


# ============================================================
# FEATURE LIST
# ============================================================

FEATURES = [
    "cross_age",
    "bearish_count_5",
    "bearish_count_3",
    "atr_ratio",
    "volatility_change",
    "body_ratio",
    "lower_wick_ratio",
    "upper_wick_ratio",
    "momentum_4",
    "momentum_8",
    "momentum_16",
    "distance_20",
    "distance_50",
    "separation",
    "separation_50_100",
    "distance_recent_high",
    "distance_recent_low",
    "ema20_slope",
    "ema50_slope",
]


# ============================================================
# ADAPTIVE FEATURE MODEL
# ============================================================
#
# For every feature:
#
# 1. Split TRAINING examples into quantile bins.
# 2. Calculate smoothed win rate in each bin.
# 3. Convert win rate into log-odds relative to baseline.
#
# This produces a continuous feature contribution.
#
# IMPORTANT:
# This model is learned ONLY from training data.
# ============================================================

def build_feature_model(
    d,
    examples,
):

    if len(examples) < 30:
        return None

    indices = np.array(
        [
            x["index"]
            for x in examples
        ],
        dtype=int,
    )

    outcomes = np.array(
        [
            x["win"]
            for x in examples
        ],
        dtype=float,
    )

    baseline = (
        outcomes.sum()
        + 2.0
    ) / (
        len(outcomes)
        + 4.0
    )

    baseline = np.clip(
        baseline,
        0.05,
        0.95,
    )

    baseline_logit = np.log(
        baseline
        / (1.0 - baseline)
    )

    model = {}

    for feature in FEATURES:

        values = np.asarray(
            d[feature][indices],
            dtype=float,
        )

        finite = np.isfinite(values)

        values = values[finite]
        local_outcomes = outcomes[
            finite
        ]

        if len(values) < 20:
            continue

        unique = np.unique(values)

        if len(unique) < 3:
            continue

        try:

            edges = np.quantile(
                values,
                np.linspace(
                    0,
                    1,
                    N_BINS + 1,
                ),
            )

        except Exception:
            continue

        edges = np.unique(edges)

        if len(edges) < 3:
            continue

        bin_values = np.digitize(
            values,
            edges[1:-1],
            right=False,
        )

        bin_models = {}

        for bin_id in range(
            len(edges) - 1
        ):

            mask = (
                bin_values
                == bin_id
            )

            count = int(
                mask.sum()
            )

            if count < 5:
                continue

            wins = float(
                local_outcomes[
                    mask
                ].sum()
            )

            # Laplace smoothing.
            win_rate = (
                wins + 2.0
            ) / (
                count + 4.0
            )

            win_rate = np.clip(
                win_rate,
                0.05,
                0.95,
            )

            logit = np.log(
                win_rate
                / (1.0 - win_rate)
            )

            contribution = (
                logit
                - baseline_logit
            )

            # Prevent one feature from
            # dominating the entire model.
            contribution = np.clip(
                contribution,
                -1.25,
                1.25,
            )

            bin_models[
                bin_id
            ] = {
                "count": count,
                "win_rate": win_rate,
                "contribution":
                    contribution,
            }

        if bin_models:

            model[
                feature
            ] = {
                "edges": edges,
                "bins": bin_models,
            }

    if not model:
        return None

    return {
        "baseline": baseline,
        "baseline_logit":
            baseline_logit,
        "features": model,
    }


# ============================================================
# SCORE INDIVIDUAL SETUPS
# ============================================================

def score_setups(
    d,
    indices,
    model,
):

    scores = np.zeros(
        len(indices),
        dtype=float,
    )

    if model is None:
        return scores

    for position, index in enumerate(
        indices
    ):

        score = 0.0
        used = 0

        for feature, config in (
            model["features"].items()
        ):

            value = d[
                feature
            ][index]

            if not np.isfinite(value):
                continue

            edges = config[
                "edges"
            ]

            bins = config[
                "bins"
            ]

            bin_id = int(
                np.digitize(
                    value,
                    edges[1:-1],
                    right=False,
                )
            )

            if bin_id not in bins:
                continue

            contribution = bins[
                bin_id
            ]["contribution"]

            score += contribution
            used += 1

        if used > 0:

            # Normalise for the number of
            # available features.
            scores[position] = (
                score
                / np.sqrt(used)
            )

    return scores


# ============================================================
# SIMULATE FILTERED SETUPS
# ============================================================

def simulate_filtered(
    d,
    indices,
    rr,
    start_idx,
    end_idx,
):

    trades = []

    next_available = start_idx

    for index in indices:

        if index < start_idx:
            continue

        if index > end_idx:
            break

        if index < next_available:
            continue

        result = single_trade(
            d,
            index,
            rr,
            end_idx,
        )

        if result is None:
            continue

        r_value, exit_index = result

        trades.append(
            float(r_value)
        )

        next_available = (
            exit_index + 1
        )

    return trades


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
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

    count = len(values)

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

    pf = (
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

    streak = 0
    longest = 0

    for value in values:

        if value < 0:

            streak += 1

            longest = max(
                longest,
                streak,
            )

        else:

            streak = 0

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
        "trades": count,
        "wins": win_count,
        "losses":
            int(losses.sum()),

        "win_rate":
            win_count
            / count
            * 100,

        "total_r":
            float(values.sum()),

        "profit_factor":
            float(pf),

        "drawdown":
            drawdown,

        "longest_losing_streak":
            longest,

        "trades_per_week":
            trades_per_week,
    }


# ============================================================
# BASE OPTIMIZER
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

        bounds = get_bounds(
            timestamps,
            train_start,
            train_end,
        )

        if bounds is None:
            continue

        start_idx, end_idx = bounds

        train_trades = (
            simulate_filtered(
                d,
                setups,
                params[0],
                start_idx,
                end_idx,
            )
        )

        train = calculate_metrics(
            train_trades,
            timestamps[start_idx],
            timestamps[end_idx],
        )

        if (
            train is None
            or train["trades"]
            < MIN_TRAIN_TRADES
        ):
            continue

        recent_bounds = get_bounds(
            timestamps,
            recent_start,
            train_end,
        )

        if recent_bounds is None:
            continue

        recent_start_idx, recent_end_idx = (
            recent_bounds
        )

        recent_trades = (
            simulate_filtered(
                d,
                setups,
                params[0],
                recent_start_idx,
                recent_end_idx,
            )
        )

        recent = calculate_metrics(
            recent_trades,
            timestamps[
                recent_start_idx
            ],
            timestamps[
                recent_end_idx
            ],
        )

        if (
            recent is None
            or recent["trades"]
            < MIN_RECENT_TRADES
        ):
            continue

        score = (
            train["total_r"]
            + recent["total_r"] * 3
            + train["win_rate"] * 0.05
            + recent["win_rate"] * 0.30
            + min(
                recent[
                    "profit_factor"
                ],
                3,
            ) * 3
            - recent["drawdown"] * 1.5
        )

        candidates.append(
            {
                "params": params,
                "setups": setups,
                "train": train,
                "recent": recent,
                "score": score,
            }
        )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return candidates[:10]


# ============================================================
# ADAPTIVE SCORE TEST
# ============================================================

def test_score_model(
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

    examples = build_training_examples(
        d,
        timestamps,
        setups,
        params[0],
        train_start,
        train_end,
    )

    if len(examples) < MIN_TRAIN_TRADES:
        return None

    model = build_feature_model(
        d,
        examples,
    )

    if model is None:
        return None

    train_indices = np.array(
        [
            x["index"]
            for x in examples
        ],
        dtype=int,
    )

    train_scores = score_setups(
        d,
        train_indices,
        model,
    )

    # --------------------------------------------------------
    # Score distribution
    # --------------------------------------------------------

    print()
    print(
        "ADAPTIVE SCORE DISTRIBUTION"
    )

    print("-" * 60)

    percentiles = [
        10,
        25,
        50,
        75,
        90,
    ]

    for p in percentiles:

        print(
            f"P{p}: "
            f"{np.percentile(train_scores, p):.3f}"
        )

    # --------------------------------------------------------
    # Test score buckets
    # --------------------------------------------------------

    bucket_rows = []

    score_edges = [
        -np.inf,
        -1.5,
        -0.75,
        -0.25,
        0.25,
        0.75,
        1.5,
        np.inf,
    ]

    print()
    print(
        "TRAINING SCORE BUCKETS"
    )

    print("-" * 60)

    for low, high in zip(
        score_edges[:-1],
        score_edges[1:],
    ):

        mask = (
            (train_scores >= low)
            & (train_scores < high)
        )

        selected = train_indices[
            mask
        ]

        if len(selected) < 5:
            continue

        wins = sum(
            examples[
                list(train_indices).index(
                    index
                )
            ]["win"]
            for index in selected
        )

        wr = (
            wins
            / len(selected)
            * 100
        )

        print(
            f"{low:7.2f} -> "
            f"{high:7.2f} | "
            f"trades "
            f"{len(selected):3d} | "
            f"WR "
            f"{wr:6.2f}%"
        )

        bucket_rows.append(
            {
                "low": low,
                "high": high,
                "trades": len(selected),
                "win_rate": wr,
            }
        )

    # --------------------------------------------------------
    # Threshold selection
    # --------------------------------------------------------

    threshold_results = []

    train_bounds = get_bounds(
        timestamps,
        train_start,
        train_end,
    )

    if train_bounds is None:
        return None

    train_start_idx, train_end_idx = (
        train_bounds
    )

    for threshold in SCORE_THRESHOLDS:

        mask = (
            train_scores
            >= threshold
        )

        selected = train_indices[
            mask
        ]

        if len(selected) < 15:
            continue

        trades = simulate_filtered(
            d,
            selected,
            params[0],
            train_start_idx,
            train_end_idx,
        )

        result = calculate_metrics(
            trades,
            timestamps[
                train_start_idx
            ],
            timestamps[
                train_end_idx
            ],
        )

        if result is None:
            continue

        if result["trades"] < 15:
            continue

        # Prefer high WR, positive R and
        # reasonable sample size.
        selection_score = (
            result["win_rate"]
            * 1.0
            + result["total_r"]
            * 4.0
            + min(
                result["profit_factor"],
                3,
            )
            * 8.0
            - result["drawdown"]
            * 2.0
        )

        # Strong penalty for tiny samples.
        if result["trades"] < 25:
            selection_score -= (
                25
                - result["trades"]
            ) * 0.5

        threshold_results.append(
            {
                "threshold":
                    threshold,
                "metrics":
                    result,
                "score":
                    selection_score,
            }
        )

    if not threshold_results:
        return None

    threshold_results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    print()
    print(
        "ADAPTIVE SCORE THRESHOLDS"
    )

    print("-" * 60)

    for row in threshold_results:

        m = row["metrics"]

        print(
            f"Threshold "
            f"{row['threshold']:6.2f} | "
            f"Trades "
            f"{m['trades']:3d} | "
            f"WR "
            f"{m['win_rate']:6.2f}% | "
            f"R "
            f"{m['total_r']:7.2f} | "
            f"PF "
            f"{m['profit_factor']:5.2f} | "
            f"DD "
            f"{m['drawdown']:5.2f}R"
        )

    best = threshold_results[0]

    return {
        "model": model,
        "threshold":
            best["threshold"],
        "training_metrics":
            best["metrics"],
        "train_indices":
            train_indices,
        "train_scores":
            train_scores,
    }


# ============================================================
# OUT OF SAMPLE
# ============================================================

def run_oos(
    d,
    timestamps,
    candidate,
    model_result,
    test_start,
    test_end,
):

    params = candidate[
        "params"
    ]

    setups = candidate[
        "setups"
    ]

    bounds = get_bounds(
        timestamps,
        test_start,
        test_end,
    )

    if bounds is None:
        return None

    start_idx, end_idx = bounds

    # --------------------------------------------------------
    # CRITICAL:
    # Score the unseen period using the
    # model learned ONLY on training data.
    # --------------------------------------------------------

    test_indices = setups[
        (setups >= start_idx)
        & (setups <= end_idx)
    ]

    scores = score_setups(
        d,
        test_indices,
        model_result["model"],
    )

    threshold = model_result[
        "threshold"
    ]

    selected = test_indices[
        scores >= threshold
    ]

    trades = simulate_filtered(
        d,
        selected,
        params[0],
        start_idx,
        end_idx,
    )

    return calculate_metrics(
        trades,
        timestamps[start_idx],
        timestamps[end_idx],
    )


# ============================================================
# RUN PERIOD
# ============================================================

def run_period(
    market,
    d,
    timestamps,
    period,
    rows,
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
        f"{market} V8: "
        f"{period_name}"
    )

    print("=" * 60)

    print(
        "PHASE 1: BASE OPTIMIZATION"
    )

    candidates = optimize_base(
        d,
        timestamps,
        train_start,
        train_end,
    )

    if not candidates:

        print(
            "No valid base strategies."
        )

        return None

    candidate = candidates[0]

    params = candidate[
        "params"
    ]

    print()
    print(
        "BEST BASE STRATEGY"
    )

    print("-" * 60)

    print(
        f"Training WR: "
        f"{candidate['train']['win_rate']:.2f}%"
    )

    print(
        f"Training trades: "
        f"{candidate['train']['trades']}"
    )

    print(
        f"Training R: "
        f"{candidate['train']['total_r']:.2f}"
    )

    print(
        f"Recent WR: "
        f"{candidate['recent']['win_rate']:.2f}%"
    )

    print(
        f"Recent trades: "
        f"{candidate['recent']['trades']}"
    )

    print()
    print(
        "PARAMETERS"
    )

    print(
        f"RR: {params[0]}"
    )

    print(
        f"Wick: {params[1]}"
    )

    print(
        f"Body: {params[2]}"
    )

    print(
        f"Separation: {params[3]}"
    )

    print(
        f"Max cross: {params[4]}"
    )

    print(
        "Hours: "
        + ",".join(
            map(
                str,
                params[5],
            )
        )
    )

    print()
    print(
        "PHASE 2: ADAPTIVE FEATURE LEARNING"
    )

    model_result = test_score_model(
        d,
        timestamps,
        candidate,
        train_start,
        train_end,
    )

    if model_result is None:

        print(
            "Adaptive model could not "
            "be built."
        )

        return None

    print()
    print(
        "SELECTED ADAPTIVE THRESHOLD"
    )

    print("-" * 60)

    print(
        f"Threshold: "
        f"{model_result['threshold']:.2f}"
    )

    tm = model_result[
        "training_metrics"
    ]

    print(
        f"Training trades: "
        f"{tm['trades']}"
    )

    print(
        f"Training WR: "
        f"{tm['win_rate']:.2f}%"
    )

    print(
        f"Training R: "
        f"{tm['total_r']:.2f}"
    )

    print()
    print(
        "PHASE 3: COMPLETELY "
        "OUT-OF-SAMPLE TEST"
    )

    print("-" * 60)

    oos = run_oos(
        d,
        timestamps,
        candidate,
        model_result,
        test_start,
        test_end,
    )

    if oos is None:

        print(
            "No OOS trades."
        )

        return None

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

    rows.append(
        {
            "market": market,
            "period": period_name,

            "rr": params[0],
            "wick": params[1],
            "body": params[2],
            "separation": params[3],
            "max_cross": params[4],

            "hours": ",".join(
                map(
                    str,
                    params[5],
                )
            ),

            "adaptive_threshold":
                model_result[
                    "threshold"
                ],

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
        f"{market} V8 ADAPTIVE "
        "SIGNAL SCORING"
    )

    print("=" * 60)

    df = load_data(
        path
    )

    timestamps = (
        df.index.to_numpy()
    )

    print(
        f"Candles: {len(df)}"
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

    for period in PERIODS:

        try:

            run_period(
                market,
                d,
                timestamps,
                period,
                rows,
            )

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
        f"optimizer_v8_results.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    return result_df


# ============================================================
# SUMMARY
# ============================================================

def market_summary(
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

    profitable = int(
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
            f"{profitable}/"
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
        "MULTI-MARKET STRATEGY "
        "OPTIMIZER V8"
    )

    print("=" * 60)

    print(
        "ADAPTIVE FEATURE LEARNING: ENABLED"
    )

    print(
        "CONTINUOUS SIGNAL SCORING: ENABLED"
    )

    print(
        "SCORE BUCKET ANALYSIS: ENABLED"
    )

    print(
        "WALK-FORWARD TESTING: ENABLED"
    )

    print(
        "CURRENT-ERA DATA: ENABLED"
    )

    print(
        "PARAMETER SEARCH: ENABLED"
    )

    print(
        "MARKETS: XAUUSD, EURUSD"
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

        summary = market_summary(
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

    print()
    print("=" * 60)

    print(
        "V8 FINAL MULTI-MARKET SUMMARY"
    )

    print("=" * 60)

    if summary_df.empty:

        print(
            "No completed market results."
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

    combined_wr = (
        total_wins
        / total_trades
        * 100
        if total_trades
        else 0
    )

    print()
    print("=" * 60)

    print(
        "COMBINED CROSS-MARKET "
        "OUT-OF-SAMPLE"
    )

    print("=" * 60)

    print(
        f"Trades: {total_trades}"
    )

    print(
        f"Wins: {total_wins}"
    )

    print(
        f"Win rate: "
        f"{combined_wr:.2f}%"
    )

    print(
        f"Total R: "
        f"{total_r:.2f}"
    )

    print()
    print("=" * 60)

    print(
        "V8 TARGET CHECK"
    )

    print("=" * 60)

    print(
        "TARGET:"
    )

    print(
        "~82% WIN RATE"
    )

    print(
        "200+ GENUINELY "
        "OUT-OF-SAMPLE TRADES"
    )

    print()

    if (
        total_trades >= 200
        and combined_wr >= 82
        and total_r > 0
    ):

        print(
            "TARGET STATUS: ACHIEVED"
        )

    elif (
        total_trades >= 100
        and combined_wr >= 75
        and total_r > 0
    ):

        print(
            "TARGET STATUS: "
            "VERY PROMISING"
        )

    elif (
        total_trades >= 50
        and combined_wr >= 65
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
        "The adaptive model is trained "
        "only on the training period."
    )

    print(
        "The OOS periods are never used "
        "to train the model."
    )

    print(
        "Do not implement live from "
        "this optimizer alone."
    )

    summary_path = (
        "data/"
        "multi_market_optimizer_v8_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        "Results saved:"
    )

    print(
        "data/"
        "xauusd_optimizer_v8_results.csv"
    )

    print(
        "data/"
        "eurusd_optimizer_v8_results.csv"
    )

    print(
        "data/"
        "multi_market_optimizer_v8_summary.csv"
    )

    print()
    print("=" * 60)

    print(
        "OPTIMIZER V8 COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
