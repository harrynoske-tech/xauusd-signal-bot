import os
import itertools
import numpy as np
import pandas as pd


# ============================================================
# MULTI-MARKET STRATEGY OPTIMIZER V10
# ============================================================
#
# PURPOSE
# -------
# V9.1 showed that we have an underlying edge, but the optimiser
# was still capable of selecting tiny high-WR samples.
#
# V10 changes the objective:
#
#   1. NO tiny-sample winners
#   2. MINIMUM TRAINING SAMPLE
#   3. CONFIDENCE / QUALITY FILTER
#   4. MARKET-SPECIFIC optimisation
#   5. WALK-FORWARD testing
#   6. COMPLETELY OOS validation
#   7. SCORE THRESHOLDS tested separately
#
# TARGET
# ------
# ~82% WIN RATE
# 200+ GENUINELY OUT-OF-SAMPLE TRADES
# POSITIVE TOTAL R
#
# IMPORTANT
# ---------
# OOS data is NEVER used for optimisation.
#
# NO LIVE TRADING
# ============================================================


MARKETS = {
    "XAUUSD": "data/XAUUSD_15m.csv",
    "EURUSD": "data/EURUSD_15m.csv",
}


# ============================================================
# WALK-FORWARD WINDOWS
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
# PARAMETERS
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


CROSS_VALUES = [
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
# V10 QUALITY REQUIREMENTS
# ============================================================

# This is the critical V10 change.
#
# A strategy with 8 trades and 100% WR is NOT allowed to win.
#
MIN_TRAIN_TRADES = 40

MIN_RECENT_TRADES = 15

MIN_THRESHOLD_TRADES = 30

MIN_OOS_TRADES_FOR_VERDICT = 20

# We are trying to find precision, so this is intentionally
# stricter than previous versions.
MIN_TRAIN_WR = 58.0

# Minimum fraction of nearby parameter combinations that
# must remain profitable.
MIN_POSITIVE_NEARBY = 0.45

# Score thresholds.
#
# Higher threshold = fewer but higher-quality signals.
SCORE_THRESHOLDS = [
    -0.50,
    -0.25,
    0.00,
    0.25,
    0.50,
    0.75,
]


MAX_HOLD_BARS = 96


# ============================================================
# UTILITY
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
# DATA
# ============================================================

def load_data(path):

    if not os.path.exists(path):

        raise RuntimeError(
            f"Missing data file: {path}"
        )

    df = pd.read_csv(path)

    df.columns = [
        str(c).strip()
        for c in df.columns
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
            "No timestamp column found."
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

    def ema(period):

        return (
            pd.Series(c)
            .ewm(
                span=period,
                adjust=False,
            )
            .mean()
            .to_numpy()
        )

    ema9 = ema(9)
    ema20 = ema(20)
    ema50 = ema(50)
    ema100 = ema(100)
    ema200 = ema(200)

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

    atr50 = (
        pd.Series(
            atr
        )
        .rolling(
            50,
            min_periods=20,
        )
        .mean()
        .to_numpy()
    )

    atr100 = (
        pd.Series(
            atr
        )
        .rolling(
            100,
            min_periods=30,
        )
        .mean()
        .to_numpy()
    )

    atr_ratio = np.divide(
        atr,
        atr100,
        out=np.ones_like(c),
        where=atr100 > 0,
    )

    atr_acceleration = np.divide(
        atr - atr50,
        atr50,
        out=np.zeros_like(c),
        where=atr50 > 0,
    )

    ema20_slope = np.divide(
        ema20 - np.roll(
            ema20,
            4,
        ),
        np.where(
            ema20 == 0,
            1,
            ema20,
        ),
    )

    ema50_slope = np.divide(
        ema50 - np.roll(
            ema50,
            8,
        ),
        np.where(
            ema50 == 0,
            1,
            ema50,
        ),
    )

    ema100_slope = np.divide(
        ema100 - np.roll(
            ema100,
            12,
        ),
        np.where(
            ema100 == 0,
            1,
            ema100,
        ),
    )

    ema200_slope = np.divide(
        ema200 - np.roll(
            ema200,
            16,
        ),
        np.where(
            ema200 == 0,
            1,
            ema200,
        ),
    )

    momentum4 = np.divide(
        c - np.roll(
            c,
            4,
        ),
        np.where(
            np.roll(c, 4) == 0,
            1,
            np.roll(c, 4),
        ),
    )

    momentum8 = np.divide(
        c - np.roll(
            c,
            8,
        ),
        np.where(
            np.roll(c, 8) == 0,
            1,
            np.roll(c, 8),
        ),
    )

    momentum16 = np.divide(
        c - np.roll(
            c,
            16,
        ),
        np.where(
            np.roll(c, 16) == 0,
            1,
            np.roll(c, 16),
        ),
    )

    separation = np.divide(
        np.abs(
            ema20 - ema50
        ),
        np.where(
            c == 0,
            1,
            c,
        ),
    )

    distance20 = np.divide(
        c - ema20,
        np.where(
            c == 0,
            1,
            c,
        ),
    )

    distance50 = np.divide(
        c - ema50,
        np.where(
            c == 0,
            1,
            c,
        ),
    )

    previous_range = np.roll(
        candle_range,
        1,
    )

    range_change = np.divide(
        candle_range,
        previous_range,
        out=np.ones_like(c),
        where=previous_range > 0,
    )

    range_expansion = np.divide(
        candle_range,
        atr,
        out=np.ones_like(c),
        where=atr > 0,
    )

    close_location = np.divide(
        c - l,
        candle_range,
        out=np.full_like(
            c,
            0.5,
        ),
        where=candle_range > 0,
    )

    trend_cross_age = np.full(
        len(c),
        9999,
        dtype=np.int32,
    )

    last_cross = -9999

    for i in range(
        1,
        len(c),
    ):

        previous_state = (
            ema20[i - 1]
            >= ema50[i - 1]
        )

        current_state = (
            ema20[i]
            >= ema50[i]
        )

        if (
            previous_state
            != current_state
        ):

            last_cross = i

        if last_cross >= 0:

            trend_cross_age[i] = (
                i - last_cross
            )

    hours = index.hour.to_numpy()

    return {
        "open": o,
        "high": h,
        "low": l,
        "close": c,

        "ema9": ema9,
        "ema20": ema20,
        "ema50": ema50,
        "ema100": ema100,
        "ema200": ema200,

        "ema20_slope": ema20_slope,
        "ema50_slope": ema50_slope,
        "ema100_slope": ema100_slope,
        "ema200_slope": ema200_slope,

        "body_ratio": body_ratio,

        "upper_wick_ratio":
            upper_wick_ratio,

        "lower_wick_ratio":
            lower_wick_ratio,

        "atr": atr,

        "atr_ratio":
            atr_ratio,

        "atr_acceleration":
            atr_acceleration,

        "momentum4":
            momentum4,

        "momentum8":
            momentum8,

        "momentum16":
            momentum16,

        "separation":
            separation,

        "distance20":
            distance20,

        "distance50":
            distance50,

        "range_change":
            range_change,

        "range_expansion":
            range_expansion,

        "close_location":
            close_location,

        "trend_cross_age":
            trend_cross_age,

        "hours":
            hours,
    }


# ============================================================
# BASE SIGNALS
# ============================================================

def trend_pullback(d, p):

    (
        rr,
        wick,
        body,
        separation,
        max_cross,
        hours,
    ) = p

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
            d["trend_cross_age"]
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


def trend_momentum(d, p):

    (
        rr,
        wick,
        body,
        separation,
        max_cross,
        hours,
    ) = p

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
            d["ema50_slope"]
            < 0
        )

        & (
            d["ema100_slope"]
            < 0
        )

        & (
            d["separation"]
            >= separation
        )

        & (
            d["momentum4"]
            < 0
        )

        & (
            d["momentum8"]
            < 0
        )

        & (
            d["body_ratio"]
            >= body
        )

        & (
            d["range_expansion"]
            >= 0.8
        )

        & (
            d["close"]
            < d["ema20"]
        )
    )

    return np.flatnonzero(
        mask
    )


def volatility_expansion(d, p):

    (
        rr,
        wick,
        body,
        separation,
        max_cross,
        hours,
    ) = p

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
            d["ema50_slope"]
            < 0
        )

        & (
            d["atr_ratio"]
            >= 1.05
        )

        & (
            d["atr_acceleration"]
            > 0
        )

        & (
            d["range_expansion"]
            >= 1.0
        )

        & (
            d["range_change"]
            >= 1.05
        )

        & (
            d["close"]
            < d["open"]
        )

        & (
            d["close"]
            < d["ema20"]
        )

        & (
            d["body_ratio"]
            >= body
        )
    )

    return np.flatnonzero(
        mask
    )


def rejection_reversal(d, p):

    (
        rr,
        wick,
        body,
        separation,
        max_cross,
        hours,
    ) = p

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
            d["lower_wick_ratio"]
            >= 0.35
        )

        & (
            d["close_location"]
            >= 0.55
        )

        & (
            d["body_ratio"]
            >= body
        )

        & (
            d["distance50"]
            < 0
        )
    )

    return np.flatnonzero(
        mask
    )


def ema_structure(d, p):

    (
        rr,
        wick,
        body,
        separation,
        max_cross,
        hours,
    ) = p

    previous_ema9 = np.roll(
        d["ema9"],
        1,
    )

    mask = (
        np.isin(
            d["hours"],
            hours,
        )

        & (
            d["ema9"]
            < d["ema20"]
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
            d["ema9"]
            < previous_ema9
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
            d["distance20"]
            < 0
        )

        & (
            d["body_ratio"]
            >= body
        )
    )

    return np.flatnonzero(
        mask
    )


SIGNALS = {
    "TREND_PULLBACK":
        trend_pullback,

    "TREND_MOMENTUM":
        trend_momentum,

    "VOLATILITY_EXPANSION":
        volatility_expansion,

    "REJECTION_REVERSAL":
        rejection_reversal,

    "EMA_STRUCTURE":
        ema_structure,
}


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate(
    d,
    indices,
    rr,
    start_idx,
    end_idx,
):

    if len(indices) == 0:
        return []

    trades = []

    next_allowed = start_idx

    for signal_idx in indices:

        signal_idx = int(
            signal_idx
        )

        if signal_idx < start_idx:
            continue

        if signal_idx > end_idx:
            break

        if signal_idx < next_allowed:
            continue

        entry = d[
            "close"
        ][signal_idx]

        atr = d[
            "atr"
        ][signal_idx]

        if not np.isfinite(atr):
            continue

        if atr <= 0:
            continue

        stop = (
            entry + atr
        )

        target = (
            entry
            - atr * rr
        )

        final_idx = min(
            signal_idx
            + MAX_HOLD_BARS,
            end_idx,
        )

        result = None
        exit_idx = final_idx

        for j in range(
            signal_idx + 1,
            final_idx + 1,
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

                result = -1.0
                exit_idx = j
                break

            if stop_hit:

                result = -1.0
                exit_idx = j
                break

            if target_hit:

                result = rr
                exit_idx = j
                break

        if result is None:

            exit_price = d[
                "close"
            ][exit_idx]

            move = (
                entry
                - exit_price
            )

            result = move / atr

            result = float(
                np.clip(
                    result,
                    -1.0,
                    rr,
                )
            )

        trades.append(
            float(result)
        )

        next_allowed = (
            exit_idx + 1
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

    wins = (
        values > 0
    )

    losses = (
        values < 0
    )

    n = len(
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

    if gross_loss > 0:

        pf = (
            gross_profit
            / gross_loss
        )

    else:

        pf = 999.0

    equity = np.cumsum(
        values
    )

    peak = np.maximum.accumulate(
        equity
    )

    dd = float(
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
        / 86400,
        1,
    )

    return {
        "trades":
            n,

        "wins":
            win_count,

        "losses":
            int(losses.sum()),

        "win_rate":
            win_count
            / n
            * 100,

        "total_r":
            float(values.sum()),

        "pf":
            float(pf),

        "dd":
            dd,

        "longest":
            longest,

        "trades_per_week":
            n / (days / 7),
    }


# ============================================================
# RAW CANDIDATE SEARCH
# ============================================================

def generate_candidates(
    d,
    timestamps,
    train_start,
    train_end,
):

    bounds = get_bounds(
        timestamps,
        train_start,
        train_end,
    )

    if bounds is None:
        return []

    start_idx, end_idx = bounds

    candidates = []

    combinations = list(
        itertools.product(
            RR_VALUES,
            WICK_VALUES,
            BODY_VALUES,
            SEPARATION_VALUES,
            CROSS_VALUES,
            HOUR_SETS,
        )
    )

    total = len(
        combinations
    )

    print(
        f"Parameter combinations: "
        f"{total}"
    )

    for signal_name, signal_function in (
        SIGNALS.items()
    ):

        print(
            f"{signal_name}"
        )

        local = []

        for count, p in enumerate(
            combinations,
            1,
        ):

            if (
                count == 1
                or count % 1000 == 0
                or count == total
            ):

                print(
                    f"  progress "
                    f"{count}/{total}"
                )

            indices = signal_function(
                d,
                p,
            )

            if len(indices) == 0:
                continue

            trades = simulate(
                d,
                indices,
                p[0],
                start_idx,
                end_idx,
            )

            m = metrics(
                trades,
                timestamps[start_idx],
                timestamps[end_idx],
            )

            if m is None:
                continue

            # ------------------------------------------------
            # CRITICAL:
            #
            # Tiny samples cannot pass.
            # ------------------------------------------------

            if (
                m["trades"]
                < MIN_TRAIN_TRADES
            ):
                continue

            if (
                m["win_rate"]
                < MIN_TRAIN_WR
            ):
                continue

            if (
                m["total_r"]
                <= 0
            ):
                continue

            # ------------------------------------------------
            # Score:
            #
            # WR is heavily weighted.
            # R and PF matter.
            # Drawdown is penalised.
            # Frequency is NOT rewarded.
            # ------------------------------------------------

            score = (
                m["win_rate"]
                * 2.0

                + m["total_r"]
                * 3.0

                + min(
                    m["pf"],
                    4.0,
                )
                * 8.0

                - m["dd"]
                * 3.0

                + np.log1p(
                    m["trades"]
                )
                * 5.0
            )

            local.append(
                {
                    "signal":
                        signal_name,

                    "params":
                        p,

                    "indices":
                        indices,

                    "metrics":
                        m,

                    "score":
                        score,
                }
            )

        local.sort(
            key=lambda x:
            x["score"],
            reverse=True,
        )

        print(
            f"  Valid candidates: "
            f"{len(local)}"
        )

        if local:

            best = local[0]

            m = best[
                "metrics"
            ]

            print(
                f"  BEST: "
                f"{m['trades']} trades | "
                f"{m['win_rate']:.2f}% | "
                f"{m['total_r']:.2f}R | "
                f"PF {m['pf']:.2f}"
            )

            candidates.extend(
                local[:20]
            )

    return candidates


# ============================================================
# SCORE MODEL
# ============================================================
#
# Rather than simply accepting every base signal, V10 creates
# a quality score from independent market features.
#
# Each feature is converted to a training percentile.
#
# This score is then tested at several thresholds.
# ============================================================

def feature_matrix(
    d,
    indices,
):

    if len(indices) == 0:

        return np.empty(
            (0, 8)
        )

    return np.column_stack(
        [
            d["separation"][
                indices
            ],

            np.abs(
                d["ema20_slope"][
                    indices
                ]
            ),

            np.abs(
                d["ema50_slope"][
                    indices
                ]
            ),

            np.abs(
                d["momentum4"][
                    indices
                ]
            ),

            np.abs(
                d["momentum8"][
                    indices
                ]
            ),

            d["body_ratio"][
                indices
            ],

            d["range_expansion"][
                indices
            ],

            d["atr_ratio"][
                indices
            ],
        ]
    )


def percentile_rank(
    values,
):

    if len(values) == 0:
        return values

    order = np.argsort(
        np.argsort(values)
    )

    return (
        order
        / max(
            len(values) - 1,
            1,
        )
    )


def build_quality_score(
    d,
    indices,
):

    x = feature_matrix(
        d,
        indices,
    )

    if len(x) == 0:

        return np.array([])

    ranks = np.column_stack(
        [
            percentile_rank(
                x[:, i]
            )
            for i in range(
                x.shape[1]
            )
        ]
    )

    # --------------------------------------------------------
    # Quality score.
    #
    # Strong trend + strong body + sufficient volatility
    # receives a higher score.
    #
    # No outcome information is used here.
    # --------------------------------------------------------

    score = (
        ranks[:, 0] * 0.15
        + ranks[:, 1] * 0.15
        + ranks[:, 2] * 0.15
        + ranks[:, 3] * 0.10
        + ranks[:, 4] * 0.10
        + ranks[:, 5] * 0.15
        + ranks[:, 6] * 0.10
        + ranks[:, 7] * 0.10
    )

    # Convert 0-1 into approximately
    # -1 to +1.
    return (
        score * 2.0
        - 1.0
    )


# ============================================================
# THRESHOLD TESTING
# ============================================================

def threshold_search(
    d,
    timestamps,
    candidate,
    train_start,
    train_end,
):

    bounds = get_bounds(
        timestamps,
        train_start,
        train_end,
    )

    if bounds is None:
        return None

    start_idx, end_idx = bounds

    indices = candidate[
        "indices"
    ]

    indices = indices[
        (indices >= start_idx)
        & (indices <= end_idx)
    ]

    if len(indices) == 0:
        return None

    scores = build_quality_score(
        d,
        indices,
    )

    if len(scores) == 0:
        return None

    best = None

    print()
    print(
        f"Threshold testing: "
        f"{candidate['signal']}"
    )

    for threshold in (
        SCORE_THRESHOLDS
    ):

        selected = indices[
            scores >= threshold
        ]

        if len(selected) == 0:
            continue

        trades = simulate(
            d,
            selected,
            candidate[
                "params"
            ][0],
            start_idx,
            end_idx,
        )

        m = metrics(
            trades,
            timestamps[start_idx],
            timestamps[end_idx],
        )

        if m is None:
            continue

        if (
            m["trades"]
            < MIN_THRESHOLD_TRADES
        ):
            continue

        # ----------------------------------------------------
        # Quality objective.
        #
        # Win rate is the dominant objective.
        # But we still need positive R and enough trades.
        # ----------------------------------------------------

        if (
            m["total_r"]
            <= 0
        ):
            continue

        score = (
            m["win_rate"]
            * 3.0

            + m["total_r"]
            * 2.0

            + min(
                m["pf"],
                5.0,
            )
            * 10.0

            + np.log1p(
                m["trades"]
            )
            * 4.0

            - m["dd"]
            * 3.0
        )

        print(
            f"  Threshold "
            f"{threshold:5.2f} | "
            f"Trades {m['trades']:3d} | "
            f"WR {m['win_rate']:6.2f}% | "
            f"R {m['total_r']:7.2f} | "
            f"PF {m['pf']:5.2f}"
        )

        result = {
            "candidate":
                candidate,

            "threshold":
                threshold,

            "indices":
                selected,

            "score":
                score,

            "metrics":
                m,
        }

        if (
            best is None
            or score
            > best["score"]
        ):

            best = result

    return best


# ============================================================
# PARAMETER STABILITY
# ============================================================

def stability_test(
    d,
    timestamps,
    candidate,
    train_start,
    train_end,
):

    p = candidate[
        "params"
    ]

    rr, wick, body, sep, cross, hours = p

    rr_values = [
        x for x in RR_VALUES
        if abs(x - rr) <= 0.30
    ]

    wick_values = [
        x for x in WICK_VALUES
        if abs(x - wick) <= 0.10
    ]

    body_values = [
        x for x in BODY_VALUES
        if abs(x - body) <= 0.10
    ]

    sep_values = [
        x for x in SEPARATION_VALUES
        if abs(x - sep) <= 0.0003
    ]

    cross_values = [
        x for x in CROSS_VALUES
        if abs(x - cross) <= 10
    ]

    hours_values = [
        hours
    ]

    nearby = []

    for params in itertools.product(
        rr_values,
        wick_values,
        body_values,
        sep_values,
        cross_values,
        hours_values,
    ):

        function = SIGNALS[
            candidate["signal"]
        ]

        indices = function(
            d,
            params,
        )

        bounds = get_bounds(
            timestamps,
            train_start,
            train_end,
        )

        if bounds is None:
            continue

        start_idx, end_idx = bounds

        indices = indices[
            (indices >= start_idx)
            & (indices <= end_idx)
        ]

        if len(indices) == 0:
            continue

        trades = simulate(
            d,
            indices,
            params[0],
            start_idx,
            end_idx,
        )

        m = metrics(
            trades,
            timestamps[start_idx],
            timestamps[end_idx],
        )

        if m is None:
            continue

        if (
            m["trades"]
            < MIN_THRESHOLD_TRADES
        ):
            continue

        nearby.append(
            m
        )

    if not nearby:

        return {
            "nearby":
                0,

            "median_wr":
                0,

            "median_r":
                0,

            "positive_fraction":
                0,

            "stable":
                False,
        }

    median_wr = float(
        np.median(
            [
                x["win_rate"]
                for x in nearby
            ]
        )
    )

    median_r = float(
        np.median(
            [
                x["total_r"]
                for x in nearby
            ]
        )
    )

    positive_fraction = float(
        np.mean(
            [
                x["total_r"] > 0
                for x in nearby
            ]
        )
    )

    stable = (
        median_r > 0
        and positive_fraction
        >= MIN_POSITIVE_NEARBY
    )

    return {
        "nearby":
            len(nearby),

        "median_wr":
            median_wr,

        "median_r":
            median_r,

        "positive_fraction":
            positive_fraction,

        "stable":
            stable,
    }


# ============================================================
# WALK-FORWARD PERIOD
# ============================================================

def run_period(
    market,
    d,
    timestamps,
    period,
    rows,
):

    (
        name,
        train_start,
        train_end,
        test_start,
        test_end,
    ) = period

    print()
    print("=" * 60)

    print(
        f"{market} V10: {name}"
    )

    print("=" * 60)

    print(
        "PHASE 1: ROBUST BASE SEARCH"
    )

    candidates = generate_candidates(
        d,
        timestamps,
        train_start,
        train_end,
    )

    if not candidates:

        print(
            "No robust candidates."
        )

        return None

    # ========================================================
    # Keep only strongest candidates.
    # ========================================================

    candidates.sort(
        key=lambda x:
        x["score"],
        reverse=True,
    )

    candidates = candidates[
        :25
    ]

    print()
    print(
        f"Candidates surviving "
        f"minimum sample filter: "
        f"{len(candidates)}"
    )

    # ========================================================
    # THRESHOLD SEARCH
    # ========================================================

    print()
    print(
        "PHASE 2: PRECISION "
        "THRESHOLD SEARCH"
    )

    threshold_candidates = []

    for candidate in candidates:

        result = threshold_search(
            d,
            timestamps,
            candidate,
            train_start,
            train_end,
        )

        if result is not None:

            threshold_candidates.append(
                result
            )

    if not threshold_candidates:

        print(
            "No threshold strategy "
            "passed minimum sample "
            "and positive-R requirements."
        )

        return None

    threshold_candidates.sort(
        key=lambda x:
        x["score"],
        reverse=True,
    )

    # ========================================================
    # STABILITY
    # ========================================================

    print()
    print(
        "PHASE 3: PARAMETER "
        "STABILITY"
    )

    best = None

    for result in threshold_candidates[
        :10
    ]:

        candidate = result[
            "candidate"
        ]

        stability = stability_test(
            d,
            timestamps,
            candidate,
            train_start,
            train_end,
        )

        print()
        print(
            f"{candidate['signal']} "
            f"| threshold "
            f"{result['threshold']:.2f}"
        )

        print(
            f"Nearby: "
            f"{stability['nearby']}"
        )

        print(
            f"Median nearby WR: "
            f"{stability['median_wr']:.2f}%"
        )

        print(
            f"Median nearby R: "
            f"{stability['median_r']:.2f}R"
        )

        print(
            f"Positive nearby: "
            f"{stability['positive_fraction'] * 100:.1f}%"
        )

        print(
            "Stability: "
            + (
                "PASS"
                if stability["stable"]
                else "FAIL"
            )
        )

        if not stability["stable"]:
            continue

        final_score = (
            result["score"]
            + stability[
                "positive_fraction"
            ]
            * 50
        )

        result[
            "stability"
        ] = stability

        result[
            "final_score"
        ] = final_score

        if (
            best is None
            or final_score
            > best["final_score"]
        ):

            best = result

    if best is None:

        print()
        print(
            "No stable strategy "
            "survived."
        )

        return None

    # ========================================================
    # SELECTED STRATEGY
    # ========================================================

    candidate = best[
        "candidate"
    ]

    train = best[
        "metrics"
    ]

    print()
    print(
        "SELECTED TRAINING STRATEGY"
    )

    print("-" * 60)

    print(
        f"Signal: "
        f"{candidate['signal']}"
    )

    print(
        f"Threshold: "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Training trades: "
        f"{train['trades']}"
    )

    print(
        f"Training WR: "
        f"{train['win_rate']:.2f}%"
    )

    print(
        f"Training R: "
        f"{train['total_r']:.2f}"
    )

    print(
        f"Training PF: "
        f"{train['pf']:.2f}"
    )

    print(
        f"Training DD: "
        f"{train['dd']:.2f}R"
    )

    print(
        f"RR: "
        f"{candidate['params'][0]}"
    )

    print(
        f"Wick: "
        f"{candidate['params'][1]}"
    )

    print(
        f"Body: "
        f"{candidate['params'][2]}"
    )

    print(
        f"Separation: "
        f"{candidate['params'][3]}"
    )

    print(
        f"Max cross: "
        f"{candidate['params'][4]}"
    )

    print(
        f"Hours: "
        f"{','.join(map(str, candidate['params'][5]))}"
    )

    # ========================================================
    # OOS
    # ========================================================

    print()
    print(
        "PHASE 4: COMPLETELY "
        "OUT-OF-SAMPLE TEST"
    )

    print("-" * 60)

    bounds = get_bounds(
        timestamps,
        test_start,
        test_end,
    )

    if bounds is None:
        return None

    test_start_idx, test_end_idx = bounds

    base_indices = candidate[
        "indices"
    ]

    oos_indices = base_indices[
        (base_indices >= test_start_idx)
        & (
            base_indices
            <= test_end_idx
        )
    ]

    if len(oos_indices) == 0:

        print(
            "No OOS base signals."
        )

        return None

    oos_scores = build_quality_score(
        d,
        oos_indices,
    )

    selected_oos = oos_indices[
        oos_scores
        >= best["threshold"]
    ]

    if len(selected_oos) == 0:

        print(
            "No OOS signals passed "
            "quality threshold."
        )

        return None

    trades = simulate(
        d,
        selected_oos,
        candidate["params"][0],
        test_start_idx,
        test_end_idx,
    )

    oos = metrics(
        trades,
        timestamps[
            test_start_idx
        ],
        timestamps[
            test_end_idx
        ],
    )

    if oos is None:
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
        f"{oos['pf']:.2f}"
    )

    print(
        f"Max drawdown: "
        f"{oos['dd']:.2f}R"
    )

    print(
        f"Longest losing streak: "
        f"{oos['longest']}"
    )

    print(
        f"Trades/week: "
        f"{oos['trades_per_week']:.2f}"
    )

    rows.append(
        {
            "market":
                market,

            "period":
                name,

            "signal":
                candidate["signal"],

            "threshold":
                best["threshold"],

            "rr":
                candidate["params"][0],

            "wick":
                candidate["params"][1],

            "body":
                candidate["params"][2],

            "separation":
                candidate["params"][3],

            "max_cross":
                candidate["params"][4],

            "hours":
                ",".join(
                    map(
                        str,
                        candidate[
                            "params"
                        ][5],
                    )
                ),

            "train_trades":
                train["trades"],

            "train_wr":
                train["win_rate"],

            "train_r":
                train["total_r"],

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

            "oos_pf":
                oos["pf"],

            "oos_dd":
                oos["dd"],

            "oos_losing_streak":
                oos["longest"],

            "oos_trades_per_week":
                oos["trades_per_week"],
        }
    )

    return oos


# ============================================================
# MARKET
# ============================================================

def run_market(
    market,
    path,
):

    print()
    print("=" * 60)

    print(
        f"{market} V10 PRECISION "
        "OPTIMIZER"
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
            print("=" * 60)

            print(
                f"{market} PERIOD FAILED"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            print("=" * 60)

    result = pd.DataFrame(
        rows
    )

    output = (
        "data/"
        f"{market.lower()}_"
        "optimizer_v10_results.csv"
    )

    result.to_csv(
        output,
        index=False,
    )

    return result


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    market,
    result,
):

    if result.empty:
        return None

    trades = int(
        result[
            "oos_trades"
        ].sum()
    )

    wins = int(
        result[
            "oos_wins"
        ].sum()
    )

    total_r = float(
        result[
            "oos_total_r"
        ].sum()
    )

    periods = len(
        result
    )

    profitable = int(
        (
            result[
                "oos_total_r"
            ]
            > 0
        ).sum()
    )

    stable = int(
        (
            result[
                "oos_win_rate"
            ]
            >= 60
        ).sum()
    )

    wr = (
        wins
        / trades
        * 100
        if trades
        else 0
    )

    avg_frequency = float(
        result[
            "oos_trades_per_week"
        ].mean()
    )

    avg_dd = float(
        result[
            "oos_dd"
        ].mean()
    )

    if (
        trades >= 200
        and wr >= 82
        and total_r > 0
    ):

        verdict = (
            "TARGET ACHIEVED"
        )

    elif (
        trades >= 100
        and wr >= 75
        and total_r > 0
    ):

        verdict = (
            "VERY PROMISING"
        )

    elif (
        trades >= 50
        and wr >= 65
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
                wr,
                2,
            ),

        "oos_total_r":
            round(
                total_r,
                2,
            ),

        "profitable_periods":
            f"{profitable}/{periods}",

        "stable_periods":
            f"{stable}/{periods}",

        "avg_trades_per_week":
            round(
                avg_frequency,
                3,
            ),

        "avg_drawdown":
            round(
                avg_dd,
                2,
            ),

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
        "OPTIMIZER V10"
    )

    print("=" * 60)

    print(
        "PRECISION FILTERING: ENABLED"
    )

    print(
        "MINIMUM SAMPLE FILTER: ENABLED"
    )

    print(
        "QUALITY SCORE: ENABLED"
    )

    print(
        "THRESHOLD SEARCH: ENABLED"
    )

    print(
        "PARAMETER STABILITY: ENABLED"
    )

    print(
        "MARKET-SPECIFIC OPTIMISATION: ENABLED"
    )

    print(
        "WALK-FORWARD TESTING: ENABLED"
    )

    print(
        "GENUINE OOS VALIDATION: ENABLED"
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

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            print("=" * 60)

            results[
                market
            ] = pd.DataFrame()

    # ========================================================
    # SUMMARY
    # ========================================================

    summaries = []

    for market, result in (
        results.items()
    ):

        summary = summarize(
            market,
            result,
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
        "V10 FINAL MULTI-MARKET "
        "SUMMARY"
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

    # ========================================================
    # COMBINED OOS
    # ========================================================

    total_trades = 0
    total_wins = 0
    total_r = 0.0
    profitable_periods = 0
    total_periods = 0

    for result in (
        results.values()
    ):

        if result.empty:
            continue

        total_trades += int(
            result[
                "oos_trades"
            ].sum()
        )

        total_wins += int(
            result[
                "oos_wins"
            ].sum()
        )

        total_r += float(
            result[
                "oos_total_r"
            ].sum()
        )

        profitable_periods += int(
            (
                result[
                    "oos_total_r"
                ]
                > 0
            ).sum()
        )

        total_periods += len(
            result
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
        f"Trades: "
        f"{total_trades}"
    )

    print(
        f"Wins: "
        f"{total_wins}"
    )

    print(
        f"Win rate: "
        f"{combined_wr:.2f}%"
    )

    print(
        f"Total R: "
        f"{total_r:.2f}"
    )

    print(
        f"Profitable periods: "
        f"{profitable_periods}/"
        f"{total_periods}"
    )

    # ========================================================
    # TARGET
    # ========================================================

    print()
    print("=" * 60)

    print(
        "V10 TARGET CHECK"
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

    print(
        "POSITIVE TOTAL R"
    )

    print()

    if (
        total_trades >= 200
        and combined_wr >= 82
        and total_r > 0
    ):

        print(
            "TARGET STATUS: "
            "ACHIEVED"
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

    # ========================================================
    # SAFETY
    # ========================================================

    print()
    print(
        "IMPORTANT"
    )

    print(
        "OOS data was never used "
        "for optimisation."
    )

    print(
        "Tiny high-win-rate samples "
        "are explicitly rejected."
    )

    print(
        "Each market is optimised "
        "independently."
    )

    print(
        "This is research only."
    )

    print(
        "DO NOT IMPLEMENT LIVE "
        "FROM THIS OPTIMIZER ALONE."
    )

    # ========================================================
    # SAVE
    # ========================================================

    output = (
        "data/"
        "multi_market_optimizer_v10_summary.csv"
    )

    summary_df.to_csv(
        output,
        index=False,
    )

    print()
    print(
        "Results saved:"
    )

    print(
        "data/"
        "xauusd_optimizer_v10_results.csv"
    )

    print(
        "data/"
        "eurusd_optimizer_v10_results.csv"
    )

    print(
        "data/"
        "multi_market_optimizer_v10_summary.csv"
    )

    print()
    print("=" * 60)

    print(
        "OPTIMIZER V10 COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
