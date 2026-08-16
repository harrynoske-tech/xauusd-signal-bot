import os
import itertools
import numpy as np
import pandas as pd


# ============================================================
# MULTI-MARKET STRATEGY OPTIMIZER V9.1
# ============================================================
#
# FAST MULTI-SIGNAL ENSEMBLE
#
# V9 was too computationally expensive:
# 5 signal families x 4,320 combinations x 3 periods
# x 2 markets.
#
# V9.1 uses a TWO-STAGE SEARCH:
#
# PHASE 1:
#   Fast coarse search across all signal families.
#
# PHASE 2:
#   Only the strongest candidates receive the full test.
#
# This allows the complete walk-forward process to finish.
#
# SIGNAL FAMILIES
# ----------------
# 1. TREND_PULLBACK
# 2. TREND_MOMENTUM
# 3. VOLATILITY_EXPANSION
# 4. REJECTION_REVERSAL
# 5. EMA_STRUCTURE
#
# IMPORTANT
# ---------
# OOS data is NEVER used for optimisation.
#
# TARGET
# ------
# ~82% win rate
# 200+ genuinely OOS trades
# Positive total R
# Robust across XAUUSD + EURUSD
#
# NO LIVE TRADING
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
# PHASE 1 — COARSE SEARCH
# ============================================================

COARSE_RR = [
    0.60,
    0.90,
    1.20,
]

COARSE_WICK = [
    0.20,
    0.30,
]

COARSE_BODY = [
    0.20,
    0.30,
]

COARSE_SEPARATION = [
    0.0005,
    0.0010,
]

COARSE_CROSS = [
    20,
    40,
]

COARSE_HOURS = [
    (3, 4, 5),
    (3, 4, 5, 12, 13),
]


# ============================================================
# PHASE 2 — FINE SEARCH
# ============================================================

FINE_RR = [
    0.50,
    0.60,
    0.75,
    0.90,
    1.00,
    1.25,
]

FINE_WICK = [
    0.20,
    0.25,
    0.30,
    0.35,
]

FINE_BODY = [
    0.20,
    0.25,
    0.30,
    0.35,
]

FINE_SEPARATION = [
    0.0005,
    0.0008,
    0.0010,
]

FINE_CROSS = [
    20,
    30,
    40,
]

FINE_HOURS = [
    (2, 3, 4),
    (3, 4, 5),
    (2, 3, 4, 5),
    (3, 4, 5, 12, 13),
    (2, 3, 4, 5, 12, 13),
]


# ============================================================
# CONTROL PARAMETERS
# ============================================================

MIN_TRAIN_TRADES = 20
MIN_SIGNAL_TRADES = 10

TOP_COARSE_PER_SIGNAL = 8

MAX_ENSEMBLE_SIGNALS = 3

MAX_HOLD_BARS = 96

# Require at least this many trades for a
# candidate to be considered statistically useful.
MIN_USEFUL_TRADES = 15


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

    separation20_50 = np.divide(
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

    high8 = (
        pd.Series(h)
        .rolling(
            8,
            min_periods=1,
        )
        .max()
        .to_numpy()
    )

    low8 = (
        pd.Series(l)
        .rolling(
            8,
            min_periods=1,
        )
        .min()
        .to_numpy()
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

    range_expansion = np.divide(
        candle_range,
        atr,
        out=np.ones_like(c),
        where=atr > 0,
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

        if previous_state != current_state:

            last_cross = i

        if last_cross >= 0:

            trend_cross_age[i] = (
                i - last_cross
            )

    bearish = (
        c < o
    )

    bullish = (
        c > o
    )

    bearish3 = (
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

    bearish5 = (
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

    bullish3 = (
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

    bullish5 = (
        pd.Series(
            bullish.astype(int)
        )
        .rolling(
            5,
            min_periods=1,
        )
        .sum()
        .to_numpy()
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
        "atr_ratio": atr_ratio,
        "atr_acceleration":
            atr_acceleration,

        "momentum4": momentum4,
        "momentum8": momentum8,
        "momentum16": momentum16,

        "separation20_50":
            separation20_50,

        "distance20": distance20,
        "distance50": distance50,

        "high8": high8,
        "low8": low8,

        "close_location":
            close_location,

        "range_expansion":
            range_expansion,

        "range_change":
            range_change,

        "trend_cross_age":
            trend_cross_age,

        "bearish3": bearish3,
        "bearish5": bearish5,

        "bullish3": bullish3,
        "bullish5": bullish5,

        "hours": hours,
    }


# ============================================================
# TRADE ENGINE
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

            # A trade that reaches neither
            # target nor stop is closed at
            # the final available price.
            exit_price = d[
                "close"
            ][exit_idx]

            move = (
                entry
                - exit_price
            )

            result = (
                move / atr
            )

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

    wins = (
        values > 0
    )

    losses = (
        values < 0
    )

    trade_count = len(
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

    drawdown = float(
        np.max(
            peak - equity
        )
    )

    losing_streak = 0
    longest_streak = 0

    for value in values:

        if value < 0:

            losing_streak += 1

            longest_streak = max(
                longest_streak,
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
        / 86400,
        1,
    )

    return {
        "trades":
            trade_count,

        "wins":
            win_count,

        "losses":
            int(losses.sum()),

        "win_rate":
            win_count
            / trade_count
            * 100,

        "total_r":
            float(values.sum()),

        "profit_factor":
            float(pf),

        "drawdown":
            drawdown,

        "longest_losing_streak":
            longest_streak,

        "trades_per_week":
            trade_count
            / (days / 7),
    }


# ============================================================
# SIGNAL FAMILIES
# ============================================================

def signal_trend_pullback(
    d,
    p,
):

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
            d["separation20_50"]
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


def signal_trend_momentum(
    d,
    p,
):

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
            d["separation20_50"]
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


def signal_volatility_expansion(
    d,
    p,
):

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


def signal_rejection_reversal(
    d,
    p,
):

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
            d["bearish5"]
            >= 3
        )

        & (
            d["distance50"]
            < 0
        )
    )

    return np.flatnonzero(
        mask
    )


def signal_ema_structure(
    d,
    p,
):

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
            < np.roll(
                d["ema9"],
                1,
            )
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
        signal_trend_pullback,

    "TREND_MOMENTUM":
        signal_trend_momentum,

    "VOLATILITY_EXPANSION":
        signal_volatility_expansion,

    "REJECTION_REVERSAL":
        signal_rejection_reversal,

    "EMA_STRUCTURE":
        signal_ema_structure,
}


# ============================================================
# FAST SEARCH
# ============================================================

def search_signal(
    d,
    timestamps,
    signal_name,
    signal_function,
    train_start,
    train_end,
    fine=False,
):

    if fine:

        grid = itertools.product(
            FINE_RR,
            FINE_WICK,
            FINE_BODY,
            FINE_SEPARATION,
            FINE_CROSS,
            FINE_HOURS,
        )

    else:

        grid = itertools.product(
            COARSE_RR,
            COARSE_WICK,
            COARSE_BODY,
            COARSE_SEPARATION,
            COARSE_CROSS,
            COARSE_HOURS,
        )

    bounds = get_bounds(
        timestamps,
        train_start,
        train_end,
    )

    if bounds is None:
        return []

    start_idx, end_idx = bounds

    candidates = []

    for p in grid:

        indices = signal_function(
            d,
            p,
        )

        if len(indices) < MIN_SIGNAL_TRADES:

            continue

        trades = simulate(
            d,
            indices,
            p[0],
            start_idx,
            end_idx,
        )

        m = calculate_metrics(
            trades,
            timestamps[start_idx],
            timestamps[end_idx],
        )

        if m is None:
            continue

        if m["trades"] < MIN_SIGNAL_TRADES:
            continue

        # ----------------------------------------------------
        # Training score.
        #
        # Win rate is important, but we deliberately include
        # total R, PF, sample size and drawdown so the optimiser
        # cannot simply select tiny high-WR samples.
        # ----------------------------------------------------

        score = (
            m["win_rate"]
            * 1.2
            + m["total_r"]
            * 3.0
            + min(
                m["profit_factor"],
                4.0,
            )
            * 8.0
            + np.log1p(
                m["trades"]
            )
            * 3.0
            - m["drawdown"]
            * 2.5
        )

        if (
            m["trades"]
            < MIN_USEFUL_TRADES
        ):

            score -= 10

        candidates.append(
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

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return candidates[
        :TOP_COARSE_PER_SIGNAL
    ]


# ============================================================
# ENSEMBLE
# ============================================================

def build_ensemble(
    d,
    timestamps,
    candidates,
    train_start,
    train_end,
):

    valid = []

    for signal_name in SIGNALS:

        signal_candidates = [
            x
            for x in candidates
            if x["signal"]
            == signal_name
        ]

        if not signal_candidates:
            continue

        valid.append(
            signal_candidates[0]
        )

    if not valid:
        return None

    print()
    print(
        "TOP SIGNAL CANDIDATES"
    )

    print("-" * 60)

    for c in valid:

        m = c["metrics"]

        print(
            f"{c['signal']:24s} "
            f"Trades {m['trades']:3d} | "
            f"WR {m['win_rate']:6.2f}% | "
            f"R {m['total_r']:7.2f} | "
            f"PF {m['profit_factor']:5.2f}"
        )

    bounds = get_bounds(
        timestamps,
        train_start,
        train_end,
    )

    if bounds is None:
        return None

    start_idx, end_idx = bounds

    best = None

    # --------------------------------------------------------
    # Test single signals and combinations.
    # Maximum 3 signals.
    # --------------------------------------------------------

    for size in range(
        1,
        min(
            MAX_ENSEMBLE_SIGNALS,
            len(valid),
        ) + 1,
    ):

        for combination in itertools.combinations(
            valid,
            size,
        ):

            union = np.unique(
                np.concatenate(
                    [
                        x["indices"]
                        for x in combination
                    ]
                )
            )

            # Use the RR belonging to the strongest
            # candidate in the ensemble.
            strongest = max(
                combination,
                key=lambda x:
                x["score"],
            )

            rr = strongest[
                "params"
            ][0]

            trades = simulate(
                d,
                union,
                rr,
                start_idx,
                end_idx,
            )

            m = calculate_metrics(
                trades,
                timestamps[start_idx],
                timestamps[end_idx],
            )

            if m is None:
                continue

            if (
                m["trades"]
                < MIN_TRAIN_TRADES
            ):
                continue

            score = (
                m["win_rate"]
                * 1.5
                + m["total_r"]
                * 3.0
                + min(
                    m["profit_factor"],
                    4.0,
                )
                * 10.0
                + np.log1p(
                    m["trades"]
                )
                * 4.0
                - m["drawdown"]
                * 2.0
            )

            # ------------------------------------------------
            # Penalise duplicate signals.
            # ------------------------------------------------

            duplicate_penalty = 0.0

            for a in range(
                len(combination)
            ):

                for b in range(
                    a + 1,
                    len(combination)
                ):

                    ia = set(
                        combination[a][
                            "indices"
                        ].tolist()
                    )

                    ib = set(
                        combination[b][
                            "indices"
                        ].tolist()
                    )

                    union_size = len(
                        ia | ib
                    )

                    if union_size:

                        overlap = (
                            len(
                                ia & ib
                            )
                            / union_size
                        )

                        duplicate_penalty += (
                            overlap * 15
                        )

            score -= (
                duplicate_penalty
            )

            candidate = {
                "signals":
                    combination,

                "indices":
                    union,

                "rr":
                    rr,

                "metrics":
                    m,

                "score":
                    score,
            }

            if (
                best is None
                or candidate["score"]
                > best["score"]
            ):

                best = candidate

    if best is None:
        return None

    print()
    print(
        "SELECTED TRAINING ENSEMBLE"
    )

    print("-" * 60)

    for signal in best[
        "signals"
    ]:

        print(
            f"  {signal['signal']}"
        )

    m = best[
        "metrics"
    ]

    print()
    print(
        f"Training trades: "
        f"{m['trades']}"
    )

    print(
        f"Training WR: "
        f"{m['win_rate']:.2f}%"
    )

    print(
        f"Training R: "
        f"{m['total_r']:.2f}"
    )

    print(
        f"Training PF: "
        f"{m['profit_factor']:.2f}"
    )

    print(
        f"Training DD: "
        f"{m['drawdown']:.2f}R"
    )

    print(
        f"RR: "
        f"{best['rr']}"
    )

    return best


# ============================================================
# OOS TEST
# ============================================================

def test_oos(
    d,
    timestamps,
    ensemble,
    test_start,
    test_end,
):

    bounds = get_bounds(
        timestamps,
        test_start,
        test_end,
    )

    if bounds is None:
        return None

    start_idx, end_idx = bounds

    indices = ensemble[
        "indices"
    ]

    oos_indices = indices[
        (indices >= start_idx)
        & (indices <= end_idx)
    ]

    if len(oos_indices) == 0:
        return None

    trades = simulate(
        d,
        oos_indices,
        ensemble["rr"],
        start_idx,
        end_idx,
    )

    return calculate_metrics(
        trades,
        timestamps[start_idx],
        timestamps[end_idx],
    )


# ============================================================
# ONE WALK-FORWARD PERIOD
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
        f"{market} V9.1: "
        f"{period_name}"
    )

    print("=" * 60)

    # ========================================================
    # PHASE 1
    # ========================================================

    print(
        "PHASE 1: FAST COARSE SEARCH"
    )

    all_candidates = []

    for signal_name, signal_function in (
        SIGNALS.items()
    ):

        print(
            f"{signal_name}: "
            "coarse search"
        )

        candidates = search_signal(
            d,
            timestamps,
            signal_name,
            signal_function,
            train_start,
            train_end,
            fine=False,
        )

        all_candidates.extend(
            candidates
        )

        if candidates:

            best = candidates[0]
            m = best["metrics"]

            print(
                f"  Best: "
                f"{m['trades']} trades | "
                f"{m['win_rate']:.2f}% WR | "
                f"{m['total_r']:.2f}R"
            )

        else:

            print(
                "  No viable candidate."
            )

    if not all_candidates:

        print(
            "No coarse candidates."
        )

        return None

    # ========================================================
    # PHASE 2
    # ========================================================

    print()
    print(
        "PHASE 2: FINE SEARCH"
    )

    # Only refine the best few coarse candidates.
    #
    # This is the major speed improvement over V9.
    # ========================================================

    coarse_best = {}

    for candidate in all_candidates:

        name = candidate[
            "signal"
        ]

        if (
            name not in coarse_best
            or candidate["score"]
            > coarse_best[name][
                "score"
            ]
        ):

            coarse_best[name] = (
                candidate
            )

    refined = []

    for signal_name, signal_function in (
        SIGNALS.items()
    ):

        if signal_name not in coarse_best:
            continue

        print(
            f"{signal_name}: "
            "fine search"
        )

        candidates = search_signal(
            d,
            timestamps,
            signal_name,
            signal_function,
            train_start,
            train_end,
            fine=True,
        )

        refined.extend(
            candidates
        )

        if candidates:

            best = candidates[0]
            m = best["metrics"]

            print(
                f"  Fine best: "
                f"{m['trades']} trades | "
                f"{m['win_rate']:.2f}% WR | "
                f"{m['total_r']:.2f}R"
            )

    if not refined:

        print(
            "No refined candidates."
        )

        return None

    # ========================================================
    # PHASE 3
    # ========================================================

    print()
    print(
        "PHASE 3: ENSEMBLE SELECTION"
    )

    ensemble = build_ensemble(
        d,
        timestamps,
        refined,
        train_start,
        train_end,
    )

    if ensemble is None:

        print(
            "No valid ensemble."
        )

        return None

    # ========================================================
    # PHASE 4 — OOS
    # ========================================================

    print()
    print(
        "PHASE 4: COMPLETELY "
        "OUT-OF-SAMPLE TEST"
    )

    print("-" * 60)

    oos = test_oos(
        d,
        timestamps,
        ensemble,
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

    signal_names = "|".join(
        x["signal"]
        for x in ensemble[
            "signals"
        ]
    )

    output_rows.append(
        {
            "market":
                market,

            "period":
                period_name,

            "signals":
                signal_names,

            "rr":
                ensemble["rr"],

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
# MARKET
# ============================================================

def run_market(
    market,
    path,
):

    print()
    print("=" * 60)

    print(
        f"{market} V9.1 "
        "MULTI-SIGNAL ENSEMBLE"
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

    output_rows = []

    for period in PERIODS:

        try:

            run_period(
                market,
                d,
                timestamps,
                period,
                output_rows,
            )

        except Exception as error:

            print()
            print(
                "=" * 60
            )

            print(
                f"{market} PERIOD FAILED"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                "=" * 60
            )

    result = pd.DataFrame(
        output_rows
    )

    output_path = (
        "data/"
        f"{market.lower()}_"
        "optimizer_v9_1_results.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    return result


# ============================================================
# SUMMARY
# ============================================================

def make_summary(
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

    profitable_periods = int(
        (
            result[
                "oos_total_r"
            ]
            > 0
        ).sum()
    )

    stable_periods = int(
        (
            result[
                "oos_win_rate"
            ]
            >= 55
        ).sum()
    )

    periods = len(
        result
    )

    win_rate = (
        wins
        / trades
        * 100
        if trades
        else 0
    )

    average_frequency = float(
        result[
            "oos_trades_per_week"
        ].mean()
    )

    average_dd = float(
        result[
            "oos_drawdown"
        ].mean()
    )

    if (
        trades >= 200
        and win_rate >= 82
        and total_r > 0
        and profitable_periods >= 2
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

        "stable_periods":
            f"{stable_periods}/"
            f"{periods}",

        "avg_trades_per_week":
            round(
                average_frequency,
                3,
            ),

        "avg_drawdown":
            round(
                average_dd,
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
        "OPTIMIZER V9.1"
    )

    print("=" * 60)

    print(
        "MULTI-SIGNAL ENSEMBLE: ENABLED"
    )

    print(
        "TWO-STAGE SEARCH: ENABLED"
    )

    print(
        "COARSE SEARCH: ENABLED"
    )

    print(
        "FINE SEARCH: ENABLED"
    )

    print(
        "WALK-FORWARD TESTING: ENABLED"
    )

    print(
        "CURRENT-ERA DATA: ENABLED"
    )

    print(
        "OOS VALIDATION: ENABLED"
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

    # ========================================================
    # MARKET SUMMARY
    # ========================================================

    summaries = []

    for market, result in (
        results.items()
    ):

        summary = make_summary(
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
        "V9.1 FINAL MULTI-MARKET "
        "SUMMARY"
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
        "V9.1 TARGET CHECK"
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
        "IMPORTANT:"
    )

    print(
        "OOS data is never used "
        "for optimisation."
    )

    print(
        "Training and OOS periods "
        "remain separated."
    )

    print(
        "This is research only."
    )

    print(
        "Do not implement live "
        "from this optimizer alone."
    )

    # ========================================================
    # SAVE
    # ========================================================

    summary_path = (
        "data/"
        "multi_market_optimizer_v9_1_summary.csv"
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
        "xauusd_optimizer_v9_1_results.csv"
    )

    print(
        "data/"
        "eurusd_optimizer_v9_1_results.csv"
    )

    print(
        "data/"
        "multi_market_optimizer_v9_1_summary.csv"
    )

    print()
    print("=" * 60)

    print(
        "OPTIMIZER V9.1 COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
