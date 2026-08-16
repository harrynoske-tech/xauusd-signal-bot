import os
import itertools
import numpy as np
import pandas as pd


# ============================================================
# MULTI-MARKET STRATEGY OPTIMIZER V9
# ============================================================
#
# MULTI-SIGNAL ENSEMBLE
#
# V8 proved that adaptive scoring can separate setups during
# training, but the OOS sample was too small.
#
# V9 therefore tests several independently defined signal
# families and combines only signals that demonstrate an edge
# during TRAINING.
#
# SIGNALS
# -------
# 1. TREND PULLBACK
# 2. TREND MOMENTUM
# 3. VOLATILITY EXPANSION
# 4. REJECTION / REVERSAL
# 5. EMA STRUCTURE
#
# IMPORTANT:
# - No OOS data is used for optimisation.
# - No live trading.
# - Every OOS result is genuinely unseen by the optimiser.
#
# TARGET:
# ~82% win rate
# 200+ genuinely OOS trades
# Positive R
# Robust across XAUUSD + EURUSD
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


MIN_TRAIN_TRADES = 25
MIN_SIGNAL_TRADES = 12
MIN_OOS_TRADES = 5

RECENT_DAYS = 365


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
# DATA
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

    atr50 = (
        pd.Series(atr)
        .rolling(
            50,
            min_periods=20,
        )
        .mean()
        .to_numpy()
    )

    atr100 = (
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
        ema20 - np.roll(ema20, 4),
        np.where(
            ema20 == 0,
            1,
            ema20,
        ),
    )

    ema50_slope = np.divide(
        ema50 - np.roll(ema50, 8),
        np.where(
            ema50 == 0,
            1,
            ema50,
        ),
    )

    ema100_slope = np.divide(
        ema100 - np.roll(ema100, 12),
        np.where(
            ema100 == 0,
            1,
            ema100,
        ),
    )

    ema200_slope = np.divide(
        ema200 - np.roll(ema200, 16),
        np.where(
            ema200 == 0,
            1,
            ema200,
        ),
    )

    momentum4 = np.divide(
        c - np.roll(c, 4),
        np.where(
            np.roll(c, 4) == 0,
            1,
            np.roll(c, 4),
        ),
    )

    momentum8 = np.divide(
        c - np.roll(c, 8),
        np.where(
            np.roll(c, 8) == 0,
            1,
            np.roll(c, 8),
        ),
    )

    momentum16 = np.divide(
        c - np.roll(c, 16),
        np.where(
            np.roll(c, 16) == 0,
            1,
            np.roll(c, 16),
        ),
    )

    separation20_50 = np.divide(
        np.abs(ema20 - ema50),
        np.where(
            c == 0,
            1,
            c,
        ),
    )

    separation50_100 = np.divide(
        np.abs(ema50 - ema100),
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

    high20 = (
        pd.Series(h)
        .rolling(
            20,
            min_periods=1,
        )
        .max()
        .to_numpy()
    )

    low20 = (
        pd.Series(l)
        .rolling(
            20,
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

    prior_range = np.roll(
        candle_range,
        1,
    )

    range_change = np.divide(
        candle_range,
        prior_range,
        out=np.ones_like(c),
        where=prior_range > 0,
    )

    # Age since EMA20/EMA50 relationship changed.
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

        "ema20_slope":
            ema20_slope,

        "ema50_slope":
            ema50_slope,

        "ema100_slope":
            ema100_slope,

        "ema200_slope":
            ema200_slope,

        "body_ratio":
            body_ratio,

        "upper_wick_ratio":
            upper_wick_ratio,

        "lower_wick_ratio":
            lower_wick_ratio,

        "atr":
            atr,

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

        "separation20_50":
            separation20_50,

        "separation50_100":
            separation50_100,

        "distance20":
            distance20,

        "distance50":
            distance50,

        "high8":
            high8,

        "low8":
            low8,

        "high20":
            high20,

        "low20":
            low20,

        "close_location":
            close_location,

        "range_expansion":
            range_expansion,

        "range_change":
            range_change,

        "trend_cross_age":
            trend_cross_age,

        "bearish3":
            bearish3,

        "bearish5":
            bearish5,

        "bullish3":
            bullish3,

        "bullish5":
            bullish5,

        "hours":
            hours,
    }


# ============================================================
# TRADE ENGINE
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


def simulate(
    d,
    indices,
    rr,
    start_idx,
    end_idx,
):

    trades = []

    exits = []

    next_available = (
        start_idx
    )

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

        r_value, exit_idx = result

        trades.append(
            float(r_value)
        )

        exits.append(
            int(exit_idx)
        )

        next_available = (
            exit_idx + 1
        )

    return trades, exits


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
            count,

        "wins":
            win_count,

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
            dd,

        "longest_losing_streak":
            longest,

        "trades_per_week":
            count
            / (days / 7),
    }


# ============================================================
# SIGNAL FAMILIES
# ============================================================

def trend_pullback(
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

        & (
            d["close"]
            > d["low8"]
        )
    )

    return np.flatnonzero(mask)


def trend_momentum(
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
            d["momentum16"]
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

    return np.flatnonzero(mask)


def volatility_expansion(
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

    return np.flatnonzero(mask)


def rejection_reversal(
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

    return np.flatnonzero(mask)


def ema_structure(
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

    return np.flatnonzero(mask)


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
# SIGNAL OPTIMISATION
# ============================================================

def optimise_signal(
    d,
    timestamps,
    signal_name,
    signal_function,
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

    best = None

    print()
    print(
        f"{signal_name}: "
        f"{len(combinations)} "
        f"combinations"
    )

    bounds = get_bounds(
        timestamps,
        train_start,
        train_end,
    )

    if bounds is None:
        return None

    start_idx, end_idx = bounds

    for number, params in enumerate(
        combinations,
        start=1,
    ):

        if (
            number == 1
            or number % 1000 == 0
            or number == len(combinations)
        ):

            print(
                f"{signal_name} progress: "
                f"{number}/"
                f"{len(combinations)}",
                flush=True,
            )

        indices = signal_function(
            d,
            params,
        )

        if len(indices) < MIN_SIGNAL_TRADES:
            continue

        trades, _ = simulate(
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

        if m["trades"] < MIN_SIGNAL_TRADES:
            continue

        # We want a genuine edge, not just
        # a high win rate with tiny sample.
        score = (
            m["win_rate"]
            * 1.0
            + m["total_r"]
            * 3.0
            + min(
                m["profit_factor"],
                3,
            )
            * 8.0
            - m["drawdown"]
            * 2.0
        )

        if m["trades"] < 20:
            score -= 5

        candidate = {
            "signal":
                signal_name,

            "params":
                params,

            "indices":
                indices,

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

    return best


# ============================================================
# ENSEMBLE SELECTION
# ============================================================

def build_ensemble(
    d,
    timestamps,
    signal_candidates,
    train_start,
    train_end,
):

    valid = [
        x
        for x in signal_candidates
        if x is not None
    ]

    if not valid:
        return None

    print()
    print(
        "SIGNAL CANDIDATES"
    )

    print("-" * 60)

    for candidate in valid:

        m = candidate[
            "metrics"
        ]

        print(
            f"{candidate['signal']:24s} "
            f"Trades {m['trades']:3d} | "
            f"WR {m['win_rate']:6.2f}% | "
            f"R {m['total_r']:7.2f} | "
            f"PF {m['profit_factor']:5.2f}"
        )

    # --------------------------------------------------------
    # Only retain signals that show a meaningful training edge.
    # --------------------------------------------------------

    strong = []

    for candidate in valid:

        m = candidate[
            "metrics"
        ]

        if (
            m["trades"] >= MIN_SIGNAL_TRADES
            and m["win_rate"] >= 55
            and m["total_r"] > 0
            and m["profit_factor"] > 1.05
        ):

            strong.append(
                candidate
            )

    if not strong:

        print()
        print(
            "No strong signals."
        )

        return None

    # --------------------------------------------------------
    # Test all useful combinations.
    # --------------------------------------------------------

    best = None

    for count in range(
        1,
        min(
            len(strong),
            4,
        ) + 1,
    ):

        for combination in itertools.combinations(
            strong,
            count,
        ):

            union = np.unique(
                np.concatenate(
                    [
                        x["indices"]
                        for x in combination
                    ]
                )
            )

            bounds = get_bounds(
                timestamps,
                train_start,
                train_end,
            )

            if bounds is None:
                continue

            start_idx, end_idx = bounds

            trades, exits = simulate(
                d,
                union,
                combination[0]["params"][0],
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

            if m["trades"] < MIN_TRAIN_TRADES:
                continue

            # Reward high WR, positive R,
            # PF and sample size.
            score = (
                m["win_rate"]
                * 1.5
                + m["total_r"]
                * 3.0
                + min(
                    m["profit_factor"],
                    3,
                )
                * 10
                + np.log1p(
                    m["trades"]
                ) * 5
                - m["drawdown"]
                * 2
            )

            # Penalise overly correlated
            # combinations which only duplicate
            # the same trades.
            trade_sets = [
                set(
                    x["indices"]
                    .tolist()
                )
                for x in combination
            ]

            overlap_penalty = 0.0

            for a in range(
                len(trade_sets)
            ):

                for b in range(
                    a + 1,
                    len(trade_sets)
                ):

                    union_size = len(
                        trade_sets[a]
                        | trade_sets[b]
                    )

                    intersection = len(
                        trade_sets[a]
                        & trade_sets[b]
                    )

                    if union_size > 0:

                        overlap_penalty += (
                            intersection
                            / union_size
                            * 10
                        )

            score -= (
                overlap_penalty
            )

            ensemble = {
                "signals":
                    combination,

                "indices":
                    union,

                "metrics":
                    m,

                "score":
                    score,
            }

            if (
                best is None
                or score
                > best["score"]
            ):

                best = ensemble

    if best is None:
        return None

    print()
    print(
        "SELECTED TRAINING ENSEMBLE"
    )

    print("-" * 60)

    print(
        "Signals:"
    )

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
        f"Trades: "
        f"{m['trades']}"
    )

    print(
        f"Win rate: "
        f"{m['win_rate']:.2f}%"
    )

    print(
        f"Total R: "
        f"{m['total_r']:.2f}"
    )

    print(
        f"Profit factor: "
        f"{m['profit_factor']:.2f}"
    )

    print(
        f"Drawdown: "
        f"{m['drawdown']:.2f}R"
    )

    return best


# ============================================================
# OOS ENSEMBLE
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

    rr = ensemble[
        "signals"
    ][0]["params"][0]

    trades, _ = simulate(
        d,
        oos_indices,
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
# RUN ONE PERIOD
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
        f"{market} V9: "
        f"{period_name}"
    )

    print("=" * 60)

    candidates = []

    for signal_name, signal_function in (
        SIGNALS.items()
    ):

        try:

            candidate = optimise_signal(
                d,
                timestamps,
                signal_name,
                signal_function,
                train_start,
                train_end,
            )

            if candidate is not None:
                candidates.append(
                    candidate
                )

        except Exception as error:

            print()
            print(
                f"{signal_name} FAILED: "
                f"{type(error).__name__}: "
                f"{error}"
            )

    ensemble = build_ensemble(
        d,
        timestamps,
        candidates,
        train_start,
        train_end,
    )

    if ensemble is None:
        print(
            "No valid ensemble."
        )
        return None

    print()
    print(
        "COMPLETELY OUT-OF-SAMPLE TEST"
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

    rows.append(
        {
            "market":
                market,

            "period":
                period_name,

            "signals":
                "|".join(
                    x["signal"]
                    for x in ensemble[
                        "signals"
                    ]
                ),

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
        f"{market} V9 MULTI-SIGNAL "
        "ENSEMBLE"
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
            print(
                f"{market} PERIOD FAILED"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

    result = pd.DataFrame(
        rows
    )

    path_out = (
        f"data/"
        f"{market.lower()}_"
        f"optimizer_v9_results.csv"
    )

    result.to_csv(
        path_out,
        index=False,
    )

    return result


# ============================================================
# SUMMARY
# ============================================================

def summarize(
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
        "MULTI-MARKET STRATEGY "
        "OPTIMIZER V9"
    )

    print("=" * 60)

    print(
        "MULTI-SIGNAL ENSEMBLE: ENABLED"
    )

    print(
        "TREND PULLBACK: ENABLED"
    )

    print(
        "TREND MOMENTUM: ENABLED"
    )

    print(
        "VOLATILITY EXPANSION: ENABLED"
    )

    print(
        "REJECTION / REVERSAL: ENABLED"
    )

    print(
        "EMA STRUCTURE: ENABLED"
    )

    print(
        "WALK-FORWARD TESTING: ENABLED"
    )

    print(
        "CURRENT-ERA DATA: ENABLED"
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
        "V9 FINAL MULTI-MARKET SUMMARY"
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

    for result in results.values():

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

    combined_wr = (
        total_wins
        / total_trades
        * 100
        if total_trades
        else 0
    )

    profitable_periods = 0
    total_periods = 0

    for result in results.values():

        if result.empty:
            continue

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

    print()
    print("=" * 60)

    print(
        "V9 TARGET CHECK"
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
        "OOS data is never used "
        "to optimise the signal."
    )

    print(
        "Do not implement live "
        "from this optimizer alone."
    )

    summary_path = (
        "data/"
        "multi_market_optimizer_v9_summary.csv"
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
        "xauusd_optimizer_v9_results.csv"
    )

    print(
        "data/"
        "eurusd_optimizer_v9_results.csv"
    )

    print(
        "data/"
        "multi_market_optimizer_v9_summary.csv"
    )

    print()
    print("=" * 60)

    print(
        "OPTIMIZER V9 COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
