import os
import itertools
import numpy as np
import pandas as pd


# ============================================================
# MULTI-MARKET STRATEGY OPTIMIZER V6
# REGIME-AWARE / CURRENT-ERA / WALK-FORWARD
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
# BASE STRATEGY PARAMETERS
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
    (2, 3),
    (3, 4),
    (4, 5),
    (2, 3, 4),
    (3, 4, 5),
    (2, 3, 4, 5),
    (3, 4, 5, 12, 13),
    (2, 3, 4, 5, 12, 13),
]


# ============================================================
# REGIME PARAMETERS
# ============================================================

# ATR relative to its longer-term average.
ATR_RATIO_VALUES = [
    0.75,
    0.90,
    1.00,
    1.10,
    1.25,
]

# Trend strength using EMA separation.
TREND_VALUES = [
    0.0005,
    0.0010,
    0.0015,
    0.0020,
]

# Maximum distance from EMA20 when entering.
PULLBACK_VALUES = [
    0.0010,
    0.0015,
    0.0020,
    0.0030,
]

# Minimum recent momentum.
MOMENTUM_VALUES = [
    0.0000,
    0.0005,
    0.0010,
]

# Current-era importance.
RECENT_DAYS = 365

MIN_TRAIN_TRADES = 25
MIN_RECENT_TRADES = 10
MIN_TEST_TRADES = 5

# Desired frequency.
TARGET_TPW_LOW = 0.05
TARGET_TPW_HIGH = 0.30


# ============================================================
# TIME
# ============================================================

def utc_timestamp(value):

    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")


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
        str(c).strip()
        for c in df.columns
    ]

    time_col = None

    for c in [
        "time",
        "Time",
        "timestamp",
        "Timestamp",
        "date",
        "Date",
    ]:
        if c in df.columns:
            time_col = c
            break

    if time_col is None:
        raise RuntimeError(
            "No timestamp column found."
        )

    df[time_col] = pd.to_datetime(
        df[time_col],
        utc=True,
    )

    df = df.set_index(time_col)

    rename = {}

    for c in df.columns:

        name = str(c).lower()

        if name == "open":
            rename[c] = "Open"

        elif name == "high":
            rename[c] = "High"

        elif name == "low":
            rename[c] = "Low"

        elif name == "close":
            rename[c] = "Close"

    df = df.rename(
        columns=rename
    )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    for c in required:

        if c not in df.columns:
            raise RuntimeError(
                f"Missing column: {c}"
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

    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)

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

    # --------------------------------------------------------
    # EMA separation
    # --------------------------------------------------------

    separation = (
        np.abs(
            ema20 - ema50
        )
        / np.where(
            c == 0,
            1,
            c,
        )
    )

    # --------------------------------------------------------
    # Candle structure
    # --------------------------------------------------------

    candle_range = h - l

    body_ratio = np.divide(
        np.abs(c - o),
        candle_range,
        out=np.zeros_like(c),
        where=candle_range > 0,
    )

    upper_wick_ratio = np.divide(
        h - np.maximum(o, c),
        candle_range,
        out=np.zeros_like(c),
        where=candle_range > 0,
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = np.roll(c, 1)

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
            min_periods=20,
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

    momentum = (
        c
        - np.roll(c, 8)
    ) / np.where(
        np.roll(c, 8) == 0,
        1,
        np.roll(c, 8),
    )

    # --------------------------------------------------------
    # Pullback distance
    # --------------------------------------------------------

    ema_distance = (
        np.abs(
            c - ema20
        )
        / np.where(
            c == 0,
            1,
            c,
        )
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

        bearish_cross = (
            ema20[i - 1]
            >= ema50[i - 1]
            and
            ema20[i]
            < ema50[i]
        )

        if bearish_cross:
            last_cross = i

        if last_cross >= 0:
            cross_age[i] = (
                i - last_cross
            )

    # --------------------------------------------------------
    # Recent high
    # --------------------------------------------------------

    recent_high = (
        pd.Series(h)
        .rolling(
            8,
            min_periods=1,
        )
        .max()
        .to_numpy()
    )

    return {
        "open": o,
        "high": h,
        "low": l,
        "close": c,

        "ema20": ema20,
        "ema50": ema50,
        "ema100": ema100,

        "ema20_slope":
            ema20_slope,

        "ema50_slope":
            ema50_slope,

        "separation":
            separation,

        "body_ratio":
            body_ratio,

        "upper_wick_ratio":
            upper_wick_ratio,

        "atr": atr,

        "atr_ratio":
            atr_ratio,

        "momentum":
            momentum,

        "ema_distance":
            ema_distance,

        "cross_age":
            cross_age,

        "recent_high":
            recent_high,

        "hours":
            df.index.hour.to_numpy(),
    }


# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_signals(
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
        atr_ratio_min,
        trend_min,
        pullback_max,
        momentum_min,
    ) = params

    mask = (

        # Session
        np.isin(
            d["hours"],
            hours,
        )

        # Bearish structure
        & (
            d["ema20"]
            < d["ema50"]
        )

        & (
            d["ema50"]
            < d["ema100"]
        )

        # Trend slope
        & (
            d["ema20_slope"]
            < 0
        )

        & (
            d["ema50_slope"]
            < 0
        )

        # EMA separation
        & (
            d["separation"]
            >= separation
        )

        # Recent bearish cross
        & (
            d["cross_age"]
            <= max_cross
        )

        # Volatility regime
        & (
            d["atr_ratio"]
            >= atr_ratio_min
        )

        # Trend strength
        & (
            d["separation"]
            >= trend_min
        )

        # Pullback isn't too deep
        & (
            d["ema_distance"]
            <= pullback_max
        )

        # Momentum confirmation
        & (
            d["momentum"]
            <= -momentum_min
        )

        # Bearish candle
        & (
            d["close"]
            < d["open"]
        )

        # Rejection wick
        & (
            d[
                "upper_wick_ratio"
            ]
            >= wick
        )

        # Real candle body
        & (
            d["body_ratio"]
            >= body
        )

        # Price below EMA
        & (
            d["close"]
            < d["ema20"]
        )
    )

    return np.flatnonzero(
        mask
    )


# ============================================================
# DATE BOUNDS
# ============================================================

def get_bounds(
    timestamps,
    start,
    end,
):

    start_ts = utc_timestamp(start)
    end_ts = utc_timestamp(end)

    left = np.flatnonzero(
        timestamps >= start_ts
    )

    right = np.flatnonzero(
        timestamps <= end_ts
    )

    if (
        len(left) == 0
        or len(right) == 0
    ):
        return None

    return (
        int(left[0]),
        int(right[-1]),
    )


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate(
    d,
    signals,
    rr,
    start_index,
    end_index,
):

    trades = []

    next_free = start_index

    for i in signals:

        if i < start_index:
            continue

        if i > end_index:
            break

        if i < next_free:
            continue

        entry = d["close"][i]

        stop = d["recent_high"][i]

        risk = stop - entry

        if risk <= 0:
            continue

        target = (
            entry
            - risk * rr
        )

        result = None
        exit_index = None

        for j in range(
            i + 1,
            end_index + 1,
        ):

            hit_stop = (
                d["high"][j]
                >= stop
            )

            hit_target = (
                d["low"][j]
                <= target
            )

            # Conservative handling if
            # both happen in the same candle.
            if (
                hit_stop
                and hit_target
            ):

                result = -1.0
                exit_index = j
                break

            if hit_stop:

                result = -1.0
                exit_index = j
                break

            if hit_target:

                result = rr
                exit_index = j
                break

        if result is not None:

            trades.append(
                result
            )

            next_free = (
                exit_index + 1
            )

    return trades


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    trades,
    days,
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

    loss_count = int(
        losses.sum()
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

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = 999.0

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

    longest_loss_streak = 0
    current_streak = 0

    for value in values:

        if value < 0:

            current_streak += 1

            longest_loss_streak = max(
                longest_loss_streak,
                current_streak,
            )

        else:

            current_streak = 0

    weeks = max(
        days / 7.0,
        1e-9,
    )

    return {
        "trades":
            count,

        "wins":
            win_count,

        "losses":
            loss_count,

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

        "longest_loss_streak":
            longest_loss_streak,

        "trades_per_week":
            count / weeks,
    }


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    d,
    timestamps,
    params,
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

    start_index, end_index = (
        bounds
    )

    rr = params[0]

    signals = generate_signals(
        d,
        params,
    )

    trades = simulate(
        d,
        signals,
        rr,
        start_index,
        end_index,
    )

    days = max(
        (
            timestamps[end_index]
            - timestamps[start_index]
        ).astype(
            "timedelta64[s]"
        ).astype(float)
        / 86400,
        1,
    )

    return calculate_metrics(
        trades,
        days,
    )


# ============================================================
# OPTIMIZER SCORE
# ============================================================

def score_strategy(
    training,
    recent,
):

    if (
        training is None
        or recent is None
    ):
        return -1e12

    if (
        training["trades"]
        < MIN_TRAIN_TRADES
    ):
        return -1e12

    if (
        recent["trades"]
        < MIN_RECENT_TRADES
    ):
        return -1e12

    score = 0.0

    # --------------------------------------------------------
    # Historical performance
    # --------------------------------------------------------

    score += (
        training["win_rate"]
        * 1.0
    )

    score += (
        training["total_r"]
        * 0.75
    )

    score += (
        min(
            training["profit_factor"],
            3.0,
        )
        * 5
    )

    score -= (
        training["drawdown"]
        * 1.0
    )

    # --------------------------------------------------------
    # CURRENT ERA IS HEAVILY WEIGHTED
    # --------------------------------------------------------

    score += (
        recent["win_rate"]
        * 8.0
    )

    score += (
        recent["total_r"]
        * 4.0
    )

    score += (
        min(
            recent["profit_factor"],
            3.0,
        )
        * 25
    )

    score -= (
        recent["drawdown"]
        * 5.0
    )

    # --------------------------------------------------------
    # SELECTIVITY
    # --------------------------------------------------------

    tpw = recent[
        "trades_per_week"
    ]

    if (
        TARGET_TPW_LOW
        <= tpw
        <= TARGET_TPW_HIGH
    ):

        score += 50

    elif tpw <= 0.50:

        score += 15

    else:

        score -= (
            tpw * 20
        )

    # --------------------------------------------------------
    # Penalise long losing streaks
    # --------------------------------------------------------

    score -= (
        recent[
            "longest_loss_streak"
        ]
        * 3
    )

    return score


# ============================================================
# OPTIMISE ONE PERIOD
# ============================================================

def optimise_period(
    d,
    timestamps,
    train_start,
    train_end,
):

    train_bounds = get_bounds(
        timestamps,
        train_start,
        train_end,
    )

    if train_bounds is None:
        return None

    recent_end = utc_timestamp(
        train_end
    )

    recent_start = (
        recent_end
        - pd.Timedelta(
            days=RECENT_DAYS
        )
    )

    combinations = itertools.product(
        RR_VALUES,
        WICK_VALUES,
        BODY_VALUES,
        SEPARATION_VALUES,
        MAX_CROSS_VALUES,
        HOUR_SETS,
        ATR_RATIO_VALUES,
        TREND_VALUES,
        PULLBACK_VALUES,
        MOMENTUM_VALUES,
    )

    best = None

    total = 0

    for params in combinations:

        total += 1

        training = evaluate(
            d,
            timestamps,
            params,
            train_start,
            train_end,
        )

        if training is None:
            continue

        if (
            training["trades"]
            < MIN_TRAIN_TRADES
        ):
            continue

        recent = evaluate(
            d,
            timestamps,
            params,
            recent_start,
            train_end,
        )

        if recent is None:
            continue

        if (
            recent["trades"]
            < MIN_RECENT_TRADES
        ):
            continue

        score = score_strategy(
            training,
            recent,
        )

        if (
            best is None
            or score
            > best["score"]
        ):

            best = {
                "score":
                    score,

                "params":
                    params,

                "training":
                    training,

                "recent":
                    recent,
            }

    print(
        f"Combinations tested: {total}"
    )

    return best


# ============================================================
# PARAMETER NEIGHBOUR TEST
# ============================================================

def stability_test(
    d,
    timestamps,
    params,
    train_start,
    train_end,
):

    (
        rr,
        wick,
        body,
        separation,
        max_cross,
        hours,
        atr_ratio_min,
        trend_min,
        pullback_max,
        momentum_min,
    ) = params

    def neighbours(
        values,
        selected,
        distance,
    ):

        return [
            x
            for x in values
            if abs(
                x - selected
            ) <= distance
        ]

    rr_values = neighbours(
        RR_VALUES,
        rr,
        0.25,
    )

    wick_values = neighbours(
        WICK_VALUES,
        wick,
        0.10,
    )

    body_values = neighbours(
        BODY_VALUES,
        body,
        0.10,
    )

    separation_values = neighbours(
        SEPARATION_VALUES,
        separation,
        0.0005,
    )

    cross_values = neighbours(
        MAX_CROSS_VALUES,
        max_cross,
        10,
    )

    atr_values = neighbours(
        ATR_RATIO_VALUES,
        atr_ratio_min,
        0.15,
    )

    trend_values = neighbours(
        TREND_VALUES,
        trend_min,
        0.0005,
    )

    pullback_values = neighbours(
        PULLBACK_VALUES,
        pullback_max,
        0.001,
    )

    momentum_values = neighbours(
        MOMENTUM_VALUES,
        momentum_min,
        0.0005,
    )

    hour_values = [
        hours
    ]

    for h in HOUR_SETS:

        overlap = len(
            set(h)
            & set(hours)
        )

        if overlap >= max(
            1,
            len(hours) - 1,
        ):

            if h not in hour_values:
                hour_values.append(h)

    results = []

    for candidate in itertools.product(
        rr_values,
        wick_values,
        body_values,
        separation_values,
        cross_values,
        hour_values,
        atr_values,
        trend_values,
        pullback_values,
        momentum_values,
    ):

        result = evaluate(
            d,
            timestamps,
            candidate,
            train_start,
            train_end,
        )

        if result is None:
            continue

        if (
            result["trades"]
            < MIN_TRAIN_TRADES
        ):
            continue

        results.append(
            result
        )

    if not results:

        return {
            "count": 0,
            "median_win_rate": 0,
            "median_r": 0,
            "positive_fraction": 0,
            "stable": False,
        }

    win_rates = np.array(
        [
            r["win_rate"]
            for r in results
        ]
    )

    total_r = np.array(
        [
            r["total_r"]
            for r in results
        ]
    )

    median_wr = float(
        np.median(
            win_rates
        )
    )

    median_r = float(
        np.median(
            total_r
        )
    )

    positive_fraction = float(
        np.mean(
            total_r > 0
        )
    )

    stable = (
        len(results) >= 10
        and median_wr >= 55
        and positive_fraction >= 0.60
    )

    return {
        "count":
            len(results),

        "median_win_rate":
            median_wr,

        "median_r":
            median_r,

        "positive_fraction":
            positive_fraction,

        "stable":
            stable,
    }


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
        f"{market} V6 "
        "REGIME-AWARE OPTIMIZER"
    )

    print("=" * 60)

    df = load_data(
        path
    )

    d = prepare_indicators(
        df
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

    results = []

    for (
        period,
        train_start,
        train_end,
        test_start,
        test_end,
    ) in PERIODS:

        print()
        print("=" * 60)

        print(
            f"{market} V6: "
            f"{period}"
        )

        print("=" * 60)

        print(
            "Optimising regime-aware "
            "strategy..."
        )

        best = optimise_period(
            d,
            timestamps,
            train_start,
            train_end,
        )

        if best is None:

            print(
                "NO VALID STRATEGY"
            )

            continue

        params = best["params"]

        training = best["training"]
        recent = best["recent"]

        (
            rr,
            wick,
            body,
            separation,
            max_cross,
            hours,
            atr_ratio_min,
            trend_min,
            pullback_max,
            momentum_min,
        ) = params

        print()
        print(
            "BEST TRAINING STRATEGY"
        )

        print("-" * 60)

        print(
            f"Training trades: "
            f"{training['trades']}"
        )

        print(
            f"Training win rate: "
            f"{training['win_rate']:.2f}%"
        )

        print(
            f"Training R: "
            f"{training['total_r']:.2f}"
        )

        print()

        print(
            f"Recent trades: "
            f"{recent['trades']}"
        )

        print(
            f"Recent win rate: "
            f"{recent['win_rate']:.2f}%"
        )

        print(
            f"Recent R: "
            f"{recent['total_r']:.2f}"
        )

        print(
            f"Recent PF: "
            f"{recent['profit_factor']:.2f}"
        )

        print(
            f"Recent trades/week: "
            f"{recent['trades_per_week']:.2f}"
        )

        print()
        print(
            "BASE PARAMETERS"
        )

        print(
            f"RR: {rr}"
        )

        print(
            f"Wick: {wick}"
        )

        print(
            f"Body: {body}"
        )

        print(
            f"EMA separation: "
            f"{separation}"
        )

        print(
            f"Max cross: "
            f"{max_cross}"
        )

        print(
            "Hours: "
            + ",".join(
                map(
                    str,
                    hours,
                )
            )
        )

        print()
        print(
            "REGIME FILTERS"
        )

        print(
            f"ATR ratio minimum: "
            f"{atr_ratio_min}"
        )

        print(
            f"Trend minimum: "
            f"{trend_min}"
        )

        print(
            f"Pullback maximum: "
            f"{pullback_max}"
        )

        print(
            f"Momentum minimum: "
            f"{momentum_min}"
        )

        # ----------------------------------------------------
        # Stability
        # ----------------------------------------------------

        print()
        print(
            "PARAMETER + REGIME "
            "STABILITY"
        )

        print("-" * 60)

        stability = stability_test(
            d,
            timestamps,
            params,
            train_start,
            train_end,
        )

        print(
            f"Nearby strategies: "
            f"{stability['count']}"
        )

        print(
            f"Median nearby win rate: "
            f"{stability['median_win_rate']:.2f}%"
        )

        print(
            f"Median nearby R: "
            f"{stability['median_r']:.2f}R"
        )

        print(
            f"Positive nearby strategies: "
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

        # ----------------------------------------------------
        # OOS
        # ----------------------------------------------------

        test = evaluate(
            d,
            timestamps,
            params,
            test_start,
            test_end,
        )

        print()
        print(
            "OUT-OF-SAMPLE RESULT"
        )

        print("-" * 60)

        if test is None:

            print(
                "No OOS trades."
            )

            continue

        print(
            f"Trades: "
            f"{test['trades']}"
        )

        print(
            f"Wins: "
            f"{test['wins']}"
        )

        print(
            f"Losses: "
            f"{test['losses']}"
        )

        print(
            f"Win rate: "
            f"{test['win_rate']:.2f}%"
        )

        print(
            f"Total R: "
            f"{test['total_r']:.2f}"
        )

        print(
            f"Profit factor: "
            f"{test['profit_factor']:.2f}"
        )

        print(
            f"Max drawdown: "
            f"{test['drawdown']:.2f}R"
        )

        print(
            f"Longest losing streak: "
            f"{test['longest_loss_streak']}"
        )

        print(
            f"Trades/week: "
            f"{test['trades_per_week']:.2f}"
        )

        results.append(
            {
                "market":
                    market,

                "period":
                    period,

                "train_trades":
                    training["trades"],

                "train_win_rate":
                    training["win_rate"],

                "train_r":
                    training["total_r"],

                "recent_trades":
                    recent["trades"],

                "recent_win_rate":
                    recent["win_rate"],

                "recent_r":
                    recent["total_r"],

                "recent_pf":
                    recent["profit_factor"],

                "test_trades":
                    test["trades"],

                "test_wins":
                    test["wins"],

                "test_losses":
                    test["losses"],

                "test_win_rate":
                    test["win_rate"],

                "test_r":
                    test["total_r"],

                "test_pf":
                    test["profit_factor"],

                "test_drawdown":
                    test["drawdown"],

                "test_losing_streak":
                    test[
                        "longest_loss_streak"
                    ],

                "test_tpw":
                    test[
                        "trades_per_week"
                    ],

                "stable":
                    stability["stable"],

                "nearby_count":
                    stability["count"],

                "nearby_median_win_rate":
                    stability[
                        "median_win_rate"
                    ],

                "nearby_median_r":
                    stability[
                        "median_r"
                    ],

                "nearby_positive_fraction":
                    stability[
                        "positive_fraction"
                    ],

                "rr": rr,
                "wick": wick,
                "body": body,
                "separation":
                    separation,
                "max_cross":
                    max_cross,

                "hours":
                    ",".join(
                        map(
                            str,
                            hours,
                        )
                    ),

                "atr_ratio_min":
                    atr_ratio_min,

                "trend_min":
                    trend_min,

                "pullback_max":
                    pullback_max,

                "momentum_min":
                    momentum_min,
            }
        )

    result_df = pd.DataFrame(
        results
    )

    output = (
        f"data/"
        f"{market.lower()}_"
        f"optimizer_v6_results.csv"
    )

    result_df.to_csv(
        output,
        index=False,
    )

    return result_df


# ============================================================
# MARKET SUMMARY
# ============================================================

def make_summary(
    market,
    df,
):

    if df.empty:
        return None

    trades = int(
        df[
            "test_trades"
        ].sum()
    )

    wins = int(
        df[
            "test_wins"
        ].sum()
    )

    total_r = float(
        df[
            "test_r"
        ].sum()
    )

    profitable = int(
        (
            df[
                "test_r"
            ]
            > 0
        ).sum()
    )

    stable = int(
        df[
            "stable"
        ].sum()
    )

    periods = len(df)

    win_rate = (
        wins
        / trades
        * 100
        if trades
        else 0
    )

    avg_tpw = float(
        df[
            "test_tpw"
        ].mean()
    )

    avg_dd = float(
        df[
            "test_drawdown"
        ].mean()
    )

    if (
        trades >= 30
        and win_rate >= 65
        and total_r > 0
        and profitable >= 2
        and stable >= 2
    ):

        verdict = "ROBUST"

    elif (
        trades >= 20
        and win_rate >= 60
        and total_r > 0
        and profitable >= 2
    ):

        verdict = "PROMISING"

    else:

        verdict = "NOT ROBUST"

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
            f"{profitable}/{periods}",

        "stable_periods":
            f"{stable}/{periods}",

        "avg_trades_per_week":
            round(
                avg_tpw,
                2,
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
        "OPTIMIZER V6"
    )

    print("=" * 60)

    print(
        "REGIME FILTERING: ENABLED"
    )

    print(
        "CURRENT-ERA WEIGHTING: ENABLED"
    )

    print(
        "WALK-FORWARD TESTING: ENABLED"
    )

    print(
        "PARAMETER STABILITY: ENABLED"
    )

    print(
        "TARGET: HIGH WIN RATE + "
        "POSITIVE R + LOW FREQUENCY"
    )

    print(
        "MARKETS: XAUUSD, EURUSD"
    )

    print(
        "NO LIVE TRADING"
    )

    market_results = {}

    for market, path in (
        MARKETS.items()
    ):

        try:

            market_results[
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

    summary_rows = []

    for market, df in (
        market_results.items()
    ):

        if (
            df is None
            or df.empty
        ):
            continue

        row = make_summary(
            market,
            df,
        )

        if row is not None:
            summary_rows.append(
                row
            )

    summary = pd.DataFrame(
        summary_rows
    )

    print()
    print("=" * 60)

    print(
        "V6 FINAL MULTI-MARKET "
        "SUMMARY"
    )

    print("=" * 60)

    if summary.empty:

        print(
            "No completed results."
        )

    else:

        print(
            summary.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # COMBINED RESULT
    # --------------------------------------------------------

    combined_trades = 0
    combined_wins = 0
    combined_r = 0.0

    profitable_periods = 0
    stable_periods = 0
    total_periods = 0

    for df in (
        market_results.values()
    ):

        if (
            df is None
            or df.empty
        ):
            continue

        combined_trades += int(
            df[
                "test_trades"
            ].sum()
        )

        combined_wins += int(
            df[
                "test_wins"
            ].sum()
        )

        combined_r += float(
            df[
                "test_r"
            ].sum()
        )

        profitable_periods += int(
            (
                df[
                    "test_r"
                ]
                > 0
            ).sum()
        )

        stable_periods += int(
            df[
                "stable"
            ].sum()
        )

        total_periods += len(
            df
        )

    combined_win_rate = (
        combined_wins
        / combined_trades
        * 100
        if combined_trades
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
        f"{combined_trades}"
    )

    print(
        f"Wins: "
        f"{combined_wins}"
    )

    print(
        f"Win rate: "
        f"{combined_win_rate:.2f}%"
    )

    print(
        f"Total R: "
        f"{combined_r:.2f}"
    )

    print(
        f"Profitable periods: "
        f"{profitable_periods}/"
        f"{total_periods}"
    )

    print(
        f"Stable periods: "
        f"{stable_periods}/"
        f"{total_periods}"
    )

    # --------------------------------------------------------
    # FINAL VERDICT
    # --------------------------------------------------------

    print()
    print("=" * 60)

    print(
        "V6 FINAL VERDICT"
    )

    print("=" * 60)

    robust_markets = 0

    for row in summary_rows:

        if (
            row["verdict"]
            == "ROBUST"
        ):
            robust_markets += 1

    if (
        len(summary_rows) >= 2
        and robust_markets >= 1
        and combined_win_rate >= 65
        and combined_r > 0
        and profitable_periods
        >= total_periods * 0.60
        and stable_periods
        >= total_periods * 0.60
    ):

        verdict = (
            "STRONG CANDIDATE"
        )

    elif (
        combined_win_rate >= 60
        and combined_r > 0
        and profitable_periods
        >= total_periods * 0.50
    ):

        verdict = "PROMISING"

    else:

        verdict = (
            "NOT ROBUST YET"
        )

    print(
        f"VERDICT: {verdict}"
    )

    print()

    if verdict == "STRONG CANDIDATE":

        print(
            "The regime-aware "
            "strategy has produced "
            "a strong cross-market "
            "out-of-sample result."
        )

        print(
            "Further validation is "
            "required before any "
            "live implementation."
        )

    elif verdict == "PROMISING":

        print(
            "The regime filters "
            "appear to improve "
            "the underlying edge."
        )

        print(
            "Continue validation "
            "before implementation."
        )

    else:

        print(
            "The regime filters "
            "have not produced "
            "a sufficiently robust "
            "edge yet."
        )

        print(
            "Do not implement live."
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    summary.to_csv(
        "data/"
        "multi_market_optimizer_v6_summary.csv",
        index=False,
    )

    print()
    print(
        "Results saved:"
    )

    print(
        "data/"
        "xauusd_optimizer_v6_results.csv"
    )

    print(
        "data/"
        "eurusd_optimizer_v6_results.csv"
    )

    print(
        "data/"
        "multi_market_optimizer_v6_summary.csv"
    )

    print()
    print("=" * 60)

    print(
        "OPTIMIZER V6 COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
