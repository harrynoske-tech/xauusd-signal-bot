import os
import itertools
import numpy as np
import pandas as pd


# ============================================================
# MULTI-MARKET STRATEGY OPTIMIZER V6.1
# FAST REGIME-AWARE OPTIMIZER
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
# BASE PARAMETERS
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
# REGIME PARAMETERS
# ============================================================

ATR_RATIO_VALUES = [
    0.90,
    1.00,
    1.10,
]

TREND_VALUES = [
    0.0005,
    0.0010,
]

PULLBACK_VALUES = [
    0.0015,
    0.0020,
    0.0030,
]

MOMENTUM_VALUES = [
    0.0000,
    0.0005,
]


# ============================================================
# SETTINGS
# ============================================================

RECENT_DAYS = 365

MIN_TRAIN_TRADES = 20
MIN_RECENT_TRADES = 10
MIN_TEST_TRADES = 5

TOP_BASE_STRATEGIES = 20
TOP_FINAL_STRATEGIES = 10

STABILITY_LIMIT = 100

# We still want relatively low frequency,
# but the optimizer should actively search
# around the user's goal of approximately
# one trade per week.
TARGET_TPW = 1.0


# ============================================================
# TIME
# ============================================================

def utc_timestamp(value):

    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")


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
            "No timestamp column found."
        )

    df[time_column] = pd.to_datetime(
        df[time_column],
        utc=True,
    )

    df = df.set_index(
        time_column
    )

    rename = {}

    for column in df.columns:

        name = str(
            column
        ).lower()

        if name == "open":
            rename[column] = "Open"

        elif name == "high":
            rename[column] = "High"

        elif name == "low":
            rename[column] = "Low"

        elif name == "close":
            rename[column] = "Close"

    df = df.rename(
        columns=rename
    )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    for column in required:

        if column not in df.columns:
            raise RuntimeError(
                f"Missing column: {column}"
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

    ema20_previous = np.roll(
        ema20,
        4,
    )

    ema50_previous = np.roll(
        ema50,
        4,
    )

    ema20_slope = (
        ema20 - ema20_previous
    ) / np.where(
        ema20 == 0,
        1,
        ema20,
    )

    ema50_slope = (
        ema50 - ema50_previous
    ) / np.where(
        ema50 == 0,
        1,
        ema50,
    )

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

    previous_8_close = np.roll(
        c,
        8,
    )

    momentum = (
        c - previous_8_close
    ) / np.where(
        previous_8_close == 0,
        1,
        previous_8_close,
    )

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
# BOUNDS
# ============================================================

def get_bounds(
    timestamps,
    start,
    end,
):

    start_ts = utc_timestamp(
        start
    )

    end_ts = utc_timestamp(
        end
    )

    left = np.searchsorted(
        timestamps,
        start_ts,
        side="left",
    )

    right = np.searchsorted(
        timestamps,
        end_ts,
        side="right",
    ) - 1

    if (
        left >= len(timestamps)
        or right < left
    ):
        return None

    return (
        int(left),
        int(right),
    )


# ============================================================
# BASE SIGNALS
# ============================================================

def generate_base_signals(
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
# REGIME FILTER
# ============================================================

def apply_regime_filter(
    d,
    signals,
    regime,
):

    (
        atr_min,
        trend_min,
        pullback_max,
        momentum_min,
    ) = regime

    if len(signals) == 0:
        return signals

    keep = (

        (
            d["atr_ratio"][signals]
            >= atr_min
        )

        & (
            d["separation"][signals]
            >= trend_min
        )

        & (
            d["ema_distance"][signals]
            <= pullback_max
        )

        & (
            d["momentum"][signals]
            <= -momentum_min
        )
    )

    return signals[
        keep
    ]


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

        stop = d[
            "recent_high"
        ][i]

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

    win_count = int(
        wins.sum()
    )

    count = len(values)

    losses = values < 0

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

    longest = 0
    current = 0

    for value in values:

        if value < 0:

            current += 1
            longest = max(
                longest,
                current,
            )

        else:

            current = 0

    days = max(
        (
            pd.Timestamp(
                end_time
            )
            - pd.Timestamp(
                start_time
            )
        ).total_seconds()
        / 86400.0,
        1.0,
    )

    weeks = days / 7.0

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

        "longest_loss_streak":
            longest,

        "trades_per_week":
            count
            / weeks,
    }


# ============================================================
# FAST EVALUATION
# ============================================================

def evaluate_signals(
    d,
    timestamps,
    signals,
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

    start_index, end_index = (
        bounds
    )

    trades = simulate(
        d,
        signals,
        rr,
        start_index,
        end_index,
    )

    return metrics(
        trades,
        timestamps[start_index],
        timestamps[end_index],
    )


# ============================================================
# SCORE
# ============================================================

def strategy_score(
    train,
    recent,
):

    if (
        train is None
        or recent is None
    ):
        return -1e9

    if train[
        "trades"
    ] < MIN_TRAIN_TRADES:

        return -1e9

    if recent[
        "trades"
    ] < MIN_RECENT_TRADES:

        return -1e9

    score = 0.0

    # Profit matters heavily.
    score += (
        train["total_r"]
        * 1.5
    )

    score += (
        recent["total_r"]
        * 5.0
    )

    # Win rate matters.
    score += (
        train["win_rate"]
        * 0.50
    )

    score += (
        recent["win_rate"]
        * 3.0
    )

    # Profit factor.
    score += (
        min(
            train["profit_factor"],
            3.0,
        )
        * 5
    )

    score += (
        min(
            recent["profit_factor"],
            3.0,
        )
        * 15
    )

    # Drawdown penalty.
    score -= (
        train["drawdown"]
        * 1.0
    )

    score -= (
        recent["drawdown"]
        * 3.0
    )

    # Prefer around one trade/week.
    tpw = recent[
        "trades_per_week"
    ]

    frequency_penalty = abs(
        tpw - TARGET_TPW
    )

    score -= (
        frequency_penalty
        * 8.0
    )

    # Losing streak penalty.
    score -= (
        recent[
            "longest_loss_streak"
        ]
        * 3.0
    )

    return score


# ============================================================
# OPTIMISE BASE STRATEGIES
# ============================================================

def optimise_base(
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

    total = len(
        combinations
    )

    print(
        f"BASE COMBINATIONS: {total}"
    )

    recent_end = utc_timestamp(
        train_end
    )

    recent_start = (
        recent_end
        - pd.Timedelta(
            days=RECENT_DAYS
        )
    )

    candidates = []

    for number, params in enumerate(
        combinations,
        start=1,
    ):

        if (
            number == 1
            or number % 250 == 0
            or number == total
        ):

            print(
                f"Base progress: "
                f"{number}/{total} "
                f"("
                f"{number / total * 100:.1f}%"
                f")",
                flush=True,
            )

        rr = params[0]

        signals = generate_base_signals(
            d,
            params,
        )

        train = evaluate_signals(
            d,
            timestamps,
            signals,
            rr,
            train_start,
            train_end,
        )

        if (
            train is None
            or train["trades"]
            < MIN_TRAIN_TRADES
        ):
            continue

        recent = evaluate_signals(
            d,
            timestamps,
            signals,
            rr,
            recent_start,
            train_end,
        )

        if (
            recent is None
            or recent["trades"]
            < MIN_RECENT_TRADES
        ):
            continue

        score = strategy_score(
            train,
            recent,
        )

        candidates.append(
            {
                "params":
                    params,

                "signals":
                    signals,

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

    return candidates[
        :TOP_BASE_STRATEGIES
    ]


# ============================================================
# REGIME OPTIMISATION
# ============================================================

def optimise_regime(
    d,
    timestamps,
    base_candidates,
    train_start,
    train_end,
):

    regimes = list(
        itertools.product(
            ATR_RATIO_VALUES,
            TREND_VALUES,
            PULLBACK_VALUES,
            MOMENTUM_VALUES,
        )
    )

    total = (
        len(base_candidates)
        * len(regimes)
    )

    print(
        f"REGIME COMBINATIONS: "
        f"{total}"
    )

    recent_end = utc_timestamp(
        train_end
    )

    recent_start = (
        recent_end
        - pd.Timedelta(
            days=RECENT_DAYS
        )
    )

    candidates = []

    completed = 0

    for base in base_candidates:

        base_params = base[
            "params"
        ]

        rr = base_params[0]

        for regime in regimes:

            completed += 1

            if (
                completed == 1
                or completed % 250 == 0
                or completed == total
            ):

                print(
                    f"Regime progress: "
                    f"{completed}/{total} "
                    f"("
                    f"{completed / total * 100:.1f}%"
                    f")",
                    flush=True,
                )

            signals = apply_regime_filter(
                d,
                base["signals"],
                regime,
            )

            train = evaluate_signals(
                d,
                timestamps,
                signals,
                rr,
                train_start,
                train_end,
            )

            if (
                train is None
                or train["trades"]
                < MIN_TRAIN_TRADES
            ):
                continue

            recent = evaluate_signals(
                d,
                timestamps,
                signals,
                rr,
                recent_start,
                train_end,
            )

            if (
                recent is None
                or recent["trades"]
                < MIN_RECENT_TRADES
            ):
                continue

            score = strategy_score(
                train,
                recent,
            )

            candidates.append(
                {
                    "base_params":
                        base_params,

                    "regime":
                        regime,

                    "signals":
                        signals,

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

    return candidates[
        :TOP_FINAL_STRATEGIES
    ]


# ============================================================
# STABILITY
# ============================================================

def stability_test(
    d,
    timestamps,
    candidate,
    train_start,
    train_end,
):

    base = candidate[
        "base_params"
    ]

    regime = candidate[
        "regime"
    ]

    rr = base[0]

    nearby = []

    # Small neighbourhood around the
    # selected strategy.
    base_candidates = []

    for rr_value in RR_VALUES:

        if abs(
            rr_value - base[0]
        ) <= 0.25:

            base_candidates.append(
                (
                    rr_value,
                    base[1],
                    base[2],
                    base[3],
                    base[4],
                    base[5],
                )
            )

    for wick in WICK_VALUES:

        if abs(
            wick - base[1]
        ) <= 0.10:

            base_candidates.append(
                (
                    base[0],
                    wick,
                    base[2],
                    base[3],
                    base[4],
                    base[5],
                )
            )

    for body in BODY_VALUES:

        if abs(
            body - base[2]
        ) <= 0.10:

            base_candidates.append(
                (
                    base[0],
                    base[1],
                    body,
                    base[3],
                    base[4],
                    base[5],
                )
            )

    unique_base = []

    for item in base_candidates:

        if item not in unique_base:

            unique_base.append(
                item
            )

    regime_candidates = []

    for atr in ATR_RATIO_VALUES:

        if abs(
            atr - regime[0]
        ) <= 0.15:

            regime_candidates.append(
                (
                    atr,
                    regime[1],
                    regime[2],
                    regime[3],
                )
            )

    for trend in TREND_VALUES:

        if abs(
            trend - regime[1]
        ) <= 0.0005:

            regime_candidates.append(
                (
                    regime[0],
                    trend,
                    regime[2],
                    regime[3],
                )
            )

    for pullback in PULLBACK_VALUES:

        if abs(
            pullback - regime[2]
        ) <= 0.001:

            regime_candidates.append(
                (
                    regime[0],
                    regime[1],
                    pullback,
                    regime[3],
                )
            )

    for momentum in MOMENTUM_VALUES:

        if abs(
            momentum - regime[3]
        ) <= 0.0005:

            regime_candidates.append(
                (
                    regime[0],
                    regime[1],
                    regime[2],
                    momentum,
                )
            )

    all_candidates = list(
        itertools.product(
            unique_base[:25],
            regime_candidates[:25],
        )
    )

    if len(
        all_candidates
    ) > STABILITY_LIMIT:

        all_candidates = (
            all_candidates[
                :STABILITY_LIMIT
            ]
        )

    for base_params, regime_params in (
        all_candidates
    ):

        signals = generate_base_signals(
            d,
            base_params,
        )

        signals = apply_regime_filter(
            d,
            signals,
            regime_params,
        )

        result = evaluate_signals(
            d,
            timestamps,
            signals,
            base_params[0],
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

        nearby.append(
            result
        )

    if not nearby:

        return {
            "count": 0,
            "median_win_rate": 0,
            "median_r": 0,
            "positive_fraction": 0,
            "stable": False,
        }

    win_rates = np.array(
        [
            x["win_rate"]
            for x in nearby
        ]
    )

    total_r = np.array(
        [
            x["total_r"]
            for x in nearby
        ]
    )

    median_win_rate = float(
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
        median_win_rate >= 55
        and median_r > 0
        and positive_fraction >= 0.60
    )

    return {
        "count":
            len(nearby),

        "median_win_rate":
            median_win_rate,

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
        f"{market} V6.1 "
        "REGIME-AWARE OPTIMIZER"
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

    results = []

    for (
        period_name,
        train_start,
        train_end,
        test_start,
        test_end,
    ) in PERIODS:

        print()
        print("=" * 60)

        print(
            f"{market} V6.1: "
            f"{period_name}"
        )

        print("=" * 60)

        print(
            "PHASE 1: BASE OPTIMIZATION"
        )

        base_candidates = optimise_base(
            d,
            timestamps,
            train_start,
            train_end,
        )

        if not base_candidates:

            print(
                "No valid base strategies."
            )

            continue

        print(
            f"Top base strategies retained: "
            f"{len(base_candidates)}"
        )

        print()
        print(
            "PHASE 2: REGIME OPTIMIZATION"
        )

        final_candidates = optimise_regime(
            d,
            timestamps,
            base_candidates,
            train_start,
            train_end,
        )

        if not final_candidates:

            print(
                "No valid regime strategies."
            )

            continue

        best = final_candidates[0]

        base = best[
            "base_params"
        ]

        regime = best[
            "regime"
        ]

        train = best[
            "train"
        ]

        recent = best[
            "recent"
        ]

        print()
        print(
            "BEST REGIME-AWARE STRATEGY"
        )

        print("-" * 60)

        print(
            f"Training trades: "
            f"{train['trades']}"
        )

        print(
            f"Training win rate: "
            f"{train['win_rate']:.2f}%"
        )

        print(
            f"Training R: "
            f"{train['total_r']:.2f}"
        )

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
            "PARAMETERS"
        )

        print(
            f"RR: {base[0]}"
        )

        print(
            f"Wick: {base[1]}"
        )

        print(
            f"Body: {base[2]}"
        )

        print(
            f"Separation: {base[3]}"
        )

        print(
            f"Max cross: {base[4]}"
        )

        print(
            "Hours: "
            + ",".join(
                map(
                    str,
                    base[5],
                )
            )
        )

        print()
        print(
            "REGIME FILTERS"
        )

        print(
            f"ATR ratio minimum: "
            f"{regime[0]}"
        )

        print(
            f"Trend minimum: "
            f"{regime[1]}"
        )

        print(
            f"Pullback maximum: "
            f"{regime[2]}"
        )

        print(
            f"Momentum minimum: "
            f"{regime[3]}"
        )

        print()
        print(
            "PARAMETER STABILITY TEST"
        )

        print("-" * 60)

        stability = stability_test(
            d,
            timestamps,
            best,
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

        print()
        print(
            "OUT-OF-SAMPLE RESULT"
        )

        print("-" * 60)

        test = evaluate_signals(
            d,
            timestamps,
            best["signals"],
            base[0],
            test_start,
            test_end,
        )

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
                    period_name,

                "train_trades":
                    train["trades"],

                "train_win_rate":
                    train["win_rate"],

                "train_r":
                    train["total_r"],

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

                "rr":
                    base[0],

                "wick":
                    base[1],

                "body":
                    base[2],

                "separation":
                    base[3],

                "max_cross":
                    base[4],

                "hours":
                    ",".join(
                        map(
                            str,
                            base[5],
                        )
                    ),

                "atr_ratio_min":
                    regime[0],

                "trend_min":
                    regime[1],

                "pullback_max":
                    regime[2],

                "momentum_min":
                    regime[3],
            }
        )

    result_df = pd.DataFrame(
        results
    )

    output = (
        f"data/"
        f"{market.lower()}_"
        f"optimizer_v6_1_results.csv"
    )

    result_df.to_csv(
        output,
        index=False,
    )

    return result_df


# ============================================================
# SUMMARY
# ============================================================

def make_summary(
    market,
    df,
):

    if (
        df is None
        or df.empty
    ):
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
        if trades > 0
        else 0
    )

    avg_tpw = float(
        df[
            "test_tpw"
        ].mean()
    )

    avg_drawdown = float(
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
                avg_drawdown,
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
        "OPTIMIZER V6.1"
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
        "TWO-STAGE SEARCH: ENABLED"
    )

    print(
        "TARGET: HIGH WIN RATE + "
        "POSITIVE R + ~1 TRADE/WEEK"
    )

    print(
        "MARKETS: XAUUSD, EURUSD"
    )

    print(
        "NO LIVE TRADING"
    )

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

    summary_rows = []

    for market, df in (
        results.items()
    ):

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
        "V6.1 FINAL MULTI-MARKET "
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

    combined_trades = 0
    combined_wins = 0
    combined_r = 0.0
    profitable_periods = 0
    stable_periods = 0
    total_periods = 0

    for df in (
        results.values()
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

        total_periods += len(df)

    combined_win_rate = (
        combined_wins
        / combined_trades
        * 100
        if combined_trades > 0
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

    robust_markets = sum(
        1
        for row in summary_rows
        if row["verdict"]
        == "ROBUST"
    )

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

        verdict = "STRONG CANDIDATE"

    elif (
        combined_win_rate >= 60
        and combined_r > 0
        and profitable_periods
        >= total_periods * 0.50
    ):

        verdict = "PROMISING"

    else:

        verdict = "NOT ROBUST YET"

    print()
    print("=" * 60)

    print(
        "V6.1 FINAL VERDICT"
    )

    print("=" * 60)

    print(
        f"VERDICT: {verdict}"
    )

    print()

    if verdict == "STRONG CANDIDATE":

        print(
            "Strong cross-market "
            "out-of-sample candidate."
        )

        print(
            "Further validation is "
            "required before live use."
        )

    elif verdict == "PROMISING":

        print(
            "Promising evidence of "
            "a cross-market edge."
        )

        print(
            "Continue validation."
        )

    else:

        print(
            "The strategy is not "
            "robust enough yet."
        )

        print(
            "Do not implement live."
        )

    summary.to_csv(
        "data/"
        "multi_market_optimizer_v6_1_summary.csv",
        index=False,
    )

    print()
    print(
        "Results saved:"
    )

    print(
        "data/"
        "xauusd_optimizer_v6_1_results.csv"
    )

    print(
        "data/"
        "eurusd_optimizer_v6_1_results.csv"
    )

    print(
        "data/"
        "multi_market_optimizer_v6_1_summary.csv"
    )

    print()
    print("=" * 60)

    print(
        "OPTIMIZER V6.1 COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
