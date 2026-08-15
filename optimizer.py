import os
import itertools
import numpy as np
import pandas as pd


# ============================================================
# MULTI-MARKET STRATEGY OPTIMIZER V6.2
# REGIME FILTER COMPARISON
# ============================================================

MARKETS = {
    "XAUUSD": "data/XAUUSD_15m.csv",
    "EURUSD": "data/EURUSD_15m.csv",
}

# Three genuine walk-forward tests.
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
# BASE SEARCH SPACE
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
# REGIME FILTER SEARCH
# ============================================================

ATR_VALUES = [
    0.90,
    1.00,
    1.10,
]

TREND_VALUES = [
    0.0000,
    0.0005,
    0.0010,
]

MOMENTUM_VALUES = [
    0.0000,
    0.0005,
    0.0010,
]

PULLBACK_VALUES = [
    0.0015,
    0.0020,
    0.0030,
]


# ============================================================
# SETTINGS
# ============================================================

RECENT_DAYS = 365

MIN_TRAIN_TRADES = 15
MIN_RECENT_TRADES = 8
MIN_OOS_TRADES = 5

TOP_BASE = 15
TOP_FILTERS = 20

STABILITY_SAMPLE = 60

TARGET_TRADES_PER_WEEK = 1.0


# ============================================================
# TIME HELPERS
# ============================================================

def utc(value):

    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")


def bounds(index, start, end):

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

    ema20_prev = np.roll(
        ema20,
        4,
    )

    ema50_prev = np.roll(
        ema50,
        4,
    )

    ema20_slope = (
        ema20 - ema20_prev
    ) / np.where(
        ema20 == 0,
        1,
        ema20,
    )

    ema50_slope = (
        ema50 - ema50_prev
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

    momentum = (
        c - np.roll(c, 8)
    ) / np.where(
        np.roll(c, 8) == 0,
        1,
        np.roll(c, 8),
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
        "ema20_slope": ema20_slope,
        "ema50_slope": ema50_slope,
        "separation": separation,
        "body_ratio": body_ratio,
        "upper_wick_ratio":
            upper_wick_ratio,
        "atr_ratio": atr_ratio,
        "momentum": momentum,
        "ema_distance":
            ema_distance,
        "cross_age": cross_age,
        "recent_high":
            recent_high,
        "hours":
            df.index.hour.to_numpy(),
    }


# ============================================================
# BASE SIGNAL
# ============================================================

def base_signal(
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
# REGIME FILTERS
#
# IMPORTANT:
# V6.2 does NOT blindly stack all filters.
#
# Each filter is tested independently,
# then useful combinations are tested.
# ============================================================

def regime_signal(
    d,
    signals,
    filter_name,
    value,
):

    if len(signals) == 0:
        return signals

    if filter_name == "NONE":

        return signals

    if filter_name == "ATR":

        keep = (
            d["atr_ratio"][signals]
            >= value
        )

    elif filter_name == "TREND":

        keep = (
            d["separation"][signals]
            >= value
        )

    elif filter_name == "MOMENTUM":

        keep = (
            d["momentum"][signals]
            <= -value
        )

    elif filter_name == "PULLBACK":

        keep = (
            d["ema_distance"][signals]
            <= value
        )

    else:

        return signals

    return signals[
        keep
    ]


def combination_signal(
    d,
    signals,
    atr_min,
    trend_min,
    momentum_min,
    pullback_max,
):

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
            d["momentum"][signals]
            <= -momentum_min
        )

        & (
            d["ema_distance"][signals]
            <= pullback_max
        )
    )

    return signals[
        keep
    ]


# ============================================================
# SIMULATE
# ============================================================

def simulate(
    d,
    signals,
    rr,
    start_idx,
    end_idx,
):

    trades = []

    next_free = start_idx

    for i in signals:

        if i < start_idx:
            continue

        if i > end_idx:
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

            next_free = (
                exit_index + 1
            )

    return trades


# ============================================================
# METRICS
# ============================================================

def get_metrics(
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

    days = max(
        (
            pd.Timestamp(end_time)
            - pd.Timestamp(start_time)
        ).total_seconds()
        / 86400.0,
        1,
    )

    weeks = days / 7

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
            pf,

        "drawdown":
            drawdown,

        "longest_loss_streak":
            longest,

        "trades_per_week":
            count
            / weeks,
    }


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    d,
    timestamps,
    signals,
    rr,
    start,
    end,
):

    b = bounds(
        timestamps,
        start,
        end,
    )

    if b is None:
        return None

    start_idx, end_idx = b

    trades = simulate(
        d,
        signals,
        rr,
        start_idx,
        end_idx,
    )

    return get_metrics(
        trades,
        timestamps[start_idx],
        timestamps[end_idx],
    )


# ============================================================
# SCORING
#
# Current era gets strong weighting,
# but not enough to completely dominate
# the historical evidence.
# ============================================================

def score(
    training,
    recent,
):

    if training is None:
        return -1e9

    if recent is None:
        return -1e9

    if (
        training["trades"]
        < MIN_TRAIN_TRADES
    ):
        return -1e9

    if (
        recent["trades"]
        < MIN_RECENT_TRADES
    ):
        return -1e9

    result = 0.0

    # Historical profitability.
    result += (
        training["total_r"]
        * 1.0
    )

    # Recent/current-era profitability.
    result += (
        recent["total_r"]
        * 4.0
    )

    # Win rate.
    result += (
        training["win_rate"]
        * 0.35
    )

    result += (
        recent["win_rate"]
        * 2.0
    )

    # Profit factor.
    result += (
        min(
            training["profit_factor"],
            3.0,
        )
        * 3.0
    )

    result += (
        min(
            recent["profit_factor"],
            3.0,
        )
        * 10.0
    )

    # Drawdown.
    result -= (
        training["drawdown"]
        * 0.5
    )

    result -= (
        recent["drawdown"]
        * 2.0
    )

    # Losing streak.
    result -= (
        recent[
            "longest_loss_streak"
        ]
        * 1.5
    )

    # Do NOT heavily reward tiny samples.
    if recent["trades"] < 12:
        result -= 8.0

    # Frequency preference.
    tpw = recent[
        "trades_per_week"
    ]

    result -= (
        abs(
            tpw
            - TARGET_TRADES_PER_WEEK
        )
        * 2.0
    )

    return result


# ============================================================
# BASE OPTIMIZER
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

    recent_end = utc(
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

        signals = base_signal(
            d,
            params,
        )

        training = evaluate(
            d,
            timestamps,
            signals,
            params[0],
            train_start,
            train_end,
        )

        if (
            training is None
            or training["trades"]
            < MIN_TRAIN_TRADES
        ):
            continue

        recent = evaluate(
            d,
            timestamps,
            signals,
            params[0],
            recent_start,
            train_end,
        )

        if (
            recent is None
            or recent["trades"]
            < MIN_RECENT_TRADES
        ):
            continue

        candidates.append(
            {
                "params":
                    params,

                "signals":
                    signals,

                "training":
                    training,

                "recent":
                    recent,

                "score":
                    score(
                        training,
                        recent,
                    ),
            }
        )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return candidates[
        :TOP_BASE
    ]


# ============================================================
# REGIME SEARCH
#
# Instead of forcing every filter on,
# we compare:
#
# NONE
# ATR
# TREND
# MOMENTUM
# PULLBACK
# ATR + TREND
# TREND + MOMENTUM
# ATR + TREND + MOMENTUM
#
# This tells us whether the regime logic
# actually improves the strategy.
# ============================================================

def optimise_filters(
    d,
    timestamps,
    base_candidates,
    train_start,
    train_end,
):

    recent_end = utc(
        train_end
    )

    recent_start = (
        recent_end
        - pd.Timedelta(
            days=RECENT_DAYS
        )
    )

    filter_tests = []

    for base in base_candidates:

        params = base[
            "params"
        ]

        rr = params[0]

        # No filter.
        filter_tests.append(
            (
                base,
                "NONE",
                None,
            )
        )

        # Individual filters.
        for value in ATR_VALUES:

            filter_tests.append(
                (
                    base,
                    "ATR",
                    value,
                )
            )

        for value in TREND_VALUES:

            filter_tests.append(
                (
                    base,
                    "TREND",
                    value,
                )
            )

        for value in MOMENTUM_VALUES:

            filter_tests.append(
                (
                    base,
                    "MOMENTUM",
                    value,
                )
            )

        for value in PULLBACK_VALUES:

            filter_tests.append(
                (
                    base,
                    "PULLBACK",
                    value,
                )
            )

        # Two-filter combinations.
        for atr in ATR_VALUES:

            for trend in TREND_VALUES:

                filter_tests.append(
                    (
                        base,
                        "ATR+TREND",
                        (
                            atr,
                            trend,
                        ),
                    )
                )

        for trend in TREND_VALUES:

            for momentum in (
                MOMENTUM_VALUES
            ):

                filter_tests.append(
                    (
                        base,
                        "TREND+MOMENTUM",
                        (
                            trend,
                            momentum,
                        ),
                    )
                )

        # Three-filter combination.
        for atr in ATR_VALUES:

            for trend in TREND_VALUES:

                for momentum in (
                    MOMENTUM_VALUES
                ):

                    filter_tests.append(
                        (
                            base,
                            "ATR+TREND+MOMENTUM",
                            (
                                atr,
                                trend,
                                momentum,
                            ),
                        )
                    )

    total = len(
        filter_tests
    )

    print(
        f"REGIME FILTER TESTS: {total}"
    )

    candidates = []

    for number, item in enumerate(
        filter_tests,
        start=1,
    ):

        if (
            number == 1
            or number % 250 == 0
            or number == total
        ):

            print(
                f"Filter progress: "
                f"{number}/{total} "
                f"("
                f"{number / total * 100:.1f}%"
                f")",
                flush=True,
            )

        base, filter_name, value = item

        params = base[
            "params"
        ]

        signals = base[
            "signals"
        ]

        if filter_name == "NONE":

            filtered = signals

        elif filter_name in [
            "ATR",
            "TREND",
            "MOMENTUM",
            "PULLBACK",
        ]:

            filtered = regime_signal(
                d,
                signals,
                filter_name,
                value,
            )

        elif filter_name == "ATR+TREND":

            atr, trend = value

            filtered = combination_signal(
                d,
                signals,
                atr,
                trend,
                0.0,
                999.0,
            )

        elif filter_name == "TREND+MOMENTUM":

            trend, momentum = value

            filtered = combination_signal(
                d,
                signals,
                0.0,
                trend,
                momentum,
                999.0,
            )

        elif (
            filter_name
            == "ATR+TREND+MOMENTUM"
        ):

            atr, trend, momentum = value

            filtered = combination_signal(
                d,
                signals,
                atr,
                trend,
                momentum,
                999.0,
            )

        else:

            filtered = signals

        training = evaluate(
            d,
            timestamps,
            filtered,
            params[0],
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
            filtered,
            params[0],
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

        candidates.append(
            {
                "base":
                    base,

                "filter_name":
                    filter_name,

                "filter_value":
                    value,

                "signals":
                    filtered,

                "training":
                    training,

                "recent":
                    recent,

                "score":
                    score(
                        training,
                        recent,
                    ),
            }
        )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return candidates[
        :TOP_FILTERS
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
        "base"
    ]

    params = base[
        "params"
    ]

    nearby = []

    rr_choices = [
        x
        for x in RR_VALUES
        if abs(
            x - params[0]
        ) <= 0.25
    ]

    wick_choices = [
        x
        for x in WICK_VALUES
        if abs(
            x - params[1]
        ) <= 0.10
    ]

    body_choices = [
        x
        for x in BODY_VALUES
        if abs(
            x - params[2]
        ) <= 0.10
    ]

    # Keep stability reasonably small.
    combinations = list(
        itertools.product(
            rr_choices,
            wick_choices,
            body_choices,
        )
    )

    if len(combinations) > STABILITY_SAMPLE:

        combinations = (
            combinations[
                :STABILITY_SAMPLE
            ]
        )

    for rr, wick, body in (
        combinations
    ):

        nearby_params = (
            rr,
            wick,
            body,
            params[3],
            params[4],
            params[5],
        )

        signals = base_signal(
            d,
            nearby_params,
        )

        result = evaluate(
            d,
            timestamps,
            signals,
            rr,
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

    median_win = float(
        np.median(
            win_rates
        )
    )

    median_r = float(
        np.median(
            total_r
        )
    )

    positive = float(
        np.mean(
            total_r > 0
        )
    )

    stable = (
        median_win >= 55
        and median_r >= 0
        and positive >= 0.55
    )

    return {
        "count":
            len(nearby),

        "median_win_rate":
            median_win,

        "median_r":
            median_r,

        "positive_fraction":
            positive,

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
        f"{market} V6.2 "
        "REGIME-COMPARISON OPTIMIZER"
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

    all_results = []

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
            f"{market}: "
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
                "No valid base strategy."
            )

            all_results.append(
                {
                    "market": market,
                    "period": period_name,
                    "status": "NO_BASE",
                    "test_trades": 0,
                    "test_wins": 0,
                    "test_losses": 0,
                    "test_win_rate": 0,
                    "test_r": 0,
                    "test_pf": 0,
                    "test_drawdown": 0,
                    "stable": False,
                }
            )

            continue

        print(
            f"Base candidates retained: "
            f"{len(base_candidates)}"
        )

        print()
        print(
            "PHASE 2: REGIME COMPARISON"
        )

        candidates = optimise_filters(
            d,
            timestamps,
            base_candidates,
            train_start,
            train_end,
        )

        if not candidates:

            print(
                "No valid regime candidates."
            )

            all_results.append(
                {
                    "market": market,
                    "period": period_name,
                    "status": "NO_FILTER",
                    "test_trades": 0,
                    "test_wins": 0,
                    "test_losses": 0,
                    "test_win_rate": 0,
                    "test_r": 0,
                    "test_pf": 0,
                    "test_drawdown": 0,
                    "stable": False,
                }
            )

            continue

        print()
        print(
            "TOP REGIME STRATEGIES"
        )

        print("-" * 60)

        for rank, candidate in enumerate(
            candidates[:10],
            start=1,
        ):

            print(
                f"{rank:02d} | "
                f"{candidate['filter_name']} "
                f"{candidate['filter_value']} | "
                f"Train "
                f"{candidate['training']['win_rate']:.1f}% | "
                f"Recent "
                f"{candidate['recent']['win_rate']:.1f}% | "
                f"Recent R "
                f"{candidate['recent']['total_r']:.2f} | "
                f"Recent PF "
                f"{candidate['recent']['profit_factor']:.2f} | "
                f"Trades "
                f"{candidate['recent']['trades']}"
            )

        best = candidates[0]

        base = best[
            "base"
        ]

        params = base[
            "params"
        ]

        print()
        print(
            "SELECTED TRAINING STRATEGY"
        )

        print("-" * 60)

        print(
            f"Filter: "
            f"{best['filter_name']}"
        )

        print(
            f"Filter value: "
            f"{best['filter_value']}"
        )

        print(
            f"Training trades: "
            f"{best['training']['trades']}"
        )

        print(
            f"Training win rate: "
            f"{best['training']['win_rate']:.2f}%"
        )

        print(
            f"Training R: "
            f"{best['training']['total_r']:.2f}"
        )

        print(
            f"Recent trades: "
            f"{best['recent']['trades']}"
        )

        print(
            f"Recent win rate: "
            f"{best['recent']['win_rate']:.2f}%"
        )

        print(
            f"Recent R: "
            f"{best['recent']['total_r']:.2f}"
        )

        print(
            f"Recent PF: "
            f"{best['recent']['profit_factor']:.2f}"
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
            "PARAMETER STABILITY"
        )

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

        oos = evaluate(
            d,
            timestamps,
            best["signals"],
            params[0],
            test_start,
            test_end,
        )

        if oos is None:

            print(
                "No OOS trades."
            )

            oos = {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "total_r": 0,
                "profit_factor": 0,
                "drawdown": 0,
                "longest_loss_streak": 0,
                "trades_per_week": 0,
            }

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
            f"{oos['longest_loss_streak']}"
        )

        print(
            f"Trades/week: "
            f"{oos['trades_per_week']:.2f}"
        )

        all_results.append(
            {
                "market":
                    market,

                "period":
                    period_name,

                "status":
                    "COMPLETE",

                "filter":
                    best[
                        "filter_name"
                    ],

                "filter_value":
                    str(
                        best[
                            "filter_value"
                        ]
                    ),

                "train_trades":
                    best[
                        "training"
                    ][
                        "trades"
                    ],

                "train_win_rate":
                    best[
                        "training"
                    ][
                        "win_rate"
                    ],

                "recent_trades":
                    best[
                        "recent"
                    ][
                        "trades"
                    ],

                "recent_win_rate":
                    best[
                        "recent"
                    ][
                        "win_rate"
                    ],

                "recent_r":
                    best[
                        "recent"
                    ][
                        "total_r"
                    ],

                "test_trades":
                    oos["trades"],

                "test_wins":
                    oos["wins"],

                "test_losses":
                    oos["losses"],

                "test_win_rate":
                    oos["win_rate"],

                "test_r":
                    oos["total_r"],

                "test_pf":
                    oos["profit_factor"],

                "test_drawdown":
                    oos["drawdown"],

                "test_losing_streak":
                    oos[
                        "longest_loss_streak"
                    ],

                "test_tpw":
                    oos[
                        "trades_per_week"
                    ],

                "stable":
                    stability[
                        "stable"
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
            }
        )

    result = pd.DataFrame(
        all_results
    )

    output = (
        f"data/"
        f"{market.lower()}_"
        f"optimizer_v6_2_results.csv"
    )

    result.to_csv(
        output,
        index=False,
    )

    return result


# ============================================================
# MARKET SUMMARY
# ============================================================

def market_summary(
    market,
    df,
):

    if df.empty:
        return None

    completed = df[
        df["status"]
        == "COMPLETE"
    ]

    if completed.empty:
        return None

    trades = int(
        completed[
            "test_trades"
        ].sum()
    )

    wins = int(
        completed[
            "test_wins"
        ].sum()
    )

    total_r = float(
        completed[
            "test_r"
        ].sum()
    )

    profitable = int(
        (
            completed[
                "test_r"
            ]
            > 0
        ).sum()
    )

    stable = int(
        completed[
            "stable"
        ].sum()
    )

    periods = len(
        completed
    )

    win_rate = (
        wins / trades * 100
        if trades
        else 0
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
        "OPTIMIZER V6.2"
    )

    print("=" * 60)

    print(
        "REGIME FILTER COMPARISON: ENABLED"
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
        "FILTER STACKING: SELECTIVE"
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

            results[
                market
            ] = pd.DataFrame()

    summaries = []

    for market, df in (
        results.items()
    ):

        row = market_summary(
            market,
            df,
        )

        if row is not None:

            summaries.append(
                row
            )

    summary = pd.DataFrame(
        summaries
    )

    print()
    print("=" * 60)

    print(
        "V6.2 FINAL MULTI-MARKET "
        "SUMMARY"
    )

    print("=" * 60)

    if summary.empty:

        print(
            "No completed market results."
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
    completed_periods = 0

    for df in results.values():

        if df.empty:
            continue

        complete = df[
            df["status"]
            == "COMPLETE"
        ]

        if complete.empty:
            continue

        combined_trades += int(
            complete[
                "test_trades"
            ].sum()
        )

        combined_wins += int(
            complete[
                "test_wins"
            ].sum()
        )

        combined_r += float(
            complete[
                "test_r"
            ].sum()
        )

        profitable_periods += int(
            (
                complete[
                    "test_r"
                ]
                > 0
            ).sum()
        )

        stable_periods += int(
            complete[
                "stable"
            ].sum()
        )

        completed_periods += len(
            complete
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
        f"{completed_periods}"
    )

    print(
        f"Stable periods: "
        f"{stable_periods}/"
        f"{completed_periods}"
    )

    # ========================================================
    # FINAL VERDICT
    # ========================================================

    robust_markets = 0

    for row in summaries:

        if row["verdict"] == "ROBUST":
            robust_markets += 1

    if (
        len(summaries) == 2
        and robust_markets >= 1
        and combined_trades >= 50
        and combined_win_rate >= 65
        and combined_r > 0
        and profitable_periods
        >= completed_periods * 0.60
        and stable_periods
        >= completed_periods * 0.60
    ):

        verdict = "STRONG CANDIDATE"

    elif (
        combined_trades >= 40
        and combined_win_rate >= 60
        and combined_r > 0
        and profitable_periods
        >= completed_periods * 0.50
    ):

        verdict = "PROMISING"

    else:

        verdict = "NOT ROBUST YET"

    print()
    print("=" * 60)

    print(
        "V6.2 FINAL VERDICT"
    )

    print("=" * 60)

    print(
        f"VERDICT: {verdict}"
    )

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
            "Promising cross-market "
            "edge."
        )

        print(
            "Continue validation before "
            "considering live deployment."
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
        "multi_market_optimizer_v6_2_summary.csv",
        index=False,
    )

    print()
    print(
        "Results saved:"
    )

    print(
        "data/"
        "xauusd_optimizer_v6_2_results.csv"
    )

    print(
        "data/"
        "eurusd_optimizer_v6_2_results.csv"
    )

    print(
        "data/"
        "multi_market_optimizer_v6_2_summary.csv"
    )

    print()
    print("=" * 60)

    print(
        "OPTIMIZER V6.2 COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
