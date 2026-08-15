import os
import itertools
import numpy as np
import pandas as pd


# ============================================================
# MULTI-MARKET STRATEGY OPTIMIZER V5
# REGIME-STABLE / CURRENT-ERA / WALK-FORWARD
# ============================================================

MARKETS = {
    "XAUUSD": "data/XAUUSD_15m.csv",
    "EURUSD": "data/EURUSD_15m.csv",
}

# ------------------------------------------------------------
# Rolling walk-forward periods
#
# We deliberately use more recent training windows so that
# old market structure does not dominate the optimisation.
# ------------------------------------------------------------

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
# PARAMETER SEARCH SPACE
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
    (3, 4, 5, 12, 13),
    (2, 3, 4, 5, 12, 13),
]


# ============================================================
# REQUIREMENTS
# ============================================================

MIN_TRAIN_TRADES = 25
MIN_RECENT_TRADES = 12
MIN_TEST_TRADES = 5

# Preferred frequency.
# This does NOT force a trade every week.
TARGET_TPW_LOW = 0.05
TARGET_TPW_HIGH = 0.35

# Current-era weighting.
RECENT_DAYS = 365


# ============================================================
# TIMEZONE
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

    time_column = None

    for c in [
        "time",
        "Time",
        "timestamp",
        "Timestamp",
        "date",
        "Date",
    ]:

        if c in df.columns:
            time_column = c
            break

    if time_column is None:
        raise RuntimeError(
            "Missing time column"
        )

    df[time_column] = pd.to_datetime(
        df[time_column],
        utc=True,
    )

    df = df.set_index(
        time_column
    )

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

    for c in [
        "Open",
        "High",
        "Low",
        "Close",
    ]:

        if c not in df.columns:
            raise RuntimeError(
                f"Missing {c} column"
            )

    df = (
        df[
            [
                "Open",
                "High",
                "Low",
                "Close",
            ]
        ]
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

    ema20_slope = (
        ema20
        - np.roll(ema20, 4)
    )

    ema50_slope = (
        ema50
        - np.roll(ema50, 4)
    )

    separation = np.divide(
        np.abs(
            ema20 - ema50
        ),
        c,
        out=np.zeros_like(c),
        where=c != 0,
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

        if (
            ema20[i - 1]
            >= ema50[i - 1]
            and
            ema20[i]
            < ema50[i]
        ):

            last_cross = i

        if last_cross >= 0:
            cross_age[i] = (
                i - last_cross
            )

    previous_close = np.roll(c, 1)
    previous_ema20 = np.roll(
        ema20,
        1,
    )

    previous_distance = np.divide(
        np.abs(
            previous_close
            - previous_ema20
        ),
        previous_close,
        out=np.ones_like(c),
        where=previous_close != 0,
    )

    current_distance = np.divide(
        np.abs(
            c - ema20
        ),
        c,
        out=np.ones_like(c),
        where=c != 0,
    )

    pullback = (
        (
            previous_distance
            <= 0.002
        )
        |
        (
            current_distance
            <= 0.002
        )
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
        "ema20_slope": ema20_slope,
        "ema50_slope": ema50_slope,
        "separation": separation,
        "body_ratio": body_ratio,
        "upper_wick_ratio":
            upper_wick_ratio,
        "cross_age": cross_age,
        "pullback": pullback,
        "recent_high": recent_high,
        "hours": df.index.hour.to_numpy(),
    }


# ============================================================
# SIGNALS
# ============================================================

def generate_signals(
    d,
    wick,
    body,
    separation,
    max_cross,
    hours,
):

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

        & d["pullback"]

        & (
            d["close"]
            < d["open"]
        )

        & (
            d[
                "upper_wick_ratio"
            ]
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
# BOUNDS
# ============================================================

def get_bounds(
    timestamps,
    start,
    end,
):

    start_ts = utc_timestamp(start)
    end_ts = utc_timestamp(end)

    start_indices = np.flatnonzero(
        timestamps >= start_ts
    )

    end_indices = np.flatnonzero(
        timestamps <= end_ts
    )

    if (
        len(start_indices) == 0
        or len(end_indices) == 0
    ):
        return None

    return (
        int(start_indices[0]),
        int(end_indices[-1]),
    )


# ============================================================
# SIMULATOR
# ============================================================

def simulate(
    d,
    signals,
    rr,
    start_index,
    end_index,
):

    results = []

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

            if (
                d["high"][j]
                >= stop
            ):

                result = -1.0
                exit_index = j
                break

            if (
                d["low"][j]
                <= target
            ):

                result = rr
                exit_index = j
                break

        if result is not None:

            results.append(
                result
            )

            next_free = (
                exit_index + 1
            )

    return results


# ============================================================
# METRICS
# ============================================================

def metrics(
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

    n = len(values)

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
        (
            peak - equity
        ).max()
    )

    # Longest losing streak
    longest_loss_streak = 0
    current_loss_streak = 0

    for value in values:

        if value < 0:

            current_loss_streak += 1

            longest_loss_streak = max(
                longest_loss_streak,
                current_loss_streak,
            )

        else:

            current_loss_streak = 0

    weeks = max(
        days / 7.0,
        1e-9,
    )

    return {
        "trades": n,
        "wins": win_count,
        "losses": loss_count,
        "win_rate":
            win_count / n * 100,
        "total_r":
            float(values.sum()),
        "profit_factor": pf,
        "drawdown": drawdown,
        "longest_loss_streak":
            longest_loss_streak,
        "trades_per_week":
            n / weeks,
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

    (
        rr,
        wick,
        body,
        separation,
        max_cross,
        hours,
    ) = params

    signals = generate_signals(
        d,
        wick,
        body,
        separation,
        max_cross,
        hours,
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
        ).total_seconds()
        / 86400,
        1,
    )

    return metrics(
        trades,
        days,
    )


# ============================================================
# CURRENT-ERA SCORE
# ============================================================

def candidate_score(
    full,
    recent,
):

    if (
        full is None
        or recent is None
    ):
        return -1e12

    if (
        full["trades"]
        < MIN_TRAIN_TRADES
    ):
        return -1e12

    if (
        recent["trades"]
        < MIN_RECENT_TRADES
    ):
        return -1e12

    score = 0.0

    # Historical performance.
    score += (
        full["win_rate"]
        * 1.5
    )

    score += (
        full["total_r"]
        * 0.75
    )

    score += (
        min(
            full["profit_factor"],
            3.0,
        )
        * 5.0
    )

    score -= (
        full["drawdown"]
        * 1.0
    )

    # CURRENT ERA HEAVILY WEIGHTED.
    score += (
        recent["win_rate"]
        * 8.0
    )

    score += (
        recent["total_r"]
        * 3.0
    )

    score += (
        min(
            recent[
                "profit_factor"
            ],
            3.0,
        )
        * 22.0
    )

    score -= (
        recent["drawdown"]
        * 4.0
    )

    # Prefer selective strategies.
    tpw = recent[
        "trades_per_week"
    ]

    if (
        TARGET_TPW_LOW
        <= tpw
        <= TARGET_TPW_HIGH
    ):

        score += 40.0

    elif tpw <= 0.75:

        score += 15.0

    else:

        score -= (
            tpw * 15.0
        )

    return score


# ============================================================
# OPTIMISE ONE TRAINING WINDOW
# ============================================================

def optimise_window(
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

    train_start_index, train_end_index = (
        train_bounds
    )

    train_start_ts = utc_timestamp(
        train_start
    )

    train_end_ts = utc_timestamp(
        train_end
    )

    recent_start = max(
        train_start_ts,
        train_end_ts
        - pd.Timedelta(
            days=RECENT_DAYS
        ),
    )

    recent_bounds = get_bounds(
        timestamps,
        recent_start,
        train_end_ts,
    )

    if recent_bounds is None:
        return None

    recent_start_index, recent_end_index = (
        recent_bounds
    )

    full_days = max(
        (
            timestamps[
                train_end_index
            ]
            - timestamps[
                train_start_index
            ]
        ).total_seconds()
        / 86400,
        1,
    )

    recent_days = max(
        (
            timestamps[
                recent_end_index
            ]
            - timestamps[
                recent_start_index
            ]
        ).total_seconds()
        / 86400,
        1,
    )

    best = None

    combinations = itertools.product(
        RR_VALUES,
        WICK_VALUES,
        BODY_VALUES,
        SEPARATION_VALUES,
        MAX_CROSS_VALUES,
        HOUR_SETS,
    )

    for params in combinations:

        (
            rr,
            wick,
            body,
            separation,
            max_cross,
            hours,
        ) = params

        signals = generate_signals(
            d,
            wick,
            body,
            separation,
            max_cross,
            hours,
        )

        full_trades = simulate(
            d,
            signals,
            rr,
            train_start_index,
            train_end_index,
        )

        recent_trades = simulate(
            d,
            signals,
            rr,
            recent_start_index,
            recent_end_index,
        )

        full = metrics(
            full_trades,
            full_days,
        )

        recent = metrics(
            recent_trades,
            recent_days,
        )

        score = candidate_score(
            full,
            recent,
        )

        if (
            best is None
            or score
            > best["score"]
        ):

            best = {
                "score": score,
                "params": params,
                "full": full,
                "recent": recent,
            }

    return best


# ============================================================
# PARAMETER STABILITY TEST
# ============================================================

def stability_test(
    d,
    timestamps,
    winning_params,
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
    ) = winning_params

    # Nearby parameter combinations.
    nearby_rr = [
        x for x in RR_VALUES
        if abs(x - rr) <= 0.25
    ]

    nearby_wick = [
        x for x in WICK_VALUES
        if abs(x - wick) <= 0.10
    ]

    nearby_body = [
        x for x in BODY_VALUES
        if abs(x - body) <= 0.10
    ]

    nearby_sep = [
        x for x in SEPARATION_VALUES
        if abs(x - separation)
        <= 0.0005
    ]

    nearby_cross = [
        x for x in MAX_CROSS_VALUES
        if abs(x - max_cross)
        <= 10
    ]

    nearby_hours = [
        hours,
    ]

    # Add neighbouring hour sets.
    for h in HOUR_SETS:

        if h not in nearby_hours:

            overlap = len(
                set(h)
                & set(hours)
            )

            if overlap >= max(
                1,
                len(hours) - 1,
            ):

                nearby_hours.append(
                    h
                )

    candidates = []

    for params in itertools.product(
        nearby_rr,
        nearby_wick,
        nearby_body,
        nearby_sep,
        nearby_cross,
        nearby_hours,
    ):

        result = evaluate(
            d,
            timestamps,
            params,
            train_start,
            train_end,
        )

        if (
            result is None
            or result["trades"]
            < MIN_TRAIN_TRADES
        ):
            continue

        candidates.append(
            {
                "params": params,
                "metrics": result,
            }
        )

    if not candidates:

        return {
            "stable": False,
            "count": 0,
            "median_win_rate": 0,
            "median_r": 0,
            "positive_fraction": 0,
        }

    win_rates = np.array(
        [
            x["metrics"][
                "win_rate"
            ]
            for x in candidates
        ]
    )

    total_r = np.array(
        [
            x["metrics"][
                "total_r"
            ]
            for x in candidates
        ]
    )

    positive_fraction = float(
        np.mean(
            total_r > 0
        )
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

    stable = (
        len(candidates) >= 5
        and median_win_rate >= 55
        and positive_fraction >= 0.60
    )

    return {
        "stable": stable,
        "count": len(candidates),
        "median_win_rate":
            median_win_rate,
        "median_r":
            median_r,
        "positive_fraction":
            positive_fraction,
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
        f"{market} V5 "
        "REGIME-STABLE OPTIMIZER"
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

    period_results = []

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
            "Optimising current-era "
            "weighted training..."
        )

        best = optimise_window(
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

        full = best["full"]
        recent = best["recent"]

        (
            rr,
            wick,
            body,
            separation,
            max_cross,
            hours,
        ) = params

        print()
        print(
            "BEST TRAINING STRATEGY"
        )

        print("-" * 60)

        print(
            f"Training trades: "
            f"{full['trades']}"
        )

        print(
            f"Training win rate: "
            f"{full['win_rate']:.2f}%"
        )

        print(
            f"Training R: "
            f"{full['total_r']:.2f}"
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
            "PARAMETERS"
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
            f"Separation: "
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

        # ----------------------------------------------------
        # Stability
        # ----------------------------------------------------

        print()
        print(
            "PARAMETER STABILITY TEST"
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
            f"Nearby combinations: "
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
            f"Stability: "
            f"{'PASS' if stability['stable'] else 'FAIL'}"
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

        period_results.append(
            {
                "market": market,
                "period": period_name,

                "train_trades":
                    full["trades"],

                "train_win_rate":
                    full["win_rate"],

                "train_r":
                    full["total_r"],

                "recent_trades":
                    recent["trades"],

                "recent_win_rate":
                    recent["win_rate"],

                "recent_r":
                    recent["total_r"],

                "recent_pf":
                    recent[
                        "profit_factor"
                    ],

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
                    test[
                        "profit_factor"
                    ],

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
            }
        )

    result_df = pd.DataFrame(
        period_results
    )

    result_path = (
        f"data/"
        f"{market.lower()}_"
        f"optimizer_v5_results.csv"
    )

    result_df.to_csv(
        result_path,
        index=False,
    )

    return result_df


# ============================================================
# FINAL MARKET SUMMARY
# ============================================================

def market_summary(
    market,
    df,
):

    if df.empty:
        return None

    trades = int(
        df["test_trades"].sum()
    )

    wins = int(
        df["test_wins"].sum()
    )

    total_r = float(
        df["test_r"].sum()
    )

    profitable_periods = int(
        (
            df["test_r"] > 0
        ).sum()
    )

    stable_periods = int(
        df["stable"].sum()
    )

    periods = len(df)

    win_rate = (
        wins / trades * 100
        if trades > 0
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

    # Strong robustness requirement.
    if (
        trades >= 30
        and profitable_periods >= 2
        and stable_periods >= 2
        and total_r > 0
        and win_rate >= 60
    ):

        verdict = "ROBUST"

    elif (
        trades >= 20
        and profitable_periods >= 2
        and total_r >= 0
        and win_rate >= 55
    ):

        verdict = "PROMISING"

    else:

        verdict = "NOT ROBUST"

    return {
        "market": market,
        "oos_trades": trades,
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
            f"{profitable_periods}/{periods}",
        "stable_periods":
            f"{stable_periods}/{periods}",
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
        "verdict": verdict,
    }


# ============================================================
# CROSS-MARKET SUMMARY
# ============================================================

def main():

    print("=" * 60)

    print(
        "MULTI-MARKET STRATEGY "
        "OPTIMIZER V5"
    )

    print("=" * 60)

    print(
        "REGIME-STABLE TESTING: ENABLED"
    )

    print(
        "CURRENT-ERA WEIGHTING: ENABLED"
    )

    print(
        "PARAMETER STABILITY: ENABLED"
    )

    print(
        "TARGET FREQUENCY: ~1 TRADE/WEEK"
    )

    print(
        "MARKETS: XAUUSD, EURUSD"
    )

    print(
        "NO LIVE TRADING"
    )

    all_results = {}

    for market, path in (
        MARKETS.items()
    ):

        try:

            all_results[market] = (
                run_market(
                    market,
                    path,
                )
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
        all_results.items()
    ):

        if df is None:
            continue

        if df.empty:
            continue

        row = market_summary(
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
        "V5 MULTI-MARKET SUMMARY"
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
    # Combined cross-market result
    # --------------------------------------------------------

    print()
    print("=" * 60)

    print(
        "COMBINED CROSS-MARKET TEST"
    )

    print("=" * 60)

    combined_trades = 0
    combined_wins = 0
    combined_r = 0.0

    profitable_periods = 0
    stable_periods = 0
    total_periods = 0

    for df in (
        all_results.values()
    ):

        if df is None or df.empty:
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

    if combined_trades > 0:

        combined_wr = (
            combined_wins
            / combined_trades
            * 100
        )

    else:

        combined_wr = 0.0

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
        f"{combined_wr:.2f}%"
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
        "V5 FINAL VERDICT"
    )

    print("=" * 60)

    robust_markets = 0

    for row in summary_rows:

        if row[
            "verdict"
        ] == "ROBUST":

            robust_markets += 1

    if (
        len(summary_rows) >= 2
        and robust_markets >= 2
        and combined_wr >= 60
        and combined_r > 0
        and stable_periods
        >= max(
            3,
            total_periods // 2,
        )
    ):

        verdict = (
            "STRONG CANDIDATE"
        )

    elif (
        combined_r > 0
        and combined_wr >= 55
        and profitable_periods
        >= total_periods * 0.60
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
            "The strategy shows "
            "cross-market and "
            "regime stability."
        )

        print(
            "Proceed to deeper "
            "validation before "
            "considering any live "
            "implementation."
        )

    elif verdict == "PROMISING":

        print(
            "The strategy shows "
            "evidence of an edge, "
            "but more validation "
            "is required."
        )

    else:

        print(
            "The strategy is not "
            "stable enough yet."
        )

        print(
            "Do not implement live."
        )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary.to_csv(
        "data/"
        "multi_market_optimizer_v5_summary.csv",
        index=False,
    )

    print()
    print(
        "Results saved:"
    )

    print(
        "data/"
        "xauusd_optimizer_v5_results.csv"
    )

    print(
        "data/"
        "eurusd_optimizer_v5_results.csv"
    )

    print(
        "data/"
        "multi_market_optimizer_v5_summary.csv"
    )

    print()
    print("=" * 60)

    print(
        "OPTIMIZER V5 COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
