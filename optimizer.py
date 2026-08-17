# ============================================================
# MULTI-MARKET STRATEGY OPTIMIZER V11.5
# ============================================================
#
# V11.3 BASELINE:
#   418 OOS trades
#   72.01% OOS win rate
#   +3.40R
#
# V11.5 GOAL:
#   >=75% OOS win rate
#   >=200 genuine OOS trades
#   positive total R
#
# IMPORTANT:
#   NO LIVE TRADING
#   OOS DATA IS NEVER USED FOR OPTIMISATION
#
# DESIGN:
#   - Fast targeted search
#   - No million-combination fine search
#   - Rejection/reversal signal
#   - Limited trend/volatility filters
#   - Walk-forward validation
#   - Parameter stability
#   - Strict OOS testing
# ============================================================

import os
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

MARKETS = {
    "XAUUSD": "data/XAUUSD_15m.csv",
    "EURUSD": "data/EURUSD_15m.csv",
}

RESULT_FILES = {
    "XAUUSD": "data/xauusd_optimizer_v11_5_results.csv",
    "EURUSD": "data/eurusd_optimizer_v11_5_results.csv",
}

SUMMARY_FILE = (
    "data/multi_market_optimizer_v11_5_summary.csv"
)

WALK_FORWARD = [
    (
        "2021-2023 -> 2024",
        "2021-01-01",
        "2023-12-31 23:59:59",
        "2024-01-01",
        "2024-12-31 23:59:59",
    ),
    (
        "2022-2024 -> 2025",
        "2022-01-01",
        "2024-12-31 23:59:59",
        "2025-01-01",
        "2025-12-31 23:59:59",
    ),
    (
        "2023-2025 -> 2026",
        "2023-01-01",
        "2025-12-31 23:59:59",
        "2026-01-01",
        "2026-12-31 23:59:59",
    ),
]


# ============================================================
# TARGETED SEARCH SPACE
# ============================================================

RR_VALUES = [
    0.40,
    0.50,
    0.60,
    0.75,
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
]

THRESHOLD_VALUES = [
    -0.25,
    0.00,
    0.25,
]

HOUR_SETS = [
    (3, 4, 5),
    (4, 5),
    (3, 4, 5, 12, 13),
    (4, 5, 12, 13),
]

TREND_FILTERS = [
    "none",
    "aligned",
]

VOL_FILTERS = [
    "none",
    "normal",
]

COOLDOWN_VALUES = [
    0,
    4,
    8,
    12,
]

MIN_TRAIN_TRADES = 40
MIN_STABILITY_TRADES = 20
MAX_STABILITY_CANDIDATES = 8


# ============================================================
# HEADER
# ============================================================

print("=" * 60, flush=True)
print("MULTI-MARKET STRATEGY OPTIMIZER V11.5", flush=True)
print("=" * 60, flush=True)
print("V11.3 BASELINE REFINEMENT", flush=True)
print("TARGET: >=75% WR + >=200 OOS TRADES + POSITIVE R", flush=True)
print("FAST TARGETED SEARCH: ENABLED", flush=True)
print("REJECTION / REVERSAL: ENABLED", flush=True)
print("TREND FILTER: SELECTIVE", flush=True)
print("VOLATILITY FILTER: SELECTIVE", flush=True)
print("WALK-FORWARD TESTING: ENABLED", flush=True)
print("PARAMETER STABILITY: ENABLED", flush=True)
print("STRICT OOS VALIDATION: ENABLED", flush=True)
print("NO LIVE TRADING", flush=True)
print("=" * 60, flush=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path):

    print(f"Loading {path}...", flush=True)

    if not os.path.exists(path):
        raise RuntimeError(
            f"Data file not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise RuntimeError(
            f"Data file is empty: {path}"
        )

    rename = {}

    for column in df.columns:

        clean = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

        rename[column] = clean

    df = df.rename(columns=rename)

    time_candidates = [
        "time",
        "datetime",
        "date",
        "timestamp",
        "timestamp_utc",
        "datetime_utc",
        "utc_time",
    ]

    time_column = None

    for name in time_candidates:

        if name in df.columns:

            time_column = name
            break

    if time_column is None:

        for column in df.columns:

            if column in {
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vol",
            }:
                continue

            try:

                parsed = pd.to_datetime(
                    df[column],
                    utc=True,
                    errors="coerce",
                )

                if parsed.notna().mean() >= 0.90:

                    time_column = column
                    break

            except Exception:
                pass

    if time_column is None:

        raise RuntimeError(
            "Could not find datetime column."
        )

    df["time"] = pd.to_datetime(
        df[time_column],
        utc=True,
        errors="coerce",
    )

    aliases = {
        "open": ["open", "o", "open_price"],
        "high": ["high", "h", "high_price"],
        "low": ["low", "l", "low_price"],
        "close": ["close", "c", "close_price"],
    }

    for target, names in aliases.items():

        if target in df.columns:
            continue

        for name in names:

            if name in df.columns:

                df[target] = df[name]
                break

    required = [
        "time",
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{path} missing required columns: {missing}"
        )

    for column in [
        "open",
        "high",
        "low",
        "close",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=required
    )

    valid = (
        (df["high"] >= df["low"])
        &
        (df["high"] >= df["open"])
        &
        (df["high"] >= df["close"])
        &
        (df["low"] <= df["open"])
        &
        (df["low"] <= df["close"])
    )

    df = df.loc[valid]

    df = (
        df
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    print(
        f"Candles: {len(df)}",
        flush=True,
    )

    print(
        f"Range: {df['time'].min()} -> "
        f"{df['time'].max()}",
        flush=True,
    )

    return df


# ============================================================
# INDICATORS
# ============================================================

def prepare_indicators(df):

    print(
        "Preparing indicators...",
        flush=True,
    )

    df = df.copy()

    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    candle_range = h - l
    body = (c - o).abs()

    df["body_ratio"] = np.where(
        candle_range > 0,
        body / candle_range,
        np.nan,
    )

    df["upper_wick"] = np.where(
        candle_range > 0,
        (
            h -
            np.maximum(o, c)
        ) / candle_range,
        np.nan,
    )

    df["lower_wick"] = np.where(
        candle_range > 0,
        (
            np.minimum(o, c) -
            l
        ) / candle_range,
        np.nan,
    )

    df["ema20"] = c.ewm(
        span=20,
        adjust=False,
    ).mean()

    df["ema50"] = c.ewm(
        span=50,
        adjust=False,
    ).mean()

    df["ema100"] = c.ewm(
        span=100,
        adjust=False,
    ).mean()

    previous_close = c.shift(1)

    tr = pd.concat(
        [
            h - l,
            (h - previous_close).abs(),
            (l - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr14"] = tr.rolling(
        14,
        min_periods=14,
    ).mean()

    df["atr50"] = df["atr14"].rolling(
        50,
        min_periods=30,
    ).mean()

    df["atr_ratio"] = (
        df["atr14"] /
        df["atr50"]
    )

    df["momentum5"] = (
        c / c.shift(5) - 1.0
    )

    df["momentum10"] = (
        c / c.shift(10) - 1.0
    )

    rolling_high = h.rolling(
        20,
        min_periods=20,
    ).max()

    rolling_low = l.rolling(
        20,
        min_periods=20,
    ).min()

    rolling_range = (
        rolling_high -
        rolling_low
    )

    df["range_position"] = (
        c - rolling_low
    ) / rolling_range

    df["bullish"] = c > o
    df["bearish"] = c < o

    df["ema_bull"] = (
        (df["ema20"] > df["ema50"])
        &
        (df["ema50"] > df["ema100"])
    )

    df["ema_bear"] = (
        (df["ema20"] < df["ema50"])
        &
        (df["ema50"] < df["ema100"])
    )

    df["trend_strength"] = (
        (
            df["ema20"] -
            df["ema50"]
        ).abs()
        /
        df["atr14"]
    )

    print(
        "Indicators ready.",
        flush=True,
    )

    return df


# ============================================================
# SIGNAL
# ============================================================

def build_signal(
    df,
    wick,
    body,
):

    lower = df[
        "lower_wick"
    ].to_numpy()

    upper = df[
        "upper_wick"
    ].to_numpy()

    body_ratio = df[
        "body_ratio"
    ].to_numpy()

    bullish = df[
        "bullish"
    ].to_numpy()

    bearish = df[
        "bearish"
    ].to_numpy()

    position = df[
        "range_position"
    ].to_numpy()

    momentum5 = df[
        "momentum5"
    ].to_numpy()

    momentum10 = df[
        "momentum10"
    ].to_numpy()

    score = np.zeros(
        len(df),
        dtype=np.float32,
    )

    # Lower rejection
    score += np.where(
        lower >= wick,
        1.0,
        0.0,
    )

    # Upper rejection
    score -= np.where(
        upper >= wick,
        1.0,
        0.0,
    )

    # Small body
    score += np.where(
        body_ratio <= body,
        0.5,
        0.0,
    )

    # Candle direction
    score += np.where(
        bullish,
        0.25,
        0.0,
    )

    score -= np.where(
        bearish,
        0.25,
        0.0,
    )

    # Lower-range rejection
    score += np.where(
        bullish &
        (position <= 0.35),
        0.50,
        0.0,
    )

    # Upper-range rejection
    score -= np.where(
        bearish &
        (position >= 0.65),
        0.50,
        0.0,
    )

    # Momentum confirmation
    score += np.where(
        bullish &
        (momentum5 > 0) &
        (momentum10 > 0),
        0.25,
        0.0,
    )

    score -= np.where(
        bearish &
        (momentum5 < 0) &
        (momentum10 < 0),
        0.25,
        0.0,
    )

    score[
        ~np.isfinite(score)
    ] = np.nan

    return score


# ============================================================
# BACKTEST
# ============================================================

def backtest(
    df,
    score,
    rr,
    hours,
    threshold,
    cooldown,
    trend_filter,
    vol_filter,
):

    high = df[
        "high"
    ].to_numpy(
        dtype=np.float64
    )

    low = df[
        "low"
    ].to_numpy(
        dtype=np.float64
    )

    opens = df[
        "open"
    ].to_numpy(
        dtype=np.float64
    )

    atr = df[
        "atr14"
    ].to_numpy(
        dtype=np.float64
    )

    atr_ratio = df[
        "atr_ratio"
    ].to_numpy()

    ema_bull = df[
        "ema_bull"
    ].to_numpy()

    ema_bear = df[
        "ema_bear"
    ].to_numpy()

    momentum5 = df[
        "momentum5"
    ].to_numpy()

    hour = df[
        "time"
    ].dt.hour.to_numpy()

    mask = (
        np.isfinite(score)
        &
        np.isfinite(atr)
        &
        (atr > 0)
        &
        np.isfinite(atr_ratio)
        &
        np.isin(hour, hours)
        &
        (np.abs(score) >= abs(threshold))
    )

    if vol_filter == "normal":

        mask &= (
            (atr_ratio >= 0.80)
            &
            (atr_ratio <= 1.60)
        )

    if trend_filter == "aligned":

        long_ok = (
            ema_bull
            &
            (momentum5 > -0.0005)
        )

        short_ok = (
            ema_bear
            &
            (momentum5 < 0.0005)
        )

        mask &= (
            long_ok |
            short_ok
        )

    indices = np.flatnonzero(
        mask
    )

    if len(indices) == 0:

        return np.empty(
            0,
            dtype=np.float64,
        )

    indices = indices[
        indices < len(df) - 1
    ]

    if len(indices) == 0:

        return np.empty(
            0,
            dtype=np.float64,
        )

    selected = []

    last_index = -10**9

    for index in indices:

        if (
            index -
            last_index
            >= cooldown
        ):

            selected.append(index)
            last_index = index

    if not selected:

        return np.empty(
            0,
            dtype=np.float64,
        )

    selected = np.asarray(
        selected,
        dtype=np.int64,
    )

    # Direction is determined by the signal score.
    direction = np.where(
        score[selected] >= 0,
        1,
        -1,
    )

    results = []

    max_hold = 48

    for n, signal_index in enumerate(
        selected
    ):

        entry_index = (
            signal_index + 1
        )

        if entry_index >= len(df):
            continue

        entry = opens[
            entry_index
        ]

        risk = atr[
            signal_index
        ]

        if (
            not np.isfinite(risk)
            or risk <= 0
        ):
            continue

        side = direction[n]

        if side == 1:

            stop = entry - risk
            target = (
                entry +
                risk * rr
            )

        else:

            stop = entry + risk
            target = (
                entry -
                risk * rr
            )

        end = min(
            entry_index + max_hold,
            len(df),
        )

        highs = high[
            entry_index:end
        ]

        lows = low[
            entry_index:end
        ]

        if side == 1:

            target_hits = (
                highs >= target
            )

            stop_hits = (
                lows <= stop
            )

        else:

            target_hits = (
                lows <= target
            )

            stop_hits = (
                highs >= stop
            )

        target_positions = np.flatnonzero(
            target_hits
        )

        stop_positions = np.flatnonzero(
            stop_hits
        )

        first_target = (
            target_positions[0]
            if len(target_positions)
            else 10**9
        )

        first_stop = (
            stop_positions[0]
            if len(stop_positions)
            else 10**9
        )

        # Conservative assumption:
        # if TP and SL occur on the same candle,
        # count it as a loss.

        if first_stop <= first_target:

            if first_stop != 10**9:

                results.append(
                    -1.0
                )

        elif first_target != 10**9:

            results.append(
                float(rr)
            )

    return np.asarray(
        results,
        dtype=np.float64,
    )


# ============================================================
# PERFORMANCE
# ============================================================

def performance(results):

    if (
        results is None
        or len(results) == 0
    ):

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "losing_streak": 0,
        }

    results = np.asarray(
        results,
        dtype=float,
    )

    wins = results > 0
    losses = results < 0

    trades = len(results)

    wins_count = int(
        wins.sum()
    )

    losses_count = int(
        losses.sum()
    )

    win_rate = (
        wins_count /
        trades *
        100.0
    )

    gross_profit = float(
        results[wins].sum()
    )

    gross_loss = abs(
        float(
            results[losses].sum()
        )
    )

    profit_factor = (
        gross_profit /
        gross_loss
        if gross_loss > 0
        else 999.0
    )

    equity = np.cumsum(
        results
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

    longest = 0
    current = 0

    for value in results:

        if value < 0:

            current += 1
            longest = max(
                longest,
                current,
            )

        else:

            current = 0

    return {
        "trades": trades,
        "wins": wins_count,
        "losses": losses_count,
        "win_rate": win_rate,
        "total_r": float(
            results.sum()
        ),
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "losing_streak": longest,
    }


# ============================================================
# CANDIDATE SCORE
# ============================================================

def candidate_score(p):

    if (
        p["trades"] <
        MIN_TRAIN_TRADES
    ):

        return -999999.0

    if (
        p["total_r"] <= 0
    ):

        return -999999.0

    wr_component = (
        p["win_rate"] *
        0.50
    )

    r_component = (
        min(
            p["total_r"],
            50.0,
        )
        / 50.0
        * 100.0
        * 0.25
    )

    sample_component = (
        min(
            p["trades"],
            300,
        )
        / 300.0
        * 100.0
        * 0.15
    )

    pf_component = (
        min(
            p["profit_factor"],
            2.5,
        )
        / 2.5
        * 100.0
        * 0.10
    )

    dd_penalty = (
        min(
            p["max_drawdown"],
            15.0,
        )
        * 0.50
    )

    return (
        wr_component
        +
        r_component
        +
        sample_component
        +
        pf_component
        -
        dd_penalty
    )


# ============================================================
# OPTIMISATION
# ============================================================

def optimise_training(df):

    signal_cache = {}

    for wick, body in itertools.product(
        WICK_VALUES,
        BODY_VALUES,
    ):

        signal_cache[
            (wick, body)
        ] = build_signal(
            df,
            wick,
            body,
        )

    combinations = list(
        itertools.product(
            RR_VALUES,
            WICK_VALUES,
            BODY_VALUES,
            THRESHOLD_VALUES,
            HOUR_SETS,
            TREND_FILTERS,
            VOL_FILTERS,
            COOLDOWN_VALUES,
        )
    )

    print(
        f"TARGETED COMBINATIONS: "
        f"{len(combinations)}",
        flush=True,
    )

    candidates = []

    total = len(
        combinations
    )

    for i, params in enumerate(
        combinations,
        1,
    ):

        if (
            i == 1
            or i % 500 == 0
            or i == total
        ):

            print(
                f"Search progress: "
                f"{i}/{total}",
                flush=True,
            )

        (
            rr,
            wick,
            body,
            threshold,
            hours,
            trend_filter,
            vol_filter,
            cooldown,
        ) = params

        score = signal_cache[
            (wick, body)
        ]

        results = backtest(
            df,
            score,
            rr,
            hours,
            threshold,
            cooldown,
            trend_filter,
            vol_filter,
        )

        p = performance(
            results
        )

        quality = candidate_score(
            p
        )

        if quality <= -999999.0:
            continue

        candidates.append(
            {
                "params": params,
                "performance": p,
                "quality": quality,
            }
        )

    candidates.sort(
        key=lambda x:
        x["quality"],
        reverse=True,
    )

    print(
        f"VALID CANDIDATES: "
        f"{len(candidates)}",
        flush=True,
    )

    return candidates


# ============================================================
# STABILITY
# ============================================================

def stability_test(
    df,
    candidate,
):

    (
        rr,
        wick,
        body,
        threshold,
        hours,
        trend_filter,
        vol_filter,
        cooldown,
    ) = candidate["params"]

    nearby_rr = sorted(
        set(
            [
                rr,
                max(
                    0.40,
                    rr - 0.10,
                ),
                min(
                    0.75,
                    rr + 0.10,
                ),
            ]
        )
    )

    nearby_wick = sorted(
        set(
            [
                wick,
                max(
                    0.20,
                    wick - 0.05,
                ),
                min(
                    0.35,
                    wick + 0.05,
                ),
            ]
        )
    )

    nearby_body = sorted(
        set(
            [
                body,
                max(
                    0.20,
                    body - 0.05,
                ),
                min(
                    0.30,
                    body + 0.05,
                ),
            ]
        )
    )

    nearby_threshold = sorted(
        set(
            [
                threshold,
                max(
                    -0.25,
                    threshold - 0.25,
                ),
                min(
                    0.25,
                    threshold + 0.25,
                ),
            ]
        )
    )

    nearby_cooldown = sorted(
        set(
            [
                cooldown,
                max(
                    0,
                    cooldown - 4,
                ),
                cooldown + 4,
            ]
        )
    )

    combinations = list(
        itertools.product(
            nearby_rr,
            nearby_wick,
            nearby_body,
            nearby_threshold,
            nearby_cooldown,
        )
    )

    # Hard cap to keep runtime predictable.
    combinations = combinations[:60]

    nearby = []

    for (
        rr2,
        wick2,
        body2,
        threshold2,
        cooldown2,
    ) in combinations:

        score = build_signal(
            df,
            wick2,
            body2,
        )

        results = backtest(
            df,
            score,
            rr2,
            hours,
            threshold2,
            cooldown2,
            trend_filter,
            vol_filter,
        )

        p = performance(
            results
        )

        if (
            p["trades"] >=
            MIN_STABILITY_TRADES
        ):

            nearby.append(p)

    if len(nearby) < 5:

        return {
            "stable": False,
            "nearby": len(nearby),
            "median_wr": 0.0,
            "median_r": 0.0,
            "positive_pct": 0.0,
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

    positive_pct = float(
        np.mean(
            [
                x["total_r"] > 0
                for x in nearby
            ]
        ) * 100.0
    )

    stable = (
        median_wr >= 60.0
        and
        median_r > 0
        and
        positive_pct >= 60.0
    )

    return {
        "stable": stable,
        "nearby": len(nearby),
        "median_wr": median_wr,
        "median_r": median_r,
        "positive_pct": positive_pct,
    }


# ============================================================
# SELECT STRATEGY
# ============================================================

def select_strategy(df):

    candidates = optimise_training(
        df
    )

    if not candidates:

        return None

    print(
        "PHASE 2: STABILITY VALIDATION",
        flush=True,
    )

    max_stability = min(
        len(candidates),
        MAX_STABILITY_CANDIDATES,
    )

    stable_candidates = []

    for i in range(
        max_stability
    ):

        candidate = candidates[i]

        print(
            f"Stability candidate "
            f"{i + 1}/{max_stability}",
            flush=True,
        )

        stability = stability_test(
            df,
            candidate,
        )

        candidate[
            "stability"
        ] = stability

        if stability[
            "stable"
        ]:

            stable_candidates.append(
                candidate
            )

    if not stable_candidates:

        print(
            "NO STABLE CANDIDATES.",
            flush=True,
        )

        return None

    stable_candidates.sort(
        key=lambda x: (
            x[
                "stability"
            ][
                "positive_pct"
            ],
            x[
                "stability"
            ][
                "median_wr"
            ],
            x[
                "performance"
            ][
                "win_rate"
            ],
            x[
                "performance"
            ][
                "total_r"
            ],
        ),
        reverse=True,
    )

    return stable_candidates[0]


# ============================================================
# OOS
# ============================================================

def run_oos(
    df,
    strategy,
):

    (
        rr,
        wick,
        body,
        threshold,
        hours,
        trend_filter,
        vol_filter,
        cooldown,
    ) = strategy["params"]

    score = build_signal(
        df,
        wick,
        body,
    )

    results = backtest(
        df,
        score,
        rr,
        hours,
        threshold,
        cooldown,
        trend_filter,
        vol_filter,
    )

    return performance(
        results
    )


# ============================================================
# MARKET
# ============================================================

def run_market(
    market,
    path,
):

    print()
    print("=" * 60, flush=True)
    print(
        f"{market} V11.5",
        flush=True,
    )
    print("=" * 60, flush=True)

    df = load_data(
        path
    )

    df = prepare_indicators(
        df
    )

    all_results = []

    for period in WALK_FORWARD:

        (
            period_name,
            train_start,
            train_end,
            oos_start,
            oos_end,
        ) = period

        print()
        print("=" * 60, flush=True)
        print(
            f"{market}: {period_name}",
            flush=True,
        )
        print("=" * 60, flush=True)

        train = df[
            (df["time"] >= train_start)
            &
            (df["time"] <= train_end)
        ].copy()

        oos = df[
            (df["time"] >= oos_start)
            &
            (df["time"] <= oos_end)
        ].copy()

        train.reset_index(
            drop=True,
            inplace=True,
        )

        oos.reset_index(
            drop=True,
            inplace=True,
        )

        print(
            f"Training candles: "
            f"{len(train)}",
            flush=True,
        )

        print(
            f"OOS candles: "
            f"{len(oos)}",
            flush=True,
        )

        print(
            "Optimising...",
            flush=True,
        )

        strategy = select_strategy(
            train
        )

        if strategy is None:

            all_results.append(
                {
                    "market": market,
                    "period": period_name,
                    "strategy_found": False,
                    "stable": False,
                    "train_trades": 0,
                    "train_wr": 0.0,
                    "train_r": 0.0,
                    "oos_trades": 0,
                    "oos_wins": 0,
                    "oos_losses": 0,
                    "oos_wr": 0.0,
                    "oos_r": 0.0,
                    "oos_pf": 0.0,
                    "oos_dd": 0.0,
                    "oos_losing_streak": 0,
                }
            )

            print(
                "NO STABLE STRATEGY.",
                flush=True,
            )

            continue

        p_train = strategy[
            "performance"
        ]

        stability = strategy[
            "stability"
        ]

        params = strategy[
            "params"
        ]

        print()
        print(
            "SELECTED STRATEGY",
            flush=True,
        )
        print("-" * 60, flush=True)

        print(
            f"RR: {params[0]}",
            flush=True,
        )

        print(
            f"Wick: {params[1]}",
            flush=True,
        )

        print(
            f"Body: {params[2]}",
            flush=True,
        )

        print(
            f"Threshold: {params[3]}",
            flush=True,
        )

        print(
            f"Hours: {params[4]}",
            flush=True,
        )

        print(
            f"Trend filter: {params[5]}",
            flush=True,
        )

        print(
            f"Volatility filter: {params[6]}",
            flush=True,
        )

        print(
            f"Cooldown: {params[7]}",
            flush=True,
        )

        print(
            f"Training trades: "
            f"{p_train['trades']}",
            flush=True,
        )

        print(
            f"Training WR: "
            f"{p_train['win_rate']:.2f}%",
            flush=True,
        )

        print(
            f"Training R: "
            f"{p_train['total_r']:.2f}",
            flush=True,
        )

        print()
        print(
            "PARAMETER STABILITY",
            flush=True,
        )

        print(
            f"Nearby: "
            f"{stability['nearby']}",
            flush=True,
        )

        print(
            f"Median nearby WR: "
            f"{stability['median_wr']:.2f}%",
            flush=True,
        )

        print(
            f"Median nearby R: "
            f"{stability['median_r']:.2f}R",
            flush=True,
        )

        print(
            f"Positive nearby: "
            f"{stability['positive_pct']:.1f}%",
            flush=True,
        )

        print(
            "Stability: PASS",
            flush=True,
        )

        print()
        print(
            "OUT-OF-SAMPLE RESULT",
            flush=True,
        )
        print("-" * 60, flush=True)

        p_oos = run_oos(
            oos,
            strategy,
        )

        print(
            f"Trades: "
            f"{p_oos['trades']}",
            flush=True,
        )

        print(
            f"Wins: "
            f"{p_oos['wins']}",
            flush=True,
        )

        print(
            f"Losses: "
            f"{p_oos['losses']}",
            flush=True,
        )

        print(
            f"Win rate: "
            f"{p_oos['win_rate']:.2f}%",
            flush=True,
        )

        print(
            f"Total R: "
            f"{p_oos['total_r']:.2f}",
            flush=True,
        )

        print(
            f"Profit factor: "
            f"{p_oos['profit_factor']:.2f}",
            flush=True,
        )

        print(
            f"Max drawdown: "
            f"{p_oos['max_drawdown']:.2f}R",
            flush=True,
        )

        print(
            f"Longest losing streak: "
            f"{p_oos['losing_streak']}",
            flush=True,
        )

        all_results.append(
            {
                "market": market,
                "period": period_name,
                "strategy_found": True,
                "stable": True,
                "train_trades": p_train["trades"],
                "train_wr": p_train["win_rate"],
                "train_r": p_train["total_r"],
                "oos_trades": p_oos["trades"],
                "oos_wins": p_oos["wins"],
                "oos_losses": p_oos["losses"],
                "oos_wr": p_oos["win_rate"],
                "oos_r": p_oos["total_r"],
                "oos_pf": p_oos["profit_factor"],
                "oos_dd": p_oos["max_drawdown"],
                "oos_losing_streak":
                    p_oos["losing_streak"],
                "rr": params[0],
                "wick": params[1],
                "body": params[2],
                "threshold": params[3],
                "hours": ",".join(
                    map(
                        str,
                        params[4],
                    )
                ),
                "trend_filter": params[5],
                "vol_filter": params[6],
                "cooldown": params[7],
            }
        )

    result_df = pd.DataFrame(
        all_results
    )

    result_df.to_csv(
        RESULT_FILES[market],
        index=False,
    )

    return result_df


# ============================================================
# MAIN
# ============================================================

def main():

    market_results = {}

    for market, path in MARKETS.items():

        try:

            market_results[
                market
            ] = run_market(
                market,
                path,
            )

        except Exception as error:

            print()
            print("=" * 60, flush=True)
            print(
                f"{market} FAILED",
                flush=True,
            )

            print(
                f"{type(error).__name__}: "
                f"{error}",
                flush=True,
            )

            print("=" * 60, flush=True)

            market_results[
                market
            ] = pd.DataFrame()

    summary_rows = []

    total_trades = 0
    total_wins = 0
    total_r = 0.0

    total_periods = 0
    profitable_periods = 0
    stable_periods = 0

    for market, result_df in (
        market_results.items()
    ):

        if result_df.empty:
            continue

        oos_trades = int(
            result_df[
                "oos_trades"
            ].sum()
        )

        wins = int(
            result_df[
                "oos_wins"
            ].sum()
        )

        market_r = float(
            result_df[
                "oos_r"
            ].sum()
        )

        periods = len(
            result_df
        )

        profitable = int(
            (
                result_df[
                    "oos_r"
                ] > 0
            ).sum()
        )

        stable = int(
            result_df[
                "stable"
            ].sum()
        )

        win_rate = (
            wins /
            oos_trades *
            100.0
            if oos_trades > 0
            else 0.0
        )

        total_trades += oos_trades
        total_wins += wins
        total_r += market_r

        total_periods += periods
        profitable_periods += profitable
        stable_periods += stable

        if (
            oos_trades >= 200
            and win_rate >= 75.0
            and market_r > 0
            and profitable >= 2
        ):

            verdict = "TARGET RANGE"

        elif (
            oos_trades >= 200
            and win_rate >= 72.0
            and market_r > 0
        ):

            verdict = "STRONG BASELINE"

        elif (
            oos_trades >= 100
            and win_rate >= 70.0
            and market_r > 0
        ):

            verdict = "PROMISING"

        else:

            verdict = "NOT THERE YET"

        summary_rows.append(
            {
                "market": market,
                "oos_trades": oos_trades,
                "oos_win_rate": round(
                    win_rate,
                    2,
                ),
                "oos_total_r": round(
                    market_r,
                    2,
                ),
                "profitable_periods":
                    f"{profitable}/{periods}",
                "stable_periods":
                    f"{stable}/{periods}",
                "verdict": verdict,
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    combined_wr = (
        total_wins /
        total_trades *
        100.0
        if total_trades > 0
        else 0.0
    )

    print()
    print("=" * 60, flush=True)
    print(
        "V11.5 FINAL MULTI-MARKET SUMMARY",
        flush=True,
    )
    print("=" * 60, flush=True)

    if summary_df.empty:

        print(
            "NO COMPLETED MARKET RESULTS",
            flush=True,
        )

    else:

        print(
            summary_df.to_string(
                index=False
            ),
            flush=True,
        )

    print()
    print("=" * 60, flush=True)
    print(
        "COMBINED CROSS-MARKET OOS",
        flush=True,
    )
    print("=" * 60, flush=True)

    print(
        f"Trades: {total_trades}",
        flush=True,
    )

    print(
        f"Wins: {total_wins}",
        flush=True,
    )

    print(
        f"Win rate: {combined_wr:.2f}%",
        flush=True,
    )

    print(
        f"Total R: {total_r:.2f}",
        flush=True,
    )

    print(
        f"Profitable periods: "
        f"{profitable_periods}/{total_periods}",
        flush=True,
    )

    print(
        f"Stable periods: "
        f"{stable_periods}/{total_periods}",
        flush=True,
    )

    print()
    print("=" * 60, flush=True)
    print(
        "V11.5 TARGET CHECK",
        flush=True,
    )
    print("=" * 60, flush=True)

    print(
        "TARGET: >=75% WIN RATE",
        flush=True,
    )

    print(
        "TARGET: >=200 GENUINELY "
        "OUT-OF-SAMPLE TRADES",
        flush=True,
    )

    print(
        "TARGET: POSITIVE TOTAL R",
        flush=True,
    )

    if (
        total_trades >= 200
        and
        combined_wr >= 75.0
        and
        total_r > 0
    ):

        print(
            "TARGET STATUS: ACHIEVED",
            flush=True,
        )

    elif (
        total_trades >= 200
        and
        combined_wr >= 72.0
        and
        total_r > 0
    ):

        print(
            "TARGET STATUS: STRONG BASELINE",
            flush=True,
        )

    else:

        print(
            "TARGET STATUS: NOT ACHIEVED YET",
            flush=True,
        )

    print()
    print("=" * 60, flush=True)
    print(
        "IMPORTANT",
        flush=True,
    )
    print("=" * 60, flush=True)

    print(
        "OOS data is NEVER used to "
        "optimise its own period.",
        flush=True,
    )

    print(
        "Every walk-forward period "
        "is independently tested.",
        flush=True,
    )

    print(
        "This is research only.",
        flush=True,
    )

    print(
        "DO NOT IMPLEMENT LIVE.",
        flush=True,
    )

    print()
    print(
        "Results saved:",
        flush=True,
    )

    print(
        RESULT_FILES["XAUUSD"],
        flush=True,
    )

    print(
        RESULT_FILES["EURUSD"],
        flush=True,
    )

    print(
        SUMMARY_FILE,
        flush=True,
    )

    print()
    print("=" * 60, flush=True)
    print(
        "OPTIMIZER V11.5 COMPLETE",
        flush=True,
    )
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
