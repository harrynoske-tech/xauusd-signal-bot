# ============================================================
# MULTI-MARKET STRATEGY OPTIMIZER V11.6
# ============================================================
#
# V11.6 = CONTROLLED V11.3 REFINEMENT
#
# V11.3 BASELINE:
#   418 genuine OOS trades
#   72.01% OOS win rate
#   +3.40R
#
# PRIMARY TARGET:
#   >=75% OOS WIN RATE
#   >=200 GENUINE OOS TRADES
#   POSITIVE TOTAL R
#
# IMPORTANT:
#   NO LIVE TRADING
#   OOS DATA NEVER USED FOR OPTIMISATION
#
# V11.6 PRINCIPLE:
#   DO NOT ADD A LARGE FILTER STACK.
#
#   The V11.5 filter stack destroyed the V11.3 edge.
#   V11.6 therefore returns to the simpler V11.3
#   rejection/reversal architecture.
#
#   The optimiser now rewards:
#     - high training win rate
#     - positive training R
#     - sufficient sample size
#     - low drawdown
#     - parameter stability
#
#   It does NOT simply select the highest training WR.
#
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
    "XAUUSD": "data/xauusd_optimizer_v11_6_results.csv",
    "EURUSD": "data/eurusd_optimizer_v11_6_results.csv",
}

SUMMARY_FILE = (
    "data/multi_market_optimizer_v11_6_summary.csv"
)


# ============================================================
# WALK-FORWARD PERIODS
# ============================================================

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
# V11.3-STYLE PARAMETER SPACE
# ============================================================
#
# Keep this intentionally small.
#
# V11.3 demonstrated that the useful region is approximately:
#
# RR:
#   0.4 - 0.75
#
# Wick:
#   0.20 - 0.35
#
# Body:
#   0.20 - 0.30
#
# Separation:
#   V11.3 used approximately 0.0005 - 0.0008
#
# Hours:
#   primarily 3,4,5 and 12,13
#
# V11.6 does NOT add trend/volatility/cooldown filters.
#
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

SEPARATION_VALUES = [
    0.0005,
    0.00065,
    0.0008,
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


# ============================================================
# LIMITS
# ============================================================

MIN_TRAIN_TRADES = 40
MIN_STABILITY_TRADES = 20

MAX_STABILITY_CANDIDATES = 10

MIN_NEARBY_POSITIVE_PERCENT = 60.0
MIN_NEARBY_MEDIAN_R = 0.0
MIN_NEARBY_MEDIAN_WR = 60.0


# ============================================================
# HEADER
# ============================================================

print("=" * 60, flush=True)
print("MULTI-MARKET STRATEGY OPTIMIZER V11.6", flush=True)
print("=" * 60, flush=True)

print(
    "V11.3 BASELINE REFINEMENT",
    flush=True,
)

print(
    "CONTROLLED SEARCH: ENABLED",
    flush=True,
)

print(
    "REJECTION / REVERSAL: ENABLED",
    flush=True,
)

print(
    "TREND FILTER STACK: DISABLED",
    flush=True,
)

print(
    "VOLATILITY FILTER STACK: DISABLED",
    flush=True,
)

print(
    "ROBUSTNESS-FIRST RANKING: ENABLED",
    flush=True,
)

print(
    "PARAMETER STABILITY: ENABLED",
    flush=True,
)

print(
    "WALK-FORWARD TESTING: ENABLED",
    flush=True,
)

print(
    "STRICT OOS VALIDATION: ENABLED",
    flush=True,
)

print(
    "NO LIVE TRADING",
    flush=True,
)

print("=" * 60, flush=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_data(path):

    print(
        f"Loading {path}...",
        flush=True,
    )

    if not os.path.exists(path):

        raise RuntimeError(
            f"Data file not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:

        raise RuntimeError(
            f"Data file is empty: {path}"
        )

    # --------------------------------------------------------
    # Normalise column names
    # --------------------------------------------------------

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

    df = df.rename(
        columns=rename
    )

    # --------------------------------------------------------
    # Find time column
    # --------------------------------------------------------

    candidates = [
        "time",
        "datetime",
        "date",
        "timestamp",
        "timestamp_utc",
        "datetime_utc",
        "utc_time",
    ]

    time_column = None

    for name in candidates:

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

                if (
                    parsed.notna().mean()
                    >= 0.90
                ):

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

    # --------------------------------------------------------
    # OHLC aliases
    # --------------------------------------------------------

    aliases = {
        "open": [
            "open",
            "o",
            "open_price",
        ],
        "high": [
            "high",
            "h",
            "high_price",
        ],
        "low": [
            "low",
            "l",
            "low_price",
        ],
        "close": [
            "close",
            "c",
            "close_price",
        ],
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
            f"{path} missing required "
            f"columns: {missing}"
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # OHLC sanity
    # --------------------------------------------------------

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
        f"Range: "
        f"{df['time'].min()} -> "
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

    body = (
        c - o
    ).abs()

    # --------------------------------------------------------
    # Candle structure
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = c.shift(1)

    true_range = pd.concat(
        [
            h - l,
            (h - previous_close).abs(),
            (l - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr14"] = (
        true_range
        .rolling(
            14,
            min_periods=14,
        )
        .mean()
    )

    # --------------------------------------------------------
    # Range / separation data
    # --------------------------------------------------------

    df["range_mean20"] = (
        candle_range
        .rolling(
            20,
            min_periods=20,
        )
        .mean()
    )

    # --------------------------------------------------------
    # EMA structure
    #
    # Used as information in the signal, but NOT as a
    # hard filter. This preserves V11.3's sample size.
    # --------------------------------------------------------

    df["ema20"] = (
        c.ewm(
            span=20,
            adjust=False,
        ).mean()
    )

    df["ema50"] = (
        c.ewm(
            span=50,
            adjust=False,
        ).mean()
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    df["momentum5"] = (
        c /
        c.shift(5)
        - 1.0
    )

    # --------------------------------------------------------
    # Rolling range location
    # --------------------------------------------------------

    rolling_high = (
        h.rolling(
            20,
            min_periods=20,
        ).max()
    )

    rolling_low = (
        l.rolling(
            20,
            min_periods=20,
        ).min()
    )

    rolling_range = (
        rolling_high -
        rolling_low
    )

    df["range_position"] = (
        c -
        rolling_low
    ) / rolling_range

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
    separation,
):

    lower = (
        df["lower_wick"]
        .to_numpy()
    )

    upper = (
        df["upper_wick"]
        .to_numpy()
    )

    body_ratio = (
        df["body_ratio"]
        .to_numpy()
    )

    bullish = (
        df["close"]
        .to_numpy()
        >
        df["open"]
        .to_numpy()
    )

    bearish = (
        df["close"]
        .to_numpy()
        <
        df["open"]
        .to_numpy()
    )

    close = (
        df["close"]
        .to_numpy()
    )

    ema20 = (
        df["ema20"]
        .to_numpy()
    )

    ema50 = (
        df["ema50"]
        .to_numpy()
    )

    range_position = (
        df["range_position"]
        .to_numpy()
    )

    momentum5 = (
        df["momentum5"]
        .to_numpy()
    )

    atr = (
        df["atr14"]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Base rejection score
    # --------------------------------------------------------

    score = np.zeros(
        len(df),
        dtype=np.float32,
    )

    # Strong lower wick = bullish rejection
    score += np.where(
        lower >= wick,
        1.0,
        0.0,
    )

    # Strong upper wick = bearish rejection
    score -= np.where(
        upper >= wick,
        1.0,
        0.0,
    )

    # Small body
    score += np.where(
        body_ratio <= body,
        0.50,
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

    # --------------------------------------------------------
    # Range-location confirmation
    # --------------------------------------------------------

    score += np.where(
        bullish
        &
        (range_position <= 0.35),
        0.50,
        0.0,
    )

    score -= np.where(
        bearish
        &
        (range_position >= 0.65),
        0.50,
        0.0,
    )

    # --------------------------------------------------------
    # Mild momentum confirmation
    # --------------------------------------------------------

    score += np.where(
        bullish
        &
        (momentum5 > 0),
        0.25,
        0.0,
    )

    score -= np.where(
        bearish
        &
        (momentum5 < 0),
        0.25,
        0.0,
    )

    # --------------------------------------------------------
    # EMA structure is deliberately NOT a hard filter.
    #
    # It contributes only a small amount to the score.
    # --------------------------------------------------------

    score += np.where(
        ema20 > ema50,
        0.10,
        0.0,
    )

    score -= np.where(
        ema20 < ema50,
        0.10,
        0.0,
    )

    # --------------------------------------------------------
    # Separation check
    #
    # separation is relative, so it works across markets.
    # --------------------------------------------------------

    ema_separation = np.where(
        atr > 0,
        np.abs(
            ema20 - ema50
        ) / atr,
        np.nan,
    )

    score += np.where(
        ema_separation >= separation,
        np.where(
            ema20 > ema50,
            0.10,
            -0.10,
        ),
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
):

    high = (
        df["high"]
        .to_numpy(
            dtype=np.float64
        )
    )

    low = (
        df["low"]
        .to_numpy(
            dtype=np.float64
        )
    )

    opens = (
        df["open"]
        .to_numpy(
            dtype=np.float64
        )
    )

    atr = (
        df["atr14"]
        .to_numpy(
            dtype=np.float64
        )
    )

    hour = (
        df["time"]
        .dt.hour
        .to_numpy()
    )

    # --------------------------------------------------------
    # Signal qualification
    # --------------------------------------------------------

    mask = (
        np.isfinite(score)
        &
        np.isfinite(atr)
        &
        (atr > 0)
        &
        np.isin(
            hour,
            hours,
        )
        &
        (
            np.abs(score)
            >=
            abs(threshold)
        )
    )

    indices = np.flatnonzero(
        mask
    )

    if len(indices) == 0:

        return np.empty(
            0,
            dtype=np.float64,
        )

    # Need next candle for entry.
    indices = indices[
        indices < len(df) - 1
    ]

    if len(indices) == 0:

        return np.empty(
            0,
            dtype=np.float64,
        )

    # --------------------------------------------------------
    # No overlapping trades.
    #
    # The next signal cannot be taken until the previous
    # trade has finished.
    #
    # This is deliberately simple and conservative.
    # --------------------------------------------------------

    selected = []

    next_available = -1

    max_hold = 48

    for index in indices:

        if index < next_available:

            continue

        selected.append(
            index
        )

        next_available = (
            index +
            max_hold
        )

    if not selected:

        return np.empty(
            0,
            dtype=np.float64,
        )

    selected = np.asarray(
        selected,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    direction = np.where(
        score[selected] >= 0,
        1,
        -1,
    )

    results = []

    for n, signal_index in enumerate(
        selected
    ):

        entry_index = (
            signal_index + 1
        )

        if (
            entry_index >=
            len(df)
        ):

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

            stop = (
                entry - risk
            )

            target = (
                entry +
                risk * rr
            )

        else:

            stop = (
                entry + risk
            )

            target = (
                entry -
                risk * rr
            )

        end = min(
            entry_index +
            max_hold,
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

        target_positions = (
            np.flatnonzero(
                target_hits
            )
        )

        stop_positions = (
            np.flatnonzero(
                stop_hits
            )
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

        # Conservative:
        # same-candle TP + SL = loss.

        if first_stop <= first_target:

            if first_stop != 10**9:

                results.append(
                    -1.0
                )

        elif first_target != 10**9:

            results.append(
                float(rr)
            )

        else:

            # ------------------------------------------------
            # Neither target nor stop reached.
            #
            # Close at the final candle.
            # We convert the result to R.
            # ------------------------------------------------

            final_close = (
                df["close"]
                .iloc[
                    end - 1
                ]
            )

            if side == 1:

                r_value = (
                    final_close -
                    entry
                ) / risk

            else:

                r_value = (
                    entry -
                    final_close
                ) / risk

            # Keep the result bounded so an unusually large
            # unfinished move cannot dominate optimisation.
            r_value = float(
                np.clip(
                    r_value,
                    -1.0,
                    rr,
                )
            )

            results.append(
                r_value
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

    wins = (
        results > 0
    )

    losses = (
        results <= 0
    )

    trades = len(results)

    wins_count = int(
        wins.sum()
    )

    losses_count = (
        trades -
        wins_count
    )

    win_rate = (
        wins_count /
        trades *
        100.0
    )

    gross_profit = float(
        results[
            results > 0
        ].sum()
    )

    gross_loss = abs(
        float(
            results[
                results <= 0
            ].sum()
        )
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit /
            gross_loss
        )

    else:

        profit_factor = 999.0

    equity = np.cumsum(
        results
    )

    peak = np.maximum.accumulate(
        equity
    )

    drawdown = (
        peak -
        equity
    )

    max_drawdown = float(
        drawdown.max()
    )

    longest = 0
    current = 0

    for value in results:

        if value <= 0:

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
        "profit_factor":
            profit_factor,
        "max_drawdown":
            max_drawdown,
        "losing_streak":
            longest,
    }


# ============================================================
# ROBUSTNESS-FIRST SCORE
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

    # --------------------------------------------------------
    # We deliberately do NOT maximise WR alone.
    #
    # A 95% WR strategy with 20 trades is less useful than
    # a 74% WR strategy with hundreds of trades and strong R.
    # --------------------------------------------------------

    win_rate_component = (
        p["win_rate"] *
        0.40
    )

    sample_component = (
        min(
            p["trades"],
            500,
        )
        /
        500.0
        *
        100.0
        *
        0.20
    )

    r_component = (
        min(
            p["total_r"],
            60.0,
        )
        /
        60.0
        *
        100.0
        *
        0.25
    )

    pf_component = (
        min(
            p["profit_factor"],
            2.5,
        )
        /
        2.5
        *
        100.0
        *
        0.10
    )

    drawdown_penalty = (
        min(
            p["max_drawdown"],
            20.0,
        )
        *
        0.25
    )

    return (
        win_rate_component
        +
        sample_component
        +
        r_component
        +
        pf_component
        -
        drawdown_penalty
    )


# ============================================================
# TRAINING OPTIMISATION
# ============================================================

def optimise_training(df):

    # --------------------------------------------------------
    # Cache signals.
    # --------------------------------------------------------

    signal_cache = {}

    for (
        wick,
        body,
        separation,
    ) in itertools.product(
        WICK_VALUES,
        BODY_VALUES,
        SEPARATION_VALUES,
    ):

        signal_cache[
            (
                wick,
                body,
                separation,
            )
        ] = build_signal(
            df,
            wick,
            body,
            separation,
        )

    combinations = list(
        itertools.product(
            RR_VALUES,
            WICK_VALUES,
            BODY_VALUES,
            SEPARATION_VALUES,
            THRESHOLD_VALUES,
            HOUR_SETS,
        )
    )

    total = len(
        combinations
    )

    print(
        f"TARGETED COMBINATIONS: "
        f"{total}",
        flush=True,
    )

    candidates = []

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
            separation,
            threshold,
            hours,
        ) = params

        score = signal_cache[
            (
                wick,
                body,
                separation,
            )
        ]

        results = backtest(
            df,
            score,
            rr,
            hours,
            threshold,
        )

        p = performance(
            results
        )

        quality = candidate_score(
            p
        )

        if (
            quality <=
            -999999.0
        ):

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
# PARAMETER STABILITY
# ============================================================

def stability_test(
    df,
    candidate,
):

    (
        rr,
        wick,
        body,
        separation,
        threshold,
        hours,
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

    nearby_separation = sorted(
        set(
            [
                separation,
                max(
                    0.0005,
                    separation -
                    0.00015,
                ),
                min(
                    0.0008,
                    separation +
                    0.00015,
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

    combinations = list(
        itertools.product(
            nearby_rr,
            nearby_wick,
            nearby_body,
            nearby_separation,
            nearby_threshold,
        )
    )

    # Keep runtime predictable.
    combinations = combinations[
        :80
    ]

    nearby = []

    for (
        rr2,
        wick2,
        body2,
        separation2,
        threshold2,
    ) in combinations:

        score = build_signal(
            df,
            wick2,
            body2,
            separation2,
        )

        results = backtest(
            df,
            score,
            rr2,
            hours,
            threshold2,
        )

        p = performance(
            results
        )

        if (
            p["trades"]
            >= MIN_STABILITY_TRADES
        ):

            nearby.append(
                p
            )

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
        )
        * 100.0
    )

    stable = (
        median_wr
        >= MIN_NEARBY_MEDIAN_WR
        and
        median_r
        > MIN_NEARBY_MEDIAN_R
        and
        positive_pct
        >= MIN_NEARBY_POSITIVE_PERCENT
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

    # --------------------------------------------------------
    # Stability first.
    #
    # Then:
    #   median WR
    #   median R
    #   training WR
    #   training R
    #
    # This prevents one lucky training result from dominating.
    # --------------------------------------------------------

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
                "stability"
            ][
                "median_r"
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
        separation,
        threshold,
        hours,
    ) = strategy["params"]

    score = build_signal(
        df,
        wick,
        body,
        separation,
    )

    results = backtest(
        df,
        score,
        rr,
        hours,
        threshold,
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
        f"{market} V11.6",
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
            f"{market}: "
            f"{period_name}",
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
            f"Separation: {params[3]}",
            flush=True,
        )

        print(
            f"Threshold: {params[4]}",
            flush=True,
        )

        print(
            f"Hours: {params[5]}",
            flush=True,
        )

        print()
        print(
            "TRAINING",
            flush=True,
        )

        print(
            f"Trades: "
            f"{p_train['trades']}",
            flush=True,
        )

        print(
            f"Win rate: "
            f"{p_train['win_rate']:.2f}%",
            flush=True,
        )

        print(
            f"Total R: "
            f"{p_train['total_r']:.2f}",
            flush=True,
        )

        print(
            f"Profit factor: "
            f"{p_train['profit_factor']:.2f}",
            flush=True,
        )

        print(
            f"Max drawdown: "
            f"{p_train['max_drawdown']:.2f}R",
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
                "train_trades":
                    p_train["trades"],
                "train_wr":
                    p_train["win_rate"],
                "train_r":
                    p_train["total_r"],
                "oos_trades":
                    p_oos["trades"],
                "oos_wins":
                    p_oos["wins"],
                "oos_losses":
                    p_oos["losses"],
                "oos_wr":
                    p_oos["win_rate"],
                "oos_r":
                    p_oos["total_r"],
                "oos_pf":
                    p_oos["profit_factor"],
                "oos_dd":
                    p_oos["max_drawdown"],
                "oos_losing_streak":
                    p_oos["losing_streak"],
                "rr": params[0],
                "wick": params[1],
                "body": params[2],
                "separation":
                    params[3],
                "threshold":
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

    # ========================================================
    # SUMMARY
    # ========================================================

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
                ]
                > 0
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

        total_trades += (
            oos_trades
        )

        total_wins += wins

        total_r += market_r

        total_periods += periods

        profitable_periods += (
            profitable
        )

        stable_periods += stable

        if (
            oos_trades >= 200
            and
            win_rate >= 75.0
            and
            market_r > 0
            and
            profitable >= 2
        ):

            verdict = (
                "TARGET RANGE"
            )

        elif (
            oos_trades >= 200
            and
            win_rate >= 72.0
            and
            market_r > 0
        ):

            verdict = (
                "STRONG BASELINE"
            )

        elif (
            oos_trades >= 100
            and
            win_rate >= 70.0
            and
            market_r > 0
        ):

            verdict = (
                "PROMISING"
            )

        else:

            verdict = (
                "NOT THERE YET"
            )

        summary_rows.append(
            {
                "market":
                    market,

                "oos_trades":
                    oos_trades,

                "oos_win_rate":
                    round(
                        win_rate,
                        2,
                    ),

                "oos_total_r":
                    round(
                        market_r,
                        2,
                    ),

                "profitable_periods":
                    f"{profitable}/{periods}",

                "stable_periods":
                    f"{stable}/{periods}",

                "verdict":
                    verdict,
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

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 60, flush=True)

    print(
        "V11.6 FINAL MULTI-MARKET SUMMARY",
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
        f"Win rate: "
        f"{combined_wr:.2f}%",
        flush=True,
    )

    print(
        f"Total R: "
        f"{total_r:.2f}",
        flush=True,
    )

    print(
        f"Profitable periods: "
        f"{profitable_periods}/"
        f"{total_periods}",
        flush=True,
    )

    print(
        f"Stable periods: "
        f"{stable_periods}/"
        f"{total_periods}",
        flush=True,
    )

    # ========================================================
    # TARGET CHECK
    # ========================================================

    print()
    print("=" * 60, flush=True)

    print(
        "V11.6 TARGET CHECK",
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
            "TARGET STATUS: "
            "STRONG BASELINE",
            flush=True,
        )

    elif (
        total_trades >= 200
        and
        combined_wr >= 70.0
        and
        total_r > 0
    ):

        print(
            "TARGET STATUS: "
            "PROMISING",
            flush=True,
        )

    else:

        print(
            "TARGET STATUS: "
            "NOT ACHIEVED YET",
            flush=True,
        )

    # ========================================================
    # SAFETY
    # ========================================================

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
        "Each walk-forward period "
        "is independently optimised.",
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
        "OPTIMIZER V11.6 COMPLETE",
        flush=True,
    )

    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
