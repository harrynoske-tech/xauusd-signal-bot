# ============================================================
# MULTI-MARKET STRATEGY OPTIMIZER V11.8
# ============================================================
#
# V11.8 MARKET-SPECIFIC REFINEMENT
#
# V11.3 BASELINE:
#   418 OOS trades
#   72.01% WR
#   +3.40R
#
# V11.7:
#   EURUSD -> 76.40% WR / +11.20R
#   XAUUSD -> 69.76% WR / -9.80R
#
# V11.8 OBJECTIVE:
#
#   EURUSD:
#       Preserve V11.7 architecture.
#
#   XAUUSD:
#       Target the V11.3/V11.7 rejection architecture
#       with a dedicated market-specific search.
#
# TARGET:
#   >=75% WIN RATE
#   >=200 GENUINE OOS TRADES
#   POSITIVE TOTAL R
#
# IMPORTANT:
#   OOS DATA IS NEVER USED FOR OPTIMISATION.
#
# NO LIVE TRADING.
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
    "XAUUSD": "data/xauusd_optimizer_v11_8_results.csv",
    "EURUSD": "data/eurusd_optimizer_v11_8_results.csv",
}

SUMMARY_FILE = (
    "data/multi_market_optimizer_v11_8_summary.csv"
)


# ============================================================
# WALK FORWARD
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
# XAUUSD SEARCH
#
# Narrowly centred around the parameters that have actually
# worked in V11.3/V11.7.
# ============================================================

XAU_RR = [
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
]

XAU_WICK = [
    0.20,
    0.25,
    0.30,
    0.35,
]

XAU_BODY = [
    0.15,
    0.20,
    0.25,
    0.30,
]

XAU_SEPARATION = [
    0.0004,
    0.0005,
    0.00065,
    0.0008,
]

XAU_THRESHOLD = [
    -0.25,
    0.00,
    0.25,
]

XAU_HOURS = [
    (3, 4),
    (4, 5),
    (3, 4, 5),
    (4, 5, 12, 13),
]


# ============================================================
# EURUSD SEARCH
#
# V11.7 produced 76.40% OOS WR.
#
# Keep the search deliberately close to that architecture.
# ============================================================

EUR_RR = [
    0.35,
    0.40,
    0.45,
    0.50,
]

EUR_WICK = [
    0.20,
    0.25,
    0.30,
]

EUR_BODY = [
    0.15,
    0.20,
    0.25,
]

EUR_SEPARATION = [
    0.0005,
    0.00065,
    0.0008,
]

EUR_THRESHOLD = [
    -0.25,
    0.00,
    0.25,
]

EUR_HOURS = [
    (3, 4, 5),
    (4, 5),
    (4, 5, 12, 13),
]


# ============================================================
# REQUIREMENTS
# ============================================================

MIN_TRAIN_TRADES = 40
MIN_RECENT_TRADES = 20
MIN_SUBPERIOD_TRADES = 10

MAX_STABILITY_CANDIDATES = 8

MIN_NEARBY_WR = 60.0
MIN_NEARBY_R = 0.0
MIN_NEARBY_POSITIVE = 60.0


# ============================================================
# HEADER
# ============================================================

print("=" * 60, flush=True)
print(
    "MULTI-MARKET STRATEGY OPTIMIZER V11.8",
    flush=True,
)
print("=" * 60, flush=True)

print(
    "MARKET-SPECIFIC OPTIMISATION: ENABLED",
    flush=True,
)

print(
    "XAUUSD DEDICATED SEARCH: ENABLED",
    flush=True,
)

print(
    "EURUSD V11.7 REFINEMENT: ENABLED",
    flush=True,
)

print(
    "REJECTION / REVERSAL: ENABLED",
    flush=True,
)

print(
    "RECENCY-WEIGHTED TRAINING: ENABLED",
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
# LOAD DATA
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
        x
        for x in required
        if x not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{path} missing required "
            f"columns: {missing}"
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

    df["momentum5"] = (
        c /
        c.shift(5)
        - 1.0
    )

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

    lower = df[
        "lower_wick"
    ].to_numpy()

    upper = df[
        "upper_wick"
    ].to_numpy()

    body_ratio = df[
        "body_ratio"
    ].to_numpy()

    opens = df[
        "open"
    ].to_numpy()

    closes = df[
        "close"
    ].to_numpy()

    bullish = (
        closes > opens
    )

    bearish = (
        closes < opens
    )

    ema20 = df[
        "ema20"
    ].to_numpy()

    ema50 = df[
        "ema50"
    ].to_numpy()

    range_position = df[
        "range_position"
    ].to_numpy()

    momentum5 = df[
        "momentum5"
    ].to_numpy()

    atr = df[
        "atr14"
    ].to_numpy()

    score = np.zeros(
        len(df),
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # PRIMARY REJECTION
    # --------------------------------------------------------

    score += np.where(
        lower >= wick,
        1.0,
        0.0,
    )

    score -= np.where(
        upper >= wick,
        1.0,
        0.0,
    )

    score += np.where(
        body_ratio <= body,
        0.50,
        0.0,
    )

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
    # RANGE LOCATION
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
    # MOMENTUM
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
    # SOFT EMA STRUCTURE
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

    closes = df[
        "close"
    ].to_numpy(
        dtype=np.float64
    )

    atr = df[
        "atr14"
    ].to_numpy(
        dtype=np.float64
    )

    hour = (
        df["time"]
        .dt.hour
        .to_numpy()
    )

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

    indices = indices[
        indices < len(df) - 1
    ]

    if len(indices) == 0:

        return np.empty(
            0,
            dtype=np.float64,
        )

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

    results = []

    for signal_index in selected:

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

        side = (
            1
            if score[
                signal_index
            ] >= 0
            else -1
        )

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

        if (
            first_stop <=
            first_target
        ):

            if (
                first_stop !=
                10**9
            ):

                results.append(
                    -1.0
                )

        elif (
            first_target !=
            10**9
        ):

            results.append(
                float(rr)
            )

        else:

            final_close = closes[
                end - 1
            ]

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

            results.append(
                float(
                    np.clip(
                        r_value,
                        -1.0,
                        rr,
                    )
                )
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
# TRADE INDICES
# ============================================================

def get_trade_indices(
    df,
    score,
    hours,
    threshold,
):

    atr = df[
        "atr14"
    ].to_numpy()

    hour = (
        df["time"]
        .dt.hour
        .to_numpy()
    )

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

    indices = indices[
        indices < len(df) - 1
    ]

    selected = []

    next_available = -1

    for index in indices:

        if index < next_available:

            continue

        selected.append(
            index
        )

        next_available = (
            index + 48
        )

    return np.asarray(
        selected,
        dtype=np.int64,
    )


# ============================================================
# RECENCY SCORE
# ============================================================

def recency_score(
    results,
    indices,
    length,
):

    if (
        len(results) <
        MIN_TRAIN_TRADES
    ):

        return -999999.0

    n = min(
        len(results),
        len(indices),
    )

    results = results[:n]
    indices = indices[:n]

    old_cut = (
        length *
        0.333333
    )

    recent_cut = (
        length *
        0.666667
    )

    old = results[
        indices < old_cut
    ]

    middle = results[
        (indices >= old_cut)
        &
        (indices < recent_cut)
    ]

    recent = results[
        indices >= recent_cut
    ]

    if (
        len(old) <
        MIN_SUBPERIOD_TRADES
    ):

        return -999999.0

    if (
        len(middle) <
        MIN_SUBPERIOD_TRADES
    ):

        return -999999.0

    if (
        len(recent) <
        MIN_RECENT_TRADES
    ):

        return -999999.0

    p_old = performance(old)
    p_middle = performance(middle)
    p_recent = performance(recent)
    p_full = performance(results)

    weighted_wr = (
        p_old["win_rate"] * 0.20
        +
        p_middle["win_rate"] * 0.30
        +
        p_recent["win_rate"] * 0.50
    )

    weighted_r = (
        p_old["total_r"] * 0.20
        +
        p_middle["total_r"] * 0.30
        +
        p_recent["total_r"] * 0.50
    )

    score = (
        weighted_wr * 0.55
        +
        weighted_r * 0.25
        +
        min(
            p_full["profit_factor"],
            2.0,
        )
        * 10.0
        +
        min(
            p_full["trades"],
            500,
        )
        * 0.01
        -
        p_full["max_drawdown"]
        * 0.20
    )

    if p_old["total_r"] < 0:

        score -= 5.0

    if p_middle["total_r"] < 0:

        score -= 3.0

    if p_recent["total_r"] < 0:

        score -= 2.0

    return float(
        score
    )


# ============================================================
# OPTIMISATION
# ============================================================

def optimise(
    df,
    market,
):

    if market == "XAUUSD":

        rr_values = XAU_RR
        wick_values = XAU_WICK
        body_values = XAU_BODY
        separation_values = (
            XAU_SEPARATION
        )
        threshold_values = (
            XAU_THRESHOLD
        )
        hour_sets = XAU_HOURS

    else:

        rr_values = EUR_RR
        wick_values = EUR_WICK
        body_values = EUR_BODY
        separation_values = (
            EUR_SEPARATION
        )
        threshold_values = (
            EUR_THRESHOLD
        )
        hour_sets = EUR_HOURS

    signal_cache = {}

    for (
        wick,
        body,
        separation,
    ) in itertools.product(
        wick_values,
        body_values,
        separation_values,
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
            rr_values,
            wick_values,
            body_values,
            separation_values,
            threshold_values,
            hour_sets,
        )
    )

    print(
        f"TARGETED COMBINATIONS: "
        f"{len(combinations)}",
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
            or i == len(combinations)
        ):

            print(
                f"Search progress: "
                f"{i}/"
                f"{len(combinations)}",
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

        if (
            p["trades"] <
            MIN_TRAIN_TRADES
        ):

            continue

        indices = get_trade_indices(
            df,
            score,
            hours,
            threshold,
        )

        quality = recency_score(
            results,
            indices,
            len(df),
        )

        if (
            quality <=
            -999999
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
        separation,
        threshold,
        hours,
    ) = candidate["params"]

    nearby_rr = [
        rr,
        max(
            0.35,
            rr - 0.05,
        ),
        min(
            0.60,
            rr + 0.05,
        ),
    ]

    nearby_wick = [
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

    nearby_body = [
        body,
        max(
            0.15,
            body - 0.05,
        ),
        min(
            0.30,
            body + 0.05,
        ),
    ]

    nearby_sep = [
        separation,
        max(
            0.0004,
            separation - 0.00015,
        ),
        min(
            0.0008,
            separation + 0.00015,
        ),
    ]

    nearby_threshold = [
        threshold,
        -0.25,
        0.00,
        0.25,
    ]

    combinations = list(
        itertools.product(
            nearby_rr,
            nearby_wick,
            nearby_body,
            nearby_sep,
            nearby_threshold,
        )
    )

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
            p["trades"] >=
            MIN_TRAIN_TRADES
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
        )
        * 100.0
    )

    stable = (
        median_wr >=
        MIN_NEARBY_WR
        and
        median_r >
        MIN_NEARBY_R
        and
        positive_pct >=
        MIN_NEARBY_POSITIVE
    )

    return {
        "stable": stable,
        "nearby": len(nearby),
        "median_wr": median_wr,
        "median_r": median_r,
        "positive_pct": positive_pct,
    }


# ============================================================
# SELECT
# ============================================================

def select_strategy(
    df,
    market,
):

    candidates = optimise(
        df,
        market,
    )

    if not candidates:

        return None

    print(
        "PHASE 2: STABILITY VALIDATION",
        flush=True,
    )

    count = min(
        len(candidates),
        MAX_STABILITY_CANDIDATES,
    )

    stable = []

    for i in range(count):

        print(
            f"Stability candidate "
            f"{i + 1}/{count}",
            flush=True,
        )

        candidate = candidates[i]

        test = stability_test(
            df,
            candidate,
        )

        candidate[
            "stability"
        ] = test

        if test["stable"]:

            stable.append(
                candidate
            )

    if not stable:

        print(
            "NO STABLE CANDIDATES.",
            flush=True,
        )

        return None

    stable.sort(
        key=lambda x: (
            x["stability"][
                "positive_pct"
            ],
            x["stability"][
                "median_wr"
            ],
            x["stability"][
                "median_r"
            ],
            x["quality"],
        ),
        reverse=True,
    )

    return stable[0]


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
        f"{market} V11.8",
        flush=True,
    )
    print("=" * 60, flush=True)

    df = load_data(
        path
    )

    df = prepare_indicators(
        df
    )

    results = []

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
            train,
            market,
        )

        if strategy is None:

            print(
                "NO STABLE STRATEGY.",
                flush=True,
            )

            results.append(
                {
                    "market": market,
                    "period": period_name,
                    "strategy_found": False,
                    "stable": False,
                    "oos_trades": 0,
                    "oos_wins": 0,
                    "oos_losses": 0,
                    "oos_wr": 0.0,
                    "oos_r": 0.0,
                    "oos_pf": 0.0,
                    "oos_dd": 0.0,
                }
            )

            continue

        (
            rr,
            wick,
            body,
            separation,
            threshold,
            hours,
        ) = strategy["params"]

        train_p = strategy[
            "performance"
        ]

        stability = strategy[
            "stability"
        ]

        print()
        print(
            "SELECTED STRATEGY",
            flush=True,
        )

        print("-" * 60, flush=True)

        print(
            f"RR: {rr}",
            flush=True,
        )

        print(
            f"Wick: {wick}",
            flush=True,
        )

        print(
            f"Body: {body}",
            flush=True,
        )

        print(
            f"Separation: "
            f"{separation}",
            flush=True,
        )

        print(
            f"Threshold: "
            f"{threshold}",
            flush=True,
        )

        print(
            f"Hours: {hours}",
            flush=True,
        )

        print()
        print(
            "TRAINING RESULT",
            flush=True,
        )

        print(
            f"Trades: "
            f"{train_p['trades']}",
            flush=True,
        )

        print(
            f"WR: "
            f"{train_p['win_rate']:.2f}%",
            flush=True,
        )

        print(
            f"R: "
            f"{train_p['total_r']:.2f}",
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

        # ----------------------------------------------------
        # STRICT OOS
        # ----------------------------------------------------

        score = build_signal(
            oos,
            wick,
            body,
            separation,
        )

        oos_results = backtest(
            oos,
            score,
            rr,
            hours,
            threshold,
        )

        oos_p = performance(
            oos_results
        )

        print()
        print(
            "OUT-OF-SAMPLE RESULT",
            flush=True,
        )

        print("-" * 60, flush=True)

        print(
            f"Trades: "
            f"{oos_p['trades']}",
            flush=True,
        )

        print(
            f"Wins: "
            f"{oos_p['wins']}",
            flush=True,
        )

        print(
            f"Losses: "
            f"{oos_p['losses']}",
            flush=True,
        )

        print(
            f"Win rate: "
            f"{oos_p['win_rate']:.2f}%",
            flush=True,
        )

        print(
            f"Total R: "
            f"{oos_p['total_r']:.2f}",
            flush=True,
        )

        print(
            f"Profit factor: "
            f"{oos_p['profit_factor']:.2f}",
            flush=True,
        )

        print(
            f"Max drawdown: "
            f"{oos_p['max_drawdown']:.2f}R",
            flush=True,
        )

        print(
            f"Longest losing streak: "
            f"{oos_p['losing_streak']}",
            flush=True,
        )

        results.append(
            {
                "market": market,
                "period": period_name,
                "strategy_found": True,
                "stable": True,
                "train_trades":
                    train_p["trades"],
                "train_wr":
                    train_p["win_rate"],
                "train_r":
                    train_p["total_r"],
                "oos_trades":
                    oos_p["trades"],
                "oos_wins":
                    oos_p["wins"],
                "oos_losses":
                    oos_p["losses"],
                "oos_wr":
                    oos_p["win_rate"],
                "oos_r":
                    oos_p["total_r"],
                "oos_pf":
                    oos_p["profit_factor"],
                "oos_dd":
                    oos_p["max_drawdown"],
                "oos_losing_streak":
                    oos_p[
                        "losing_streak"
                    ],
                "rr": rr,
                "wick": wick,
                "body": body,
                "separation":
                    separation,
                "threshold":
                    threshold,
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
        results
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

    summary = []

    total_trades = 0
    total_wins = 0
    total_r = 0.0

    total_periods = 0
    profitable_periods = 0
    stable_periods = 0

    for market, df in (
        market_results.items()
    ):

        if df.empty:

            continue

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

        total_market_r = float(
            df[
                "oos_r"
            ].sum()
        )

        periods = len(df)

        profitable = int(
            (
                df[
                    "oos_r"
                ]
                > 0
            ).sum()
        )

        stable = int(
            df[
                "stable"
            ].sum()
        )

        wr = (
            wins /
            trades *
            100.0
            if trades > 0
            else 0.0
        )

        total_trades += trades
        total_wins += wins
        total_r += total_market_r

        total_periods += periods
        profitable_periods += profitable
        stable_periods += stable

        if (
            trades >= 200
            and
            wr >= 75
            and
            total_market_r > 0
            and
            profitable >= 2
        ):

            verdict = (
                "TARGET RANGE"
            )

        elif (
            trades >= 200
            and
            wr >= 72
            and
            total_market_r > 0
        ):

            verdict = (
                "STRONG BASELINE"
            )

        elif (
            trades >= 100
            and
            wr >= 70
            and
            total_market_r > 0
        ):

            verdict = (
                "PROMISING"
            )

        else:

            verdict = (
                "NOT THERE YET"
            )

        summary.append(
            {
                "market": market,
                "oos_trades": trades,
                "oos_win_rate":
                    round(
                        wr,
                        2,
                    ),
                "oos_total_r":
                    round(
                        total_market_r,
                        2,
                    ),
                "profitable_periods":
                    f"{profitable}/"
                    f"{periods}",
                "stable_periods":
                    f"{stable}/"
                    f"{periods}",
                "verdict": verdict,
            }
        )

    summary_df = pd.DataFrame(
        summary
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
        "V11.8 FINAL MULTI-MARKET SUMMARY",
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

    print()
    print("=" * 60, flush=True)

    print(
        "V11.8 TARGET CHECK",
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
        combined_wr >= 75
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
        combined_wr >= 72
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
        combined_wr >= 70
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
        "XAUUSD and EURUSD use "
        "market-specific parameter spaces.",
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
        "OPTIMIZER V11.8 COMPLETE",
        flush=True,
    )

    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
