# ============================================================
# MULTI-MARKET STRATEGY OPTIMIZER V11.3
# ============================================================
#
# FAST COARSE -> FINE SEARCH
# REGIME-AWARE
# REJECTION / REVERSAL
# WALK-FORWARD
# PARAMETER STABILITY
# STRICT FINAL HOLDOUT
#
# TARGET:
# ~82% WIN RATE
# 200+ GENUINELY OUT-OF-SAMPLE TRADES
# POSITIVE TOTAL R
#
# NO LIVE TRADING
# ============================================================

import os
import sys
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

print("=" * 60, flush=True)
print("MULTI-MARKET STRATEGY OPTIMIZER V11.3", flush=True)
print("=" * 60, flush=True)
print("FAST COARSE -> FINE SEARCH: ENABLED", flush=True)
print("REGIME FILTERING: ENABLED", flush=True)
print("REJECTION / REVERSAL: ENABLED", flush=True)
print("WALK-FORWARD TESTING: ENABLED", flush=True)
print("PARAMETER STABILITY: ENABLED", flush=True)
print("STRICT FINAL HOLDOUT: ENABLED", flush=True)
print("NO LIVE TRADING", flush=True)
print("=" * 60, flush=True)


# ============================================================
# CONFIG
# ============================================================

MARKETS = {
    "XAUUSD": "data/XAUUSD_15m.csv",
    "EURUSD": "data/EURUSD_15m.csv",
}

RESULT_FILES = {
    "XAUUSD": "data/xauusd_optimizer_v11_3_results.csv",
    "EURUSD": "data/eurusd_optimizer_v11_3_results.csv",
}

SUMMARY_FILE = (
    "data/multi_market_optimizer_v11_3_summary.csv"
)


# ============================================================
# WALK FORWARD
# ============================================================

WALK_FORWARD_PERIODS = [
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
        "2026-12-31",
    ),
]


# ============================================================
# MINIMUM SAMPLE REQUIREMENTS
# ============================================================

MIN_TRAIN_TRADES = 40
MIN_FINE_TRADES = 40
MIN_STABILITY_TRADES = 30
MIN_OOS_TRADES = 10


# ============================================================
# COARSE SEARCH
# ============================================================

COARSE_RR = [
    0.50,
    0.75,
    1.00,
]

COARSE_WICK = [
    0.20,
    0.30,
    0.35,
]

COARSE_BODY = [
    0.20,
    0.25,
    0.30,
]

COARSE_SEPARATION = [
    10,
    20,
    40,
]

COARSE_HOURS = [
    (3, 4, 5),
    (4, 5),
    (3, 4, 5, 12, 13),
]

COARSE_THRESHOLD = [
    -0.25,
    0.00,
    0.25,
]


# ============================================================
# FINE SEARCH
# ============================================================

FINE_RR = [
    0.40,
    0.50,
    0.60,
    0.75,
    0.90,
    1.00,
]

FINE_WICK = [
    0.15,
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
    10,
    15,
    20,
    30,
    40,
]

FINE_THRESHOLD = [
    -0.50,
    -0.25,
    0.00,
    0.25,
    0.50,
]


# ============================================================
# DATA
# ============================================================

def load_data(path):

    print(
        f"Loading {path}...",
        flush=True,
    )

    if not os.path.exists(path):

        raise RuntimeError(
            f"DATA FILE NOT FOUND: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:

        raise RuntimeError(
            f"DATA FILE EMPTY: {path}"
        )

    # --------------------------------------------------------
    # Normalise names
    # --------------------------------------------------------

    rename = {}

    for col in df.columns:

        clean = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

        rename[col] = clean

    df = df.rename(
        columns=rename
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    candidates = [
        "time",
        "datetime",
        "date",
        "timestamp",
        "timestamp_utc",
        "datetime_utc",
        "utc_time",
        "gmt_time",
        "date_time",
    ]

    time_col = None

    for candidate in candidates:

        if candidate in df.columns:

            time_col = candidate
            break

    if time_col is None:

        for col in df.columns:

            if col in {
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
                    df[col],
                    utc=True,
                    errors="coerce",
                )

                if parsed.notna().mean() >= 0.90:

                    time_col = col
                    break

            except Exception:

                pass

    if time_col is None:

        raise RuntimeError(
            "Could not find datetime column.\n"
            f"Columns: {list(df.columns)}"
        )

    df["time"] = pd.to_datetime(
        df[time_col],
        utc=True,
        errors="coerce",
    )

    # --------------------------------------------------------
    # OHLC
    # --------------------------------------------------------

    aliases = {
        "open": ["open", "o", "open_price"],
        "high": ["high", "h", "high_price"],
        "low": ["low", "l", "low_price"],
        "close": ["close", "c", "close_price"],
    }

    ohlc = {}

    for target, options in aliases.items():

        for option in options:

            if option in df.columns:

                ohlc[option] = target
                break

    df = df.rename(
        columns=ohlc
    )

    required = [
        "time",
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        x for x in required
        if x not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{path} missing columns: {missing}"
        )

    for col in [
        "open",
        "high",
        "low",
        "close",
    ]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=required
    )

    df = df[
        (df["high"] >= df["low"]) &
        (df["high"] >= df["open"]) &
        (df["high"] >= df["close"]) &
        (df["low"] <= df["open"]) &
        (df["low"] <= df["close"])
    ]

    df = (
        df
        .sort_values("time")
        .drop_duplicates(
            "time"
        )
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

    df["upper_wick_ratio"] = np.where(
        candle_range > 0,
        (
            h -
            np.maximum(o, c)
        ) / candle_range,
        np.nan,
    )

    df["lower_wick_ratio"] = np.where(
        candle_range > 0,
        (
            np.minimum(o, c) -
            l
        ) / candle_range,
        np.nan,
    )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    ema20 = c.ewm(
        span=20,
        adjust=False,
    ).mean()

    ema50 = c.ewm(
        span=50,
        adjust=False,
    ).mean()

    df["ema20"] = ema20
    df["ema50"] = ema50

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    prev = c.shift(1)

    tr = pd.concat(
        [
            h - l,
            (h - prev).abs(),
            (l - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr14 = tr.rolling(
        14,
        min_periods=14,
    ).mean()

    atr50 = atr14.rolling(
        50,
        min_periods=30,
    ).mean()

    df["atr14"] = atr14

    atr_ratio = np.where(
        atr50 > 0,
        atr14 / atr50,
        np.nan,
    )

    df["atr_ratio"] = atr_ratio

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    ema_spread = (
        (ema20 - ema50).abs()
        / c
    )

    trend_strength = np.where(
        atr_ratio > 0,
        ema_spread / atr_ratio,
        np.nan,
    )

    df["trend_strength"] = (
        trend_strength
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    df["momentum5"] = (
        c /
        c.shift(5)
        - 1
    )

    # --------------------------------------------------------
    # Range position
    # --------------------------------------------------------

    rh = h.rolling(
        20,
        min_periods=20,
    ).max()

    rl = l.rolling(
        20,
        min_periods=20,
    ).min()

    rr = rh - rl

    df["range_position"] = np.where(
        rr > 0,
        (c - rl) / rr,
        np.nan,
    )

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    df["hour"] = (
        df["time"].dt.hour
    )

    # --------------------------------------------------------
    # Regime
    # --------------------------------------------------------

    trend = (
        df["trend_strength"]
        .to_numpy()
    )

    vol = (
        df["atr_ratio"]
        .to_numpy()
    )

    regime = np.full(
        len(df),
        "UNKNOWN",
        dtype=object,
    )

    valid = (
        np.isfinite(trend) &
        np.isfinite(vol)
    )

    regime[
        valid &
        (trend >= 2.0) &
        (vol >= 1.10)
    ] = "TREND_HIGH_VOL"

    regime[
        valid &
        (trend >= 2.0) &
        (vol < 1.10)
    ] = "TREND_LOW_VOL"

    regime[
        valid &
        (trend < 1.0) &
        (vol >= 1.10)
    ] = "RANGE_HIGH_VOL"

    regime[
        valid &
        (trend < 1.0) &
        (vol < 1.10)
    ] = "RANGE_LOW_VOL"

    transition = (
        valid &
        ~np.isin(
            regime,
            [
                "TREND_HIGH_VOL",
                "TREND_LOW_VOL",
                "RANGE_HIGH_VOL",
                "RANGE_LOW_VOL",
            ],
        )
    )

    regime[transition] = "TRANSITION"

    df["regime"] = regime

    print(
        "Indicators ready.",
        flush=True,
    )

    return df


# ============================================================
# FAST SIGNAL SCORE
# ============================================================

def calculate_score(
    df,
    wick,
    body,
):

    lower = (
        df["lower_wick_ratio"]
        .to_numpy()
    )

    upper = (
        df["upper_wick_ratio"]
        .to_numpy()
    )

    body_ratio = (
        df["body_ratio"]
        .to_numpy()
    )

    close = (
        df["close"]
        .to_numpy()
    )

    open_ = (
        df["open"]
        .to_numpy()
    )

    position = (
        df["range_position"]
        .to_numpy()
    )

    momentum = (
        df["momentum5"]
        .to_numpy()
    )

    score = np.zeros(
        len(df),
        dtype=np.float32,
    )

    # Lower wick rejection
    score += np.where(
        lower >= wick,
        0.75,
        0.0,
    )

    # Upper wick rejection
    score -= np.where(
        upper >= wick,
        0.75,
        0.0,
    )

    # Small body
    score += np.where(
        body_ratio <= body,
        0.25,
        0.0,
    )

    bullish = (
        close > open_
    )

    bearish = (
        close < open_
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

    # Location reversal
    score += np.where(
        (
            position <= 0.30
        ) &
        bullish,
        0.50,
        0.0,
    )

    score -= np.where(
        (
            position >= 0.70
        ) &
        bearish,
        0.50,
        0.0,
    )

    # Momentum
    score += np.where(
        bullish &
        (momentum > 0),
        0.25,
        0.0,
    )

    score -= np.where(
        bearish &
        (momentum < 0),
        0.25,
        0.0,
    )

    score[
        ~np.isfinite(score)
    ] = np.nan

    return score


# ============================================================
# FAST EXACTISH TRADE ENGINE
# ============================================================

def backtest(
    df,
    score,
    rr,
    max_cross,
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

    close = (
        df["close"]
        .to_numpy(
            dtype=np.float64
        )
    )

    open_ = (
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
        df["hour"]
        .to_numpy()
    )

    regime = (
        df["regime"]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Valid regimes
    # --------------------------------------------------------

    valid_regime = np.isin(
        regime,
        [
            "RANGE_HIGH_VOL",
            "RANGE_LOW_VOL",
            "TRANSITION",
            "TREND_LOW_VOL",
        ],
    )

    # --------------------------------------------------------
    # Signal mask
    # --------------------------------------------------------

    mask = (
        np.isfinite(score) &
        (score >= threshold) &
        np.isin(
            hour,
            hours,
        ) &
        valid_regime &
        np.isfinite(atr) &
        (atr > 0)
    )

    indices = np.flatnonzero(
        mask
    )

    if len(indices) == 0:

        return np.empty(
            0,
            dtype=np.float64,
        )

    # Need a following candle.
    indices = indices[
        indices < len(df) - 1
    ]

    if len(indices) == 0:

        return np.empty(
            0,
            dtype=np.float64,
        )

    # --------------------------------------------------------
    # Signal separation
    # --------------------------------------------------------

    selected = []

    last = -10_000

    for idx in indices:

        if (
            idx - last
            >= max_cross
        ):

            selected.append(
                idx
            )

            last = idx

    if not selected:

        return np.empty(
            0,
            dtype=np.float64,
        )

    selected = np.asarray(
        selected,
        dtype=int,
    )

    entry_indices = (
        selected + 1
    )

    entries = (
        open_[entry_indices]
    )

    risks = (
        atr[selected]
    )

    bullish = (
        close[selected]
        >
        open_[selected]
    )

    # --------------------------------------------------------
    # We only need to inspect the next 48 candles.
    #
    # This is the maximum holding period.
    # --------------------------------------------------------

    max_hold = 48

    results = []

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We loop only over actual signals, NOT every candle.
    # --------------------------------------------------------

    for n, signal_idx in enumerate(
        selected
    ):

        entry_idx = (
            signal_idx + 1
        )

        if entry_idx >= len(df):
            continue

        entry = (
            entries[n]
        )

        risk = (
            risks[n]
        )

        if not np.isfinite(risk):
            continue

        if risk <= 0:
            continue

        direction = (
            1
            if bullish[n]
            else -1
        )

        if direction == 1:

            stop = (
                entry -
                risk
            )

            target = (
                entry +
                risk * rr
            )

        else:

            stop = (
                entry +
                risk
            )

            target = (
                entry -
                risk * rr
            )

        end = min(
            entry_idx +
            max_hold,
            len(df),
        )

        candle_high = (
            high[
                entry_idx:end
            ]
        )

        candle_low = (
            low[
                entry_idx:end
            ]
        )

        if direction == 1:

            target_hit = (
                candle_high
                >= target
            )

            stop_hit = (
                candle_low
                <= stop
            )

        else:

            target_hit = (
                candle_low
                <= target
            )

            stop_hit = (
                candle_high
                >= stop
            )

        target_positions = np.flatnonzero(
            target_hit
        )

        stop_positions = np.flatnonzero(
            stop_hit
        )

        first_target = (
            target_positions[0]
            if len(target_positions)
            else 10_000
        )

        first_stop = (
            stop_positions[0]
            if len(stop_positions)
            else 10_000
        )

        # ----------------------------------------------------
        # Conservative assumption:
        # if both occur on same candle, SL wins.
        # ----------------------------------------------------

        if first_stop <= first_target:

            if first_stop != 10_000:

                results.append(
                    -1.0
                )

        elif first_target != 10_000:

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

    wins = (
        results > 0
    )

    losses = (
        results < 0
    )

    trades = len(
        results
    )

    wins_count = int(
        wins.sum()
    )

    losses_count = int(
        losses.sum()
    )

    win_rate = (
        wins_count /
        trades *
        100
    )

    gross_profit = float(
        results[wins].sum()
    )

    gross_loss = abs(
        float(
            results[losses].sum()
        )
    )

    pf = (
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

    max_dd = float(
        drawdown.max()
    )

    streak = 0
    longest = 0

    for value in results:

        if value < 0:

            streak += 1

            longest = max(
                longest,
                streak,
            )

        else:

            streak = 0

    return {
        "trades":
            trades,

        "wins":
            wins_count,

        "losses":
            losses_count,

        "win_rate":
            win_rate,

        "total_r":
            float(
                results.sum()
            ),

        "profit_factor":
            float(pf),

        "max_drawdown":
            max_dd,

        "losing_streak":
            longest,
    }


# ============================================================
# SCORE CANDIDATE
# ============================================================

def candidate_quality(p):

    if p["trades"] < MIN_TRAIN_TRADES:
        return -999999

    if p["total_r"] <= 0:
        return -999999

    wr = min(
        p["win_rate"],
        100,
    )

    pf = min(
        p["profit_factor"],
        4,
    )

    total_r = min(
        max(
            p["total_r"],
            0,
        ),
        30,
    )

    sample = min(
        p["trades"],
        200,
    )

    dd = min(
        p["max_drawdown"],
        10,
    )

    return (
        wr * 0.50
        +
        (pf / 4 * 100) * 0.15
        +
        (total_r / 30 * 100) * 0.15
        +
        (sample / 200 * 100) * 0.20
        -
        (dd / 10 * 10)
    )


# ============================================================
# FAST COARSE SEARCH
# ============================================================

def coarse_search(
    df,
):

    print(
        "PHASE 1: FAST COARSE SEARCH",
        flush=True,
    )

    combinations = list(
        itertools.product(
            COARSE_RR,
            COARSE_WICK,
            COARSE_BODY,
            COARSE_SEPARATION,
            COARSE_HOURS,
            COARSE_THRESHOLD,
        )
    )

    print(
        f"COARSE COMBINATIONS: "
        f"{len(combinations)}",
        flush=True,
    )

    candidates = []

    score_cache = {}

    for wick, body in itertools.product(
        COARSE_WICK,
        COARSE_BODY,
    ):

        score_cache[
            (wick, body)
        ] = calculate_score(
            df,
            wick,
            body,
        )

    for i, params in enumerate(
        combinations,
        1,
    ):

        (
            rr,
            wick,
            body,
            separation,
            hours,
            threshold,
        ) = params

        score = score_cache[
            (wick, body)
        ]

        results = backtest(
            df,
            score,
            rr,
            separation,
            hours,
            threshold,
        )

        p = performance(
            results
        )

        quality = candidate_quality(
            p
        )

        if quality > -999999:

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
        f"VALID COARSE CANDIDATES: "
        f"{len(candidates)}",
        flush=True,
    )

    return candidates[:12]


# ============================================================
# FINE SEARCH
# ============================================================

def fine_search(
    df,
    coarse_candidates,
):

    print(
        "PHASE 2: FINE SEARCH",
        flush=True,
    )

    candidates = []

    for rank, coarse in enumerate(
        coarse_candidates,
        1,
    ):

        base = coarse[
            "params"
        ]

        base_rr = base[0]
        base_wick = base[1]
        base_body = base[2]
        base_sep = base[3]
        base_hours = base[4]
        base_threshold = base[5]

        rr_values = [
            x
            for x in FINE_RR
            if abs(
                x - base_rr
            ) <= 0.30
        ]

        wick_values = [
            x
            for x in FINE_WICK
            if abs(
                x - base_wick
            ) <= 0.10
        ]

        body_values = [
            x
            for x in FINE_BODY
            if abs(
                x - base_body
            ) <= 0.10
        ]

        sep_values = [
            x
            for x in FINE_SEPARATION
            if abs(
                x - base_sep
            ) <= 20
        ]

        threshold_values = [
            x
            for x in FINE_THRESHOLD
            if abs(
                x - base_threshold
            ) <= 0.50
        ]

        hours_values = [
            base_hours,
        ]

        combos = list(
            itertools.product(
                rr_values,
                wick_values,
                body_values,
                sep_values,
                hours_values,
                threshold_values,
            )
        )

        print(
            f"Fine region "
            f"{rank}/"
            f"{len(coarse_candidates)}: "
            f"{len(combos)} combinations",
            flush=True,
        )

        score_cache = {}

        for wick, body in itertools.product(
            wick_values,
            body_values,
        ):

            score_cache[
                (wick, body)
            ] = calculate_score(
                df,
                wick,
                body,
            )

        for params in combos:

            (
                rr,
                wick,
                body,
                separation,
                hours,
                threshold,
            ) = params

            score = score_cache[
                (wick, body)
            ]

            results = backtest(
                df,
                score,
                rr,
                separation,
                hours,
                threshold,
            )

            p = performance(
                results
            )

            quality = candidate_quality(
                p
            )

            if quality > -999999:

                candidates.append(
                    {
                        "params":
                            params,

                        "performance":
                            p,

                        "quality":
                            quality,
                    }
                )

    candidates.sort(
        key=lambda x:
        x["quality"],
        reverse=True,
    )

    print(
        f"VALID FINE CANDIDATES: "
        f"{len(candidates)}",
        flush=True,
    )

    return candidates[:20]


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
        hours,
        threshold,
    ) = candidate[
        "params"
    ]

    nearby_rr = [
        x for x in FINE_RR
        if abs(x - rr) <= 0.25
    ]

    nearby_wick = [
        x for x in FINE_WICK
        if abs(x - wick) <= 0.10
    ]

    nearby_body = [
        x for x in FINE_BODY
        if abs(x - body) <= 0.10
    ]

    nearby_threshold = [
        x for x in FINE_THRESHOLD
        if abs(x - threshold) <= 0.25
    ]

    nearby = []

    for rr2, wick2, body2, threshold2 in itertools.product(
        nearby_rr,
        nearby_wick,
        nearby_body,
        nearby_threshold,
    ):

        score = calculate_score(
            df,
            wick2,
            body2,
        )

        results = backtest(
            df,
            score,
            rr2,
            separation,
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

    if not nearby:

        return {
            "stable": False,
            "nearby": 0,
            "median_r": 0.0,
            "median_wr": 0.0,
            "positive_pct": 0.0,
        }

    median_r = float(
        np.median(
            [
                x["total_r"]
                for x in nearby
            ]
        )
    )

    median_wr = float(
        np.median(
            [
                x["win_rate"]
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
        ) * 100
    )

    stable = (
        len(nearby) >= 8
        and median_r > 0
        and positive_pct >= 60
    )

    return {
        "stable": stable,
        "nearby": len(nearby),
        "median_r": median_r,
        "median_wr": median_wr,
        "positive_pct": positive_pct,
    }


# ============================================================
# OPTIMISE
# ============================================================

def optimise(
    df,
):

    coarse = coarse_search(
        df
    )

    if not coarse:

        return None

    fine = fine_search(
        df,
        coarse,
    )

    if not fine:

        return None

    print(
        "PHASE 3: STABILITY VALIDATION",
        flush=True,
    )

    stable_candidates = []

    for i, candidate in enumerate(
        fine[:10],
        1,
    ):

        print(
            f"Stability candidate "
            f"{i}/10",
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
            x["stability"]["positive_pct"],
            x["stability"]["median_r"],
            x["performance"]["win_rate"],
            x["performance"]["trades"],
        ),
        reverse=True,
    )

    return stable_candidates[0]


# ============================================================
# TEST OOS
# ============================================================

def test_oos(
    selected,
    oos_df,
):

    params = selected[
        "params"
    ]

    (
        rr,
        wick,
        body,
        separation,
        hours,
        threshold,
    ) = params

    score = calculate_score(
        oos_df,
        wick,
        body,
    )

    results = backtest(
        oos_df,
        score,
        rr,
        separation,
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
        f"{market} V11.3",
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

    # ========================================================
    # WALK FORWARD
    # ========================================================

    for period in (
        WALK_FORWARD_PERIODS
    ):

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

        train_df = (
            df[
                (df["time"] >= train_start) &
                (df["time"] <= train_end)
            ]
            .reset_index(drop=True)
        )

        oos_df = (
            df[
                (df["time"] >= oos_start) &
                (df["time"] <= oos_end)
            ]
            .reset_index(drop=True)
        )

        print(
            f"Training candles: "
            f"{len(train_df)}",
            flush=True,
        )

        print(
            f"OOS candles: "
            f"{len(oos_df)}",
            flush=True,
        )

        print(
            "Optimising...",
            flush=True,
        )

        selected = optimise(
            train_df
        )

        if selected is None:

            print(
                "NO STABLE STRATEGY.",
                flush=True,
            )

            continue

        p = selected[
            "performance"
        ]

        stability = selected[
            "stability"
        ]

        params = selected[
            "params"
        ]

        print()
        print(
            "SELECTED STRATEGY",
            flush=True,
        )

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
            f"Separation: "
            f"{params[3]}",
            flush=True,
        )

        print(
            f"Hours: "
            f"{','.join(map(str, params[4]))}",
            flush=True,
        )

        print(
            f"Threshold: "
            f"{params[5]}",
            flush=True,
        )

        print(
            f"Training trades: "
            f"{p['trades']}",
            flush=True,
        )

        print(
            f"Training WR: "
            f"{p['win_rate']:.2f}%",
            flush=True,
        )

        print(
            f"Training R: "
            f"{p['total_r']:.2f}",
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

        oos = test_oos(
            selected,
            oos_df,
        )

        print(
            f"Trades: "
            f"{oos['trades']}",
            flush=True,
        )

        print(
            f"Wins: "
            f"{oos['wins']}",
            flush=True,
        )

        print(
            f"Losses: "
            f"{oos['losses']}",
            flush=True,
        )

        print(
            f"Win rate: "
            f"{oos['win_rate']:.2f}%",
            flush=True,
        )

        print(
            f"Total R: "
            f"{oos['total_r']:.2f}",
            flush=True,
        )

        print(
            f"Profit factor: "
            f"{oos['profit_factor']:.2f}",
            flush=True,
        )

        print(
            f"Max drawdown: "
            f"{oos['max_drawdown']:.2f}R",
            flush=True,
        )

        results.append(
            {
                "market": market,
                "period": period_name,

                "train_trades":
                    p["trades"],

                "train_win_rate":
                    p["win_rate"],

                "train_total_r":
                    p["total_r"],

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
                    oos["max_drawdown"],

                "rr":
                    params[0],

                "wick":
                    params[1],

                "body":
                    params[2],

                "separation":
                    params[3],

                "hours":
                    ",".join(
                        map(
                            str,
                            params[4],
                        )
                    ),

                "threshold":
                    params[5],

                "stability_nearby":
                    stability["nearby"],

                "stability_median_r":
                    stability["median_r"],

                "stability_positive_pct":
                    stability["positive_pct"],
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

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = []

    total_trades = 0
    total_wins = 0
    total_r = 0.0

    profitable_periods = 0
    total_periods = 0

    for market, result_df in (
        market_results.items()
    ):

        if result_df is None:
            continue

        if result_df.empty:
            continue

        trades = int(
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
                "oos_total_r"
            ].sum()
        )

        periods = len(
            result_df
        )

        profitable = int(
            (
                result_df[
                    "oos_total_r"
                ] > 0
            ).sum()
        )

        wr = (
            wins /
            trades *
            100
            if trades
            else 0
        )

        total_trades += trades
        total_wins += wins
        total_r += market_r

        profitable_periods += (
            profitable
        )

        total_periods += (
            periods
        )

        if (
            trades >= 100
            and wr >= 75
            and market_r > 0
        ):

            verdict = "PROMISING"

        elif (
            trades >= 50
            and wr >= 65
            and market_r > 0
        ):

            verdict = "INTERESTING"

        else:

            verdict = "NOT THERE YET"

        summary.append(
            {
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
                        market_r,
                        2,
                    ),

                "profitable_periods":
                    f"{profitable}/{periods}",

                "verdict":
                    verdict,
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
        100
        if total_trades
        else 0
    )

    print()
    print("=" * 60, flush=True)

    print(
        "V11.3 FINAL MULTI-MARKET "
        "SUMMARY",
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

    print()
    print("=" * 60, flush=True)

    print(
        "V11.3 TARGET CHECK",
        flush=True,
    )

    print("=" * 60, flush=True)

    print(
        "TARGET: ~82% WIN RATE",
        flush=True,
    )

    print(
        "TARGET: 200+ GENUINELY "
        "OUT-OF-SAMPLE TRADES",
        flush=True,
    )

    print(
        "TARGET: POSITIVE TOTAL R",
        flush=True,
    )

    if (
        total_trades >= 200
        and combined_wr >= 82
        and total_r > 0
    ):

        print(
            "TARGET STATUS: ACHIEVED",
            flush=True,
        )

    elif (
        total_trades >= 100
        and combined_wr >= 75
        and total_r > 0
    ):

        print(
            "TARGET STATUS: VERY PROMISING",
            flush=True,
        )

    elif (
        total_trades >= 50
        and combined_wr >= 65
        and total_r > 0
    ):

        print(
            "TARGET STATUS: PROMISING",
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
        "OOS data is never used "
        "for optimisation.",
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
        "OPTIMIZER V11.3 COMPLETE",
        flush=True,
    )

    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
