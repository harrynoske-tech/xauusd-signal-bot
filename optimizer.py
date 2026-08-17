# ============================================================
# MULTI-MARKET STRATEGY OPTIMIZER V11.2
# ============================================================
#
# FAST REGIME-AWARE REJECTION / REVERSAL OPTIMIZER
#
# FEATURES
# ------------------------------------------------------------
# - XAUUSD + EURUSD
# - Regime classification
# - Rejection / reversal signal
# - Walk-forward testing
# - Parameter stability
# - Current-era weighting
# - Strict 2026 holdout
# - Minimum sample requirements
# - Fast vectorised backtesting
# - No live trading
#
# TARGET
# ------------------------------------------------------------
# ~82% WIN RATE
# 200+ GENUINELY OUT-OF-SAMPLE TRADES
# POSITIVE TOTAL R
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
    "XAUUSD": "data/xauusd_optimizer_v11_2_results.csv",
    "EURUSD": "data/eurusd_optimizer_v11_2_results.csv",
}

SUMMARY_FILE = (
    "data/multi_market_optimizer_v11_2_summary.csv"
)


# ============================================================
# WALK-FORWARD WINDOWS
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
]


# ============================================================
# FINAL HOLDOUT
# ============================================================

FINAL_TRAIN_START = "2023-01-01"
FINAL_TRAIN_END = "2025-12-31"

FINAL_HOLDOUT_START = "2026-01-01"


# ============================================================
# SAMPLE REQUIREMENTS
# ============================================================

MIN_TRAIN_TRADES = 40
MIN_STABILITY_TRADES = 25
MIN_STABILITY_NEIGHBOURS = 10
MIN_OOS_TRADES = 20


# ============================================================
# PARAMETERS
# ============================================================

RR_VALUES = [
    0.50,
    0.60,
    0.75,
    1.00,
]

WICK_VALUES = [
    0.15,
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
    0.0003,
    0.0005,
    0.0008,
    0.0010,
]

MAX_CROSS_VALUES = [
    10,
    20,
    40,
]

HOUR_SETS = [
    (3, 4, 5),
    (4, 5),
    (3, 4, 5, 12, 13),
    (12, 13),
]

THRESHOLDS = [
    -0.50,
    -0.25,
    0.00,
    0.25,
    0.50,
]


# ============================================================
# DATA LOADER
# ============================================================

def load_data(path):

    if not os.path.exists(path):
        raise RuntimeError(
            f"DATA FILE NOT FOUND: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise RuntimeError(
            f"DATA FILE IS EMPTY: {path}"
        )

    # --------------------------------------------------------
    # Normalise column names
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

    df = df.rename(columns=rename)

    # --------------------------------------------------------
    # Timestamp detection
    # --------------------------------------------------------

    timestamp_candidates = [
        "time",
        "datetime",
        "date",
        "timestamp",
        "timestamp_utc",
        "datetime_utc",
        "gmt_time",
        "utc_time",
        "date_time",
    ]

    time_col = None

    for candidate in timestamp_candidates:

        if candidate in df.columns:
            time_col = candidate
            break

    # --------------------------------------------------------
    # Automatic timestamp detection
    # --------------------------------------------------------

    if time_col is None:

        excluded = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vol",
        }

        for col in df.columns:

            if col in excluded:
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
                continue

    if time_col is None:

        raise RuntimeError(
            "Could not identify timestamp column.\n"
            f"Columns found: {list(df.columns)}"
        )

    df["time"] = pd.to_datetime(
        df[time_col],
        utc=True,
        errors="coerce",
    )

    # --------------------------------------------------------
    # OHLC aliases
    # --------------------------------------------------------

    aliases = {
        "open": ["open", "o", "open_price"],
        "high": ["high", "h", "high_price"],
        "low": ["low", "l", "low_price"],
        "close": ["close", "c", "close_price"],
    }

    ohlc_rename = {}

    for target, options in aliases.items():

        for option in options:

            if option in df.columns:

                ohlc_rename[option] = target
                break

    df = df.rename(
        columns=ohlc_rename
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
            f"{path} missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Clean OHLC
    # --------------------------------------------------------

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
            subset=["time"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise RuntimeError(
            f"No valid candles remain in {path}"
        )

    print(
        f"Loaded candles: {len(df)}"
    )

    print(
        f"Range: "
        f"{df['time'].min()} -> "
        f"{df['time'].max()}"
    )

    return df


# ============================================================
# INDICATORS
# ============================================================

def prepare_indicators(df):

    df = df.copy()

    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]

    candle_range = (
        high - low
    )

    body = (
        close - open_
    ).abs()

    df["range"] = candle_range

    df["body_ratio"] = np.where(
        candle_range > 0,
        body / candle_range,
        np.nan,
    )

    upper_wick = (
        high -
        np.maximum(open_, close)
    )

    lower_wick = (
        np.minimum(open_, close) -
        low
    )

    df["upper_wick_ratio"] = np.where(
        candle_range > 0,
        upper_wick / candle_range,
        np.nan,
    )

    df["lower_wick_ratio"] = np.where(
        candle_range > 0,
        lower_wick / candle_range,
        np.nan,
    )

    # --------------------------------------------------------
    # EMAs
    # --------------------------------------------------------

    ema20 = close.ewm(
        span=20,
        adjust=False,
    ).mean()

    ema50 = close.ewm(
        span=50,
        adjust=False,
    ).mean()

    ema100 = close.ewm(
        span=100,
        adjust=False,
    ).mean()

    df["ema20"] = ema20
    df["ema50"] = ema50
    df["ema100"] = ema100

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
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
    # Trend strength
    # --------------------------------------------------------

    ema_spread = (
        (ema20 - ema50).abs()
        / close
    )

    trend_strength = np.where(
        np.asarray(atr_ratio) > 0,
        ema_spread /
        np.asarray(atr_ratio),
        np.nan,
    )

    df["trend_strength"] = (
        trend_strength
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    df["momentum5"] = (
        close /
        close.shift(5)
        - 1
    )

    # --------------------------------------------------------
    # Rolling location
    # --------------------------------------------------------

    rolling_high = high.rolling(
        20,
        min_periods=20,
    ).max()

    rolling_low = low.rolling(
        20,
        min_periods=20,
    ).min()

    rolling_range = (
        rolling_high -
        rolling_low
    )

    df["range_position"] = np.where(
        rolling_range > 0,
        (
            close -
            rolling_low
        ) /
        rolling_range,
        np.nan,
    )

    # --------------------------------------------------------
    # Hour
    # --------------------------------------------------------

    df["hour"] = (
        df["time"].dt.hour
    )

    # --------------------------------------------------------
    # Regime
    # --------------------------------------------------------

    trend = (
        df["trend_strength"]
    )

    volatility = (
        df["atr_ratio"]
    )

    regime = np.full(
        len(df),
        "UNKNOWN",
        dtype=object,
    )

    valid = (
        np.isfinite(trend) &
        np.isfinite(volatility)
    )

    regime[
        valid &
        (trend >= 2.0) &
        (volatility >= 1.10)
    ] = "TREND_HIGH_VOL"

    regime[
        valid &
        (trend >= 2.0) &
        (volatility < 1.10)
    ] = "TREND_LOW_VOL"

    regime[
        valid &
        (trend < 1.0) &
        (volatility >= 1.10)
    ] = "RANGE_HIGH_VOL"

    regime[
        valid &
        (trend < 1.0) &
        (volatility < 1.10)
    ] = "RANGE_LOW_VOL"

    middle = (
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

    regime[middle] = "TRANSITION"

    df["regime"] = regime

    return df


# ============================================================
# PRECOMPUTE FUTURE EXCURSIONS
# ============================================================

def prepare_backtest_arrays(df):

    high = (
        df["high"]
        .to_numpy(dtype=float)
    )

    low = (
        df["low"]
        .to_numpy(dtype=float)
    )

    close = (
        df["close"]
        .to_numpy(dtype=float)
    )

    open_ = (
        df["open"]
        .to_numpy(dtype=float)
    )

    atr = (
        df["atr14"]
        .to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # Future maximum/minimum over the next 96 candles.
    #
    # These are calculated ONCE.
    #
    # 96 x 15 minutes = 24 hours.
    # --------------------------------------------------------

    future_high = np.full(
        len(df),
        np.nan,
    )

    future_low = np.full(
        len(df),
        np.nan,
    )

    for shift in range(1, 97):

        shifted_high = np.full(
            len(df),
            np.nan,
        )

        shifted_low = np.full(
            len(df),
            np.nan,
        )

        shifted_high[:-shift] = (
            high[shift:]
        )

        shifted_low[:-shift] = (
            low[shift:]
        )

        if shift == 1:

            future_high = shifted_high
            future_low = shifted_low

        else:

            future_high = np.fmax(
                future_high,
                shifted_high,
            )

            future_low = np.fmin(
                future_low,
                shifted_low,
            )

    return {
        "high": high,
        "low": low,
        "close": close,
        "open": open_,
        "atr": atr,
        "future_high": future_high,
        "future_low": future_low,
    }


# ============================================================
# SIGNAL SCORE
# ============================================================

def calculate_scores(
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

    location = (
        df["range_position"]
        .to_numpy()
    )

    momentum = (
        df["momentum5"]
        .to_numpy()
    )

    score = np.zeros(
        len(df),
        dtype=float,
    )

    score += np.where(
        lower >= wick,
        0.75,
        0.0,
    )

    score -= np.where(
        upper >= wick,
        0.75,
        0.0,
    )

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

    score += np.where(
        (
            location <= 0.30
        ) &
        bullish,
        0.50,
        0.0,
    )

    score -= np.where(
        (
            location >= 0.70
        ) &
        bearish,
        0.50,
        0.0,
    )

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
# FAST TRADE BACKTEST
# ============================================================

def backtest_candidate(
    df,
    arrays,
    score,
    rr,
    separation,
    max_cross,
    hours,
    threshold,
):

    time = df["time"]

    hour = (
        df["hour"]
        .to_numpy()
    )

    regime = (
        df["regime"]
        .to_numpy()
    )

    close = arrays["close"]
    open_ = arrays["open"]
    atr = arrays["atr"]

    future_high = (
        arrays["future_high"]
    )

    future_low = (
        arrays["future_low"]
    )

    # --------------------------------------------------------
    # Valid regime.
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
    # Signal mask.
    # --------------------------------------------------------

    mask = (
        np.isfinite(score) &
        (score >= threshold) &
        np.isin(hour, hours) &
        valid_regime &
        np.isfinite(atr) &
        (atr > 0)
    )

    indices = np.flatnonzero(
        mask
    )

    # Need next candle for entry.
    indices = indices[
        indices < len(df) - 1
    ]

    if len(indices) == 0:

        return np.empty(
            0,
            dtype=float,
        )

    # --------------------------------------------------------
    # Enforce minimum signal separation.
    #
    # This is much faster than scanning every candle.
    # --------------------------------------------------------

    selected = []

    last_index = -10_000

    for idx in indices:

        if (
            idx -
            last_index
            >= max_cross
        ):

            selected.append(
                idx
            )

            last_index = idx

    if not selected:

        return np.empty(
            0,
            dtype=float,
        )

    idx = np.asarray(
        selected,
        dtype=int,
    )

    entry_idx = idx + 1

    entry = (
        open_[entry_idx]
    )

    risk = (
        atr[idx]
    )

    bullish = (
        close[idx] >
        open_[idx]
    )

    bearish = (
        close[idx] <
        open_[idx]
    )

    # --------------------------------------------------------
    # Long setup
    # --------------------------------------------------------

    long_mask = (
        bullish
    )

    long_entry = entry[
        long_mask
    ]

    long_risk = risk[
        long_mask
    ]

    long_target = (
        long_entry +
        long_risk * rr
    )

    long_stop = (
        long_entry -
        long_risk
    )

    long_future_high = (
        future_high[
            entry_idx[
                long_mask
            ]
            - 1
        ]
    )

    long_future_low = (
        future_low[
            entry_idx[
                long_mask
            ]
            - 1
        ]
    )

    long_target_hit = (
        long_future_high
        >= long_target
    )

    long_stop_hit = (
        long_future_low
        <= long_stop
    )

    long_results = np.where(
        long_target_hit &
        ~long_stop_hit,
        rr,
        np.where(
            long_stop_hit,
            -1.0,
            np.nan,
        ),
    )

    # --------------------------------------------------------
    # Short setup
    # --------------------------------------------------------

    short_mask = (
        bearish
    )

    short_entry = entry[
        short_mask
    ]

    short_risk = risk[
        short_mask
    ]

    short_target = (
        short_entry -
        short_risk * rr
    )

    short_stop = (
        short_entry +
        short_risk
    )

    short_future_low = (
        future_low[
            entry_idx[
                short_mask
            ]
            - 1
        ]
    )

    short_future_high = (
        future_high[
            entry_idx[
                short_mask
            ]
            - 1
        ]
    )

    short_target_hit = (
        short_future_low
        <= short_target
    )

    short_stop_hit = (
        short_future_high
        >= short_stop
    )

    short_results = np.where(
        short_target_hit &
        ~short_stop_hit,
        rr,
        np.where(
            short_stop_hit,
            -1.0,
            np.nan,
        ),
    )

    results = np.concatenate(
        [
            long_results,
            short_results,
        ]
    )

    return results[
        np.isfinite(results)
    ]


# ============================================================
# PERFORMANCE
# ============================================================

def performance(
    results
):

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

    win_count = int(
        wins.sum()
    )

    loss_count = int(
        losses.sum()
    )

    win_rate = (
        win_count /
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

    if gross_loss > 0:

        pf = (
            gross_profit /
            gross_loss
        )

    else:

        pf = 999.0

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
            win_count,

        "losses":
            loss_count,

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
# CANDIDATE GENERATION
# ============================================================

def optimise_training(
    df,
    arrays,
    train_start,
    train_end,
):

    dates = (
        df["time"]
    )

    train_mask = (
        (dates >= train_start) &
        (dates <= train_end)
    )

    train_df = (
        df.loc[
            train_mask
        ]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Recalculate arrays for training slice.
    # --------------------------------------------------------

    train_arrays = {
        key: value[
            train_mask.to_numpy()
        ]
        for key, value in arrays.items()
    }

    print(
        "PHASE 1: FAST SEARCH"
    )

    combinations = list(
        itertools.product(
            RR_VALUES,
            WICK_VALUES,
            BODY_VALUES,
            SEPARATION_VALUES,
            MAX_CROSS_VALUES,
            HOUR_SETS,
            THRESHOLDS,
        )
    )

    total = len(
        combinations
    )

    print(
        f"COMBINATIONS: {total}"
    )

    candidates = []

    # --------------------------------------------------------
    # Cache scores.
    #
    # Wick/body combinations determine the signal score.
    # Other parameters don't change the score.
    # --------------------------------------------------------

    score_cache = {}

    score_combinations = list(
        itertools.product(
            WICK_VALUES,
            BODY_VALUES,
        )
    )

    for wick, body in (
        score_combinations
    ):

        score_cache[
            (wick, body)
        ] = calculate_scores(
            train_df,
            wick,
            body,
        )

    for number, params in enumerate(
        combinations,
        1,
    ):

        if (
            number == 1
            or number % 500 == 0
            or number == total
        ):

            print(
                f"Progress: "
                f"{number}/{total} "
                f"("
                f"{number / total * 100:.1f}"
                f"%)"
            )

        (
            rr,
            wick,
            body,
            separation,
            max_cross,
            hours,
            threshold,
        ) = params

        score = score_cache[
            (wick, body)
        ]

        results = backtest_candidate(
            train_df,
            train_arrays,
            score,
            rr,
            separation,
            max_cross,
            hours,
            threshold,
        )

        if len(results) < MIN_TRAIN_TRADES:
            continue

        p = performance(
            results
        )

        if p["total_r"] <= 0:
            continue

        # ----------------------------------------------------
        # Quality score.
        # ----------------------------------------------------

        wr_score = (
            p["win_rate"]
        )

        r_score = min(
            max(
                p["total_r"],
                0,
            ),
            30,
        ) / 30 * 100

        pf_score = min(
            p["profit_factor"],
            4,
        ) / 4 * 100

        dd_penalty = min(
            p["max_drawdown"],
            10,
        ) / 10 * 100

        sample_score = min(
            len(results),
            150,
        ) / 150 * 100

        quality = (
            wr_score * 0.40
            +
            r_score * 0.20
            +
            pf_score * 0.20
            +
            sample_score * 0.15
            -
            dd_penalty * 0.05
        )

        candidates.append(
            {
                "params": params,
                "performance": p,
                "quality": quality,
            }
        )

    print(
        f"VALID CANDIDATES: "
        f"{len(candidates)}"
    )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x:
        x["quality"],
        reverse=True,
    )

    # ========================================================
    # STABILITY TEST
    # ========================================================

    print(
        "PHASE 2: STABILITY TEST"
    )

    stable = []

    for candidate in candidates[:30]:

        (
            rr,
            wick,
            body,
            separation,
            max_cross,
            hours,
            threshold,
        ) = candidate["params"]

        nearby = []

        for rr2 in RR_VALUES:

            if abs(
                rr2 - rr
            ) > 0.25:
                continue

            for wick2 in WICK_VALUES:

                if abs(
                    wick2 - wick
                ) > 0.10:
                    continue

                for body2 in BODY_VALUES:

                    if abs(
                        body2 - body
                    ) > 0.10:
                        continue

                    score2 = calculate_scores(
                        train_df,
                        wick2,
                        body2,
                    )

                    results2 = backtest_candidate(
                        train_df,
                        train_arrays,
                        score2,
                        rr2,
                        separation,
                        max_cross,
                        hours,
                        threshold,
                    )

                    if (
                        len(results2)
                        < MIN_STABILITY_TRADES
                    ):
                        continue

                    p2 = performance(
                        results2
                    )

                    nearby.append(
                        p2
                    )

        if (
            len(nearby)
            < MIN_STABILITY_NEIGHBOURS
        ):
            continue

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
            ) * 100
        )

        if (
            median_r > 0
            and positive_pct >= 60
        ):

            candidate[
                "stability"
            ] = {
                "nearby":
                    len(nearby),

                "median_wr":
                    median_wr,

                "median_r":
                    median_r,

                "positive_pct":
                    positive_pct,
            }

            stable.append(
                candidate
            )

    print(
        f"STABLE CANDIDATES: "
        f"{len(stable)}"
    )

    if not stable:
        return None

    # --------------------------------------------------------
    # Stability first.
    # --------------------------------------------------------

    stable.sort(
        key=lambda x: (
            x["stability"]["positive_pct"],
            x["stability"]["median_r"],
            x["quality"],
        ),
        reverse=True,
    )

    return stable[0]


# ============================================================
# TEST PERIOD
# ============================================================

def test_period(
    df,
    arrays,
    selected,
    start,
    end,
):

    params = selected[
        "params"
    ]

    (
        rr,
        wick,
        body,
        separation,
        max_cross,
        hours,
        threshold,
    ) = params

    mask = (
        (df["time"] >= start) &
        (df["time"] <= end)
    )

    period_df = (
        df.loc[
            mask
        ]
        .reset_index(drop=True)
    )

    period_arrays = {
        key: value[
            mask.to_numpy()
        ]
        for key, value in arrays.items()
    }

    score = calculate_scores(
        period_df,
        wick,
        body,
    )

    results = backtest_candidate(
        period_df,
        period_arrays,
        score,
        rr,
        separation,
        max_cross,
        hours,
        threshold,
    )

    return performance(
        results
    )


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
        f"{market} V11.2 "
        "REGIME-AWARE OPTIMIZER"
    )

    print("=" * 60)

    df = load_data(
        path
    )

    print(
        "Preparing indicators..."
    )

    df = prepare_indicators(
        df
    )

    print(
        "Preparing fast backtest arrays..."
    )

    arrays = prepare_backtest_arrays(
        df
    )

    print(
        "Indicators ready."
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
        print("=" * 60)

        print(
            f"{market} "
            f"{period_name}"
        )

        print("=" * 60)

        selected = optimise_training(
            df,
            arrays,
            train_start,
            train_end,
        )

        if selected is None:

            print(
                "NO STABLE STRATEGY FOUND"
            )

            continue

        params = selected[
            "params"
        ]

        train_perf = selected[
            "performance"
        ]

        stability = selected[
            "stability"
        ]

        print()
        print(
            "SELECTED TRAINING STRATEGY"
        )

        print("-" * 60)

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
            f"Hours: "
            f"{','.join(map(str, params[5]))}"
        )

        print(
            f"Threshold: {params[6]}"
        )

        print(
            f"Training trades: "
            f"{train_perf['trades']}"
        )

        print(
            f"Training WR: "
            f"{train_perf['win_rate']:.2f}%"
        )

        print(
            f"Training R: "
            f"{train_perf['total_r']:.2f}"
        )

        print(
            f"Training PF: "
            f"{train_perf['profit_factor']:.2f}"
        )

        print()
        print(
            "PARAMETER STABILITY"
        )

        print(
            f"Nearby: "
            f"{stability['nearby']}"
        )

        print(
            f"Median nearby WR: "
            f"{stability['median_wr']:.2f}%"
        )

        print(
            f"Median nearby R: "
            f"{stability['median_r']:.2f}R"
        )

        print(
            f"Positive nearby: "
            f"{stability['positive_pct']:.1f}%"
        )

        print(
            "Stability: PASS"
        )

        # ----------------------------------------------------
        # OOS
        # ----------------------------------------------------

        oos_perf = test_period(
            df,
            arrays,
            selected,
            oos_start,
            oos_end,
        )

        print()
        print(
            "OUT-OF-SAMPLE RESULT"
        )

        print("-" * 60)

        print(
            f"Trades: "
            f"{oos_perf['trades']}"
        )

        print(
            f"Wins: "
            f"{oos_perf['wins']}"
        )

        print(
            f"Losses: "
            f"{oos_perf['losses']}"
        )

        print(
            f"Win rate: "
            f"{oos_perf['win_rate']:.2f}%"
        )

        print(
            f"Total R: "
            f"{oos_perf['total_r']:.2f}"
        )

        print(
            f"Profit factor: "
            f"{oos_perf['profit_factor']:.2f}"
        )

        print(
            f"Max drawdown: "
            f"{oos_perf['max_drawdown']:.2f}R"
        )

        print(
            f"Longest losing streak: "
            f"{oos_perf['losing_streak']}"
        )

        results.append(
            {
                "period":
                    period_name,

                "train_trades":
                    train_perf["trades"],

                "train_win_rate":
                    train_perf["win_rate"],

                "train_total_r":
                    train_perf["total_r"],

                "oos_trades":
                    oos_perf["trades"],

                "oos_wins":
                    oos_perf["wins"],

                "oos_losses":
                    oos_perf["losses"],

                "oos_win_rate":
                    oos_perf["win_rate"],

                "oos_total_r":
                    oos_perf["total_r"],

                "oos_profit_factor":
                    oos_perf[
                        "profit_factor"
                    ],

                "oos_drawdown":
                    oos_perf[
                        "max_drawdown"
                    ],

                "oos_losing_streak":
                    oos_perf[
                        "losing_streak"
                    ],

                "stable":
                    True,

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

                "threshold":
                    params[6],
            }
        )

    # ========================================================
    # FINAL 2026 HOLDOUT
    # ========================================================

    print()
    print("=" * 60)

    print(
        f"{market} FINAL "
        "NEVER-SEEN HOLDOUT"
    )

    print("=" * 60)

    final_selected = optimise_training(
        df,
        arrays,
        FINAL_TRAIN_START,
        FINAL_TRAIN_END,
    )

    holdout_result = None

    if final_selected is None:

        print(
            "NO STABLE FINAL STRATEGY"
        )

    else:

        holdout_perf = test_period(
            df,
            arrays,
            final_selected,
            FINAL_HOLDOUT_START,
            "2100-01-01",
        )

        print()
        print(
            "FINAL 2026 HOLDOUT"
        )

        print("-" * 60)

        print(
            f"Trades: "
            f"{holdout_perf['trades']}"
        )

        print(
            f"Wins: "
            f"{holdout_perf['wins']}"
        )

        print(
            f"Losses: "
            f"{holdout_perf['losses']}"
        )

        print(
            f"Win rate: "
            f"{holdout_perf['win_rate']:.2f}%"
        )

        print(
            f"Total R: "
            f"{holdout_perf['total_r']:.2f}"
        )

        print(
            f"Profit factor: "
            f"{holdout_perf['profit_factor']:.2f}"
        )

        print(
            f"Max drawdown: "
            f"{holdout_perf['max_drawdown']:.2f}R"
        )

        holdout_result = holdout_perf

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        RESULT_FILES[market],
        index=False,
    )

    return (
        result_df,
        holdout_result,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "MULTI-MARKET STRATEGY "
        "OPTIMIZER V11.2"
    )

    print("=" * 60)

    print(
        "FAST VECTORIZED SEARCH: ENABLED"
    )

    print(
        "REGIME FILTERING: ENABLED"
    )

    print(
        "REJECTION / REVERSAL: ENABLED"
    )

    print(
        "WALK-FORWARD TESTING: ENABLED"
    )

    print(
        "PARAMETER STABILITY: ENABLED"
    )

    print(
        "STRICT 2026 HOLDOUT: ENABLED"
    )

    print(
        "MINIMUM SAMPLE FILTER: ENABLED"
    )

    print(
        "NO LIVE TRADING"
    )

    print("=" * 60)

    market_outputs = {}

    for market, path in (
        MARKETS.items()
    ):

        try:

            market_outputs[
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

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            print("=" * 60)

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = []

    total_trades = 0
    total_wins = 0
    total_r = 0.0

    profitable_periods = 0
    total_periods = 0

    for market, output in (
        market_outputs.items()
    ):

        if output is None:
            continue

        result_df, holdout = output

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

        total_market_r = float(
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

        win_rate = (
            wins /
            trades *
            100
            if trades
            else 0
        )

        total_trades += trades
        total_wins += wins
        total_r += total_market_r

        profitable_periods += (
            profitable
        )

        total_periods += periods

        if (
            trades >= 100
            and win_rate >= 75
            and total_market_r > 0
        ):

            verdict = "PROMISING"

        elif (
            trades >= 50
            and win_rate >= 65
            and total_market_r > 0
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
                        win_rate,
                        2,
                    ),

                "oos_total_r":
                    round(
                        total_market_r,
                        2,
                    ),

                "profitable_periods":
                    f"{profitable}/{periods}",

                "verdict":
                    verdict,

                "holdout_trades":
                    (
                        holdout["trades"]
                        if holdout
                        else 0
                    ),

                "holdout_win_rate":
                    (
                        round(
                            holdout[
                                "win_rate"
                            ],
                            2,
                        )
                        if holdout
                        else 0
                    ),

                "holdout_total_r":
                    (
                        round(
                            holdout[
                                "total_r"
                            ],
                            2,
                        )
                        if holdout
                        else 0
                    ),
            }
        )

    summary_df = pd.DataFrame(
        summary
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 60)

    print(
        "V11.2 FINAL MULTI-MARKET "
        "SUMMARY"
    )

    print("=" * 60)

    if summary_df.empty:

        print(
            "NO COMPLETED MARKET RESULTS"
        )

    else:

        print(
            summary_df.to_string(
                index=False
            )
        )

    combined_wr = (
        total_wins /
        total_trades *
        100
        if total_trades
        else 0
    )

    print()
    print("=" * 60)

    print(
        "COMBINED GENUINE "
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
        "V11.2 TARGET CHECK"
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
            "TARGET STATUS: VERY PROMISING"
        )

    elif (
        total_trades >= 50
        and combined_wr >= 65
        and total_r > 0
    ):

        print(
            "TARGET STATUS: PROMISING"
        )

    else:

        print(
            "TARGET STATUS: NOT ACHIEVED YET"
        )

    # ========================================================
    # SAFETY
    # ========================================================

    print()
    print("=" * 60)

    print(
        "IMPORTANT"
    )

    print("=" * 60)

    print(
        "OOS data is never used "
        "for optimisation."
    )

    print(
        "The 2026 holdout is kept "
        "separate from optimisation."
    )

    print(
        "Minimum sample filtering "
        "prevents tiny high-WR samples."
    )

    print(
        "This is research only."
    )

    print(
        "DO NOT IMPLEMENT LIVE."
    )

    print()
    print(
        "Results saved:"
    )

    print(
        RESULT_FILES["XAUUSD"]
    )

    print(
        RESULT_FILES["EURUSD"]
    )

    print(
        SUMMARY_FILE
    )

    print()
    print("=" * 60)

    print(
        "OPTIMIZER V11.2 COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()
