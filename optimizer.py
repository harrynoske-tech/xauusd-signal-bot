# ============================================================
# MULTI-MARKET STRATEGY OPTIMIZER V11.4
# ============================================================
#
# OBJECTIVE
# ------------------------------------------------------------
# Move the V11.3 baseline:
#
# 418 OOS trades
# 72.01% OOS WR
# +3.40R
#
# Toward:
#
# >=75% OOS WIN RATE
# >=200 GENUINE OOS TRADES
# POSITIVE TOTAL R
#
# ============================================================
#
# V11.4 CHANGES
# ------------------------------------------------------------
# 1. Keeps walk-forward testing
# 2. Keeps strict OOS separation
# 3. Keeps parameter stability
# 4. Adds rejection-quality scoring
# 5. Adds volatility-quality filtering
# 6. Adds EMA alignment filtering
# 7. Adds range-location filtering
# 8. Adds momentum confirmation
# 9. Optimises signal threshold
# 10. Optimises cooldown
# 11. Optimises TP/SL RR
# 12. Optimises market-specific settings
# 13. Does NOT require every period to produce a strategy
# 14. Reports every OOS period
# 15. Keeps 2026 as the latest OOS period
#
# NO LIVE TRADING
# ============================================================

import os
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# HEADER
# ============================================================

print("=" * 60, flush=True)
print("MULTI-MARKET STRATEGY OPTIMIZER V11.4", flush=True)
print("=" * 60, flush=True)

print(
    "REJECTION QUALITY FILTER: ENABLED",
    flush=True,
)

print(
    "EMA ALIGNMENT FILTER: ENABLED",
    flush=True,
)

print(
    "VOLATILITY FILTER: ENABLED",
    flush=True,
)

print(
    "RANGE LOCATION FILTER: ENABLED",
    flush=True,
)

print(
    "MOMENTUM CONFIRMATION: ENABLED",
    flush=True,
)

print(
    "WALK-FORWARD TESTING: ENABLED",
    flush=True,
)

print(
    "PARAMETER STABILITY: ENABLED",
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
# MARKETS
# ============================================================

MARKETS = {
    "XAUUSD": "data/XAUUSD_15m.csv",
    "EURUSD": "data/EURUSD_15m.csv",
}


RESULT_FILES = {
    "XAUUSD":
        "data/xauusd_optimizer_v11_4_results.csv",

    "EURUSD":
        "data/eurusd_optimizer_v11_4_results.csv",
}


SUMMARY_FILE = (
    "data/multi_market_optimizer_v11_4_summary.csv"
)


# ============================================================
# WALK-FORWARD PERIODS
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
# MINIMUMS
# ============================================================

MIN_TRAIN_TRADES = 35

MIN_STABILITY_TRADES = 25

MIN_STABILITY_NEIGHBOURS = 8

MIN_OOS_TRADES = 10


# ============================================================
# COARSE PARAMETER SEARCH
# ============================================================

COARSE_RR = [
    0.40,
    0.50,
    0.60,
    0.75,
]


COARSE_WICK = [
    0.20,
    0.25,
    0.30,
    0.35,
]


COARSE_BODY = [
    0.20,
    0.25,
    0.30,
]


COARSE_THRESHOLD = [
    -0.25,
    0.00,
    0.25,
    0.50,
]


COARSE_COOLDOWN = [
    10,
    20,
    30,
]


COARSE_VOL_MIN = [
    0.85,
    1.00,
    1.10,
]


COARSE_VOL_MAX = [
    1.50,
    1.75,
    2.00,
]


# ============================================================
# FINE PARAMETER SEARCH
# ============================================================

FINE_RR = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.75,
]


FINE_WICK = [
    0.18,
    0.20,
    0.22,
    0.25,
    0.28,
    0.30,
    0.35,
]


FINE_BODY = [
    0.18,
    0.20,
    0.22,
    0.25,
    0.28,
    0.30,
]


FINE_THRESHOLD = [
    -0.25,
    -0.10,
    0.00,
    0.10,
    0.25,
    0.40,
    0.50,
]


FINE_COOLDOWN = [
    8,
    10,
    12,
    15,
    20,
    25,
    30,
]


FINE_VOL_MIN = [
    0.80,
    0.90,
    1.00,
    1.10,
]


FINE_VOL_MAX = [
    1.40,
    1.60,
    1.75,
    2.00,
]


# ============================================================
# DATA LOADER
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
    # Normalise columns
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
    # Find timestamp
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

    time_column = None

    for candidate in candidates:

        if candidate in df.columns:

            time_column = candidate

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
            "Could not identify datetime column."
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

    replacements = {}

    for target, options in aliases.items():

        for option in options:

            if option in df.columns:

                replacements[option] = target

                break

    df = df.rename(
        columns=replacements
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

    df = df[
        (df["high"] >= df["low"])
        &
        (df["high"] >= df["open"])
        &
        (df["high"] >= df["close"])
        &
        (df["low"] <= df["open"])
        &
        (df["low"] <= df["close"])
    ]

    df = (
        df
        .sort_values("time")
        .drop_duplicates(
            subset="time"
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

    candle_range = (
        h - l
    )

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

    df["upper_wick_ratio"] = np.where(
        candle_range > 0,
        (
            h -
            np.maximum(o, c)
        ) /
        candle_range,
        np.nan,
    )

    df["lower_wick_ratio"] = np.where(
        candle_range > 0,
        (
            np.minimum(o, c) -
            l
        ) /
        candle_range,
        np.nan,
    )

    # --------------------------------------------------------
    # EMA structure
    # --------------------------------------------------------

    ema20 = c.ewm(
        span=20,
        adjust=False,
    ).mean()

    ema50 = c.ewm(
        span=50,
        adjust=False,
    ).mean()

    ema100 = c.ewm(
        span=100,
        adjust=False,
    ).mean()

    df["ema20"] = ema20

    df["ema50"] = ema50

    df["ema100"] = ema100

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = c.shift(1)

    true_range = pd.concat(
        [
            h - l,

            (
                h -
                previous_close
            ).abs(),

            (
                l -
                previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr14 = true_range.rolling(
        14,
        min_periods=14,
    ).mean()

    atr50 = atr14.rolling(
        50,
        min_periods=30,
    ).mean()

    df["atr14"] = atr14

    df["atr_ratio"] = np.where(
        atr50 > 0,
        atr14 / atr50,
        np.nan,
    )

    # --------------------------------------------------------
    # EMA spread / trend strength
    # --------------------------------------------------------

    df["trend_strength"] = np.where(
        (
            atr14 > 0
        ) &
        np.isfinite(atr14),

        (
            (
                ema20 -
                ema50
            ).abs()
            /
            c
        )
        /
        (
            atr14 /
            c
        ),

        np.nan,
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    df["momentum5"] = (
        c /
        c.shift(5)
        - 1
    )

    df["momentum10"] = (
        c /
        c.shift(10)
        - 1
    )

    # --------------------------------------------------------
    # Range location
    # --------------------------------------------------------

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

    df["range_position"] = np.where(
        rolling_range > 0,
        (
            c -
            rolling_low
        )
        /
        rolling_range,
        np.nan,
    )

    # --------------------------------------------------------
    # Candle direction
    # --------------------------------------------------------

    df["bullish"] = (
        c > o
    )

    df["bearish"] = (
        c < o
    )

    # --------------------------------------------------------
    # EMA direction
    # --------------------------------------------------------

    df["ema_bullish"] = (
        (
            ema20 > ema50
        )
        &
        (
            ema50 > ema100
        )
    )

    df["ema_bearish"] = (
        (
            ema20 < ema50
        )
        &
        (
            ema50 < ema100
        )
    )

    # --------------------------------------------------------
    # Regime
    # --------------------------------------------------------

    trend = (
        df["trend_strength"]
        .to_numpy()
    )

    volatility = (
        df["atr_ratio"]
        .to_numpy()
    )

    regime = np.full(
        len(df),
        "UNKNOWN",
        dtype=object,
    )

    valid = (
        np.isfinite(trend)
        &
        np.isfinite(volatility)
    )

    regime[
        valid
        &
        (trend >= 2.0)
        &
        (volatility >= 1.10)
    ] = "TREND_HIGH_VOL"

    regime[
        valid
        &
        (trend >= 2.0)
        &
        (volatility < 1.10)
    ] = "TREND_LOW_VOL"

    regime[
        valid
        &
        (trend < 1.0)
        &
        (volatility >= 1.10)
    ] = "RANGE_HIGH_VOL"

    regime[
        valid
        &
        (trend < 1.0)
        &
        (volatility < 1.10)
    ] = "RANGE_LOW_VOL"

    transition = (
        valid
        &
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

    regime[
        transition
    ] = "TRANSITION"

    df["regime"] = regime

    print(
        "Indicators ready.",
        flush=True,
    )

    return df


# ============================================================
# SIGNAL SCORE
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

    momentum5 = (
        df["momentum5"]
        .to_numpy()
    )

    momentum10 = (
        df["momentum10"]
        .to_numpy()
    )

    ema_bullish = (
        df["ema_bullish"]
        .to_numpy()
    )

    ema_bearish = (
        df["ema_bearish"]
        .to_numpy()
    )

    score = np.zeros(
        len(df),
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Rejection
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Body quality
    # --------------------------------------------------------

    score += np.where(
        body_ratio <= body,
        0.25,
        0.0,
    )

    # --------------------------------------------------------
    # Candle direction
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Range location
    #
    # Long reversal preferred near lower range.
    # Short reversal preferred near upper range.
    # --------------------------------------------------------

    score += np.where(
        (
            position <= 0.30
        )
        &
        bullish,
        0.50,
        0.0,
    )

    score -= np.where(
        (
            position >= 0.70
        )
        &
        bearish,
        0.50,
        0.0,
    )

    # --------------------------------------------------------
    # Momentum confirmation
    # --------------------------------------------------------

    score += np.where(
        bullish
        &
        (momentum5 > 0)
        &
        (momentum10 > 0),
        0.25,
        0.0,
    )

    score -= np.where(
        bearish
        &
        (momentum5 < 0)
        &
        (momentum10 < 0),
        0.25,
        0.0,
    )

    # --------------------------------------------------------
    # EMA structure
    # --------------------------------------------------------

    score += np.where(
        bullish &
        ema_bullish,
        0.25,
        0.0,
    )

    score -= np.where(
        bearish &
        ema_bearish,
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
    cooldown,
    hours,
    threshold,
    vol_min,
    vol_max,
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

    atr_ratio = (
        df["atr_ratio"]
        .to_numpy()
    )

    hour = (
        df["time"]
        .dt.hour
        .to_numpy()
    )

    regime = (
        df["regime"]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Regime selection
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
    # Volatility selection
    # --------------------------------------------------------

    valid_volatility = (
        np.isfinite(atr_ratio)
        &
        (
            atr_ratio >=
            vol_min
        )
        &
        (
            atr_ratio <=
            vol_max
        )
    )

    # --------------------------------------------------------
    # Initial signal mask
    # --------------------------------------------------------

    mask = (
        np.isfinite(score)
        &
        (
            score >=
            threshold
        )
        &
        np.isin(
            hour,
            hours,
        )
        &
        valid_regime
        &
        valid_volatility
        &
        np.isfinite(atr)
        &
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

    indices = indices[
        indices <
        len(df) - 1
    ]

    if len(indices) == 0:

        return np.empty(
            0,
            dtype=np.float64,
        )

    # --------------------------------------------------------
    # Cooldown
    # --------------------------------------------------------

    selected = []

    last_signal = -100000

    for index in indices:

        if (
            index -
            last_signal
            >= cooldown
        ):

            selected.append(
                index
            )

            last_signal = index

    if not selected:

        return np.empty(
            0,
            dtype=np.float64,
        )

    selected = np.asarray(
        selected,
        dtype=int,
    )

    # --------------------------------------------------------
    # Entries
    # --------------------------------------------------------

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

    results = []

    # --------------------------------------------------------
    # Exact sequential trade resolution.
    #
    # Only actual signals are looped over.
    # --------------------------------------------------------

    max_hold = 48

    for n in range(
        len(selected)
    ):

        entry_index = (
            entry_indices[n]
        )

        if (
            entry_index >=
            len(df)
        ):

            continue

        entry = (
            entries[n]
        )

        risk = (
            risks[n]
        )

        if (
            not np.isfinite(risk)
            or risk <= 0
        ):

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
            entry_index +
            max_hold,
            len(df),
        )

        highs = (
            high[
                entry_index:end
            ]
        )

        lows = (
            low[
                entry_index:end
            ]
        )

        if direction == 1:

            target_hit = (
                highs >= target
            )

            stop_hit = (
                lows <= stop
            )

        else:

            target_hit = (
                lows <= target
            )

            stop_hit = (
                highs >= stop
            )

        target_positions = (
            np.flatnonzero(
                target_hit
            )
        )

        stop_positions = (
            np.flatnonzero(
                stop_hit
            )
        )

        first_target = (
            target_positions[0]
            if len(
                target_positions
            )
            else 100000
        )

        first_stop = (
            stop_positions[0]
            if len(
                stop_positions
            )
            else 100000
        )

        # ----------------------------------------------------
        # Conservative:
        # same candle = SL first.
        # ----------------------------------------------------

        if (
            first_stop <=
            first_target
        ):

            if (
                first_stop
                != 100000
            ):

                results.append(
                    -1.0
                )

        elif (
            first_target
            != 100000
        ):

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

    losing_streak = 0

    current_streak = 0

    for result in results:

        if result < 0:

            current_streak += 1

            losing_streak = max(
                losing_streak,
                current_streak,
            )

        else:

            current_streak = 0

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
            float(
                profit_factor
            ),

        "max_drawdown":
            max_drawdown,

        "losing_streak":
            losing_streak,
    }


# ============================================================
# QUALITY SCORE
# ============================================================

def quality_score(p):

    if (
        p["trades"]
        < MIN_TRAIN_TRADES
    ):

        return -999999

    if (
        p["total_r"]
        <= 0
    ):

        return -999999

    # --------------------------------------------------------
    # We deliberately do NOT make training WR the whole score.
    # --------------------------------------------------------

    wr = min(
        p["win_rate"],
        100,
    )

    pf = min(
        p["profit_factor"],
        3.0,
    )

    total_r = min(
        max(
            p["total_r"],
            0,
        ),
        50,
    )

    sample = min(
        p["trades"],
        300,
    )

    dd = min(
        p["max_drawdown"],
        15,
    )

    return (
        wr * 0.40
        +
        (
            pf /
            3.0 *
            100
        ) * 0.20
        +
        (
            total_r /
            50 *
            100
        ) * 0.20
        +
        (
            sample /
            300 *
            100
        ) * 0.20
        -
        dd * 0.50
    )


# ============================================================
# COARSE SEARCH
# ============================================================

def coarse_search(df):

    print(
        "PHASE 1: COARSE SEARCH",
        flush=True,
    )

    combinations = list(
        itertools.product(
            COARSE_RR,
            COARSE_WICK,
            COARSE_BODY,
            COARSE_THRESHOLD,
            COARSE_COOLDOWN,
            COARSE_VOL_MIN,
            COARSE_VOL_MAX,
        )
    )

    # --------------------------------------------------------
    # Remove impossible volatility ranges.
    # --------------------------------------------------------

    combinations = [
        x for x in combinations
        if x[5] < x[6]
    ]

    print(
        f"COARSE COMBINATIONS: "
        f"{len(combinations)}",
        flush=True,
    )

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
                f"Coarse progress: "
                f"{i}/"
                f"{len(combinations)}",
                flush=True,
            )

        (
            rr,
            wick,
            body,
            threshold,
            cooldown,
            vol_min,
            vol_max,
        ) = params

        score = score_cache[
            (wick, body)
        ]

        results = backtest(
            df,
            score,
            rr,
            cooldown,
            (3, 4, 5, 12, 13),
            threshold,
            vol_min,
            vol_max,
        )

        p = performance(
            results
        )

        q = quality_score(
            p
        )

        if q > -999999:

            candidates.append(
                {
                    "params":
                        params,

                    "performance":
                        p,

                    "quality":
                        q,
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
    coarse,
):

    print(
        "PHASE 2: FINE SEARCH",
        flush=True,
    )

    candidates = []

    for rank, candidate in enumerate(
        coarse,
        1,
    ):

        (
            base_rr,
            base_wick,
            base_body,
            base_threshold,
            base_cooldown,
            base_vol_min,
            base_vol_max,
        ) = candidate[
            "params"
        ]

        rr_values = [
            x
            for x in FINE_RR
            if abs(
                x -
                base_rr
            ) <= 0.25
        ]

        wick_values = [
            x
            for x in FINE_WICK
            if abs(
                x -
                base_wick
            ) <= 0.10
        ]

        body_values = [
            x
            for x in FINE_BODY
            if abs(
                x -
                base_body
            ) <= 0.10
        ]

        threshold_values = [
            x
            for x in FINE_THRESHOLD
            if abs(
                x -
                base_threshold
            ) <= 0.50
        ]

        cooldown_values = [
            x
            for x in FINE_COOLDOWN
            if abs(
                x -
                base_cooldown
            ) <= 15
        ]

        vol_min_values = [
            x
            for x in FINE_VOL_MIN
            if abs(
                x -
                base_vol_min
            ) <= 0.30
        ]

        vol_max_values = [
            x
            for x in FINE_VOL_MAX
            if abs(
                x -
                base_vol_max
            ) <= 0.40
        ]

        combinations = list(
            itertools.product(
                rr_values,
                wick_values,
                body_values,
                threshold_values,
                cooldown_values,
                vol_min_values,
                vol_max_values,
            )
        )

        combinations = [
            x for x in combinations
            if x[5] < x[6]
        ]

        print(
            f"Fine region "
            f"{rank}/"
            f"{len(coarse)}: "
            f"{len(combinations)} "
            f"combinations",
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

        for params in combinations:

            (
                rr,
                wick,
                body,
                threshold,
                cooldown,
                vol_min,
                vol_max,
            ) = params

            score = score_cache[
                (wick, body)
            ]

            results = backtest(
                df,
                score,
                rr,
                cooldown,
                (3, 4, 5, 12, 13),
                threshold,
                vol_min,
                vol_max,
            )

            p = performance(
                results
            )

            q = quality_score(
                p
            )

            if q > -999999:

                candidates.append(
                    {
                        "params":
                            params,

                        "performance":
                            p,

                        "quality":
                            q,
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
        threshold,
        cooldown,
        vol_min,
        vol_max,
    ) = candidate[
        "params"
    ]

    nearby_rr = [
        x
        for x in FINE_RR
        if abs(
            x - rr
        ) <= 0.15
    ]

    nearby_wick = [
        x
        for x in FINE_WICK
        if abs(
            x - wick
        ) <= 0.07
    ]

    nearby_body = [
        x
        for x in FINE_BODY
        if abs(
            x - body
        ) <= 0.07
    ]

    nearby_threshold = [
        x
        for x in FINE_THRESHOLD
        if abs(
            x - threshold
        ) <= 0.25
    ]

    nearby_cooldown = [
        x
        for x in FINE_COOLDOWN
        if abs(
            x - cooldown
        ) <= 10
    ]

    nearby_vol_min = [
        x
        for x in FINE_VOL_MIN
        if abs(
            x - vol_min
        ) <= 0.20
    ]

    nearby_vol_max = [
        x
        for x in FINE_VOL_MAX
        if abs(
            x - vol_max
        ) <= 0.30
    ]

    nearby = []

    combinations = list(
        itertools.product(
            nearby_rr,
            nearby_wick,
            nearby_body,
            nearby_threshold,
            nearby_cooldown,
            nearby_vol_min,
            nearby_vol_max,
        )
    )

    # --------------------------------------------------------
    # Cap stability testing.
    #
    # We don't need thousands of almost-identical tests.
    # --------------------------------------------------------

    if len(combinations) > 150:

        combinations = combinations[
            :150
        ]

    for params in combinations:

        (
            rr2,
            wick2,
            body2,
            threshold2,
            cooldown2,
            vol_min2,
            vol_max2,
        ) = params

        if (
            vol_min2 >=
            vol_max2
        ):

            continue

        score = calculate_score(
            df,
            wick2,
            body2,
        )

        results = backtest(
            df,
            score,
            rr2,
            cooldown2,
            (3, 4, 5, 12, 13),
            threshold2,
            vol_min2,
            vol_max2,
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

    if len(nearby) < (
        MIN_STABILITY_NEIGHBOURS
    ):

        return {
            "stable":
                False,

            "nearby":
                len(nearby),

            "median_r":
                0.0,

            "median_wr":
                0.0,

            "positive_pct":
                0.0,
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
        )
        * 100
    )

    # --------------------------------------------------------
    # Stability is deliberately demanding.
    # --------------------------------------------------------

    stable = (
        median_r > 0
        and
        positive_pct >= 60
        and
        median_wr >= 60
    )

    return {
        "stable":
            stable,

        "nearby":
            len(nearby),

        "median_r":
            median_r,

        "median_wr":
            median_wr,

        "positive_pct":
            positive_pct,
    }


# ============================================================
# COMPLETE TRAINING OPTIMISATION
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
        "PHASE 3: PARAMETER STABILITY",
        flush=True,
    )

    stable = []

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

            stable.append(
                candidate
            )

    if not stable:

        print(
            "NO STABLE CANDIDATES.",
            flush=True,
        )

        return None

    # --------------------------------------------------------
    # Rank stable candidates by:
    #
    # 1. stability
    # 2. WR
    # 3. R
    # 4. sample size
    # --------------------------------------------------------

    stable.sort(
        key=lambda x: (
            x[
                "stability"
            ][
                "positive_pct"
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

            x[
                "performance"
            ][
                "trades"
            ],
        ),

        reverse=True,
    )

    return stable[0]


# ============================================================
# OOS TEST
# ============================================================

def test_oos(
    selected,
    df,
):

    (
        rr,
        wick,
        body,
        threshold,
        cooldown,
        vol_min,
        vol_max,
    ) = selected[
        "params"
    ]

    score = calculate_score(
        df,
        wick,
        body,
    )

    results = backtest(
        df,
        score,
        rr,
        cooldown,
        (3, 4, 5, 12, 13),
        threshold,
        vol_min,
        vol_max,
    )

    return performance(
        results
    )


# ============================================================
# MARKET RUNNER
# ============================================================

def run_market(
    market,
    path,
):

    print()
    print("=" * 60, flush=True)

    print(
        f"{market} V11.4",
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

    # --------------------------------------------------------
    # Run EVERY period.
    #
    # Important:
    # We record a failed optimisation period as well.
    # We never silently remove it from the summary.
    # --------------------------------------------------------

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
                (
                    df["time"]
                    >= train_start
                )
                &
                (
                    df["time"]
                    <= train_end
                )
            ]
            .reset_index(
                drop=True
            )
        )

        oos_df = (
            df[
                (
                    df["time"]
                    >= oos_start
                )
                &
                (
                    df["time"]
                    <= oos_end
                )
            ]
            .reset_index(
                drop=True
            )
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
            "Optimising training period...",
            flush=True,
        )

        selected = optimise(
            train_df
        )

        # ----------------------------------------------------
        # No strategy
        # ----------------------------------------------------

        if selected is None:

            print(
                "NO STABLE STRATEGY.",
                flush=True,
            )

            results.append(
                {
                    "market":
                        market,

                    "period":
                        period_name,

                    "train_trades":
                        0,

                    "train_win_rate":
                        0.0,

                    "train_total_r":
                        0.0,

                    "oos_trades":
                        0,

                    "oos_wins":
                        0,

                    "oos_losses":
                        0,

                    "oos_win_rate":
                        0.0,

                    "oos_total_r":
                        0.0,

                    "oos_profit_factor":
                        0.0,

                    "oos_drawdown":
                        0.0,

                    "stable":
                        False,

                    "rr":
                        np.nan,

                    "wick":
                        np.nan,

                    "body":
                        np.nan,

                    "threshold":
                        np.nan,

                    "cooldown":
                        np.nan,

                    "vol_min":
                        np.nan,

                    "vol_max":
                        np.nan,
                }
            )

            continue

        train_perf = selected[
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
            "-" * 60,
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
            f"Threshold: {params[3]}",
            flush=True,
        )

        print(
            f"Cooldown: {params[4]}",
            flush=True,
        )

        print(
            f"Volatility min: "
            f"{params[5]}",
            flush=True,
        )

        print(
            f"Volatility max: "
            f"{params[6]}",
            flush=True,
        )

        print(
            f"Training trades: "
            f"{train_perf['trades']}",
            flush=True,
        )

        print(
            f"Training WR: "
            f"{train_perf['win_rate']:.2f}%",
            flush=True,
        )

        print(
            f"Training R: "
            f"{train_perf['total_r']:.2f}",
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
        # OOS
        # ----------------------------------------------------

        print()
        print(
            "OUT-OF-SAMPLE RESULT",
            flush=True,
        )

        print(
            "-" * 60,
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

        print(
            f"Longest losing streak: "
            f"{oos['losing_streak']}",
            flush=True,
        )

        results.append(
            {
                "market":
                    market,

                "period":
                    period_name,

                "train_trades":
                    train_perf["trades"],

                "train_win_rate":
                    train_perf["win_rate"],

                "train_total_r":
                    train_perf["total_r"],

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

                "oos_losing_streak":
                    oos["losing_streak"],

                "stable":
                    True,

                "rr":
                    params[0],

                "wick":
                    params[1],

                "body":
                    params[2],

                "threshold":
                    params[3],

                "cooldown":
                    params[4],

                "vol_min":
                    params[5],

                "vol_max":
                    params[6],
            }
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Markets
    # --------------------------------------------------------

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

            market_results[
                market
            ] = pd.DataFrame()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    summary = []

    total_trades = 0

    total_wins = 0

    total_r = 0.0

    profitable_periods = 0

    completed_periods = 0

    stable_periods = 0

    for market, df in (
        market_results.items()
    ):

        if df is None:
            continue

        if df.empty:
            continue

        oos_trades = int(
            df[
                "oos_trades"
            ].sum()
        )

        wins = int(
            df[
                "oos_wins"
            ].sum()
        )

        market_r = float(
            df[
                "oos_total_r"
            ].sum()
        )

        periods = len(
            df
        )

        completed = int(
            (
                df[
                    "oos_trades"
                ] > 0
            ).sum()
        )

        profitable = int(
            (
                df[
                    "oos_total_r"
                ] > 0
            ).sum()
        )

        stable = int(
            df[
                "stable"
            ].sum()
        )

        win_rate = (
            wins /
            oos_trades *
            100
            if oos_trades
            else 0.0
        )

        total_trades += (
            oos_trades
        )

        total_wins += (
            wins
        )

        total_r += (
            market_r
        )

        profitable_periods += (
            profitable
        )

        completed_periods += (
            completed
        )

        stable_periods += (
            stable
        )

        # ----------------------------------------------------
        # Verdict
        # ----------------------------------------------------

        if (
            oos_trades >= 200
            and win_rate >= 75
            and market_r > 0
            and profitable >= 2
        ):

            verdict = (
                "TARGET RANGE"
            )

        elif (
            oos_trades >= 100
            and win_rate >= 72
            and market_r > 0
        ):

            verdict = (
                "VERY PROMISING"
            )

        elif (
            oos_trades >= 50
            and win_rate >= 70
            and market_r > 0
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

                "completed_oos_periods":
                    f"{completed}/{periods}",

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

    combined_win_rate = (
        total_wins /
        total_trades *
        100
        if total_trades
        else 0.0
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 60, flush=True)

    print(
        "V11.4 FINAL MULTI-MARKET SUMMARY",
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
        "COMBINED GENUINE "
        "OUT-OF-SAMPLE",
        flush=True,
    )

    print("=" * 60, flush=True)

    print(
        f"Trades: "
        f"{total_trades}",
        flush=True,
    )

    print(
        f"Wins: "
        f"{total_wins}",
        flush=True,
    )

    print(
        f"Win rate: "
        f"{combined_win_rate:.2f}%",
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
        f"{completed_periods}",
        flush=True,
    )

    print(
        f"Stable periods: "
        f"{stable_periods}/"
        f"{completed_periods}",
        flush=True,
    )

    # ========================================================
    # TARGET
    # ========================================================

    print()
    print("=" * 60, flush=True)

    print(
        "V11.4 TARGET CHECK",
        flush=True,
    )

    print("=" * 60, flush=True)

    print(
        "NEAR-TERM TARGET:",
        flush=True,
    )

    print(
        ">=75% WIN RATE",
        flush=True,
    )

    print(
        ">=200 GENUINELY "
        "OUT-OF-SAMPLE TRADES",
        flush=True,
    )

    print(
        "POSITIVE TOTAL R",
        flush=True,
    )

    if (
        total_trades >= 200
        and combined_win_rate >= 75
        and total_r > 0
    ):

        print(
            "TARGET STATUS: "
            "75%+ TARGET ACHIEVED",
            flush=True,
        )

    elif (
        total_trades >= 200
        and combined_win_rate >= 72
        and total_r > 0
    ):

        print(
            "TARGET STATUS: "
            "STRONG BASELINE",
            flush=True,
        )

    elif (
        total_trades >= 100
        and combined_win_rate >= 70
        and total_r > 0
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
        "Training data is used only "
        "to select parameters.",
        flush=True,
    )

    print(
        "OOS data is never used "
        "to optimise that period.",
        flush=True,
    )

    print(
        "Every walk-forward period "
        "is reported, including failures.",
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
        "OPTIMIZER V11.4 COMPLETE",
        flush=True,
    )

    print("=" * 60, flush=True)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()
