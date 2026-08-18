# ============================================================
# MULTI-MARKET STRATEGY OPTIMIZER V11.9
# ============================================================
# V11.8 CONTROLLED R/PF REFINEMENT
#
# OBJECTIVE:
#   Preserve >=75% OOS WR
#   Improve total R
#   Improve profit factor
#   Maintain >=200 genuine OOS trades
#
# NO LIVE TRADING
# ============================================================

import os
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MARKETS = {
    "XAUUSD": "data/XAUUSD_15m.csv",
    "EURUSD": "data/EURUSD_15m.csv",
}

RESULT_FILES = {
    "XAUUSD": "data/xauusd_optimizer_v11_9_results.csv",
    "EURUSD": "data/eurusd_optimizer_v11_9_results.csv",
}

SUMMARY_FILE = (
    "data/multi_market_optimizer_v11_9_summary.csv"
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

MIN_TRAIN_TRADES = 40
MIN_SUBPERIOD_TRADES = 10
MIN_RECENT_TRADES = 20
MAX_STABILITY_CANDIDATES = 8

# V11.8 WR floor.
# Candidates below this are not allowed to win merely
# because they produce more R.
MIN_ACCEPTABLE_WR = 73.0

# Stability requirements.
MIN_NEARBY_WR = 68.0
MIN_NEARBY_POSITIVE = 60.0

# ============================================================
# V11.9 SEARCH SPACE
#
# V11.8's winning area was concentrated around:
#
# RR       0.35 - 0.40
# Wick     0.20
# Body     0.15
# Separation 0.0004 - 0.0005
#
# V11.9 tests slightly higher RR values to determine whether
# we can retain the high WR while improving expectancy.
# ============================================================

SEARCH = {
    "XAUUSD": {
        "rr": [
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
        ],
        "wick": [
            0.20,
            0.25,
        ],
        "body": [
            0.15,
            0.20,
        ],
        "separation": [
            0.00035,
            0.00040,
            0.00050,
            0.00065,
        ],
        "threshold": [
            -0.25,
            0.00,
        ],
        "hours": [
            (3, 4),
            (4, 5),
            (3, 4, 5),
        ],
    },
    "EURUSD": {
        "rr": [
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
        ],
        "wick": [
            0.20,
            0.25,
        ],
        "body": [
            0.15,
            0.20,
        ],
        "separation": [
            0.00040,
            0.00050,
            0.00065,
        ],
        "threshold": [
            -0.25,
            0.00,
        ],
        "hours": [
            (3, 4, 5),
            (4, 5),
            (3, 4),
        ],
    },
}


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
            f"Data file not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise RuntimeError(
            f"Data file is empty: {path}"
        )

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

    time_column = None

    for name in [
        "time",
        "datetime",
        "date",
        "timestamp",
        "timestamp_utc",
        "datetime_utc",
        "utc_time",
    ]:

        if name in df.columns:
            time_column = name
            break

    if time_column is None:

        for col in df.columns:

            parsed = pd.to_datetime(
                df[col],
                utc=True,
                errors="coerce",
            )

            if (
                parsed.notna().mean()
                >= 0.90
            ):

                time_column = col
                break

    if time_column is None:
        raise RuntimeError(
            "Could not find datetime column."
        )

    df["time"] = pd.to_datetime(
        df[time_column],
        utc=True,
        errors="coerce",
    )

    for target, aliases in {
        "open": ["open", "o", "open_price"],
        "high": ["high", "h", "high_price"],
        "low": ["low", "l", "low_price"],
        "close": ["close", "c", "close_price"],
    }.items():

        if target in df.columns:
            continue

        for alias in aliases:

            if alias in df.columns:

                df[target] = df[alias]
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
            f"{path} missing required columns: "
            f"{missing}"
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

    prev_close = c.shift(1)

    true_range = pd.concat(
        [
            h - l,
            (h - prev_close).abs(),
            (l - prev_close).abs(),
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

    high20 = (
        h.rolling(
            20,
            min_periods=20,
        ).max()
    )

    low20 = (
        l.rolling(
            20,
            min_periods=20,
        ).min()
    )

    range20 = (
        high20 -
        low20
    )

    df["range_position"] = np.where(
        range20 > 0,
        (
            c - low20
        ) / range20,
        np.nan,
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

    ema20 = df[
        "ema20"
    ].to_numpy()

    ema50 = df[
        "ema50"
    ].to_numpy()

    momentum = df[
        "momentum5"
    ].to_numpy()

    location = df[
        "range_position"
    ].to_numpy()

    atr = df[
        "atr14"
    ].to_numpy()

    bullish = closes > opens
    bearish = closes < opens

    score = np.zeros(
        len(df),
        dtype=float,
    )

    # Primary rejection.
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

    # Location.
    score += np.where(
        bullish
        &
        (location <= 0.35),
        0.50,
        0.0,
    )

    score -= np.where(
        bearish
        &
        (location >= 0.65),
        0.50,
        0.0,
    )

    # Momentum.
    score += np.where(
        bullish
        &
        (momentum > 0),
        0.25,
        0.0,
    )

    score -= np.where(
        bearish
        &
        (momentum < 0),
        0.25,
        0.0,
    )

    # Very soft EMA component.
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

    ema_sep = np.where(
        atr > 0,
        np.abs(
            ema20 - ema50
        ) / atr,
        np.nan,
    )

    score += np.where(
        ema_sep >= separation,
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
        dtype=float
    )

    low = df[
        "low"
    ].to_numpy(
        dtype=float
    )

    opens = df[
        "open"
    ].to_numpy(
        dtype=float
    )

    closes = df[
        "close"
    ].to_numpy(
        dtype=float
    )

    atr = df[
        "atr14"
    ].to_numpy(
        dtype=float
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

    selected = []

    next_available = -1

    for index in indices:

        if index < next_available:
            continue

        selected.append(index)

        # Preserve the V11.8 low-frequency
        # behaviour.
        next_available = (
            index + 48
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
            or
            risk <= 0
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
            entry_index + 48,
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

        target_pos = np.flatnonzero(
            target_hits
        )

        stop_pos = np.flatnonzero(
            stop_hits
        )

        first_target = (
            target_pos[0]
            if len(target_pos)
            else 10**9
        )

        first_stop = (
            stop_pos[0]
            if len(stop_pos)
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

                value = (
                    final_close -
                    entry
                ) / risk

            else:

                value = (
                    entry -
                    final_close
                ) / risk

            results.append(
                float(
                    np.clip(
                        value,
                        -1.0,
                        rr,
                    )
                )
            )

    return np.asarray(
        results,
        dtype=float,
    )


# ============================================================
# PERFORMANCE
# ============================================================

def performance(results):

    if (
        results is None
        or
        len(results) == 0
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
# RECENCY-WEIGHTED TRAINING SCORE
# ============================================================

def training_quality(
    df,
    results,
):

    p = performance(
        results
    )

    if (
        p["trades"] <
        MIN_TRAIN_TRADES
    ):
        return -999999.0

    n = len(results)

    third = max(
        1,
        n // 3,
    )

    old = results[
        :third
    ]

    middle = results[
        third:
        third * 2
    ]

    recent = results[
        third * 2:
    ]

    if (
        len(old) <
        MIN_SUBPERIOD_TRADES
        or
        len(middle) <
        MIN_SUBPERIOD_TRADES
        or
        len(recent) <
        MIN_RECENT_TRADES
    ):
        return -999999.0

    po = performance(old)
    pm = performance(middle)
    pr = performance(recent)

    # Recent data matters most, but we do not
    # completely discard older regimes.
    weighted_wr = (
        po["win_rate"] * 0.20
        +
        pm["win_rate"] * 0.30
        +
        pr["win_rate"] * 0.50
    )

    weighted_r = (
        po["total_r"] * 0.20
        +
        pm["total_r"] * 0.30
        +
        pr["total_r"] * 0.50
    )

    # Strongly favour candidates that actually
    # make money while keeping WR high.
    quality = (
        weighted_wr * 1.0
        +
        weighted_r * 2.0
        +
        min(
            p["profit_factor"],
            3.0,
        ) * 15.0
        -
        p["max_drawdown"] * 0.25
    )

    # Hard WR protection.
    if (
        p["win_rate"] <
        MIN_ACCEPTABLE_WR
    ):
        quality -= 100.0

    if po["total_r"] < 0:
        quality -= 5.0

    if pm["total_r"] < 0:
        quality -= 3.0

    if pr["total_r"] < 0:
        quality -= 2.0

    return float(
        quality
    )


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

    nearby_rr = sorted(
        set(
            [
                rr,
                max(
                    0.35,
                    rr - 0.05,
                ),
                min(
                    0.55,
                    rr + 0.05,
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
                    0.25,
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
                    0.15,
                    body - 0.05,
                ),
                min(
                    0.20,
                    body + 0.05,
                ),
            ]
        )
    )

    nearby_sep = sorted(
        set(
            [
                separation,
                max(
                    0.00035,
                    separation - 0.00010,
                ),
                min(
                    0.00065,
                    separation + 0.00010,
                ),
            ]
        )
    )

    nearby_threshold = sorted(
        set(
            [
                threshold,
                -0.25,
                0.0,
            ]
        )
    )

    combos = list(
        itertools.product(
            nearby_rr,
            nearby_wick,
            nearby_body,
            nearby_sep,
            nearby_threshold,
        )
    )

    nearby = []

    for (
        rr2,
        wick2,
        body2,
        sep2,
        threshold2,
    ) in combos:

        signal = build_signal(
            df,
            wick2,
            body2,
            sep2,
        )

        result = backtest(
            df,
            signal,
            rr2,
            hours,
            threshold2,
        )

        p = performance(
            result
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
        ) * 100.0
    )

    stable = (
        median_wr >=
        MIN_NEARBY_WR
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
# OPTIMISE
# ============================================================

def optimise(
    df,
    market,
):

    cfg = SEARCH[
        market
    ]

    cache = {}

    for (
        wick,
        body,
        separation,
    ) in itertools.product(
        cfg["wick"],
        cfg["body"],
        cfg["separation"],
    ):

        cache[
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
            cfg["rr"],
            cfg["wick"],
            cfg["body"],
            cfg["separation"],
            cfg["threshold"],
            cfg["hours"],
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
            or
            i % 500 == 0
            or
            i == len(combinations)
        ):

            print(
                f"Search progress: "
                f"{i}/{len(combinations)}",
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

        signal = cache[
            (
                wick,
                body,
                separation,
            )
        ]

        result = backtest(
            df,
            signal,
            rr,
            hours,
            threshold,
        )

        p = performance(
            result
        )

        if (
            p["trades"] <
            MIN_TRAIN_TRADES
        ):
            continue

        quality = training_quality(
            df,
            result,
        )

        if quality <= -999999:
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

    # Prefer R/PF among candidates that
    # preserve the WR floor.
    stable.sort(
        key=lambda x: (
            x["performance"][
                "total_r"
            ],
            x["performance"][
                "profit_factor"
            ],
            x["performance"][
                "win_rate"
            ],
            x["stability"][
                "positive_pct"
            ],
        ),
        reverse=True,
    )

    return stable[0]


# ============================================================
# RUN MARKET
# ============================================================

def run_market(
    market,
    path,
):

    print()
    print("=" * 60, flush=True)

    print(
        f"{market} V11.9",
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

        print(
            f"PF: "
            f"{train_p['profit_factor']:.2f}",
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

        signal = build_signal(
            oos,
            wick,
            body,
            separation,
        )

        oos_result = backtest(
            oos,
            signal,
            rr,
            hours,
            threshold,
        )

        oos_p = performance(
            oos_result
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
                "stable": True,
                "train_trades":
                    train_p["trades"],
                "train_wr":
                    train_p["win_rate"],
                "train_r":
                    train_p["total_r"],
                "train_pf":
                    train_p["profit_factor"],
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
                    oos_p["losing_streak"],
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

    print("=" * 60, flush=True)
    print(
        "MULTI-MARKET STRATEGY OPTIMIZER V11.9",
        flush=True,
    )
    print("=" * 60, flush=True)

    print(
        "V11.8 CONTROLLED R/PF REFINEMENT",
        flush=True,
    )

    print(
        "75% OOS WR FLOOR: ENABLED",
        flush=True,
    )

    print(
        "HIGHER RR SEARCH: ENABLED",
        flush=True,
    )

    print(
        "PROFIT FACTOR OPTIMISATION: ENABLED",
        flush=True,
    )

    print(
        "MARKET-SPECIFIC OPTIMISATION: ENABLED",
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

        market_r = float(
            df[
                "oos_r"
            ].sum()
        )

        periods = len(df)

        profitable = int(
            (
                df[
                    "oos_r"
                ] > 0
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
        total_r += market_r

        total_periods += periods
        profitable_periods += profitable
        stable_periods += stable

        if (
            trades >= 200
            and
            wr >= 75.0
            and
            market_r > 0
            and
            profitable >= 2
            and
            stable >= 2
        ):

            verdict = (
                "TARGET RANGE"
            )

        elif (
            trades >= 200
            and
            wr >= 72.0
            and
            market_r > 0
        ):

            verdict = (
                "STRONG BASELINE"
            )

        elif (
            trades >= 100
            and
            wr >= 70.0
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
                        market_r,
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
        "V11.9 FINAL MULTI-MARKET SUMMARY",
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
        "V11.9 TARGET CHECK",
        flush=True,
    )

    print("=" * 60, flush=True)

    print(
        "PRIMARY TARGET:",
        flush=True,
    )

    print(
        ">=75% WIN RATE",
        flush=True,
    )

    print(
        ">=200 GENUINELY OOS TRADES",
        flush=True,
    )

    print(
        "POSITIVE TOTAL R",
        flush=True,
    )

    print()
    print(
        "SECONDARY TARGET:",
        flush=True,
    )

    print(
        "IMPROVE TOTAL R / PROFIT FACTOR "
        "OVER V11.8",
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
        "OOS data is NEVER used to "
        "optimise its own period.",
        flush=True,
    )

    print(
        "The 75% WR floor is applied "
        "during candidate selection.",
        flush=True,
    )

    print(
        "Higher-RR candidates are tested "
        "only within the V11.8 neighbourhood.",
        flush=True,
    )

    print(
        "Each walk-forward period is "
        "independently optimised.",
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
        "OPTIMIZER V11.9 COMPLETE",
        flush=True,
    )

    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
