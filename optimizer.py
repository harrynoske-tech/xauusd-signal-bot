import os
import itertools
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# MULTI-MARKET STRATEGY OPTIMIZER V11.1
# ============================================================
#
# REGIME-AWARE SIGNAL VALIDATION
# REJECTION / REVERSAL PRIMARY SIGNAL
# MARKET-SPECIFIC OPTIMISATION
# WALK-FORWARD TESTING
# PARAMETER STABILITY
# STRICT FINAL HOLDOUT
# MINIMUM SAMPLE FILTER
#
# TARGET:
# ~82% WIN RATE
# 200+ GENUINELY OUT-OF-SAMPLE TRADES
# POSITIVE TOTAL R
#
# NO LIVE TRADING
# ============================================================


# ============================================================
# DATA FILES
# ============================================================

MARKETS = {
    "XAUUSD": "data/XAUUSD_15m.csv",
    "EURUSD": "data/EURUSD_15m.csv",
}


RESULT_FILES = {
    "XAUUSD":
        "data/xauusd_optimizer_v11_results.csv",

    "EURUSD":
        "data/eurusd_optimizer_v11_results.csv",
}


SUMMARY_FILE = (
    "data/multi_market_optimizer_v11_summary.csv"
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
#
# This period is NEVER used for optimisation.
#
# The latest available data is therefore kept genuinely
# unseen until the very end.
# ============================================================

FINAL_TRAIN_START = "2023-01-01"
FINAL_TRAIN_END = "2025-12-31"

FINAL_HOLDOUT_START = "2026-01-01"


# ============================================================
# MINIMUM SAMPLE REQUIREMENTS
# ============================================================

MIN_TRAIN_TRADES = 50

MIN_STABILITY_TRADES = 30

MIN_STABILITY_NEIGHBOURS = 15

MIN_OOS_TRADES = 20


# ============================================================
# PARAMETER SEARCH
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

    print(
        f"Loading: {path}"
    )

    df = pd.read_csv(path)

    if df.empty:

        raise RuntimeError(
            f"DATA FILE IS EMPTY: {path}"
        )

    # --------------------------------------------------------
    # Normalise all column names.
    # --------------------------------------------------------

    original_columns = list(
        df.columns
    )

    cleaned_columns = {}

    for column in df.columns:

        clean = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

        cleaned_columns[column] = clean

    df = df.rename(
        columns=cleaned_columns
    )

    # --------------------------------------------------------
    # Find timestamp column.
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

    time_column = None

    for candidate in timestamp_candidates:

        if candidate in df.columns:

            time_column = candidate
            break

    # --------------------------------------------------------
    # If no obvious timestamp name exists, automatically
    # search every non-OHLC column for something that parses
    # as datetime.
    # --------------------------------------------------------

    if time_column is None:

        excluded = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vol",
        }

        for column in df.columns:

            if column in excluded:
                continue

            try:

                parsed = pd.to_datetime(
                    df[column],
                    utc=True,
                    errors="coerce",
                )

                valid_ratio = (
                    parsed.notna().mean()
                )

                if valid_ratio >= 0.90:

                    time_column = column
                    break

            except Exception:

                continue

    if time_column is None:

        raise RuntimeError(
            "Could not identify timestamp column.\n"
            f"Columns found: {original_columns}"
        )

    # --------------------------------------------------------
    # Parse timestamps.
    # --------------------------------------------------------

    df[time_column] = pd.to_datetime(
        df[time_column],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=[time_column]
    )

    df = df.rename(
        columns={
            time_column: "time"
        }
    )

    # --------------------------------------------------------
    # Normalise OHLC column names.
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

    rename = {}

    for target, candidates in aliases.items():

        for candidate in candidates:

            if candidate in df.columns:

                rename[candidate] = target
                break

    df = df.rename(
        columns=rename
    )

    # --------------------------------------------------------
    # Validate OHLC.
    # --------------------------------------------------------

    required = [
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
            f"{path} is missing OHLC columns: "
            f"{missing}\n"
            f"Columns found: "
            f"{list(df.columns)}"
        )

    # --------------------------------------------------------
    # Convert OHLC to numeric.
    # --------------------------------------------------------

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=required
    )

    # --------------------------------------------------------
    # Remove bad candles.
    # --------------------------------------------------------

    df = df[
        (df["high"] >= df["low"]) &
        (df["high"] >= df["open"]) &
        (df["high"] >= df["close"]) &
        (df["low"] <= df["open"]) &
        (df["low"] <= df["close"])
    ]

    # --------------------------------------------------------
    # Sort and remove duplicate timestamps.
    # --------------------------------------------------------

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
            f"No valid OHLC candles remain in {path}"
        )

    print(
        f"Loaded candles: {len(df)}"
    )

    print(
        f"Columns: {list(df.columns)}"
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

    # --------------------------------------------------------
    # Candle structure
    # --------------------------------------------------------

    df["candle_range"] = (
        df["high"] -
        df["low"]
    )

    df["body"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["body_ratio"] = np.where(
        df["candle_range"] > 0,
        df["body"] /
        df["candle_range"],
        np.nan,
    )

    df["upper_wick"] = (
        df["high"] -
        df[
            ["open", "close"]
        ].max(axis=1)
    )

    df["lower_wick"] = (
        df[
            ["open", "close"]
        ].min(axis=1) -
        df["low"]
    )

    df["upper_wick_ratio"] = np.where(
        df["candle_range"] > 0,
        df["upper_wick"] /
        df["candle_range"],
        np.nan,
    )

    df["lower_wick_ratio"] = np.where(
        df["candle_range"] > 0,
        df["lower_wick"] /
        df["candle_range"],
        np.nan,
    )

    # --------------------------------------------------------
    # EMAs
    # --------------------------------------------------------

    df["ema20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    df["ema50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    df["ema100"] = (
        df["close"]
        .ewm(
            span=100,
            adjust=False,
        )
        .mean()
    )

    df["ema200"] = (
        df["close"]
        .ewm(
            span=200,
            adjust=False,
        )
        .mean()
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = (
        df["close"].shift(1)
    )

    tr1 = (
        df["high"] -
        df["low"]
    )

    tr2 = (
        df["high"] -
        previous_close
    ).abs()

    tr3 = (
        df["low"] -
        previous_close
    ).abs()

    df["true_range"] = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    df["atr14"] = (
        df["true_range"]
        .rolling(
            14,
            min_periods=14,
        )
        .mean()
    )

    df["atr50"] = (
        df["atr14"]
        .rolling(
            50,
            min_periods=30,
        )
        .mean()
    )

    df["atr_ratio"] = np.where(
        df["atr50"] > 0,
        df["atr14"] /
        df["atr50"],
        np.nan,
    )

    # --------------------------------------------------------
    # EMA structure
    # --------------------------------------------------------

    df["ema20_slope"] = (
        df["ema20"] -
        df["ema20"].shift(4)
    ) / df["close"]

    df["ema50_slope"] = (
        df["ema50"] -
        df["ema50"].shift(8)
    ) / df["close"]

    df["ema_spread"] = (
        (
            df["ema20"] -
            df["ema50"]
        ).abs()
        /
        df["close"]
    )

    df["trend_strength"] = np.where(
        df["atr_ratio"] > 0,
        df["ema_spread"] /
        df["atr_ratio"],
        np.nan,
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    df["momentum5"] = (
        df["close"] /
        df["close"].shift(5)
        - 1
    )

    df["momentum10"] = (
        df["close"] /
        df["close"].shift(10)
        - 1
    )

    # --------------------------------------------------------
    # 20-bar location
    # --------------------------------------------------------

    df["rolling_high20"] = (
        df["high"]
        .rolling(
            20,
            min_periods=20,
        )
        .max()
    )

    df["rolling_low20"] = (
        df["low"]
        .rolling(
            20,
            min_periods=20,
        )
        .min()
    )

    rolling_range = (
        df["rolling_high20"] -
        df["rolling_low20"]
    )

    df["range_position"] = np.where(
        rolling_range > 0,
        (
            df["close"] -
            df["rolling_low20"]
        ) /
        rolling_range,
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

    df["regime"] = "UNKNOWN"

    valid = (
        df["trend_strength"].notna() &
        df["atr_ratio"].notna()
    )

    df.loc[
        valid &
        (df["trend_strength"] >= 2.0) &
        (df["atr_ratio"] >= 1.10),
        "regime"
    ] = "TREND_HIGH_VOL"

    df.loc[
        valid &
        (df["trend_strength"] >= 2.0) &
        (df["atr_ratio"] < 1.10),
        "regime"
    ] = "TREND_LOW_VOL"

    df.loc[
        valid &
        (df["trend_strength"] < 1.0) &
        (df["atr_ratio"] >= 1.10),
        "regime"
    ] = "RANGE_HIGH_VOL"

    df.loc[
        valid &
        (df["trend_strength"] < 1.0) &
        (df["atr_ratio"] < 1.10),
        "regime"
    ] = "RANGE_LOW_VOL"

    df.loc[
        valid &
        ~df["regime"].isin(
            [
                "TREND_HIGH_VOL",
                "TREND_LOW_VOL",
                "RANGE_HIGH_VOL",
                "RANGE_LOW_VOL",
            ]
        ),
        "regime"
    ] = "TRANSITION"

    df = df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return df


# ============================================================
# SIGNAL SCORE
# ============================================================

def calculate_signal_score(
    row,
    wick_threshold,
    body_threshold,
):

    score = 0.0

    if pd.isna(
        row["body_ratio"]
    ):
        return np.nan

    # --------------------------------------------------------
    # Wick rejection
    # --------------------------------------------------------

    upper = (
        row["upper_wick_ratio"]
    )

    lower = (
        row["lower_wick_ratio"]
    )

    if lower >= wick_threshold:

        score += 0.75

    if upper >= wick_threshold:

        score -= 0.75

    # --------------------------------------------------------
    # Candle body
    # --------------------------------------------------------

    if (
        row["body_ratio"]
        <= body_threshold
    ):

        score += 0.25

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    if row["close"] > row["open"]:

        score += 0.25

    elif row["close"] < row["open"]:

        score -= 0.25

    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    position = (
        row["range_position"]
    )

    if not pd.isna(position):

        if (
            position <= 0.30
            and row["close"] >
            row["open"]
        ):

            score += 0.50

        if (
            position >= 0.70
            and row["close"] <
            row["open"]
        ):

            score -= 0.50

    # --------------------------------------------------------
    # Momentum confirmation
    # --------------------------------------------------------

    if (
        not pd.isna(
            row["momentum5"]
        )
    ):

        if (
            row["close"] >
            row["open"]
            and row["momentum5"] > 0
        ):

            score += 0.25

        elif (
            row["close"] <
            row["open"]
            and row["momentum5"] < 0
        ):

            score -= 0.25

    return score


# ============================================================
# TRADE ENGINE
# ============================================================

def generate_trades(
    df,
    rr,
    wick,
    body,
    separation,
    max_cross,
    hours,
    threshold,
):

    if df.empty:

        return pd.DataFrame(
            columns=[
                "time",
                "direction",
                "score",
                "regime",
                "result_r",
            ]
        )

    work = df.copy()

    # --------------------------------------------------------
    # Calculate score without using future information.
    # --------------------------------------------------------

    work["signal_score"] = work.apply(
        lambda row:
        calculate_signal_score(
            row,
            wick,
            body,
        ),
        axis=1,
    )

    trades = []

    last_signal_index = -999999

    # --------------------------------------------------------
    # Signal loop
    # --------------------------------------------------------

    for i in range(
        len(work) - 1
    ):

        row = work.iloc[i]

        if pd.isna(
            row["signal_score"]
        ):
            continue

        if (
            row["hour"]
            not in hours
        ):
            continue

        if (
            row["signal_score"]
            < threshold
        ):
            continue

        # ----------------------------------------------------
        # Minimum separation between signals.
        # ----------------------------------------------------

        if (
            i -
            last_signal_index
            < max_cross
        ):
            continue

        # ----------------------------------------------------
        # Only regimes in which rejection/reversal is
        # allowed.
        # ----------------------------------------------------

        if row["regime"] not in [

            "RANGE_HIGH_VOL",
            "RANGE_LOW_VOL",
            "TRANSITION",
            "TREND_LOW_VOL",

        ]:

            continue

        atr = row["atr14"]

        if pd.isna(atr):
            continue

        if atr <= 0:
            continue

        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        if (
            row["close"] >
            row["open"]
        ):

            direction = 1

        elif (
            row["close"] <
            row["open"]
        ):

            direction = -1

        else:

            continue

        # ----------------------------------------------------
        # Entry on NEXT candle.
        #
        # This prevents look-ahead bias.
        # ----------------------------------------------------

        entry_index = i + 1

        entry = (
            work.iloc[
                entry_index
            ]["open"]
        )

        risk = atr

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

        result = None
        exit_index = None

        # ----------------------------------------------------
        # Maximum holding period.
        # ----------------------------------------------------

        max_exit = min(
            entry_index + 96,
            len(work) - 1,
        )

        for j in range(
            entry_index,
            max_exit + 1,
        ):

            candle = work.iloc[j]

            if direction == 1:

                stop_hit = (
                    candle["low"]
                    <= stop
                )

                target_hit = (
                    candle["high"]
                    >= target
                )

            else:

                stop_hit = (
                    candle["high"]
                    >= stop
                )

                target_hit = (
                    candle["low"]
                    <= target
                )

            # ------------------------------------------------
            # Conservative assumption if both happen inside
            # one candle.
            # ------------------------------------------------

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

        if result is None:

            continue

        trades.append(
            {
                "time":
                    row["time"],

                "direction":
                    direction,

                "score":
                    row["signal_score"],

                "regime":
                    row["regime"],

                "result_r":
                    float(result),

                "signal_index":
                    i,

                "exit_index":
                    exit_index,
            }
        )

        last_signal_index = i

    if not trades:

        return pd.DataFrame(
            columns=[
                "time",
                "direction",
                "score",
                "regime",
                "result_r",
                "signal_index",
                "exit_index",
            ]
        )

    return pd.DataFrame(
        trades
    )


# ============================================================
# PERFORMANCE
# ============================================================

def calculate_performance(
    trades
):

    if (
        trades is None
        or len(trades) == 0
    ):

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "longest_losing_streak": 0,
        }

    values = (
        trades["result_r"]
        .astype(float)
        .to_numpy()
    )

    wins = (
        values > 0
    )

    losses = (
        values < 0
    )

    trade_count = len(
        values
    )

    win_count = int(
        wins.sum()
    )

    loss_count = int(
        losses.sum()
    )

    win_rate = (
        win_count /
        trade_count *
        100
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

        profit_factor = (
            gross_profit /
            gross_loss
        )

    else:

        profit_factor = 999.0

    equity = np.cumsum(
        values
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

    current_streak = 0
    longest_streak = 0

    for value in values:

        if value < 0:

            current_streak += 1

            longest_streak = max(
                longest_streak,
                current_streak,
            )

        else:

            current_streak = 0

    return {
        "trades":
            trade_count,

        "wins":
            win_count,

        "losses":
            loss_count,

        "win_rate":
            win_rate,

        "total_r":
            float(
                values.sum()
            ),

        "profit_factor":
            float(
                profit_factor
            ),

        "max_drawdown":
            max_drawdown,

        "longest_losing_streak":
            longest_streak,
    }


# ============================================================
# STABILITY TEST
# ============================================================

def stability_test(
    df,
    selected_params,
    training_start,
    training_end,
):

    (
        rr,
        wick,
        body,
        separation,
        max_cross,
        hours,
        threshold,
    ) = selected_params

    nearby = []

    # --------------------------------------------------------
    # Only vary parameters locally around the selected setup.
    # --------------------------------------------------------

    nearby_rr = [
        x
        for x in RR_VALUES
        if abs(x - rr) <= 0.25
    ]

    nearby_wick = [
        x
        for x in WICK_VALUES
        if abs(x - wick) <= 0.10
    ]

    nearby_body = [
        x
        for x in BODY_VALUES
        if abs(x - body) <= 0.10
    ]

    nearby_threshold = [
        x
        for x in THRESHOLDS
        if abs(x - threshold) <= 0.25
    ]

    training_df = df[
        (df["time"] >= training_start) &
        (df["time"] <= training_end)
    ].copy()

    for values in itertools.product(
        nearby_rr,
        nearby_wick,
        nearby_body,
        [separation],
        [max_cross],
        [hours],
        nearby_threshold,
    ):

        trades = generate_trades(
            training_df,
            *values,
        )

        if (
            len(trades)
            < MIN_STABILITY_TRADES
        ):

            continue

        p = calculate_performance(
            trades
        )

        nearby.append(
            p
        )

    if (
        len(nearby)
        < MIN_STABILITY_NEIGHBOURS
    ):

        return {
            "stable":
                False,

            "nearby":
                len(nearby),

            "median_wr":
                0.0,

            "median_r":
                0.0,

            "positive_pct":
                0.0,
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
        * 100
    )

    # --------------------------------------------------------
    # Stability is deliberately strict.
    # --------------------------------------------------------

    stable = (
        median_r > 0
        and positive_pct >= 60.0
    )

    return {
        "stable":
            stable,

        "nearby":
            len(nearby),

        "median_wr":
            median_wr,

        "median_r":
            median_r,

        "positive_pct":
            positive_pct,
    }


# ============================================================
# TRAINING OPTIMISATION
# ============================================================

def optimise_training(
    df,
    training_start,
    training_end,
):

    training_df = df[
        (df["time"] >= training_start) &
        (df["time"] <= training_end)
    ].copy()

    print(
        "PHASE 1: REGIME-AWARE "
        "PARAMETER SEARCH"
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
        f"TOTAL COMBINATIONS: {total}"
    )

    candidates = []

    for count, params in enumerate(
        combinations,
        1,
    ):

        if (
            count == 1
            or count % 500 == 0
            or count == total
        ):

            print(
                f"Progress: "
                f"{count}/{total} "
                f"("
                f"{count / total * 100:.1f}"
                f"%)"
            )

        trades = generate_trades(
            training_df,
            *params,
        )

        # ----------------------------------------------------
        # CRITICAL SAMPLE FILTER.
        #
        # No 8/10, 12/15 or 15/18 candidates.
        # ----------------------------------------------------

        if (
            len(trades)
            < MIN_TRAIN_TRADES
        ):

            continue

        p = calculate_performance(
            trades
        )

        if p["total_r"] <= 0:
            continue

        # ----------------------------------------------------
        # Quality score.
        #
        # Win rate matters most, but R, PF and drawdown
        # prevent tiny/highly unstable strategies winning.
        # ----------------------------------------------------

        wr_score = (
            p["win_rate"]
        )

        r_score = min(
            max(
                p["total_r"],
                0
            ),
            30
        ) / 30 * 100

        pf_score = min(
            p["profit_factor"],
            4
        ) / 4 * 100

        dd_penalty = min(
            p["max_drawdown"],
            10
        ) / 10 * 100

        sample_bonus = min(
            len(trades),
            150
        ) / 150 * 100

        quality = (
            wr_score * 0.40
            + r_score * 0.20
            + pf_score * 0.20
            + sample_bonus * 0.15
            - dd_penalty * 0.05
        )

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

    print()
    print(
        f"VALID TRAINING CANDIDATES: "
        f"{len(candidates)}"
    )

    if not candidates:

        return None

    candidates.sort(
        key=lambda x:
        x["quality"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Only the strongest candidates receive stability testing.
    # --------------------------------------------------------

    print()
    print(
        "PHASE 2: PARAMETER STABILITY"
    )

    stable_candidates = []

    for candidate in candidates[:30]:

        stability = stability_test(
            df,
            candidate["params"],
            training_start,
            training_end,
        )

        candidate[
            "stability"
        ] = stability

        if stability["stable"]:

            stable_candidates.append(
                candidate
            )

    print()
    print(
        f"STABLE CANDIDATES: "
        f"{len(stable_candidates)}"
    )

    if not stable_candidates:

        return None

    # --------------------------------------------------------
    # Select based on stability first, then quality.
    # --------------------------------------------------------

    stable_candidates.sort(
        key=lambda x: (
            x["stability"]["positive_pct"],
            x["stability"]["median_r"],
            x["quality"],
        ),
        reverse=True,
    )

    return stable_candidates[0]


# ============================================================
# WALK-FORWARD PERIOD
# ============================================================

def run_walk_forward_period(
    market,
    df,
    period,
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
        f"{market} V11: "
        f"{period_name}"
    )

    print("=" * 60)

    print(
        "Optimising training period..."
    )

    selected = optimise_training(
        df,
        train_start,
        train_end,
    )

    if selected is None:

        print(
            "NO STABLE TRAINING "
            "STRATEGY FOUND"
        )

        return None

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

    print(
        f"Training DD: "
        f"{train_perf['max_drawdown']:.2f}R"
    )

    print()
    print(
        "PARAMETER STABILITY"
    )

    print(
        f"Nearby strategies: "
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

    # --------------------------------------------------------
    # COMPLETELY OOS TEST
    # --------------------------------------------------------

    oos_df = df[
        (df["time"] >= oos_start) &
        (df["time"] <= oos_end)
    ].copy()

    print()
    print(
        "OUT-OF-SAMPLE RESULT"
    )

    print("-" * 60)

    oos_trades = generate_trades(
        oos_df,
        *params,
    )

    oos_perf = calculate_performance(
        oos_trades
    )

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
        f"{oos_perf['longest_losing_streak']}"
    )

    return {
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
            oos_perf["profit_factor"],

        "oos_drawdown":
            oos_perf["max_drawdown"],

        "oos_losing_streak":
            oos_perf[
                "longest_losing_streak"
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


# ============================================================
# FINAL HOLDOUT
# ============================================================

def run_final_holdout(
    market,
    df,
):

    print()
    print("=" * 60)

    print(
        f"{market} FINAL "
        "NEVER-SEEN HOLDOUT"
    )

    print("=" * 60)

    print(
        "Training through "
        f"{FINAL_TRAIN_END}"
    )

    print(
        "Testing from "
        f"{FINAL_HOLDOUT_START}"
    )

    selected = optimise_training(
        df,
        FINAL_TRAIN_START,
        FINAL_TRAIN_END,
    )

    if selected is None:

        print(
            "NO STABLE FINAL "
            "STRATEGY FOUND"
        )

        return None

    params = selected[
        "params"
    ]

    holdout_df = df[
        df["time"] >=
        FINAL_HOLDOUT_START
    ].copy()

    trades = generate_trades(
        holdout_df,
        *params,
    )

    p = calculate_performance(
        trades
    )

    print()
    print(
        "FINAL HOLDOUT RESULT"
    )

    print("-" * 60)

    print(
        f"Trades: {p['trades']}"
    )

    print(
        f"Wins: {p['wins']}"
    )

    print(
        f"Losses: {p['losses']}"
    )

    print(
        f"Win rate: "
        f"{p['win_rate']:.2f}%"
    )

    print(
        f"Total R: "
        f"{p['total_r']:.2f}"
    )

    print(
        f"Profit factor: "
        f"{p['profit_factor']:.2f}"
    )

    print(
        f"Max drawdown: "
        f"{p['max_drawdown']:.2f}R"
    )

    print(
        f"Longest losing streak: "
        f"{p['longest_losing_streak']}"
    )

    return {
        "market":
            market,

        "holdout_trades":
            p["trades"],

        "holdout_wins":
            p["wins"],

        "holdout_losses":
            p["losses"],

        "holdout_win_rate":
            p["win_rate"],

        "holdout_total_r":
            p["total_r"],

        "holdout_profit_factor":
            p["profit_factor"],

        "holdout_drawdown":
            p["max_drawdown"],

        "holdout_stable":
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


# ============================================================
# MARKET
# ============================================================

def run_market(
    market,
    path,
):

    print()
    print("=" * 60)

    print(
        f"{market} V11 REGIME-AWARE "
        "OPTIMIZER"
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
        "Indicators ready."
    )

    results = []

    # --------------------------------------------------------
    # Walk-forward
    # --------------------------------------------------------

    for period in (
        WALK_FORWARD_PERIODS
    ):

        try:

            result = (
                run_walk_forward_period(
                    market,
                    df,
                    period,
                )
            )

            if result is not None:

                results.append(
                    result
                )

        except Exception as error:

            print()
            print("=" * 60)

            print(
                f"{market} PERIOD FAILED"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            print("=" * 60)

    # --------------------------------------------------------
    # Final holdout
    # --------------------------------------------------------

    try:

        holdout = run_final_holdout(
            market,
            df,
        )

    except Exception as error:

        print()
        print("=" * 60)

        print(
            f"{market} HOLDOUT FAILED"
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        print("=" * 60)

        holdout = None

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        RESULT_FILES[market],
        index=False,
    )

    return result_df, holdout


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    market_results
):

    summaries = []

    total_oos_trades = 0
    total_oos_wins = 0
    total_oos_r = 0.0

    profitable_periods = 0
    total_periods = 0

    for market, result in (
        market_results.items()
    ):

        if result is None:
            continue

        result_df, holdout = result

        if result_df.empty:
            continue

        oos_trades = int(
            result_df[
                "oos_trades"
            ].sum()
        )

        oos_wins = int(
            result_df[
                "oos_wins"
            ].sum()
        )

        oos_r = float(
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
            oos_wins /
            oos_trades *
            100
            if oos_trades
            else 0
        )

        total_oos_trades += (
            oos_trades
        )

        total_oos_wins += (
            oos_wins
        )

        total_oos_r += oos_r

        profitable_periods += (
            profitable
        )

        total_periods += (
            periods
        )

        if (
            oos_trades >= 100
            and win_rate >= 75
            and oos_r > 0
        ):

            verdict = "PROMISING"

        elif (
            oos_trades >= 50
            and win_rate >= 65
            and oos_r > 0
        ):

            verdict = "INTERESTING"

        else:

            verdict = "NOT THERE YET"

        summaries.append(
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
                        oos_r,
                        2,
                    ),

                "profitable_periods":
                    f"{profitable}/{periods}",

                "verdict":
                    verdict,

                "holdout_trades":
                    (
                        holdout[
                            "holdout_trades"
                        ]
                        if holdout
                        else 0
                    ),

                "holdout_win_rate":
                    (
                        round(
                            holdout[
                                "holdout_win_rate"
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
                                "holdout_total_r"
                            ],
                            2,
                        )
                        if holdout
                        else 0
                    ),
            }
        )

    summary_df = pd.DataFrame(
        summaries
    )

    return (
        summary_df,
        total_oos_trades,
        total_oos_wins,
        total_oos_r,
        profitable_periods,
        total_periods,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "MULTI-MARKET STRATEGY "
        "OPTIMIZER V11.1"
    )

    print("=" * 60)

    print(
        "REGIME-AWARE SIGNAL "
        "VALIDATION: ENABLED"
    )

    print(
        "REJECTION / REVERSAL "
        "PRIMARY SIGNAL: ENABLED"
    )

    print(
        "MARKET-SPECIFIC "
        "OPTIMISATION: ENABLED"
    )

    print(
        "WALK-FORWARD TESTING: ENABLED"
    )

    print(
        "PARAMETER STABILITY: ENABLED"
    )

    print(
        "STRICT FINAL HOLDOUT: ENABLED"
    )

    print(
        "MINIMUM SAMPLE FILTER: ENABLED"
    )

    print(
        "NO LIVE TRADING"
    )

    print("=" * 60)

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
            print("=" * 60)

            print(
                f"{market} FAILED"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            print("=" * 60)

            market_results[
                market
            ] = None

    # ========================================================
    # SUMMARY
    # ========================================================

    (
        summary_df,
        total_trades,
        total_wins,
        total_r,
        profitable_periods,
        total_periods,
    ) = build_summary(
        market_results
    )

    print()
    print("=" * 60)

    print(
        "V11.1 FINAL MULTI-MARKET "
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

    # ========================================================
    # COMBINED OOS
    # ========================================================

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
        "V11.1 TARGET CHECK"
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

    print()

    if (
        total_trades >= 200
        and combined_wr >= 82
        and total_r > 0
    ):

        print(
            "TARGET STATUS: "
            "ACHIEVED"
        )

    elif (
        total_trades >= 100
        and combined_wr >= 75
        and total_r > 0
    ):

        print(
            "TARGET STATUS: "
            "VERY PROMISING"
        )

    elif (
        total_trades >= 50
        and combined_wr >= 65
        and total_r > 0
    ):

        print(
            "TARGET STATUS: "
            "PROMISING"
        )

    else:

        print(
            "TARGET STATUS: "
            "NOT ACHIEVED YET"
        )

    # ========================================================
    # IMPORTANT
    # ========================================================

    print()
    print("=" * 60)

    print(
        "IMPORTANT"
    )

    print("=" * 60)

    print(
        "OOS data was never used "
        "for optimisation."
    )

    print(
        "The final 2026 holdout "
        "was kept separate."
    )

    print(
        "Minimum sample filtering "
        "rejects tiny high-WR samples."
    )

    print(
        "XAUUSD and EURUSD are "
        "optimised independently."
    )

    print(
        "This is research only."
    )

    print(
        "DO NOT IMPLEMENT LIVE "
        "FROM THIS OPTIMIZER ALONE."
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
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
        "OPTIMIZER V11.1 COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()
