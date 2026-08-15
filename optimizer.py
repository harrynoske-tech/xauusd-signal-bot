import os
import pandas as pd
import numpy as np
from itertools import product


# ============================================================
# FAST XAUUSD STRATEGY OPTIMIZER
# ============================================================

DATA_FILE = "data/XAUUSD_15m.csv"
OUTPUT_FILE = "data/optimizer_results.csv"

MIN_TRADES = 30

# ------------------------------------------------------------
# PARAMETERS TO TEST
# ------------------------------------------------------------

RR_VALUES = [
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
]

WICK_VALUES = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
]

BODY_VALUES = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
]

SEPARATION_VALUES = [
    0.0008,
    0.0010,
    0.0015,
    0.0020,
    0.0030,
]

MAX_CROSS_VALUES = [
    10,
    15,
    20,
    30,
]

HOUR_SETS = [
    (2, 3),
    (2, 3, 4),
    (2, 3, 4, 5),
    (3, 4, 5),
    (2, 3, 4, 5, 12, 13),
    (2, 3, 12, 13),
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("FAST XAUUSD OPTIMIZER")
    print("=" * 60)

    if not os.path.exists(DATA_FILE):

        raise RuntimeError(
            f"Missing file: {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    time_column = None

    for column in [
        "time",
        "Time",
        "timestamp",
        "Timestamp",
        "date",
        "Date",
    ]:

        if column in df.columns:

            time_column = column
            break

    if time_column is None:

        raise RuntimeError(
            "Could not find time column."
        )

    df[time_column] = pd.to_datetime(
        df[time_column],
        utc=True
    )

    df = df.set_index(
        time_column
    )

    rename = {}

    for column in df.columns:

        lower = column.lower()

        if lower == "open":
            rename[column] = "Open"

        elif lower == "high":
            rename[column] = "High"

        elif lower == "low":
            rename[column] = "Low"

        elif lower == "close":
            rename[column] = "Close"

    df = df.rename(
        columns=rename
    )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    for column in required:

        if column not in df.columns:

            raise RuntimeError(
                f"Missing column: {column}"
            )

    df = df.sort_index()

    print(
        "Candles:",
        len(df)
    )

    print(
        "Range:",
        df.index.min(),
        "->",
        df.index.max()
    )

    return df


# ============================================================
# PRE-CALCULATE EVERYTHING ONCE
# ============================================================

def prepare_data(df):

    print()
    print("Preparing indicators...")

    close = df["Close"].astype(float)
    open_price = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    ema20 = (
        close
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
        .to_numpy()
    )

    ema50 = (
        close
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
        .to_numpy()
    )

    close_np = close.to_numpy()
    open_np = open_price.to_numpy()
    high_np = high.to_numpy()
    low_np = low.to_numpy()

    separation = (
        np.abs(
            ema20 - ema50
        )
        / close_np
    )

    ema20_slope = (
        ema20
        - np.roll(
            ema20,
            4
        )
    )

    ema50_slope = (
        ema50
        - np.roll(
            ema50,
            4
        )
    )

    candle_range = (
        high_np - low_np
    )

    body = np.abs(
        close_np - open_np
    )

    body_ratio = np.divide(
        body,
        candle_range,
        out=np.zeros_like(body),
        where=candle_range > 0
    )

    upper_wick = (
        high_np
        - np.maximum(
            open_np,
            close_np
        )
    )

    upper_wick_ratio = np.divide(
        upper_wick,
        candle_range,
        out=np.zeros_like(
            upper_wick
        ),
        where=candle_range > 0
    )

    hours = (
        df.index.hour.to_numpy()
    )

    # --------------------------------------------------------
    # BEARISH CROSS AGE
    # --------------------------------------------------------

    bearish_cross = (
        (ema20[:-1] >= ema50[:-1])
        &
        (ema20[1:] < ema50[1:])
    )

    cross_age = np.full(
        len(df),
        9999,
        dtype=np.int32
    )

    last_cross = -9999

    for i in range(
        1,
        len(df)
    ):

        if bearish_cross[i - 1]:

            last_cross = i

        if last_cross >= 0:

            cross_age[i] = (
                i - last_cross
            )

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------

    previous_close = np.roll(
        close_np,
        1
    )

    previous_ema20 = np.roll(
        ema20,
        1
    )

    previous_distance = np.divide(
        np.abs(
            previous_close
            - previous_ema20
        ),
        previous_close,
        out=np.ones_like(
            previous_close
        ),
        where=previous_close != 0
    )

    current_distance = np.divide(
        np.abs(
            close_np - ema20
        ),
        close_np,
        out=np.ones_like(
            close_np
        ),
        where=close_np != 0
    )

    pullback = (
        (previous_distance <= 0.0020)
        |
        (current_distance <= 0.0020)
    )

    # --------------------------------------------------------
    # STRUCTURAL STOP
    # --------------------------------------------------------

    recent_high = (
        pd.Series(high_np)
        .rolling(
            8,
            min_periods=1
        )
        .max()
        .to_numpy()
    )

    risk = (
        recent_high
        - close_np
    )

    return {
        "close": close_np,
        "high": high_np,
        "low": low_np,
        "open": open_np,
        "ema20": ema20,
        "ema50": ema50,
        "separation": separation,
        "ema20_slope": ema20_slope,
        "ema50_slope": ema50_slope,
        "body_ratio": body_ratio,
        "upper_wick_ratio":
            upper_wick_ratio,
        "hours": hours,
        "cross_age": cross_age,
        "pullback": pullback,
        "recent_high": recent_high,
        "risk": risk,
    }


# ============================================================
# RUN ONE CONFIGURATION
# ============================================================

def run_configuration(
    data,
    rr,
    wick,
    body,
    separation,
    max_cross,
    hours
):

    close = data["close"]
    high = data["high"]
    low = data["low"]
    open_price = data["open"]

    ema20 = data["ema20"]
    ema50 = data["ema50"]

    ema20_slope = data[
        "ema20_slope"
    ]

    ema50_slope = data[
        "ema50_slope"
    ]

    cross_age = data[
        "cross_age"
    ]

    body_ratio = data[
        "body_ratio"
    ]

    upper_wick_ratio = data[
        "upper_wick_ratio"
    ]

    candle_hours = data[
        "hours"
    ]

    pullback = data[
        "pullback"
    ]

    recent_high = data[
        "recent_high"
    ]

    # --------------------------------------------------------
    # BASE SIGNAL
    # --------------------------------------------------------

    hour_mask = np.isin(
        candle_hours,
        np.array(hours)
    )

    signal_mask = (
        hour_mask
        &
        (ema20 < ema50)
        &
        (ema20_slope < 0)
        &
        (ema50_slope < 0)
        &
        (
            data["separation"]
            >= separation
        )
        &
        (
            cross_age
            <= max_cross
        )
        &
        pullback
        &
        (close < open_price)
        &
        (
            upper_wick_ratio
            >= wick
        )
        &
        (
            body_ratio
            >= body
        )
        &
        (close < ema20)
    )

    indices = np.flatnonzero(
        signal_mask
    )

    if len(indices) == 0:

        return []

    trades = []

    # --------------------------------------------------------
    # TRADE SIMULATION
    #
    # This only loops through SIGNALS rather
    # than all 156k candles.
    # --------------------------------------------------------

    next_available = 0

    for index in indices:

        if index < next_available:
            continue

        entry = close[index]

        stop = recent_high[index]

        risk = (
            stop - entry
        )

        if risk <= 0:
            continue

        target = (
            entry
            - risk * rr
        )

        result = None

        exit_index = None

        # Search forward until SL or TP.
        for j in range(
            index + 1,
            len(close)
        ):

            candle_high = high[j]
            candle_low = low[j]

            # Conservative assumption:
            # SL wins if both are touched.
            if candle_high >= stop:

                result = -1.0

                exit_index = j

                break

            if candle_low <= target:

                result = rr

                exit_index = j

                break

        if result is None:
            continue

        trades.append(
            result
        )

        next_available = (
            exit_index + 1
        )

    return trades


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    trades,
    total_days
):

    if len(trades) < MIN_TRADES:

        return None

    trades = np.array(
        trades,
        dtype=float
    )

    wins = (
        trades > 0
    )

    losses = (
        trades < 0
    )

    win_count = int(
        wins.sum()
    )

    loss_count = int(
        losses.sum()
    )

    total_trades = len(
        trades
    )

    win_rate = (
        win_count
        / total_trades
        * 100
    )

    total_r = float(
        trades.sum()
    )

    gross_profit = float(
        trades[trades > 0].sum()
    )

    gross_loss = abs(
        float(
            trades[trades < 0].sum()
        )
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = 999.0

    equity = np.cumsum(
        trades
    )

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        running_max - equity
    )

    max_drawdown = float(
        drawdown.max()
    )

    weeks = (
        total_days / 7
    )

    trades_per_week = (
        total_trades
        / weeks
    )

    expectancy = (
        total_r
        / total_trades
    )

    return {
        "trades":
            total_trades,

        "wins":
            win_count,

        "losses":
            loss_count,

        "win_rate":
            win_rate,

        "total_r":
            total_r,

        "profit_factor":
            profit_factor,

        "expectancy":
            expectancy,

        "trades_per_week":
            trades_per_week,

        "max_drawdown":
            max_drawdown,
    }


# ============================================================
# OPTIMIZER SCORE
# ============================================================

def optimizer_score(
    metrics
):

    if metrics is None:
        return -999999

    win_rate = metrics[
        "win_rate"
    ]

    total_r = metrics[
        "total_r"
    ]

    drawdown = metrics[
        "max_drawdown"
    ]

    trades_per_week = metrics[
        "trades_per_week"
    ]

    trades = metrics[
        "trades"
    ]

    # --------------------------------------------------------
    # We want:
    #
    # 1. HIGH WIN RATE
    # 2. POSITIVE R
    # 3. ~1 TRADE/WEEK
    # 4. LOW DRAWDOWN
    # 5. Enough trades to be meaningful
    # --------------------------------------------------------

    score = 0

    score += (
        win_rate * 10
    )

    score += (
        total_r * 2
    )

    score -= (
        drawdown * 2
    )

    frequency_penalty = abs(
        trades_per_week - 1.0
    )

    score -= (
        frequency_penalty * 20
    )

    # Reward larger samples.
    score += min(
        trades,
        300
    ) * 0.05

    return score


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    data = prepare_data(
        df
    )

    total_days = (
        df.index.max()
        - df.index.min()
    ).total_seconds() / 86400

    combinations = list(
        product(
            RR_VALUES,
            WICK_VALUES,
            BODY_VALUES,
            SEPARATION_VALUES,
            MAX_CROSS_VALUES,
            HOUR_SETS,
        )
    )

    print()
    print("=" * 60)
    print("FAST OPTIMIZER STARTING")
    print("=" * 60)

    print(
        "Total combinations:",
        len(combinations)
    )

    print()

    results = []

    for number, params in enumerate(
        combinations,
        start=1
    ):

        (
            rr,
            wick,
            body,
            separation,
            max_cross,
            hours,
        ) = params

        trades = run_configuration(
            data,
            rr,
            wick,
            body,
            separation,
            max_cross,
            hours
        )

        metrics = calculate_metrics(
            trades,
            total_days
        )

        if metrics is None:
            continue

        metrics.update({

            "rr":
                rr,

            "wick":
                wick,

            "body":
                body,

            "separation":
                separation,

            "max_cross":
                max_cross,

            "hours":
                ",".join(
                    str(x)
                    for x in hours
                ),

        })

        metrics[
            "optimizer_score"
        ] = optimizer_score(
            metrics
        )

        results.append(
            metrics
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            number == 1
            or number % 100 == 0
        ):

            print(
                f"Progress: "
                f"{number}/"
                f"{len(combinations)} "
                f"| valid: "
                f"{len(results)}",
                flush=True
            )

    if not results:

        raise RuntimeError(
            "No valid strategies found."
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "optimizer_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    os.makedirs(
        "data",
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # TOP 20 OVERALL
    # ========================================================

    columns = [
        "trades",
        "trades_per_week",
        "win_rate",
        "total_r",
        "profit_factor",
        "max_drawdown",
        "rr",
        "wick",
        "body",
        "separation",
        "max_cross",
        "hours",
    ]

    print()
    print("=" * 60)
    print("TOP 20 STRATEGIES")
    print("=" * 60)

    print(
        results_df[
            columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # BEST WIN RATE
    # ========================================================

    best_win = (
        results_df[
            results_df["trades"] >= 50
        ]
        .sort_values(
            [
                "win_rate",
                "total_r"
            ],
            ascending=False
        )
        .head(10)
    )

    print()
    print("=" * 60)
    print("BEST WIN RATE — MIN 50 TRADES")
    print("=" * 60)

    print(
        best_win[
            columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # BEST ~1 TRADE/WEEK
    # ========================================================

    near_one = (
        results_df[
            (
                results_df[
                    "trades_per_week"
                ] >= 0.50
            )
            &
            (
                results_df[
                    "trades_per_week"
                ] <= 1.50
            )
            &
            (
                results_df[
                    "trades"
                ] >= 50
            )
        ]
        .sort_values(
            [
                "win_rate",
                "total_r"
            ],
            ascending=False
        )
        .head(20)
    )

    print()
    print("=" * 60)
    print("BEST STRATEGIES — ~1 TRADE/WEEK")
    print("=" * 60)

    if len(near_one):

        print(
            near_one[
                columns
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "No strategies found."
        )

    # ========================================================
    # BEST PROFIT
    # ========================================================

    best_profit = (
        results_df
        .sort_values(
            "total_r",
            ascending=False
        )
        .head(10)
    )

    print()
    print("=" * 60)
    print("BEST PROFIT")
    print("=" * 60)

    print(
        best_profit[
            columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    best = results_df.iloc[0]

    print()
    print("=" * 60)
    print("OPTIMIZER RECOMMENDATION")
    print("=" * 60)

    print(
        f"Win rate: "
        f"{best['win_rate']:.2f}%"
    )

    print(
        f"Trades: "
        f"{int(best['trades'])}"
    )

    print(
        f"Trades/week: "
        f"{best['trades_per_week']:.2f}"
    )

    print(
        f"Total R: "
        f"{best['total_r']:.2f}"
    )

    print(
        f"Profit factor: "
        f"{best['profit_factor']:.2f}"
    )

    print(
        f"Max drawdown: "
        f"{best['max_drawdown']:.2f}R"
    )

    print()
    print("PARAMETERS")
    print(
        "RR:",
        best["rr"]
    )

    print(
        "Wick:",
        best["wick"]
    )

    print(
        "Body:",
        best["body"]
    )

    print(
        "EMA separation:",
        best["separation"]
    )

    print(
        "Max cross:",
        best["max_cross"]
    )

    print(
        "Hours:",
        best["hours"]
    )

    print()
    print(
        "Results saved to:",
        OUTPUT_FILE
    )

    print()
    print("=" * 60)
    print("OPTIMIZER COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
