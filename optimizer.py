import os
import pandas as pd
import numpy as np
from itertools import product


# ============================================================
# XAUUSD OPTIMIZER V2
#
# Goal:
# Find the highest-quality strategy around ~1 trade/week
# while requiring a meaningful number of historical trades.
#
# IMPORTANT:
# This optimizer is research only.
# It does NOT place live trades.
# ============================================================

DATA_FILE = "data/XAUUSD_15m.csv"
OUTPUT_FILE = "data/optimizer_results_v2.csv"


# ============================================================
# SEARCH SPACE
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
    0.40,
    0.45,
    0.50,
]

BODY_VALUES = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
]

SEPARATION_VALUES = [
    0.0005,
    0.0008,
    0.0010,
    0.0012,
    0.0015,
    0.0020,
]

MAX_CROSS_VALUES = [
    10,
    15,
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
    (2, 3, 4, 5, 12),
    (2, 3, 4, 5, 12, 13),
    (3, 4, 5, 12, 13),
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("XAUUSD OPTIMIZER V2")
    print("=" * 60)

    if not os.path.exists(DATA_FILE):

        raise RuntimeError(
            f"Missing file: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

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
# PREPARE DATA
# ============================================================

def prepare_data(df):

    print()
    print("Preparing indicators...")

    close = df["Close"].astype(float)
    open_price = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    close_np = close.to_numpy()
    open_np = open_price.to_numpy()
    high_np = high.to_numpy()
    low_np = low.to_numpy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # EMA SLOPES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # EMA SEPARATION
    # --------------------------------------------------------

    separation = (
        np.abs(
            ema20 - ema50
        )
        / close_np
    )

    # --------------------------------------------------------
    # CANDLE STRUCTURE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # HOUR
    # --------------------------------------------------------

    hours = (
        df.index.hour.to_numpy()
    )

    # --------------------------------------------------------
    # BEARISH EMA CROSS AGE
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
            close_np
            - ema20
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
        "open": open_np,
        "high": high_np,
        "low": low_np,
        "ema20": ema20,
        "ema50": ema50,
        "ema20_slope": ema20_slope,
        "ema50_slope": ema50_slope,
        "separation": separation,
        "body_ratio": body_ratio,
        "upper_wick_ratio": upper_wick_ratio,
        "hours": hours,
        "cross_age": cross_age,
        "pullback": pullback,
        "recent_high": recent_high,
        "risk": risk,
    }


# ============================================================
# BASE SIGNAL
# ============================================================

def build_signal_mask(
    data,
    wick,
    body,
    separation,
    max_cross,
    hours
):

    hour_mask = np.isin(
        data["hours"],
        np.array(hours)
    )

    mask = (
        hour_mask
        &
        (
            data["ema20"]
            < data["ema50"]
        )
        &
        (
            data["ema20_slope"]
            < 0
        )
        &
        (
            data["ema50_slope"]
            < 0
        )
        &
        (
            data["separation"]
            >= separation
        )
        &
        (
            data["cross_age"]
            <= max_cross
        )
        &
        data["pullback"]
        &
        (
            data["close"]
            < data["open"]
        )
        &
        (
            data["upper_wick_ratio"]
            >= wick
        )
        &
        (
            data["body_ratio"]
            >= body
        )
        &
        (
            data["close"]
            < data["ema20"]
        )
    )

    return np.flatnonzero(
        mask
    )


# ============================================================
# RUN STRATEGY
# ============================================================

def run_strategy(
    data,
    indices,
    rr
):

    close = data["close"]
    high = data["high"]

    recent_high = (
        data["recent_high"]
    )

    trades = []

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

        for j in range(
            index + 1,
            len(close)
        ):

            candle_high = high[j]
            candle_low = (
                data["low"][j]
            )

            # Conservative:
            # if SL and TP occur in same
            # candle, count SL first.
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

    if not trades:
        return None

    trades = np.asarray(
        trades,
        dtype=float
    )

    total_trades = len(
        trades
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

    win_rate = (
        win_count
        / total_trades
        * 100
    )

    total_r = float(
        trades.sum()
    )

    gross_profit = float(
        trades[wins].sum()
    )

    gross_loss = abs(
        float(
            trades[losses].sum()
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

    running_peak = np.maximum.accumulate(
        equity
    )

    drawdown = (
        running_peak - equity
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

    # --------------------------------------------------------
    # LONGEST LOSING STREAK
    # --------------------------------------------------------

    longest_loss_streak = 0
    current_loss_streak = 0

    for result in trades:

        if result < 0:

            current_loss_streak += 1

            longest_loss_streak = max(
                longest_loss_streak,
                current_loss_streak
            )

        else:

            current_loss_streak = 0

    return {
        "trades": total_trades,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": win_rate,
        "total_r": total_r,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "trades_per_week": trades_per_week,
        "max_drawdown": max_drawdown,
        "longest_loss_streak":
            longest_loss_streak,
    }


# ============================================================
# RANKING
# ============================================================

def rank_strategy(
    metrics
):

    trades = metrics["trades"]
    win_rate = metrics["win_rate"]
    total_r = metrics["total_r"]
    drawdown = metrics["max_drawdown"]
    trades_per_week = (
        metrics["trades_per_week"]
    )

    # --------------------------------------------------------
    # HARD SAMPLE FILTER
    # --------------------------------------------------------

    if trades < 50:
        return -999999

    # --------------------------------------------------------
    # PRIMARY OBJECTIVE:
    #
    # Get close to ONE trade/week.
    # But do NOT reward extremely rare systems.
    # --------------------------------------------------------

    if (
        0.50
        <= trades_per_week
        <= 1.50
    ):

        frequency_score = 100

    elif (
        0.25
        <= trades_per_week
        < 0.50
    ):

        frequency_score = 50

    elif (
        1.50
        < trades_per_week
        <= 2.50
    ):

        frequency_score = 50

    else:

        frequency_score = 0

    # --------------------------------------------------------
    # WIN RATE DOMINATES
    # --------------------------------------------------------

    score = (
        win_rate * 10
    )

    # --------------------------------------------------------
    # PROFITABILITY
    # --------------------------------------------------------

    score += (
        total_r * 2
    )

    # --------------------------------------------------------
    # PROFIT FACTOR
    # --------------------------------------------------------

    score += (
        min(
            metrics["profit_factor"],
            3
        )
        * 10
    )

    # --------------------------------------------------------
    # DRAWDOWN
    # --------------------------------------------------------

    score -= (
        drawdown * 2
    )

    # --------------------------------------------------------
    # FREQUENCY
    # --------------------------------------------------------

    score += frequency_score

    # --------------------------------------------------------
    # SAMPLE SIZE
    # --------------------------------------------------------

    score += min(
        trades,
        500
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
    print("OPTIMIZER V2 STARTING")
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

        indices = build_signal_mask(
            data,
            wick,
            body,
            separation,
            max_cross,
            hours
        )

        if len(indices) == 0:
            continue

        trades = run_strategy(
            data,
            indices,
            rr
        )

        metrics = calculate_metrics(
            trades,
            total_days
        )

        if metrics is None:
            continue

        metrics.update({

            "rr": rr,

            "wick": wick,

            "body": body,

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
        ] = rank_strategy(
            metrics
        )

        results.append(
            metrics
        )

        if (
            number == 1
            or number % 250 == 0
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

    columns = [
        "trades",
        "trades_per_week",
        "win_rate",
        "total_r",
        "profit_factor",
        "max_drawdown",
        "longest_loss_streak",
        "rr",
        "wick",
        "body",
        "separation",
        "max_cross",
        "hours",
    ]

    # ========================================================
    # TOP OVERALL
    # ========================================================

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
    # BEST WIN RATE — 50+
    # ========================================================

    best_50 = (
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
        best_50[
            columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # BEST WIN RATE — 100+
    # ========================================================

    best_100 = (
        results_df[
            results_df["trades"] >= 100
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
    print("BEST WIN RATE — MIN 100 TRADES")
    print("=" * 60)

    if len(best_100):

        print(
            best_100[
                columns
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "No strategies with 100+ trades."
        )

    # ========================================================
    # BEST WIN RATE — 200+
    # ========================================================

    best_200 = (
        results_df[
            results_df["trades"] >= 200
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
    print("BEST WIN RATE — MIN 200 TRADES")
    print("=" * 60)

    if len(best_200):

        print(
            best_200[
                columns
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "No strategies with 200+ trades."
        )

    # ========================================================
    # 0.5–1.5 TRADES/WEEK
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
    print(
        "BEST WIN RATE — 0.5 TO 1.5 "
        "TRADES/WEEK"
    )
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
    # 0.75–1.25 TRADES/WEEK
    # ========================================================

    exact_frequency = (
        results_df[
            (
                results_df[
                    "trades_per_week"
                ] >= 0.75
            )
            &
            (
                results_df[
                    "trades_per_week"
                ] <= 1.25
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
    print(
        "BEST WIN RATE — 0.75 TO 1.25 "
        "TRADES/WEEK"
    )
    print("=" * 60)

    if len(exact_frequency):

        print(
            exact_frequency[
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
        results_df[
            results_df["trades"] >= 50
        ]
        .sort_values(
            "total_r",
            ascending=False
        )
        .head(10)
    )

    print()
    print("=" * 60)
    print("BEST PROFIT — MIN 50 TRADES")
    print("=" * 60)

    print(
        best_profit[
            columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # BEST PROFIT FACTOR
    # ========================================================

    best_pf = (
        results_df[
            results_df["trades"] >= 50
        ]
        .sort_values(
            [
                "profit_factor",
                "win_rate"
            ],
            ascending=False
        )
        .head(10)
    )

    print()
    print("=" * 60)
    print("BEST PROFIT FACTOR — MIN 50 TRADES")
    print("=" * 60)

    print(
        best_pf[
            columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # FINAL RECOMMENDATION
    # ========================================================

    ranked = results_df[
        results_df[
            "optimizer_score"
        ] > -999000
    ]

    if len(ranked):

        best = ranked.iloc[0]

        print()
        print("=" * 60)
        print("OPTIMIZER V2 RECOMMENDATION")
        print("=" * 60)

        print(
            "Win rate:",
            f"{best['win_rate']:.2f}%"
        )

        print(
            "Trades:",
            int(best["trades"])
        )

        print(
            "Trades/week:",
            f"{best['trades_per_week']:.2f}"
        )

        print(
            "Total R:",
            f"{best['total_r']:.2f}"
        )

        print(
            "Profit factor:",
            f"{best['profit_factor']:.2f}"
        )

        print(
            "Max drawdown:",
            f"{best['max_drawdown']:.2f}R"
        )

        print(
            "Longest losing streak:",
            int(
                best[
                    "longest_loss_streak"
                ]
            )
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
    print("=" * 60)
    print("FULL RESULTS")
    print("=" * 60)

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 60)
    print("OPTIMIZER V2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
