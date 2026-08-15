import pandas as pd
import numpy as np
from itertools import product


# ============================================================
# XAUUSD STRATEGY OPTIMIZER
# ============================================================

DATA_FILE = "data/XAUUSD_15m.csv"

MIN_TRADES = 30

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

EMA_SEPARATION_VALUES = [
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
    (2, 3, 12, 13),
    (2, 3, 4, 5, 12, 13),
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("LOADING XAUUSD DATA")
    print("=" * 60)

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
            "Could not find time column in XAUUSD data."
        )

    df[time_column] = pd.to_datetime(
        df[time_column],
        utc=True
    )

    df = df.set_index(
        time_column
    )

    rename_map = {}

    for column in df.columns:

        lower = column.lower()

        if lower == "open":
            rename_map[column] = "Open"

        elif lower == "high":
            rename_map[column] = "High"

        elif lower == "low":
            rename_map[column] = "Low"

        elif lower == "close":
            rename_map[column] = "Close"

    df = df.rename(
        columns=rename_map
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
        "15M candles:",
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
# PREPARE INDICATORS
# ============================================================

def prepare_data(df):

    df = df.copy()

    df["EMA20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["EMA50"] = (
        df["Close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    df["EMA20_SLOPE"] = (
        df["EMA20"]
        - df["EMA20"].shift(4)
    )

    df["EMA50_SLOPE"] = (
        df["EMA50"]
        - df["EMA50"].shift(4)
    )

    df["SEPARATION"] = (
        abs(
            df["EMA20"]
            - df["EMA50"]
        )
        / df["Close"]
    )

    df["RANGE"] = (
        df["High"]
        - df["Low"]
    )

    df["BODY"] = abs(
        df["Close"]
        - df["Open"]
    )

    df["BODY_RATIO"] = np.where(
        df["RANGE"] > 0,
        df["BODY"] / df["RANGE"],
        0
    )

    df["UPPER_WICK"] = (
        df["High"]
        - df[
            ["Open", "Close"]
        ].max(axis=1)
    )

    df["UPPER_WICK_RATIO"] = np.where(
        df["RANGE"] > 0,
        df["UPPER_WICK"]
        / df["RANGE"],
        0
    )

    return df


# ============================================================
# FIND CROSS
# ============================================================

def find_recent_bearish_cross(
    df,
    index,
    max_bars
):

    start = max(
        1,
        index - max_bars
    )

    for i in range(
        index,
        start - 1,
        -1
    ):

        previous = df.iloc[i - 1]
        current = df.iloc[i]

        if (
            previous["EMA20"]
            >= previous["EMA50"]
            and
            current["EMA20"]
            < current["EMA50"]
        ):

            return index - i

    return None


# ============================================================
# RUN ONE STRATEGY
# ============================================================

def run_strategy(
    df,
    rr,
    wick_threshold,
    body_threshold,
    separation_threshold,
    max_cross,
    hours
):

    trades = []

    in_trade = False

    entry = 0
    stop = 0
    target = 0

    for i in range(
        60,
        len(df)
    ):

        candle = df.iloc[i]

        timestamp = df.index[i]

        # ----------------------------------------------------
        # MANAGE OPEN TRADE
        # ----------------------------------------------------

        if in_trade:

            high = float(
                candle["High"]
            )

            low = float(
                candle["Low"]
            )

            # Stop first for conservative testing.
            if high >= stop:

                trades.append(-1.0)

                in_trade = False

                continue

            if low <= target:

                trades.append(rr)

                in_trade = False

                continue

            continue

        # ----------------------------------------------------
        # SESSION
        # ----------------------------------------------------

        if timestamp.hour not in hours:
            continue

        # ----------------------------------------------------
        # BEARISH EMA TREND
        # ----------------------------------------------------

        ema20 = float(
            candle["EMA20"]
        )

        ema50 = float(
            candle["EMA50"]
        )

        close = float(
            candle["Close"]
        )

        if ema20 >= ema50:
            continue

        if candle["EMA20_SLOPE"] >= 0:
            continue

        if candle["EMA50_SLOPE"] >= 0:
            continue

        # ----------------------------------------------------
        # EMA SEPARATION
        # ----------------------------------------------------

        if (
            candle["SEPARATION"]
            < separation_threshold
        ):
            continue

        # ----------------------------------------------------
        # RECENT CROSS
        # ----------------------------------------------------

        bars_since_cross = (
            find_recent_bearish_cross(
                df,
                i,
                max_cross
            )
        )

        if bars_since_cross is None:
            continue

        # ----------------------------------------------------
        # PULLBACK
        # ----------------------------------------------------

        previous = df.iloc[i - 1]

        previous_distance = abs(
            float(previous["Close"])
            - float(previous["EMA20"])
        ) / float(previous["Close"])

        current_distance = abs(
            close - ema20
        ) / close

        pullback_tolerance = 0.0020

        if (
            previous_distance
            > pullback_tolerance
            and
            current_distance
            > pullback_tolerance
        ):
            continue

        # ----------------------------------------------------
        # STRONG BEARISH CANDLE
        # ----------------------------------------------------

        if candle["Close"] >= candle["Open"]:
            continue

        if (
            candle["UPPER_WICK_RATIO"]
            < wick_threshold
        ):
            continue

        if (
            candle["BODY_RATIO"]
            < body_threshold
        ):
            continue

        if close >= ema20:
            continue

        # ----------------------------------------------------
        # ENTRY
        # ----------------------------------------------------

        entry = close

        lookback = 8

        recent_high = float(
            df["High"]
            .iloc[
                i - lookback + 1:
                i + 1
            ]
            .max()
        )

        stop = recent_high

        risk = stop - entry

        if risk <= 0:
            continue

        target = (
            entry
            - risk * rr
        )

        in_trade = True

    return trades


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    trades,
    df
):

    if len(trades) == 0:
        return None

    wins = [
        r for r in trades
        if r > 0
    ]

    losses = [
        r for r in trades
        if r < 0
    ]

    win_rate = (
        len(wins)
        / len(trades)
        * 100
    )

    total_r = sum(trades)

    gross_profit = sum(wins)

    gross_loss = abs(
        sum(losses)
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit
            / gross_loss
        )
    else:
        profit_factor = 999

    expectancy = (
        total_r
        / len(trades)
    )

    days = (
        df.index.max()
        - df.index.min()
    ).total_seconds() / 86400

    weeks = days / 7

    trades_per_week = (
        len(trades)
        / weeks
    )

    # --------------------------------------------------------
    # DRAWDOWN
    # --------------------------------------------------------

    equity = 0
    peak = 0
    max_drawdown = 0

    for r in trades:

        equity += r

        peak = max(
            peak,
            equity
        )

        drawdown = (
            peak - equity
        )

        max_drawdown = max(
            max_drawdown,
            drawdown
        )

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "total_r": total_r,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "trades_per_week":
            trades_per_week,
        "max_drawdown":
            max_drawdown,
    }


# ============================================================
# SCORE STRATEGY
# ============================================================

def score_strategy(metrics):

    if metrics is None:
        return -999999

    trades = metrics["trades"]
    win_rate = metrics["win_rate"]
    total_r = metrics["total_r"]
    drawdown = metrics["max_drawdown"]
    trades_per_week = (
        metrics["trades_per_week"]
    )

    if trades < MIN_TRADES:
        return -999999

    # --------------------------------------------------------
    # TARGET FREQUENCY
    # --------------------------------------------------------

    frequency_score = -abs(
        trades_per_week - 1.0
    ) * 10

    # --------------------------------------------------------
    # WIN RATE
    # --------------------------------------------------------

    win_score = (
        win_rate * 5
    )

    # --------------------------------------------------------
    # PROFIT
    # --------------------------------------------------------

    profit_score = (
        total_r * 1.5
    )

    # --------------------------------------------------------
    # DRAWDOWN PENALTY
    # --------------------------------------------------------

    drawdown_penalty = (
        drawdown * 1.0
    )

    return (
        win_score
        + profit_score
        + frequency_score
        - drawdown_penalty
    )


# ============================================================
# MAIN OPTIMIZER
# ============================================================

def main():

    df = load_data()

    df = prepare_data(df)

    combinations = list(
        product(
            RR_VALUES,
            WICK_VALUES,
            BODY_VALUES,
            EMA_SEPARATION_VALUES,
            MAX_CROSS_VALUES,
            HOUR_SETS,
        )
    )

    print()
    print("=" * 60)
    print("OPTIMIZER STARTING")
    print("=" * 60)

    print(
        "Strategy combinations:",
        len(combinations)
    )

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

        trades = run_strategy(
            df,
            rr,
            wick,
            body,
            separation,
            max_cross,
            hours,
        )

        metrics = calculate_metrics(
            trades,
            df
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
                    str(h)
                    for h in hours
                ),
        })

        metrics["optimizer_score"] = (
            score_strategy(metrics)
        )

        results.append(metrics)

        if number % 250 == 0:

            print(
                f"Tested {number}/"
                f"{len(combinations)}"
            )

    if not results:

        print(
            "No valid strategies found."
        )

        return

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "optimizer_score",
        ascending=False
    )

    # ========================================================
    # SAVE FULL RESULTS
    # ========================================================

    results_df.to_csv(
        "data/optimizer_results.csv",
        index=False
    )

    # ========================================================
    # TOP STRATEGIES
    # ========================================================

    print()
    print("=" * 60)
    print("TOP 20 STRATEGIES")
    print("=" * 60)

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

    valid_win_rate = (
        results_df[
            results_df["trades"] >= 50
        ]
        .sort_values(
            "win_rate",
            ascending=False
        )
    )

    print()
    print("=" * 60)
    print("BEST WIN RATE")
    print("=" * 60)

    if len(valid_win_rate):

        print(
            valid_win_rate[
                columns
            ]
            .head(10)
            .to_string(
                index=False
            )
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
    # BEST ~1 TRADE/WEEK
    # ========================================================

    frequency_match = (
        results_df[
            (
                results_df["trades_per_week"]
                >= 0.50
            )
            &
            (
                results_df["trades_per_week"]
                <= 1.50
            )
        ]
        .sort_values(
            [
                "win_rate",
                "total_r"
            ],
            ascending=False
        )
    )

    print()
    print("=" * 60)
    print("BEST STRATEGIES NEAR 1 TRADE/WEEK")
    print("=" * 60)

    if len(frequency_match):

        print(
            frequency_match[
                columns
            ]
            .head(20)
            .to_string(
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
        "Max cross bars:",
        best["max_cross"]
    )

    print(
        "Hours:",
        best["hours"]
    )

    print()
    print(
        "Full results saved to:"
    )

    print(
        "data/optimizer_results.csv"
    )

    print()
    print("=" * 60)
    print("OPTIMIZER COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
