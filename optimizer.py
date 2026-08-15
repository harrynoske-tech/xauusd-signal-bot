import os
import pandas as pd
import numpy as np
from itertools import product


# ============================================================
# XAUUSD OPTIMIZER V3
#
# WALK-FORWARD / OUT-OF-SAMPLE TESTING
#
# We optimise on one period, then test those exact parameters
# on completely unseen future data.
#
# NO LIVE TRADING.
# ============================================================

DATA_FILE = "data/XAUUSD_15m.csv"
OUTPUT_FILE = "data/optimizer_v3_results.csv"


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
]

BODY_VALUES = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
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
    (2, 3, 4, 5, 12, 13),
    (3, 4, 5, 12, 13),
]


# ============================================================
# WALK-FORWARD PERIODS
# ============================================================

PERIODS = [
    {
        "name": "2020-2023 -> 2024",
        "train_start": "2020-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
    },
    {
        "name": "2020-2024 -> 2025",
        "train_start": "2020-01-01",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
    },
    {
        "name": "2020-2025 -> 2026",
        "train_start": "2020-01-01",
        "train_end": "2025-12-31",
        "test_start": "2026-01-01",
        "test_end": "2026-08-14",
    },
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("XAUUSD OPTIMIZER V3")
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

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
    ]:

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
# PREPARE INDICATORS
# ============================================================

def prepare_data(df):

    print(
        "Preparing indicators..."
    )

    close = df["Close"].astype(float)
    open_price = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    close_np = close.to_numpy()
    open_np = open_price.to_numpy()
    high_np = high.to_numpy()
    low_np = low.to_numpy()

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

    separation = (
        np.abs(
            ema20 - ema50
        )
        / close_np
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
    # STOP
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
        "upper_wick_ratio":
            upper_wick_ratio,
        "hours": hours,
        "cross_age": cross_age,
        "pullback": pullback,
        "recent_high": recent_high,
    }


# ============================================================
# BUILD SIGNALS
# ============================================================

def build_signals(
    data,
    wick,
    body,
    separation,
    max_cross,
    hours,
):

    mask = (
        np.isin(
            data["hours"],
            np.array(hours)
        )
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
# SIMULATE TRADES
# ============================================================

def simulate(
    data,
    indices,
    rr,
    start_index,
    end_index,
):

    close = data["close"]
    high = data["high"]
    low = data["low"]
    recent_high = data["recent_high"]

    trades = []

    next_available = (
        start_index
    )

    for index in indices:

        if index < start_index:
            continue

        if index > end_index:
            break

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

        search_end = min(
            end_index,
            len(close) - 1
        )

        for j in range(
            index + 1,
            search_end + 1
        ):

            # Conservative assumption:
            # SL gets priority if both
            # levels occur in the candle.

            if high[j] >= stop:

                result = -1.0
                exit_index = j

                break

            if low[j] <= target:

                result = rr
                exit_index = j

                break

        if result is None:

            # Do not manufacture a result
            # if the trade is still open
            # at the end of the test window.

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

def metrics(
    trades,
    days
):

    if not trades:

        return None

    trades = np.asarray(
        trades,
        dtype=float
    )

    wins = (
        trades > 0
    )

    losses = (
        trades < 0
    )

    total = len(
        trades
    )

    win_count = int(
        wins.sum()
    )

    loss_count = int(
        losses.sum()
    )

    win_rate = (
        win_count
        / total
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

    peak = np.maximum.accumulate(
        equity
    )

    drawdown = (
        peak - equity
    )

    max_drawdown = float(
        drawdown.max()
    )

    weeks = (
        days / 7
    )

    trades_per_week = (
        total / weeks
    )

    return {
        "trades": total,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": win_rate,
        "total_r": total_r,
        "profit_factor": profit_factor,
        "max_drawdown":
            max_drawdown,
        "trades_per_week":
            trades_per_week,
    }


# ============================================================
# TRAINING SCORE
# ============================================================

def training_score(
    m
):

    if m is None:
        return -999999

    if m["trades"] < 30:
        return -999999

    score = 0

    # Strong preference for win rate.
    score += (
        m["win_rate"] * 10
    )

    # Reward positive expectancy.
    score += (
        m["total_r"] * 2
    )

    # Reward PF.
    score += (
        min(
            m["profit_factor"],
            3
        ) * 10
    )

    # Penalise drawdown.
    score -= (
        m["max_drawdown"] * 2
    )

    # Prefer useful frequency,
    # but don't force exactly 1/week.
    frequency = (
        m["trades_per_week"]
    )

    if (
        0.25
        <= frequency
        <= 1.50
    ):

        score += 40

    elif (
        0.15
        <= frequency
        <= 2.00
    ):

        score += 15

    return score


# ============================================================
# FIND BEST PARAMETERS ON TRAINING DATA
# ============================================================

def optimise_training(
    data,
    train_start,
    train_end,
    timestamps,
):

    train_start_index = (
        timestamps
        >= train_start
    )

    train_end_index = (
        timestamps
        <= train_end
    )

    start = np.flatnonzero(
        train_start_index
    )[0]

    end = np.flatnonzero(
        train_end_index
    )[-1]

    train_days = (
        timestamps[end]
        - timestamps[start]
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

    best = None

    for params in combinations:

        (
            rr,
            wick,
            body,
            separation,
            max_cross,
            hours,
        ) = params

        signals = build_signals(
            data,
            wick,
            body,
            separation,
            max_cross,
            hours,
        )

        trades = simulate(
            data,
            signals,
            rr,
            start,
            end,
        )

        m = metrics(
            trades,
            train_days
        )

        if m is None:
            continue

        score = training_score(
            m
        )

        if (
            best is None
            or score
            > best["score"]
        ):

            best = {
                "score": score,
                "rr": rr,
                "wick": wick,
                "body": body,
                "separation":
                    separation,
                "max_cross":
                    max_cross,
                "hours":
                    hours,
                "metrics": m,
            }

    return best


# ============================================================
# RUN ONE WALK-FORWARD PERIOD
# ============================================================

def run_period(
    period,
    data,
    timestamps,
):

    print()
    print("=" * 60)
    print(
        "WALK-FORWARD:",
        period["name"]
    )
    print("=" * 60)

    train_start = pd.Timestamp(
        period["train_start"],
        tz="UTC"
    )

    train_end = pd.Timestamp(
        period["train_end"],
        tz="UTC"
    )

    test_start = pd.Timestamp(
        period["test_start"],
        tz="UTC"
    )

    test_end = pd.Timestamp(
        period["test_end"],
        tz="UTC"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print("Optimising training period...")

    best = optimise_training(
        data,
        train_start,
        train_end,
        timestamps,
    )

    if best is None:

        print(
            "No valid training strategy."
        )

        return None

    print()
    print("BEST TRAINING STRATEGY")
    print("-" * 60)

    print(
        "Win rate:",
        f"{best['metrics']['win_rate']:.2f}%"
    )

    print(
        "Trades:",
        best["metrics"]["trades"]
    )

    print(
        "Total R:",
        f"{best['metrics']['total_r']:.2f}"
    )

    print(
        "Trades/week:",
        f"{best['metrics']['trades_per_week']:.2f}"
    )

    print()
    print("Parameters:")

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
        "Separation:",
        best["separation"]
    )

    print(
        "Max cross:",
        best["max_cross"]
    )

    print(
        "Hours:",
        ",".join(
            str(x)
            for x in best["hours"]
        )
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_start_mask = (
        timestamps
        >= test_start
    )

    test_end_mask = (
        timestamps
        <= test_end
    )

    test_start_index = np.flatnonzero(
        test_start_mask
    )[0]

    test_end_index = np.flatnonzero(
        test_end_mask
    )[-1]

    test_days = (
        timestamps[test_end_index]
        - timestamps[test_start_index]
    ).total_seconds() / 86400

    signals = build_signals(
        data,
        best["wick"],
        best["body"],
        best["separation"],
        best["max_cross"],
        best["hours"],
    )

    test_trades = simulate(
        data,
        signals,
        best["rr"],
        test_start_index,
        test_end_index,
    )

    test_metrics = metrics(
        test_trades,
        test_days
    )

    if test_metrics is None:

        print()
        print(
            "NO OUT-OF-SAMPLE TRADES"
        )

        return {
            "period": period["name"],
            "train_win_rate":
                best["metrics"]["win_rate"],
            "train_trades":
                best["metrics"]["trades"],
            "test_trades": 0,
            "test_win_rate": 0,
            "test_total_r": 0,
            "test_profit_factor": 0,
            "test_drawdown": 0,
            "test_trades_per_week": 0,
            "rr": best["rr"],
            "wick": best["wick"],
            "body": best["body"],
            "separation":
                best["separation"],
            "max_cross":
                best["max_cross"],
            "hours":
                ",".join(
                    str(x)
                    for x in best["hours"]
                ),
        }

    print()
    print("OUT-OF-SAMPLE RESULT")
    print("-" * 60)

    print(
        "Win rate:",
        f"{test_metrics['win_rate']:.2f}%"
    )

    print(
        "Trades:",
        test_metrics["trades"]
    )

    print(
        "Total R:",
        f"{test_metrics['total_r']:.2f}"
    )

    print(
        "Profit factor:",
        f"{test_metrics['profit_factor']:.2f}"
    )

    print(
        "Max drawdown:",
        f"{test_metrics['max_drawdown']:.2f}R"
    )

    print(
        "Trades/week:",
        f"{test_metrics['trades_per_week']:.2f}"
    )

    return {
        "period": period["name"],

        "train_win_rate":
            best["metrics"]["win_rate"],

        "train_trades":
            best["metrics"]["trades"],

        "train_total_r":
            best["metrics"]["total_r"],

        "test_trades":
            test_metrics["trades"],

        "test_wins":
            test_metrics["wins"],

        "test_losses":
            test_metrics["losses"],

        "test_win_rate":
            test_metrics["win_rate"],

        "test_total_r":
            test_metrics["total_r"],

        "test_profit_factor":
            test_metrics["profit_factor"],

        "test_drawdown":
            test_metrics["max_drawdown"],

        "test_trades_per_week":
            test_metrics[
                "trades_per_week"
            ],

        "rr":
            best["rr"],

        "wick":
            best["wick"],

        "body":
            best["body"],

        "separation":
            best["separation"],

        "max_cross":
            best["max_cross"],

        "hours":
            ",".join(
                str(x)
                for x in best["hours"]
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    data = prepare_data(
        df
    )

    timestamps = (
        df.index.to_numpy()
    )

    print()
    print("=" * 60)
    print("WALK-FORWARD TESTING")
    print("=" * 60)

    print(
        "Periods:",
        len(PERIODS)
    )

    results = []

    for period in PERIODS:

        result = run_period(
            period,
            data,
            timestamps,
        )

        if result is not None:

            results.append(
                result
            )

    if not results:

        raise RuntimeError(
            "No walk-forward results."
        )

    results_df = pd.DataFrame(
        results
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
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("WALK-FORWARD SUMMARY")
    print("=" * 60)

    summary_columns = [
        "period",
        "train_trades",
        "train_win_rate",
        "test_trades",
        "test_win_rate",
        "test_total_r",
        "test_profit_factor",
        "test_drawdown",
        "test_trades_per_week",
    ]

    print(
        results_df[
            summary_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # COMBINED OUT-OF-SAMPLE
    # ========================================================

    total_test_trades = int(
        results_df[
            "test_trades"
        ].sum()
    )

    total_test_wins = int(
        results_df[
            "test_wins"
        ].sum()
    )

    total_test_losses = int(
        results_df[
            "test_losses"
        ].sum()
    )

    total_test_r = float(
        results_df[
            "test_total_r"
        ].sum()
    )

    if total_test_trades > 0:

        combined_win_rate = (
            total_test_wins
            / total_test_trades
            * 100
        )

    else:

        combined_win_rate = 0

    gross_profit = 0
    gross_loss = 0

    for _, row in results_df.iterrows():

        wins = row[
            "test_wins"
        ]

        losses = row[
            "test_losses"
        ]

        # Reconstruct approximate gross
        # profit/loss from fixed RR.

        rr = row["rr"]

        gross_profit += (
            wins * rr
        )

        gross_loss += (
            losses * 1.0
        )

    if gross_loss > 0:

        combined_pf = (
            gross_profit
            / gross_loss
        )

    else:

        combined_pf = 999

    print()
    print("=" * 60)
    print("COMBINED OUT-OF-SAMPLE RESULT")
    print("=" * 60)

    print(
        "Total test trades:",
        total_test_trades
    )

    print(
        "Wins:",
        total_test_wins
    )

    print(
        "Losses:",
        total_test_losses
    )

    print(
        "Combined win rate:",
        f"{combined_win_rate:.2f}%"
    )

    print(
        "Combined total R:",
        f"{total_test_r:.2f}"
    )

    print(
        "Approx profit factor:",
        f"{combined_pf:.2f}"
    )

    # ========================================================
    # VERDICT
    # ========================================================

    print()
    print("=" * 60)
    print("ROBUSTNESS VERDICT")
    print("=" * 60)

    profitable_periods = int(
        (
            results_df[
                "test_total_r"
            ] > 0
        ).sum()
    )

    positive_win_periods = int(
        (
            results_df[
                "test_win_rate"
            ] >= 50
        ).sum()
    )

    print(
        "Profitable test periods:",
        f"{profitable_periods}/"
        f"{len(results_df)}"
    )

    print(
        "Test periods >=50% win rate:",
        f"{positive_win_periods}/"
        f"{len(results_df)}"
    )

    if (
        combined_win_rate >= 70
        and
        total_test_r > 0
        and
        profitable_periods
        == len(results_df)
    ):

        print()
        print(
            "VERDICT: STRONG"
        )

        print(
            "The strategy has passed a"
            " preliminary walk-forward test."
        )

    elif (
        combined_win_rate >= 60
        and
        total_test_r > 0
    ):

        print()
        print(
            "VERDICT: PROMISING"
        )

        print(
            "The strategy shows an edge,"
            " but needs further refinement."
        )

    else:

        print()
        print(
            "VERDICT: NOT ROBUST YET"
        )

        print(
            "Do not implement this version live."
        )

    print()
    print(
        "Results saved to:",
        OUTPUT_FILE
    )

    print()
    print("=" * 60)
    print("OPTIMIZER V3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
