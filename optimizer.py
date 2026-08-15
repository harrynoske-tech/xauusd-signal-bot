import os
import itertools
import numpy as np
import pandas as pd


# ============================================================
# MULTI-MARKET STRATEGY OPTIMIZER V4.2
# ============================================================

MARKETS = {
    "XAUUSD": "data/XAUUSD_15m.csv",
    "EURUSD": "data/EURUSD_15m.csv",
}

PERIODS = [
    (
        "2020-2023 -> 2024",
        "2020-01-01",
        "2023-12-31",
        "2024-01-01",
        "2024-12-31",
    ),
    (
        "2020-2024 -> 2025",
        "2020-01-01",
        "2024-12-31",
        "2025-01-01",
        "2025-12-31",
    ),
    (
        "2020-2025 -> 2026",
        "2020-01-01",
        "2025-12-31",
        "2026-01-01",
        "2026-08-14",
    ),
]


# Strategy search space
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


# Minimum sample requirements
MIN_TRAIN_TRADES = 30
MIN_RECENT_TRADES = 20

# Recent-era window
RECENT_DAYS = 365


# ============================================================
# TIMEZONE HELPER
# ============================================================

def utc_timestamp(value):
    """
    Convert any date-like value into a timezone-aware UTC Timestamp.

    Handles both:
    - timezone-naive timestamps
    - timezone-aware timestamps
    """

    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path):

    if not os.path.exists(path):
        raise RuntimeError(
            f"Missing data file: {path}"
        )

    df = pd.read_csv(path)

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
            "Missing time column"
        )

    df[time_column] = pd.to_datetime(
        df[time_column],
        utc=True,
    )

    df = df.set_index(
        time_column
    )

    rename = {}

    for column in df.columns:

        name = str(column).lower()

        if name == "open":
            rename[column] = "Open"

        elif name == "high":
            rename[column] = "High"

        elif name == "low":
            rename[column] = "Low"

        elif name == "close":
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
                f"Missing {column} column"
            )

    df = (
        df[required]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .dropna()
    )

    df = (
        df[
            ~df.index.duplicated(
                keep="first"
            )
        ]
        .sort_index()
    )

    return df


# ============================================================
# PREPARE INDICATORS
# ============================================================

def prepare_indicators(df):

    open_price = df["Open"].to_numpy(
        float
    )

    high = df["High"].to_numpy(
        float
    )

    low = df["Low"].to_numpy(
        float
    )

    close = df["Close"].to_numpy(
        float
    )

    ema20 = (
        pd.Series(close)
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
        .to_numpy()
    )

    ema50 = (
        pd.Series(close)
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
        .to_numpy()
    )

    ema20_slope = (
        ema20
        - np.roll(
            ema20,
            4,
        )
    )

    ema50_slope = (
        ema50
        - np.roll(
            ema50,
            4,
        )
    )

    separation = np.divide(
        np.abs(
            ema20 - ema50
        ),
        close,
        out=np.zeros_like(close),
        where=close != 0,
    )

    candle_range = (
        high - low
    )

    body_ratio = np.divide(
        np.abs(
            close - open_price
        ),
        candle_range,
        out=np.zeros_like(close),
        where=candle_range > 0,
    )

    upper_wick_ratio = np.divide(
        high
        - np.maximum(
            open_price,
            close,
        ),
        candle_range,
        out=np.zeros_like(close),
        where=candle_range > 0,
    )

    cross_age = np.full(
        len(close),
        9999,
        dtype=np.int32,
    )

    last_cross = -9999

    for i in range(
        1,
        len(close),
    ):

        bearish_cross = (
            ema20[i - 1]
            >= ema50[i - 1]
            and
            ema20[i]
            < ema50[i]
        )

        if bearish_cross:
            last_cross = i

        if last_cross >= 0:
            cross_age[i] = (
                i - last_cross
            )

    previous_close = np.roll(
        close,
        1,
    )

    previous_ema20 = np.roll(
        ema20,
        1,
    )

    previous_distance = np.divide(
        np.abs(
            previous_close
            - previous_ema20
        ),
        previous_close,
        out=np.ones_like(close),
        where=previous_close != 0,
    )

    current_distance = np.divide(
        np.abs(
            close - ema20
        ),
        close,
        out=np.ones_like(close),
        where=close != 0,
    )

    pullback = (
        (
            previous_distance
            <= 0.0020
        )
        |
        (
            current_distance
            <= 0.0020
        )
    )

    recent_high = (
        pd.Series(high)
        .rolling(
            8,
            min_periods=1,
        )
        .max()
        .to_numpy()
    )

    return {
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,

        "ema20": ema20,
        "ema50": ema50,

        "ema20_slope": ema20_slope,
        "ema50_slope": ema50_slope,

        "separation": separation,

        "body_ratio": body_ratio,
        "upper_wick_ratio":
            upper_wick_ratio,

        "cross_age": cross_age,

        "pullback": pullback,

        "recent_high": recent_high,

        "hours":
            df.index.hour.to_numpy(),
    }


# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_signals(
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
            hours,
        )

        & (
            data["ema20"]
            < data["ema50"]
        )

        & (
            data["ema20_slope"]
            < 0
        )

        & (
            data["ema50_slope"]
            < 0
        )

        & (
            data["separation"]
            >= separation
        )

        & (
            data["cross_age"]
            <= max_cross
        )

        & data["pullback"]

        & (
            data["close"]
            < data["open"]
        )

        & (
            data[
                "upper_wick_ratio"
            ]
            >= wick
        )

        & (
            data["body_ratio"]
            >= body
        )

        & (
            data["close"]
            < data["ema20"]
        )
    )

    return np.flatnonzero(
        mask
    )


# ============================================================
# PERIOD BOUNDS
# ============================================================

def get_bounds(
    timestamps,
    start,
    end,
):

    start_ts = utc_timestamp(
        start
    )

    end_ts = utc_timestamp(
        end
    )

    start_indices = np.flatnonzero(
        timestamps >= start_ts
    )

    end_indices = np.flatnonzero(
        timestamps <= end_ts
    )

    if (
        len(start_indices) == 0
        or len(end_indices) == 0
    ):
        return None

    return (
        int(start_indices[0]),
        int(end_indices[-1]),
    )


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate(
    data,
    signal_indices,
    rr,
    start_index,
    end_index,
):

    results = []

    next_free = start_index

    for i in signal_indices:

        if i < start_index:
            continue

        if i > end_index:
            break

        if i < next_free:
            continue

        entry = data[
            "close"
        ][i]

        stop_loss = data[
            "recent_high"
        ][i]

        risk = (
            stop_loss
            - entry
        )

        if risk <= 0:
            continue

        take_profit = (
            entry
            - risk * rr
        )

        result = None
        exit_index = None

        for j in range(
            i + 1,
            end_index + 1,
        ):

            if (
                data["high"][j]
                >= stop_loss
            ):

                result = -1.0
                exit_index = j

                break

            if (
                data["low"][j]
                <= take_profit
            ):

                result = rr
                exit_index = j

                break

        if result is not None:

            results.append(
                result
            )

            next_free = (
                exit_index + 1
            )

    return results


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    trades,
    days,
):

    if not trades:
        return None

    values = np.asarray(
        trades,
        dtype=float,
    )

    wins = values > 0
    losses = values < 0

    trade_count = len(
        values
    )

    win_count = int(
        wins.sum()
    )

    loss_count = int(
        losses.sum()
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
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = 999.0

    equity = np.cumsum(
        values
    )

    peak = np.maximum.accumulate(
        equity
    )

    drawdown = float(
        (
            peak - equity
        ).max()
    )

    weeks = max(
        days / 7.0,
        1e-9,
    )

    return {
        "trades": trade_count,

        "wins": win_count,

        "losses": loss_count,

        "win_rate":
            win_count
            / trade_count
            * 100,

        "total_r":
            float(
                values.sum()
            ),

        "profit_factor":
            profit_factor,

        "drawdown":
            drawdown,

        "trades_per_week":
            trade_count
            / weeks,
    }


# ============================================================
# EVALUATE PERIOD
# ============================================================

def evaluate_period(
    data,
    timestamps,
    params,
    start,
    end,
):

    bounds = get_bounds(
        timestamps,
        start,
        end,
    )

    if bounds is None:
        return None

    start_index, end_index = (
        bounds
    )

    (
        rr,
        wick,
        body,
        separation,
        max_cross,
        hours,
    ) = params

    signals = generate_signals(
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
        start_index,
        end_index,
    )

    days = max(
        (
            timestamps[end_index]
            - timestamps[start_index]
        ).total_seconds()
        / 86400,
        1,
    )

    return calculate_metrics(
        trades,
        days,
    )


# ============================================================
# CANDIDATE SCORING
# ============================================================

def score_candidate(
    full,
    recent,
):

    if (
        full is None
        or recent is None
    ):
        return -1e12

    if (
        full["trades"]
        < MIN_TRAIN_TRADES
    ):
        return -1e12

    if (
        recent["trades"]
        < MIN_RECENT_TRADES
    ):
        return -1e12

    score = 0.0

    # Historical robustness

    score += (
        full["win_rate"]
        * 2.0
    )

    score += (
        full["total_r"]
    )

    score += (
        min(
            full[
                "profit_factor"
            ],
            3.0,
        )
        * 8.0
    )

    score -= (
        full["drawdown"]
        * 1.5
    )

    # Current-era weighting

    score += (
        recent["win_rate"]
        * 6.0
    )

    score += (
        recent["total_r"]
        * 2.5
    )

    score += (
        min(
            recent[
                "profit_factor"
            ],
            3.0,
        )
        * 18.0
    )

    score -= (
        recent["drawdown"]
        * 3.0
    )

    # Prefer selective strategies.

    trades_per_week = (
        recent[
            "trades_per_week"
        ]
    )

    if (
        0.08
        <= trades_per_week
        <= 0.35
    ):

        score += 25.0

    elif trades_per_week <= 1.0:

        score += 10.0

    return score


# ============================================================
# OPTIMIZE
# ============================================================

def optimize(
    data,
    timestamps,
    train_start,
    train_end,
):

    training_bounds = get_bounds(
        timestamps,
        train_start,
        train_end,
    )

    if training_bounds is None:
        return None

    train_start_index, train_end_index = (
        training_bounds
    )

    train_end_date = utc_timestamp(
        train_end
    )

    train_start_date = utc_timestamp(
        train_start
    )

    recent_start_date = max(
        train_start_date,
        train_end_date
        - pd.Timedelta(
            days=RECENT_DAYS
        ),
    )

    recent_bounds = get_bounds(
        timestamps,
        recent_start_date,
        train_end_date,
    )

    if recent_bounds is None:
        return None

    recent_start_index, recent_end_index = (
        recent_bounds
    )

    full_days = max(
        (
            timestamps[
                train_end_index
            ]
            - timestamps[
                train_start_index
            ]
        ).total_seconds()
        / 86400,
        1,
    )

    recent_days = max(
        (
            timestamps[
                recent_end_index
            ]
            - timestamps[
                recent_start_index
            ]
        ).total_seconds()
        / 86400,
        1,
    )

    best = None

    combinations = itertools.product(
        RR_VALUES,
        WICK_VALUES,
        BODY_VALUES,
        SEPARATION_VALUES,
        MAX_CROSS_VALUES,
        HOUR_SETS,
    )

    for params in combinations:

        (
            rr,
            wick,
            body,
            separation,
            max_cross,
            hours,
        ) = params

        signals = generate_signals(
            data,
            wick,
            body,
            separation,
            max_cross,
            hours,
        )

        full_trades = simulate(
            data,
            signals,
            rr,
            train_start_index,
            train_end_index,
        )

        recent_trades = simulate(
            data,
            signals,
            rr,
            recent_start_index,
            recent_end_index,
        )

        full = calculate_metrics(
            full_trades,
            full_days,
        )

        recent = calculate_metrics(
            recent_trades,
            recent_days,
        )

        score = score_candidate(
            full,
            recent,
        )

        if (
            best is None
            or score
            > best["score"]
        ):

            best = {
                "score": score,
                "params": params,
                "full": full,
                "recent": recent,
            }

    return best


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
        f"{market} OPTIMIZER V4.2"
    )
    print("=" * 60)

    df = load_data(
        path
    )

    data = prepare_indicators(
        df
    )

    timestamps = (
        df.index.to_numpy()
    )

    print(
        f"Candles: {len(df)}"
    )

    print(
        f"Range: "
        f"{df.index.min()} -> "
        f"{df.index.max()}"
    )

    results = []

    for (
        period_name,
        train_start,
        train_end,
        test_start,
        test_end,
    ) in PERIODS:

        print()
        print("=" * 60)
        print(
            f"{market} WALK-FORWARD: "
            f"{period_name}"
        )
        print("=" * 60)

        print(
            "Optimising training period..."
        )

        best = optimize(
            data,
            timestamps,
            train_start,
            train_end,
        )

        if best is None:

            print(
                "NO VALID TRAINING STRATEGY"
            )

            continue

        (
            rr,
            wick,
            body,
            separation,
            max_cross,
            hours,
        ) = best["params"]

        full = best["full"]
        recent = best["recent"]

        print()
        print(
            "BEST TRAINING STRATEGY"
        )
        print("-" * 60)

        print(
            f"Training: "
            f"{full['trades']} trades | "
            f"{full['win_rate']:.2f}% | "
            f"{full['total_r']:.2f}R"
        )

        print(
            f"Recent {RECENT_DAYS}d: "
            f"{recent['trades']} trades | "
            f"{recent['win_rate']:.2f}% | "
            f"{recent['total_r']:.2f}R | "
            f"PF {recent['profit_factor']:.2f}"
        )

        print()
        print(
            "Parameters:"
        )

        print(
            f"RR: {rr}"
        )

        print(
            f"Wick: {wick}"
        )

        print(
            f"Body: {body}"
        )

        print(
            f"Separation: "
            f"{separation}"
        )

        print(
            f"Max cross: "
            f"{max_cross}"
        )

        print(
            "Hours: "
            + ",".join(
                map(
                    str,
                    hours,
                )
            )
        )

        test = evaluate_period(
            data,
            timestamps,
            best["params"],
            test_start,
            test_end,
        )

        print()
        print(
            "OUT-OF-SAMPLE RESULT"
        )
        print("-" * 60)

        if test is None:

            print(
                "No out-of-sample trades."
            )

            continue

        print(
            f"Trades: "
            f"{test['trades']}"
        )

        print(
            f"Wins: "
            f"{test['wins']}"
        )

        print(
            f"Losses: "
            f"{test['losses']}"
        )

        print(
            f"Win rate: "
            f"{test['win_rate']:.2f}%"
        )

        print(
            f"Total R: "
            f"{test['total_r']:.2f}"
        )

        print(
            f"Profit factor: "
            f"{test['profit_factor']:.2f}"
        )

        print(
            f"Max drawdown: "
            f"{test['drawdown']:.2f}R"
        )

        print(
            f"Trades/week: "
            f"{test['trades_per_week']:.2f}"
        )

        results.append(
            {
                "market": market,
                "period": period_name,

                "train_trades":
                    full["trades"],

                "train_win_rate":
                    full["win_rate"],

                "recent_trades":
                    recent["trades"],

                "recent_win_rate":
                    recent["win_rate"],

                "recent_r":
                    recent["total_r"],

                "test_trades":
                    test["trades"],

                "test_wins":
                    test["wins"],

                "test_losses":
                    test["losses"],

                "test_win_rate":
                    test["win_rate"],

                "test_r":
                    test["total_r"],

                "test_pf":
                    test[
                        "profit_factor"
                    ],

                "test_drawdown":
                    test["drawdown"],

                "test_trades_per_week":
                    test[
                        "trades_per_week"
                    ],

                "rr": rr,

                "wick": wick,

                "body": body,

                "separation":
                    separation,

                "max_cross":
                    max_cross,

                "hours":
                    ",".join(
                        map(
                            str,
                            hours,
                        )
                    ),
            }
        )

    output = pd.DataFrame(
        results
    )

    output.to_csv(
        f"data/"
        f"{market.lower()}_"
        f"optimizer_v4_2_results.csv",
        index=False,
    )

    return output


# ============================================================
# FINAL SUMMARY
# ============================================================

def main():

    print("=" * 60)

    print(
        "MULTI-MARKET STRATEGY "
        "OPTIMIZER V4.2"
    )

    print("=" * 60)

    print(
        "Current-era weighting: ENABLED"
    )

    print(
        f"Recent minimum sample: "
        f"{MIN_RECENT_TRADES} trades"
    )

    print(
        "Markets: "
        + ", ".join(
            MARKETS.keys()
        )
    )

    print(
        "NO LIVE TRADING"
    )

    all_results = {}

    for market, path in (
        MARKETS.items()
    ):

        try:

            all_results[market] = (
                run_market(
                    market,
                    path,
                )
            )

        except Exception as error:

            print()
            print("=" * 60)

            print(
                f"{market} FAILED"
            )

            print("=" * 60)

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

    summary_rows = []

    for market, df in (
        all_results.items()
    ):

        if df.empty:
            continue

        total_trades = int(
            df[
                "test_trades"
            ].sum()
        )

        total_wins = int(
            df[
                "test_wins"
            ].sum()
        )

        total_r = float(
            df[
                "test_r"
            ].sum()
        )

        profitable_periods = int(
            (
                df[
                    "test_r"
                ]
                > 0
            ).sum()
        )

        periods = len(df)

        if total_trades > 0:

            win_rate = (
                total_wins
                / total_trades
                * 100
            )

        else:

            win_rate = 0.0

        if (
            total_trades >= 20
            and total_r > 0
            and win_rate >= 55
            and profitable_periods >= 2
        ):

            verdict = "ROBUST"

        elif (
            total_trades >= 15
            and total_r >= 0
            and win_rate >= 50
        ):

            verdict = "PROMISING"

        else:

            verdict = "NOT ROBUST"

        summary_rows.append(
            {
                "market":
                    market,

                "oos_trades":
                    total_trades,

                "oos_win_rate":
                    round(
                        win_rate,
                        2,
                    ),

                "oos_total_r":
                    round(
                        total_r,
                        2,
                    ),

                "profitable_periods":
                    f"{profitable_periods}"
                    f"/{periods}",

                "verdict":
                    verdict,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print()
    print("=" * 60)

    print(
        "V4.2 FINAL "
        "MULTI-MARKET SUMMARY"
    )

    print("=" * 60)

    if not summary.empty:

        print(
            summary.to_string(
                index=False
            )
        )

    else:

        print(
            "No completed market results."
        )

    summary.to_csv(
        "data/"
        "multi_market_optimizer_v4_2_summary.csv",
        index=False,
    )

    print()
    print("=" * 60)

    print(
        "IMPORTANT"
    )

    print("=" * 60)

    print(
        "These are out-of-sample "
        "backtest results."
    )

    print(
        "Do not implement live "
        "from this optimizer alone."
    )

    print()
    print(
        "Results saved:"
    )

    print(
        "data/"
        "xauusd_optimizer_v4_2_results.csv"
    )

    print(
        "data/"
        "eurusd_optimizer_v4_2_results.csv"
    )

    print(
        "data/"
        "multi_market_optimizer_v4_2_summary.csv"
    )

    print()
    print("=" * 60)

    print(
        "OPTIMIZER V4.2 COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
