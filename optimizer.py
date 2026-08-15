import os
import itertools
import numpy as np
import pandas as pd

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

RR = [0.50, 0.60, 0.75, 0.90, 1.00, 1.25, 1.50]
WICK = [0.20, 0.25, 0.30, 0.35, 0.40]
BODY = [0.20, 0.25, 0.30, 0.35, 0.40]
SEP = [0.0005, 0.0008, 0.0010, 0.0012, 0.0015]
CROSS = [15, 20, 25, 30, 40]

HOURS = [
    (2, 3),
    (3, 4),
    (4, 5),
    (2, 3, 4),
    (3, 4, 5),
    (2, 3, 4, 5),
    (2, 3, 4, 5, 12, 13),
    (3, 4, 5, 12, 13),
]

MIN_TRAIN = 30
MIN_RECENT = 20
RECENT_DAYS = 365


def load(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Missing file: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    time_col = next(
        (
            c
            for c in [
                "time",
                "Time",
                "timestamp",
                "Timestamp",
                "date",
                "Date",
            ]
            if c in df.columns
        ),
        None,
    )

    if time_col is None:
        raise RuntimeError("Missing time column")

    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col)

    rename = {}

    for c in df.columns:
        name = str(c).lower()

        if name == "open":
            rename[c] = "Open"
        elif name == "high":
            rename[c] = "High"
        elif name == "low":
            rename[c] = "Low"
        elif name == "close":
            rename[c] = "Close"

    df = df.rename(columns=rename)

    required = ["Open", "High", "Low", "Close"]

    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Missing {col} column")

    df = (
        df[required]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )

    df = df[~df.index.duplicated()].sort_index()

    return df


def prepare(df):
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)

    ema20 = (
        pd.Series(c)
        .ewm(span=20, adjust=False)
        .mean()
        .to_numpy()
    )

    ema50 = (
        pd.Series(c)
        .ewm(span=50, adjust=False)
        .mean()
        .to_numpy()
    )

    separation = np.divide(
        np.abs(ema20 - ema50),
        c,
        out=np.zeros_like(c),
        where=c != 0,
    )

    candle_range = h - l

    body_ratio = np.divide(
        np.abs(c - o),
        candle_range,
        out=np.zeros_like(c),
        where=candle_range > 0,
    )

    upper_wick_ratio = np.divide(
        h - np.maximum(o, c),
        candle_range,
        out=np.zeros_like(c),
        where=candle_range > 0,
    )

    cross_age = np.full(
        len(c),
        9999,
        dtype=np.int32,
    )

    last_cross = -9999

    for i in range(1, len(c)):
        if (
            ema20[i - 1] >= ema50[i - 1]
            and ema20[i] < ema50[i]
        ):
            last_cross = i

        if last_cross >= 0:
            cross_age[i] = i - last_cross

    previous_close = np.roll(c, 1)
    previous_ema = np.roll(ema20, 1)

    previous_distance = np.divide(
        np.abs(previous_close - previous_ema),
        previous_close,
        out=np.ones_like(c),
        where=previous_close != 0,
    )

    current_distance = np.divide(
        np.abs(c - ema20),
        c,
        out=np.ones_like(c),
        where=c != 0,
    )

    pullback = (
        (previous_distance <= 0.0020)
        | (current_distance <= 0.0020)
    )

    recent_high = (
        pd.Series(h)
        .rolling(8, min_periods=1)
        .max()
        .to_numpy()
    )

    return {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "ema20": ema20,
        "ema50": ema50,
        "ema20_slope": ema20 - np.roll(ema20, 4),
        "ema50_slope": ema50 - np.roll(ema50, 4),
        "separation": separation,
        "body_ratio": body_ratio,
        "upper_wick_ratio": upper_wick_ratio,
        "cross_age": cross_age,
        "pullback": pullback,
        "recent_high": recent_high,
        "hours": df.index.hour.to_numpy(),
    }


def get_signals(
    d,
    wick,
    body,
    separation,
    max_cross,
    hours,
):
    mask = (
        np.isin(d["hours"], hours)
        & (d["ema20"] < d["ema50"])
        & (d["ema20_slope"] < 0)
        & (d["ema50_slope"] < 0)
        & (d["separation"] >= separation)
        & (d["cross_age"] <= max_cross)
        & d["pullback"]
        & (d["close"] < d["open"])
        & (d["upper_wick_ratio"] >= wick)
        & (d["body_ratio"] >= body)
        & (d["close"] < d["ema20"])
    )

    return np.flatnonzero(mask)


def get_bounds(timestamps, start, end):
    start_idx = np.flatnonzero(
        timestamps >= pd.Timestamp(start, tz="UTC")
    )

    end_idx = np.flatnonzero(
        timestamps <= pd.Timestamp(end, tz="UTC")
    )

    if len(start_idx) == 0 or len(end_idx) == 0:
        return None

    return int(start_idx[0]), int(end_idx[-1])


def simulate(
    d,
    signal_indices,
    rr,
    start_idx,
    end_idx,
):
    results = []

    next_free = start_idx

    for i in signal_indices:

        if i < start_idx:
            continue

        if i > end_idx:
            break

        if i < next_free:
            continue

        entry = d["close"][i]

        stop_loss = d["recent_high"][i]

        risk = stop_loss - entry

        if risk <= 0:
            continue

        take_profit = entry - risk * rr

        result = None
        exit_index = None

        for j in range(i + 1, end_idx + 1):

            if d["high"][j] >= stop_loss:
                result = -1.0
                exit_index = j
                break

            if d["low"][j] <= take_profit:
                result = rr
                exit_index = j
                break

        if result is not None:
            results.append(result)
            next_free = exit_index + 1

    return results


def calculate_metrics(trades, days):

    if not trades:
        return None

    values = np.asarray(
        trades,
        dtype=float,
    )

    wins = values > 0
    losses = values < 0

    trade_count = len(values)

    win_count = int(wins.sum())
    loss_count = int(losses.sum())

    gross_profit = float(values[wins].sum())

    gross_loss = abs(
        float(values[losses].sum())
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit / gross_loss
        )
    else:
        profit_factor = 999.0

    equity = np.cumsum(values)

    peak = np.maximum.accumulate(equity)

    drawdown = float(
        (peak - equity).max()
    )

    weeks = max(
        days / 7.0,
        1e-9,
    )

    return {
        "trades": trade_count,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": (
            win_count / trade_count * 100
        ),
        "total_r": float(values.sum()),
        "profit_factor": profit_factor,
        "drawdown": drawdown,
        "trades_per_week": (
            trade_count / weeks
        ),
    }


def evaluate_period(
    d,
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

    start_idx, end_idx = bounds

    (
        rr,
        wick,
        body,
        separation,
        max_cross,
        hours,
    ) = params

    signals = get_signals(
        d,
        wick,
        body,
        separation,
        max_cross,
        hours,
    )

    trades = simulate(
        d,
        signals,
        rr,
        start_idx,
        end_idx,
    )

    days = max(
        (
            timestamps[end_idx]
            - timestamps[start_idx]
        ).total_seconds()
        / 86400,
        1,
    )

    return calculate_metrics(
        trades,
        days,
    )


def score_candidate(
    full,
    recent,
):

    if full is None or recent is None:
        return -1e12

    if full["trades"] < MIN_TRAIN:
        return -1e12

    if recent["trades"] < MIN_RECENT:
        return -1e12

    score = 0.0

    # Full-history robustness.
    score += full["win_rate"] * 2.0
    score += full["total_r"]
    score += (
        min(full["profit_factor"], 3.0)
        * 8.0
    )
    score -= full["drawdown"] * 1.5

    # Strong current-era weighting.
    score += recent["win_rate"] * 6.0
    score += recent["total_r"] * 2.5
    score += (
        min(recent["profit_factor"], 3.0)
        * 18.0
    )
    score -= recent["drawdown"] * 3.0

    # Prefer selective strategies.
    trades_per_week = (
        recent["trades_per_week"]
    )

    if 0.08 <= trades_per_week <= 0.35:
        score += 25.0

    elif trades_per_week <= 1.0:
        score += 10.0

    return score


def optimise(
    d,
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

    train_start_idx, train_end_idx = (
        training_bounds
    )

    train_end_date = pd.Timestamp(
        train_end,
        tz="UTC",
    )

    recent_start_date = max(
        pd.Timestamp(
            train_start,
            tz="UTC",
        ),
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

    recent_start_idx, recent_end_idx = (
        recent_bounds
    )

    full_days = max(
        (
            timestamps[train_end_idx]
            - timestamps[train_start_idx]
        ).total_seconds()
        / 86400,
        1,
    )

    recent_days = max(
        (
            timestamps[recent_end_idx]
            - timestamps[recent_start_idx]
        ).total_seconds()
        / 86400,
        1,
    )

    best = None

    combinations = itertools.product(
        RR,
        WICK,
        BODY,
        SEP,
        CROSS,
        HOURS,
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

        signals = get_signals(
            d,
            wick,
            body,
            separation,
            max_cross,
            hours,
        )

        full_trades = simulate(
            d,
            signals,
            rr,
            train_start_idx,
            train_end_idx,
        )

        recent_trades = simulate(
            d,
            signals,
            rr,
            recent_start_idx,
            recent_end_idx,
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
            or score > best["score"]
        ):
            best = {
                "score": score,
                "params": params,
                "full": full,
                "recent": recent,
            }

    return best


def run_market(
    market,
    path,
):

    print()
    print("=" * 60)
    print(f"{market} OPTIMIZER V4.1")
    print("=" * 60)

    df = load(path)

    d = prepare(df)

    timestamps = df.index.to_numpy()

    print(
        f"Candles: {len(df)}"
    )

    print(
        f"Range: {df.index.min()} -> "
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

        best = optimise(
            d,
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
        print("Parameters:")
        print(f"RR: {rr}")
        print(f"Wick: {wick}")
        print(f"Body: {body}")
        print(
            f"Separation: {separation}"
        )
        print(
            f"Max cross: {max_cross}"
        )
        print(
            "Hours: "
            + ",".join(
                map(str, hours)
            )
        )

        test = evaluate_period(
            d,
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
            f"Trades: {test['trades']}"
        )

        print(
            f"Wins: {test['wins']}"
        )

        print(
            f"Losses: {test['losses']}"
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
                "train_trades": full[
                    "trades"
                ],
                "train_win_rate": full[
                    "win_rate"
                ],
                "recent_trades": recent[
                    "trades"
                ],
                "recent_win_rate": recent[
                    "win_rate"
                ],
                "recent_r": recent[
                    "total_r"
                ],
                "test_trades": test[
                    "trades"
                ],
                "test_wins": test[
                    "wins"
                ],
                "test_losses": test[
                    "losses"
                ],
                "test_win_rate": test[
                    "win_rate"
                ],
                "test_r": test[
                    "total_r"
                ],
                "test_pf": test[
                    "profit_factor"
                ],
                "test_drawdown": test[
                    "drawdown"
                ],
                "test_trades_per_week": test[
                    "trades_per_week"
                ],
                "rr": rr,
                "wick": wick,
                "body": body,
                "separation": separation,
                "max_cross": max_cross,
                "hours": ",".join(
                    map(str, hours)
                ),
            }
        )

    output = pd.DataFrame(
        results
    )

    output.to_csv(
        f"data/"
        f"{market.lower()}_"
        f"optimizer_v4_1_results.csv",
        index=False,
    )

    return output


def main():

    print("=" * 60)
    print(
        "MULTI-MARKET STRATEGY "
        "OPTIMIZER V4.1"
    )
    print("=" * 60)

    print(
        "Current-era weighting: ENABLED"
    )

    print(
        f"Recent minimum sample: "
        f"{MIN_RECENT} trades"
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

    for market, path in MARKETS.items():

        try:

            all_results[market] = (
                run_market(
                    market,
                    path,
                )
            )

        except Exception as e:

            print()
            print("=" * 60)
            print(
                f"{market} FAILED"
            )
            print("=" * 60)

            print(
                f"{type(e).__name__}: "
                f"{e}"
            )

    summary_rows = []

    for market, df in all_results.items():

        if df.empty:
            continue

        total_trades = int(
            df["test_trades"].sum()
        )

        total_wins = int(
            df["test_wins"].sum()
        )

        total_r = float(
            df["test_r"].sum()
        )

        profitable_periods = int(
            (df["test_r"] > 0).sum()
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
                "market": market,
                "oos_trades": total_trades,
                "oos_win_rate": round(
                    win_rate,
                    2,
                ),
                "oos_total_r": round(
                    total_r,
                    2,
                ),
                "profitable_periods": (
                    f"{profitable_periods}"
                    f"/{periods}"
                ),
                "verdict": verdict,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print()
    print("=" * 60)
    print(
        "V4.1 FINAL MULTI-MARKET SUMMARY"
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
        "multi_market_optimizer_v4_1_summary.csv",
        index=False,
    )

    print()
    print("=" * 60)
    print("IMPORTANT")
    print("=" * 60)

    print(
        "These are out-of-sample "
        "backtest results."
    )

    print(
        "Do not implement live from "
        "this optimizer alone."
    )

    print()
    print(
        "Results saved:"
    )

    print(
        "data/"
        "xauusd_optimizer_v4_1_results.csv"
    )

    print(
        "data/"
        "eurusd_optimizer_v4_1_results.csv"
    )

    print(
        "data/"
        "multi_market_optimizer_v4_1_summary.csv"
    )

    print()
    print("=" * 60)
    print(
        "OPTIMIZER V4.1 COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
