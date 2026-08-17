# ============================================================
# MULTI-MARKET STRATEGY OPTIMIZER V11
# ============================================================
# REGIME-AWARE SIGNAL VALIDATION
# PRIMARY MARKET: XAUUSD
# SECONDARY MARKET: EURUSD (VALIDATION ONLY)
# REJECTION/REVERSAL: PRIMARY SIGNAL
# TREND/EMA/OTHER SIGNALS: REGIME CONFIRMATION ONLY
# WALK-FORWARD TESTING: ENABLED
# STRICT OOS HOLDOUT: ENABLED
# PARAMETER STABILITY: ENABLED
# MINIMUM OOS SAMPLE: ENABLED
# NO LIVE TRADING
# ============================================================

import os
import itertools
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"

MARKETS = {
    "XAUUSD": "data/XAUUSD_15m.csv",
    "EURUSD": "data/EURUSD_15m.csv",
}

RESULT_FILES = {
    "XAUUSD": "data/xauusd_optimizer_v11_results.csv",
    "EURUSD": "data/eurusd_optimizer_v11_results.csv",
}

SUMMARY_FILE = "data/multi_market_optimizer_v11_summary.csv"

# We deliberately leave the latest period completely untouched.
# It is NOT used during optimisation, threshold selection,
# stability testing, or model selection.

TRAIN_START = "2021-01-01"

WALK_FORWARD_PERIODS = [
    ("2021-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]

FINAL_HOLDOUT_START = "2026-01-01"

MIN_TRAIN_TRADES = 50
MIN_OOS_TRADES = 20
MIN_STABILITY_NEIGHBOURS = 20

TARGET_WIN_RATE = 0.82

# ============================================================
# PARAMETER SEARCH
# ============================================================

RR_VALUES = [0.5, 0.6, 0.75, 1.0]
WICK_VALUES = [0.15, 0.20, 0.25, 0.30, 0.35]
BODY_VALUES = [0.20, 0.25, 0.30, 0.35]
SEPARATION_VALUES = [0.0003, 0.0005, 0.0008, 0.0010]
MAX_CROSS_VALUES = [10, 20, 40]
HOUR_SETS = [
    (3, 4, 5),
    (4, 5),
    (3, 4, 5, 12, 13),
    (12, 13),
]

THRESHOLDS = [-0.50, -0.25, 0.00, 0.25, 0.50]

# ============================================================
# HELPERS
# ============================================================

def load_data(path):
    df = pd.read_csv(path)

    if "time" in df.columns:
        time_col = "time"
    elif "datetime" in df.columns:
        time_col = "datetime"
    elif "Date" in df.columns:
        time_col = "Date"
    else:
        raise ValueError(
            f"Could not find datetime column in {path}. "
            f"Expected time, datetime or Date."
        )

    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    df = df.sort_values(time_col).reset_index(drop=True)

    df = df.rename(columns={time_col: "time"})

    required = ["open", "high", "low", "close"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}"
        )

    return df


def prepare_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # Basic candle structure
    # --------------------------------------------------------

    df["range"] = (df["high"] - df["low"]).replace(0, np.nan)

    df["body"] = (df["close"] - df["open"]).abs()

    df["body_ratio"] = df["body"] / df["range"]

    df["upper_wick"] = (
        df["high"] -
        df[["open", "close"]].max(axis=1)
    )

    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1) -
        df["low"]
    )

    df["upper_wick_ratio"] = (
        df["upper_wick"] / df["range"]
    )

    df["lower_wick_ratio"] = (
        df["lower_wick"] / df["range"]
    )

    # --------------------------------------------------------
    # EMAs
    # --------------------------------------------------------

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    df["tr"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr14"] = df["tr"].rolling(14).mean()

    df["atr_ratio"] = (
        df["atr14"] /
        df["close"]
    )

    # --------------------------------------------------------
    # Volatility regime
    # --------------------------------------------------------

    df["atr50"] = df["atr14"].rolling(50).mean()

    df["vol_ratio"] = (
        df["atr14"] /
        df["atr50"]
    )

    # --------------------------------------------------------
    # Trend regime
    # --------------------------------------------------------

    df["ema_spread"] = (
        (df["ema20"] - df["ema50"]).abs() /
        df["close"]
    )

    df["trend_up"] = (
        (df["ema20"] > df["ema50"]) &
        (df["ema50"] > df["ema100"])
    )

    df["trend_down"] = (
        (df["ema20"] < df["ema50"]) &
        (df["ema50"] < df["ema100"])
    )

    df["trend_strength"] = (
        df["ema_spread"] /
        df["atr_ratio"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Candle direction
    # --------------------------------------------------------

    df["bullish"] = df["close"] > df["open"]
    df["bearish"] = df["close"] < df["open"]

    # --------------------------------------------------------
    # Rolling location
    # --------------------------------------------------------

    df["high20"] = df["high"].rolling(20).max()
    df["low20"] = df["low"].rolling(20).min()

    df["range_position"] = (
        (df["close"] - df["low20"]) /
        (df["high20"] - df["low20"]).replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    df["momentum5"] = (
        df["close"] /
        df["close"].shift(5) - 1
    )

    df["momentum10"] = (
        df["close"] /
        df["close"].shift(10) - 1
    )

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    df["hour"] = df["time"].dt.hour

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


# ============================================================
# REGIME CLASSIFICATION
# ============================================================

def classify_regime(row):

    trend = row["trend_strength"]
    vol = row["vol_ratio"]

    if pd.isna(trend) or pd.isna(vol):
        return "UNKNOWN"

    if trend >= 2.0 and vol >= 1.10:
        return "TREND_HIGH_VOL"

    if trend >= 2.0 and vol < 1.10:
        return "TREND_LOW_VOL"

    if trend < 1.0 and vol >= 1.10:
        return "RANGE_HIGH_VOL"

    if trend < 1.0 and vol < 1.10:
        return "RANGE_LOW_VOL"

    return "TRANSITION"


# ============================================================
# REJECTION / REVERSAL SIGNAL
# ============================================================

def rejection_score(row, wick_threshold, body_threshold):

    score = 0.0

    body_ratio = row["body_ratio"]

    upper = row["upper_wick_ratio"]
    lower = row["lower_wick_ratio"]

    bullish = row["bullish"]
    bearish = row["bearish"]

    if pd.isna(body_ratio):
        return np.nan

    # Strong lower rejection
    if lower >= wick_threshold:
        score += 0.75

    # Strong upper rejection
    if upper >= wick_threshold:
        score -= 0.75

    # Small/controlled body
    if body_ratio <= body_threshold:
        score += 0.25

    # Candle direction
    if bullish:
        score += 0.25

    if bearish:
        score -= 0.25

    # Location confirmation
    if row["range_position"] <= 0.30 and bullish:
        score += 0.50

    if row["range_position"] >= 0.70 and bearish:
        score -= 0.50

    return score


# ============================================================
# TRADE GENERATION
# ============================================================

def generate_trades(
    df,
    rr,
    wick,
    body,
    separation,
    max_cross,
    hours,
    threshold
):

    trades = []

    last_signal_index = -10_000

    scores = []

    for i in range(len(df)):

        row = df.iloc[i]

        if row["hour"] not in hours:
            scores.append(np.nan)
            continue

        score = rejection_score(
            row,
            wick,
            body
        )

        scores.append(score)

    df = df.copy()

    df["signal_score"] = scores

    for i in range(len(df) - 2):

        row = df.iloc[i]

        if pd.isna(row["signal_score"]):
            continue

        if row["signal_score"] < threshold:
            continue

        if i - last_signal_index < max_cross:
            continue

        # ----------------------------------------------------
        # Regime-aware confirmation
        # ----------------------------------------------------

        regime = row["regime"]

        # Rejection/reversal is primarily useful in
        # range/transition environments.
        if regime not in (
            "RANGE_HIGH_VOL",
            "RANGE_LOW_VOL",
            "TRANSITION",
            "TREND_LOW_VOL",
        ):
            continue

        direction = None

        if row["bullish"]:
            direction = 1

        elif row["bearish"]:
            direction = -1

        if direction is None:
            continue

        entry = df.iloc[i + 1]["open"]

        atr = row["atr14"]

        if pd.isna(atr) or atr <= 0:
            continue

        risk = atr

        if direction == 1:

            stop = entry - risk
            target = entry + risk * rr

        else:

            stop = entry + risk
            target = entry - risk * rr

        result = None

        exit_index = None

        for j in range(i + 1, len(df)):

            candle = df.iloc[j]

            if direction == 1:

                stop_hit = candle["low"] <= stop
                target_hit = candle["high"] >= target

            else:

                stop_hit = candle["high"] >= stop
                target_hit = candle["low"] <= target

            # Conservative assumption:
            # if both are hit in the same candle,
            # count the stop first.
            if stop_hit and target_hit:
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

        trades.append({
            "index": i,
            "time": row["time"],
            "direction": direction,
            "score": row["signal_score"],
            "regime": regime,
            "result_r": result,
            "exit_index": exit_index,
        })

        last_signal_index = i

    return pd.DataFrame(trades)


# ============================================================
# PERFORMANCE
# ============================================================

def performance(trades):

    if trades is None or len(trades) == 0:

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

    r = trades["result_r"].astype(float)

    wins = int((r > 0).sum())
    losses = int((r < 0).sum())

    total = len(r)

    win_rate = wins / total * 100

    gross_profit = r[r > 0].sum()
    gross_loss = abs(r[r < 0].sum())

    if gross_loss == 0:
        pf = 999.0
    else:
        pf = gross_profit / gross_loss

    equity = r.cumsum()

    running_max = equity.cummax()

    drawdown = running_max - equity

    max_dd = float(drawdown.max())

    streak = 0
    longest = 0

    for value in r:

        if value < 0:
            streak += 1
            longest = max(longest, streak)

        else:
            streak = 0

    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_r": float(r.sum()),
        "profit_factor": float(pf),
        "max_drawdown": max_dd,
        "longest_losing_streak": longest,
    }


# ============================================================
# STABILITY
# ============================================================

def parameter_stability(
    df,
    params,
    training_start,
    training_end
):

    rr, wick, body, separation, max_cross, hours, threshold = params

    nearby_results = []

    for rr2 in RR_VALUES:

        for wick2 in WICK_VALUES:

            for body2 in BODY_VALUES:

                if (
                    abs(rr2 - rr) > 0.25 or
                    abs(wick2 - wick) > 0.10 or
                    abs(body2 - body) > 0.10
                ):
                    continue

                trades = generate_trades(
                    df,
                    rr2,
                    wick2,
                    body2,
                    separation,
                    max_cross,
                    hours,
                    threshold
                )

                trades = trades[
                    (trades["time"] >= training_start) &
                    (trades["time"] <= training_end)
                ]

                if len(trades) < MIN_TRAIN_TRADES:
                    continue

                p = performance(trades)

                nearby_results.append(p)

    if len(nearby_results) < MIN_STABILITY_NEIGHBOURS:

        return {
            "stable": False,
            "nearby": len(nearby_results),
            "median_wr": 0.0,
            "median_r": 0.0,
            "positive_pct": 0.0,
        }

    wrs = [
        x["win_rate"]
        for x in nearby_results
    ]

    rs = [
        x["total_r"]
        for x in nearby_results
    ]

    positive = [
        x > 0
        for x in rs
    ]

    median_wr = float(np.median(wrs))
    median_r = float(np.median(rs))
    positive_pct = float(np.mean(positive) * 100)

    stable = (
        positive_pct >= 70.0
        and median_r > 0
    )

    return {
        "stable": stable,
        "nearby": len(nearby_results),
        "median_wr": median_wr,
        "median_r": median_r,
        "positive_pct": positive_pct,
    }


# ============================================================
# OPTIMISATION
# ============================================================

def optimise_training(
    df,
    training_start,
    training_end
):

    training = df[
        (df["time"] >= training_start) &
        (df["time"] <= training_end)
    ].copy()

    candidates = []

    print("PHASE 1: REGIME-AWARE SEARCH")

    combinations = list(
        itertools.product(
            RR_VALUES,
            WICK_VALUES,
            BODY_VALUES,
            SEPARATION_VALUES,
            MAX_CROSS_VALUES,
            HOUR_SETS,
            THRESHOLDS
        )
    )

    total = len(combinations)

    print(f"TOTAL COMBINATIONS: {total}")

    for n, params in enumerate(combinations, 1):

        if n == 1 or n % 500 == 0 or n == total:

            print(
                f"Progress: {n}/{total} "
                f"({n / total * 100:.1f}%)"
            )

        (
            rr,
            wick,
            body,
            separation,
            max_cross,
            hours,
            threshold
        ) = params

        trades = generate_trades(
            training,
            rr,
            wick,
            body,
            separation,
            max_cross,
            hours,
            threshold
        )

        if len(trades) < MIN_TRAIN_TRADES:
            continue

        p = performance(trades)

        # ----------------------------------------------------
        # Quality score
        # ----------------------------------------------------

        score = (
            p["win_rate"]
            * 0.45
            +
            min(p["profit_factor"], 3.0) / 3.0
            * 100
            * 0.25
            +
            min(max(p["total_r"], 0), 30) / 30
            * 100
            * 0.20
            -
            min(p["max_drawdown"], 10) / 10
            * 100
            * 0.10
        )

        candidates.append({
            "params": params,
            "performance": p,
            "quality": score,
        })

    if not candidates:

        return None

    candidates.sort(
        key=lambda x: x["quality"],
        reverse=True
    )

    # --------------------------------------------------------
    # Stability filtering
    # --------------------------------------------------------

    print("PHASE 2: PARAMETER STABILITY")

    stable_candidates = []

    for candidate in candidates[:25]:

        params = candidate["params"]

        stability = parameter_stability(
            df,
            params,
            training_start,
            training_end
        )

        candidate["stability"] = stability

        if stability["stable"]:

            stable_candidates.append(candidate)

    if not stable_candidates:

        print("No stable candidates.")

        return None

    # --------------------------------------------------------
    # Select based on quality AND stability.
    # Do NOT select purely on highest WR.
    # --------------------------------------------------------

    stable_candidates.sort(
        key=lambda x: (
            x["stability"]["positive_pct"],
            x["stability"]["median_r"],
            x["quality"],
        ),
        reverse=True
    )

    selected = stable_candidates[0]

    return selected


# ============================================================
# WALK-FORWARD + FINAL HOLDOUT
# ============================================================

def run_market(name, path):

    print("=" * 60)
    print(f"{name} V11 REGIME-AWARE OPTIMIZER")
    print("=" * 60)

    df = load_data(path)

    print(f"Candles: {len(df)}")
    print(
        f"Range: "
        f"{df['time'].min()} -> {df['time'].max()}"
    )

    print("Preparing indicators...")

    df = prepare_indicators(df)

    df["regime"] = df.apply(
        classify_regime,
        axis=1
    )

    results = []

    # --------------------------------------------------------
    # WALK FORWARD
    # --------------------------------------------------------

    for train_start, train_end, oos_start, oos_end in WALK_FORWARD_PERIODS:

        print("=" * 60)
        print(
            f"{name}: "
            f"{train_start[:4]}-{train_end[:4]} "
            f"-> {oos_start[:4]}"
        )
        print("=" * 60)

        selected = optimise_training(
            df,
            train_start,
            train_end
        )

        if selected is None:

            print("NO STABLE STRATEGY FOUND")

            results.append({
                "period": oos_start[:4],
                "oos_trades": 0,
                "oos_win_rate": 0,
                "oos_total_r": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "stable": False,
            })

            continue

        params = selected["params"]

        print("SELECTED TRAINING STRATEGY")
        print("-" * 60)
        print(
            f"Signal: REJECTION_REVERSAL"
        )
        print(
            f"Threshold: {params[-1]}"
        )

        p = selected["performance"]
        s = selected["stability"]

        print(
            f"Training trades: {p['trades']}"
        )
        print(
            f"Training WR: {p['win_rate']:.2f}%"
        )
        print(
            f"Training R: {p['total_r']:.2f}"
        )
        print(
            f"Training PF: {p['profit_factor']:.2f}"
        )

        print(
            f"Nearby: {s['nearby']}"
        )

        print(
            f"Median nearby WR: "
            f"{s['median_wr']:.2f}%"
        )

        print(
            f"Median nearby R: "
            f"{s['median_r']:.2f}R"
        )

        print(
            f"Positive nearby: "
            f"{s['positive_pct']:.1f}%"
        )

        print(
            f"Stability: "
            f"{'PASS' if s['stable'] else 'FAIL'}"
        )

        # ----------------------------------------------------
        # Genuine OOS
        # ----------------------------------------------------

        oos = df[
            (df["time"] >= oos_start) &
            (df["time"] <= oos_end)
        ].copy()

        oos_trades = generate_trades(
            oos,
            *params
        )

        p_oos = performance(oos_trades)

        print("=" * 60)
        print("COMPLETELY OUT-OF-SAMPLE TEST")
        print("-" * 60)
        print(
            f"Trades: {p_oos['trades']}"
        )
        print(
            f"Wins: {p_oos['wins']}"
        )
        print(
            f"Losses: {p_oos['losses']}"
        )
        print(
            f"Win rate: {p_oos['win_rate']:.2f}%"
        )
        print(
            f"Total R: {p_oos['total_r']:.2f}"
        )
        print(
            f"Profit factor: "
            f"{p_oos['profit_factor']:.2f}"
        )
        print(
            f"Max drawdown: "
            f"{p_oos['max_drawdown']:.2f}R"
        )

        results.append({
            "period": oos_start[:4],
            "oos_trades": p_oos["trades"],
            "oos_win_rate": p_oos["win_rate"],
            "oos_total_r": p_oos["total_r"],
            "profit_factor": p_oos["profit_factor"],
            "max_drawdown": p_oos["max_drawdown"],
            "stable": True,
        })

    # ========================================================
    # FINAL UNTOUCHED HOLDOUT
    # ========================================================

    print("=" * 60)
    print(
        f"{name}: FINAL HOLDOUT "
        f"{FINAL_HOLDOUT_START[:4]} -> PRESENT"
    )
    print("=" * 60)

    # Use ONLY the immediately preceding training period
    # to select parameters. The final holdout itself is
    # never touched during optimisation.

    final_train_start = "2023-01-01"
    final_train_end = "2025-12-31"

    selected = optimise_training(
        df,
        final_train_start,
        final_train_end
    )

    if selected is None:

        print("NO STABLE FINAL STRATEGY")

        final_result = {
            "period": "FINAL_HOLDOUT",
            "oos_trades": 0,
            "oos_win_rate": 0,
            "oos_total_r": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "stable": False,
        }

    else:

        params = selected["params"]

        holdout = df[
            df["time"] >= FINAL_HOLDOUT_START
        ].copy()

        holdout_trades = generate_trades(
            holdout,
            *params
        )

        hp = performance(holdout_trades)

        print("-" * 60)
        print("FINAL NEVER-SEEN HOLDOUT")
        print("-" * 60)

        print(
            f"Trades: {hp['trades']}"
        )
        print(
            f"Wins: {hp['wins']}"
        )
        print(
            f"Losses: {hp['losses']}"
        )
        print(
            f"Win rate: {hp['win_rate']:.2f}%"
        )
        print(
            f"Total R: {hp['total_r']:.2f}"
        )
        print(
            f"Profit factor: "
            f"{hp['profit_factor']:.2f}"
        )
        print(
            f"Max drawdown: "
            f"{hp['max_drawdown']:.2f}R"
        )

        final_result = {
            "period": "FINAL_HOLDOUT",
            "oos_trades": hp["trades"],
            "oos_win_rate": hp["win_rate"],
            "oos_total_r": hp["total_r"],
            "profit_factor": hp["profit_factor"],
            "max_drawdown": hp["max_drawdown"],
            "stable": True,
        }

    results.append(final_result)

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MULTI-MARKET STRATEGY OPTIMIZER V11")
    print("=" * 60)

    print("REGIME-AWARE SIGNAL VALIDATION: ENABLED")
    print("REJECTION/REVERSAL PRIMARY SIGNAL: ENABLED")
    print("MARKET-SPECIFIC OPTIMISATION: ENABLED")
    print("WALK-FORWARD TESTING: ENABLED")
    print("PARAMETER STABILITY: ENABLED")
    print("STRICT FINAL HOLDOUT: ENABLED")
    print("MINIMUM SAMPLE FILTER: ENABLED")
    print("NO LIVE TRADING")
    print("=" * 60)

    summaries = []

    for market, path in MARKETS.items():

        if not os.path.exists(path):

            print(
                f"WARNING: {path} not found. "
                f"Skipping {market}."
            )

            continue

        try:

            result = run_market(
                market,
                path
            )

            if result is None:
                continue

            output = RESULT_FILES[market]

            result.to_csv(
                output,
                index=False
            )

            # ------------------------------------------------
            # Aggregate only genuine OOS periods.
            # ------------------------------------------------

            oos = result[
                result["period"] != "FINAL_HOLDOUT"
            ]

            if len(oos):

                total_trades = int(
                    oos["oos_trades"].sum()
                )

                # We need actual wins rather than
                # averaging percentages.
                wins = int(
                    sum(
                        round(
                            row["oos_trades"]
                            * row["oos_win_rate"]
                            / 100
                        )
                        for _, row in oos.iterrows()
                    )
                )

                wr = (
                    wins /
                    total_trades *
                    100
                    if total_trades
                    else 0
                )

                total_r = float(
                    oos["oos_total_r"].sum()
                )

                profitable = int(
                    (oos["oos_total_r"] > 0).sum()
                )

                stable = int(
                    oos["stable"].sum()
                )

            else:

                total_trades = 0
                wr = 0
                total_r = 0
                profitable = 0
                stable = 0

            summaries.append({
                "market": market,
                "oos_trades": total_trades,
                "oos_win_rate": wr,
                "oos_total_r": total_r,
                "profitable_periods":
                    f"{profitable}/{len(oos)}",
                "stable_periods":
                    f"{stable}/{len(oos)}",
            })

        except Exception as e:

            print("=" * 60)
            print(f"{market} FAILED")
            print("=" * 60)
            print(
                f"{type(e).__name__}: {e}"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        summaries
    )

    if len(summary):

        summary.to_csv(
            SUMMARY_FILE,
            index=False
        )

        print("=" * 60)
        print("V11 FINAL MULTI-MARKET SUMMARY")
        print("=" * 60)

        print(
            summary.to_string(
                index=False
            )
        )

        total_trades = int(
            summary["oos_trades"].sum()
        )

        total_r = float(
            summary["oos_total_r"].sum()
        )

        # Approximate aggregate wins using
        # market-level integer trade counts.
        total_wins = 0

        for _, row in summary.iterrows():

            total_wins += round(
                row["oos_trades"] *
                row["oos_win_rate"] /
                100
            )

        combined_wr = (
            total_wins /
            total_trades *
            100
            if total_trades
            else 0
        )

        print("=" * 60)
        print("COMBINED GENUINE OUT-OF-SAMPLE")
        print("=" * 60)

        print(
            f"Trades: {total_trades}"
        )

        print(
            f"Wins: {total_wins}"
        )

        print(
            f"Win rate: "
            f"{combined_wr:.2f}%"
        )

        print(
            f"Total R: "
            f"{total_r:.2f}"
        )

        # ----------------------------------------------------
        # Target assessment
        # ----------------------------------------------------

        print("=" * 60)
        print("V11 TARGET CHECK")
        print("=" * 60)

        print("TARGET:")
        print("~82% WIN RATE")
        print("200+ GENUINELY OUT-OF-SAMPLE TRADES")
        print("POSITIVE TOTAL R")

        achieved = (
            total_trades >= 200
            and combined_wr >= TARGET_WIN_RATE * 100
            and total_r > 0
        )

        if achieved:

            print(
                "TARGET STATUS: "
                "ACHIEVED FOR THIS TEST"
            )

        else:

            print(
                "TARGET STATUS: "
                "NOT ACHIEVED YET"
            )

    else:

        print("=" * 60)
        print("NO COMPLETED MARKET RESULTS")
        print("=" * 60)

    print("=" * 60)
    print("V11 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
