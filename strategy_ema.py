import pandas as pd


# ============================================================
# XAUUSD EMA 20/50 PULLBACK STRATEGY
# ============================================================

EMA_FAST = 20
EMA_SLOW = 50

RR = 2.5

PULLBACK_TOLERANCE = 0.0015

MAX_BARS_AFTER_CROSS = 40


def generate_signal(
    data_15m,
    data_daily=None,
    price=None
):

    if len(data_15m) < EMA_SLOW + 5:
        return {
            "signal": "NONE",
            "reason": "NOT_ENOUGH_DATA"
        }

    df = data_15m.copy()

    df["EMA20"] = (
        df["Close"]
        .ewm(
            span=EMA_FAST,
            adjust=False
        )
        .mean()
    )

    df["EMA50"] = (
        df["Close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # Most recent completed candle
    # --------------------------------------------------------

    current = df.iloc[-1]
    previous = df.iloc[-2]

    ema20 = float(
        current["EMA20"]
    )

    ema50 = float(
        current["EMA50"]
    )

    previous_ema20 = float(
        previous["EMA20"]
    )

    previous_ema50 = float(
        previous["EMA50"]
    )

    current_close = float(
        current["Close"]
    )

    current_high = float(
        current["High"]
    )

    current_low = float(
        current["Low"]
    )

    # --------------------------------------------------------
    # Find most recent EMA cross
    # --------------------------------------------------------

    cross_direction = None
    bars_since_cross = None

    search_start = max(
        1,
        len(df) - MAX_BARS_AFTER_CROSS - 1
    )

    for i in range(
        len(df) - 1,
        search_start - 1,
        -1
    ):

        a = df.iloc[i - 1]
        b = df.iloc[i]

        a_fast = float(a["EMA20"])
        a_slow = float(a["EMA50"])

        b_fast = float(b["EMA20"])
        b_slow = float(b["EMA50"])

        # Bullish cross
        if (
            a_fast <= a_slow
            and b_fast > b_slow
        ):

            cross_direction = "BUY"
            bars_since_cross = (
                len(df) - 1 - i
            )

            break

        # Bearish cross
        if (
            a_fast >= a_slow
            and b_fast < b_slow
        ):

            cross_direction = "SELL"
            bars_since_cross = (
                len(df) - 1 - i
            )

            break

    if cross_direction is None:

        return {
            "signal": "NONE",
            "reason": "NO_RECENT_CROSS"
        }

    if (
        bars_since_cross is None
        or bars_since_cross > MAX_BARS_AFTER_CROSS
    ):

        return {
            "signal": "NONE",
            "reason": "CROSS_TOO_OLD"
        }

    # --------------------------------------------------------
    # BUY setup
    # --------------------------------------------------------

    if cross_direction == "BUY":

        # Trend must still be bullish.
        if ema20 <= ema50:

            return {
                "signal": "NONE",
                "reason": "BULLISH_TREND_LOST"
            }

        # Price must pull back toward the 20 EMA.
        distance = abs(
            current_low - ema20
        )

        tolerance = (
            current_close
            * PULLBACK_TOLERANCE
        )

        if distance > tolerance:

            return {
                "signal": "NONE",
                "reason": "NO_20EMA_PULLBACK"
            }

        # Require the candle to finish back above EMA20.
        if current_close <= ema20:

            return {
                "signal": "NONE",
                "reason": "NO_BULLISH_RECLAIM"
            }

        entry = current_close

        # Fixed structural stop using pullback candle.
        stop_loss = min(
            current_low,
            ema20
        )

        risk = (
            entry - stop_loss
        )

        if risk <= 0:

            return {
                "signal": "NONE",
                "reason": "INVALID_RISK"
            }

        take_profit = (
            entry
            + risk * RR
        )

        return {
            "signal": "BUY",
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "score": 100,
            "reason": "EMA20_50_BULLISH_PULLBACK",
            "components": {
                "ema20": ema20,
                "ema50": ema50,
                "bars_since_cross":
                    bars_since_cross
            }
        }

    # --------------------------------------------------------
    # SELL setup
    # --------------------------------------------------------

    if cross_direction == "SELL":

        # Trend must still be bearish.
        if ema20 >= ema50:

            return {
                "signal": "NONE",
                "reason": "BEARISH_TREND_LOST"
            }

        # Price must pull back toward the 20 EMA.
        distance = abs(
            current_high - ema20
        )

        tolerance = (
            current_close
            * PULLBACK_TOLERANCE
        )

        if distance > tolerance:

            return {
                "signal": "NONE",
                "reason": "NO_20EMA_PULLBACK"
            }

        # Require candle to finish back below EMA20.
        if current_close >= ema20:

            return {
                "signal": "NONE",
                "reason": "NO_BEARISH_REJECTION"
            }

        entry = current_close

        # Fixed structural stop using pullback candle.
        stop_loss = max(
            current_high,
            ema20
        )

        risk = (
            stop_loss - entry
        )

        if risk <= 0:

            return {
                "signal": "NONE",
                "reason": "INVALID_RISK"
            }

        take_profit = (
            entry
            - risk * RR
        )

        return {
            "signal": "SELL",
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "score": 100,
            "reason": "EMA20_50_BEARISH_PULLBACK",
            "components": {
                "ema20": ema20,
                "ema50": ema50,
                "bars_since_cross":
                    bars_since_cross
            }
        }

    return {
        "signal": "NONE",
        "reason": "NO_SIGNAL"
    }
