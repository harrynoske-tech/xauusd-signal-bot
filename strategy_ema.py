# ============================================================
# XAUUSD EMA V5 — HIGH-CONVICTION SELL
# ============================================================

EMA_FAST = 20
EMA_SLOW = 50

RR = 1.0

MAX_BARS_AFTER_CROSS = 60
PULLBACK_TOLERANCE = 0.0020
MIN_EMA_SEPARATION = 0.0008
SWING_LOOKBACK = 8

ALLOWED_HOURS = {
    2,
    3,
    12,
    13,
}

MIN_UPPER_WICK_RATIO = 0.40
MIN_BEARISH_BODY_RATIO = 0.30


def generate_signal(data_15m, data_daily=None, price=None):

    if len(data_15m) < EMA_SLOW + 20:
        return {"signal": "NONE", "reason": "NOT_ENOUGH_DATA"}

    timestamp = data_15m.index[-1]

    if timestamp.hour not in ALLOWED_HOURS:
        return {"signal": "NONE", "reason": "OUTSIDE_SESSION"}

    df = data_15m.copy()

    df["EMA20"] = df["Close"].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()

    df["EMA50"] = df["Close"].ewm(
        span=EMA_SLOW,
        adjust=False
    ).mean()

    current = df.iloc[-1]

    ema20 = float(current["EMA20"])
    ema50 = float(current["EMA50"])

    close = float(current["Close"])
    high = float(current["High"])
    low = float(current["Low"])
    open_price = float(current["Open"])

    # --------------------------------------------------------
    # TREND STRENGTH
    # --------------------------------------------------------

    separation = abs(ema20 - ema50) / close

    if separation < MIN_EMA_SEPARATION:
        return {
            "signal": "NONE",
            "reason": "EMA_TREND_TOO_WEAK"
        }

    # --------------------------------------------------------
    # EMA SLOPE
    # --------------------------------------------------------

    ema20_previous = float(df.iloc[-5]["EMA20"])
    ema50_previous = float(df.iloc[-5]["EMA50"])

    ema20_slope = ema20 - ema20_previous
    ema50_slope = ema50 - ema50_previous

    if ema20 >= ema50:
        return {
            "signal": "NONE",
            "reason": "NOT_BEARISH"
        }

    if ema20_slope >= 0:
        return {
            "signal": "NONE",
            "reason": "EMA20_NOT_FALLING"
        }

    if ema50_slope >= 0:
        return {
            "signal": "NONE",
            "reason": "EMA50_NOT_FALLING"
        }

    # --------------------------------------------------------
    # FIND RECENT BEARISH CROSS
    # --------------------------------------------------------

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

        previous = df.iloc[i - 1]
        candle = df.iloc[i]

        previous_fast = float(previous["EMA20"])
        previous_slow = float(previous["EMA50"])

        current_fast = float(candle["EMA20"])
        current_slow = float(candle["EMA50"])

        if (
            previous_fast >= previous_slow
            and current_fast < current_slow
        ):
            bars_since_cross = len(df) - 1 - i
            break

    if bars_since_cross is None:
        return {
            "signal": "NONE",
            "reason": "NO_RECENT_BEARISH_CROSS"
        }

    if bars_since_cross > MAX_BARS_AFTER_CROSS:
        return {
            "signal": "NONE",
            "reason": "CROSS_TOO_OLD"
        }

    # --------------------------------------------------------
    # PULLBACK TO EMA20
    # --------------------------------------------------------

    previous_candle = df.iloc[-2]

    previous_close = float(
        previous_candle["Close"]
    )

    previous_ema20 = float(
        previous_candle["EMA20"]
    )

    previous_distance = abs(
        previous_close - previous_ema20
    ) / previous_close

    current_distance = abs(
        close - ema20
    ) / close

    if (
        previous_distance > PULLBACK_TOLERANCE
        and current_distance > PULLBACK_TOLERANCE
    ):
        return {
            "signal": "NONE",
            "reason": "NO_PULLBACK"
        }

    # --------------------------------------------------------
    # STRONG BEARISH REJECTION
    # --------------------------------------------------------

    candle_range = high - low

    if candle_range <= 0:
        return {
            "signal": "NONE",
            "reason": "INVALID_CANDLE"
        }

    body = abs(close - open_price)

    upper_wick = high - max(
        open_price,
        close
    )

    body_ratio = body / candle_range
    upper_wick_ratio = upper_wick / candle_range

    # Must be a bearish candle.
    if close >= open_price:
        return {
            "signal": "NONE",
            "reason": "NOT_BEARISH_CANDLE"
        }

    # Stronger upper rejection.
    if upper_wick_ratio < MIN_UPPER_WICK_RATIO:
        return {
            "signal": "WEAK_UPPER_REJECTION",
            "reason": "NONE"
        }

    # Meaningful bearish body.
    if body_ratio < MIN_BEARISH_BODY_RATIO:
        return {
            "signal": "NONE",
            "reason": "BEARISH_BODY_TOO_SMALL"
        }

    # Close below EMA20.
    if close >= ema20:
        return {
            "signal": "NONE",
            "reason": "NO_EMA20_REJECTION"
        }

    # --------------------------------------------------------
    # ENTRY / STOP / TARGET
    # --------------------------------------------------------

    entry = close

    recent_high = float(
        df["High"]
        .iloc[-SWING_LOOKBACK:]
        .max()
    )

    stop_loss = recent_high

    risk = stop_loss - entry

    if risk <= 0:
        return {
            "signal": "NONE",
            "reason": "INVALID_RISK"
        }

    take_profit = entry - (
        risk * RR
    )

    return {
        "signal": "SELL",
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "score": 100,
        "reason": "EMA_STRONG_BEARISH_REJECTION",
        "components": {
            "session_hour": timestamp.hour,
            "ema20": ema20,
            "ema50": ema50,
            "ema_separation": separation,
            "ema20_slope": ema20_slope,
            "ema50_slope": ema50_slope,
            "bars_since_cross": bars_since_cross,
            "upper_wick_ratio": upper_wick_ratio,
            "body_ratio": body_ratio
        }
    }
