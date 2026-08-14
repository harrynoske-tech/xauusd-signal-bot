import pandas as pd
import numpy as np


# ============================================================
# XAUUSD STRATEGY V8.0
# MULTI-FACTOR RESEARCH ENGINE
# ============================================================

EMA_FAST = 20
EMA_MEDIUM = 50
EMA_SLOW = 200

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14

MOMENTUM_LOOKBACK = 8
BREAKOUT_LOOKBACK = 20

MIN_SCORE = 70

ACTIVE_START_UTC = 6
ACTIVE_END_UTC = 20


# ============================================================
# CLEAN DATA
# ============================================================

def clean(data):

    if data is None or len(data) == 0:
        return pd.DataFrame()

    df = data.copy()

    required = [
        "Open",
        "High",
        "Low",
        "Close"
    ]

    for column in required:

        if column not in df.columns:
            raise ValueError(
                f"Missing column: {column}"
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df.dropna(
        subset=required
    )


# ============================================================
# EMA
# ============================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


# ============================================================
# RSI
# ============================================================

def rsi(series, period=RSI_PERIOD):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    return 100 - (
        100 / (1 + rs)
    )


# ============================================================
# ATR
# ============================================================

def atr(
    data,
    period=ATR_PERIOD
):

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(
        axis=1
    )

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# ADX
# ============================================================

def adx(
    data,
    period=ADX_PERIOD
):

    high = data["High"]
    low = data["Low"]

    up_move = (
        high.diff()
    )

    down_move = (
        -low.diff()
    )

    plus_dm = np.where(
        (
            (up_move > down_move)
            & (up_move > 0)
        ),
        up_move,
        0
    )

    minus_dm = np.where(
        (
            (down_move > up_move)
            & (down_move > 0)
        ),
        down_move,
        0
    )

    previous_close = (
        data["Close"].shift(1)
    )

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(
        axis=1
    )

    atr_value = tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    plus_di = (
        100
        * pd.Series(
            plus_dm,
            index=data.index
        ).ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        / atr_value
    )

    minus_di = (
        100
        * pd.Series(
            minus_dm,
            index=data.index
        ).ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        / atr_value
    )

    denominator = (
        plus_di
        + minus_di
    )

    dx = (
        100
        * (
            plus_di
            - minus_di
        ).abs()
        / denominator.replace(
            0,
            np.nan
        )
    )

    return dx.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# TREND
# ============================================================

def trend_score(data):

    close = data["Close"]

    fast = ema(
        close,
        EMA_FAST
    )

    medium = ema(
        close,
        EMA_MEDIUM
    )

    slow = ema(
        close,
        EMA_SLOW
    )

    price = float(
        close.iloc[-1]
    )

    fast_value = float(
        fast.iloc[-1]
    )

    medium_value = float(
        medium.iloc[-1]
    )

    slow_value = float(
        slow.iloc[-1]
    )

    bullish = 0
    bearish = 0

    if price > fast_value:
        bullish += 1
    else:
        bearish += 1

    if fast_value > medium_value:
        bullish += 1
    else:
        bearish += 1

    if medium_value > slow_value:
        bullish += 1
    else:
        bearish += 1

    if bullish == 3:

        return {
            "direction": "BUY",
            "score": 20
        }

    if bearish == 3:

        return {
            "direction": "SELL",
            "score": 20
        }

    if bullish == 2:

        return {
            "direction": "BUY",
            "score": 13
        }

    if bearish == 2:

        return {
            "direction": "SELL",
            "score": 13
        }

    return {
        "direction": "NONE",
        "score": 0
    }


# ============================================================
# MOMENTUM
# ============================================================

def momentum_score(data):

    close = data["Close"]

    if len(close) < MOMENTUM_LOOKBACK + 2:

        return {
            "direction": "NONE",
            "score": 0
        }

    current = float(
        close.iloc[-1]
    )

    previous = float(
        close.iloc[
            -1 - MOMENTUM_LOOKBACK
        ]
    )

    change = (
        current
        - previous
    )

    rsi_value = float(
        rsi(close).iloc[-1]
    )

    bullish = 0
    bearish = 0

    if change > 0:
        bullish += 1
    elif change < 0:
        bearish += 1

    if rsi_value >= 55:
        bullish += 1

    elif rsi_value <= 45:
        bearish += 1

    if bullish == 2:

        return {
            "direction": "BUY",
            "score": 20
        }

    if bearish == 2:

        return {
            "direction": "SELL",
            "score": 20
        }

    if bullish == 1:

        return {
            "direction": "BUY",
            "score": 10
        }

    if bearish == 1:

        return {
            "direction": "SELL",
            "score": 10
        }

    return {
        "direction": "NONE",
        "score": 0
    }


# ============================================================
# VOLATILITY
# ============================================================

def volatility_score(data):

    atr_values = atr(
        data
    )

    if len(atr_values) < 30:

        return {
            "score": 0,
            "regime": "UNKNOWN"
        }

    current = float(
        atr_values.iloc[-1]
    )

    median = float(
        atr_values.tail(100).median()
    )

    if median <= 0:

        return {
            "score": 0,
            "regime": "UNKNOWN"
        }

    ratio = (
        current
        / median
    )

    if ratio >= 1.20:

        return {
            "score": 10,
            "regime": "EXPANSION"
        }

    if ratio >= 0.80:

        return {
            "score": 7,
            "regime": "NORMAL"
        }

    return {
        "score": 2,
        "regime": "LOW"
    }


# ============================================================
# BREAKOUT / DISPLACEMENT
# ============================================================

def breakout_score(data):

    if len(data) < BREAKOUT_LOOKBACK + 2:

        return {
            "direction": "NONE",
            "score": 0
        }

    current = data.iloc[-1]

    previous = data.iloc[
        -BREAKOUT_LOOKBACK - 1:-1
    ]

    highest = float(
        previous["High"].max()
    )

    lowest = float(
        previous["Low"].min()
    )

    candle_range = (
        float(current["High"])
        - float(current["Low"])
    )

    atr_value = float(
        atr(data).iloc[-1]
    )

    if atr_value <= 0:

        return {
            "direction": "NONE",
            "score": 0
        }

    if (
        float(current["Close"])
        > highest
        and candle_range
        >= atr_value * 1.20
    ):

        return {
            "direction": "BUY",
            "score": 15
        }

    if (
        float(current["Close"])
        < lowest
        and candle_range
        >= atr_value * 1.20
    ):

        return {
            "direction": "SELL",
            "score": 15
        }

    return {
        "direction": "NONE",
        "score": 0
    }


# ============================================================
# PREVIOUS DAY LEVELS
# ============================================================

def daily_levels(
    data_daily,
    price
):

    if len(data_daily) < 2:

        return {
            "direction": "NONE",
            "score": 0
        }

    previous = data_daily.iloc[-2]

    high = float(
        previous["High"]
    )

    low = float(
        previous["Low"]
    )

    distance_high = abs(
        price - high
    )

    distance_low = abs(
        price - low
    )

    atr_value = float(
        atr(
            data_daily
        ).iloc[-1]
    )

    if atr_value <= 0:

        return {
            "direction": "NONE",
            "score": 0
        }

    tolerance = (
        atr_value * 0.20
    )

    if distance_low <= tolerance:

        return {
            "direction": "BUY",
            "score": 10
        }

    if distance_high <= tolerance:

        return {
            "direction": "SELL",
            "score": 10
        }

    return {
        "direction": "NONE",
        "score": 0
    }


# ============================================================
# SESSION
# ============================================================

def session_score(timestamp):

    hour = int(
        timestamp.hour
    )

    if (
        ACTIVE_START_UTC
        <= hour
        < ACTIVE_END_UTC
    ):

        return 10

    return 0


# ============================================================
# HIGHER TIMEFRAME
# ============================================================

def higher_timeframe_score(
    data_15m
):

    close = data_15m["Close"]

    if len(close) < 200:

        return {
            "direction": "NONE",
            "score": 0
        }

    fast = ema(
        close,
        50
    )

    slow = ema(
        close,
        200
    )

    fast_value = float(
        fast.iloc[-1]
    )

    slow_value = float(
        slow.iloc[-1]
    )

    if fast_value > slow_value:

        return {
            "direction": "BUY",
            "score": 15
        }

    if fast_value < slow_value:

        return {
            "direction": "SELL",
            "score": 15
        }

    return {
        "direction": "NONE",
        "score": 0
    }


# ============================================================
# SCORE ENGINE
# ============================================================

def generate_signal(
    data_15m,
    data_daily,
    current_price
):

    data_15m = clean(
        data_15m
    )

    data_daily = clean(
        data_daily
    )

    if (
        len(data_15m) < 250
        or len(data_daily) < 50
    ):

        return {
            "signal": "NONE",
            "score": 0,
            "reason":
                "INSUFFICIENT_DATA"
        }

    price = float(
        current_price
    )

    trend = trend_score(
        data_15m
    )

    momentum = momentum_score(
        data_15m
    )

    volatility = volatility_score(
        data_15m
    )

    breakout = breakout_score(
        data_15m
    )

    daily = daily_levels(
        data_daily,
        price
    )

    htf = higher_timeframe_score(
        data_15m
    )

    session = session_score(
        data_15m.index[-1]
    )

    scores = {
        "trend": trend,
        "momentum": momentum,
        "breakout": breakout,
        "daily": daily,
        "htf": htf
    }

    buy_score = 0
    sell_score = 0

    for component in scores.values():

        direction = component[
            "direction"
        ]

        score = component[
            "score"
        ]

        if direction == "BUY":
            buy_score += score

        elif direction == "SELL":
            sell_score += score

    # Volatility and session are neutral
    # filters rather than directional signals.

    if session == 0:

        return {
            "signal": "NONE",
            "score": 0,
            "reason":
                "OUTSIDE_ACTIVE_SESSION"
        }

    if volatility[
        "regime"
    ] == "LOW":

        return {
            "signal": "NONE",
            "score": 0,
            "reason":
                "LOW_VOLATILITY"
        }

    if buy_score > sell_score:

        direction = "BUY"
        score = buy_score

    elif sell_score > buy_score:

        direction = "SELL"
        score = sell_score

    else:

        return {
            "signal": "NONE",
            "score": 0,
            "reason":
                "NO_DIRECTIONAL_ALIGNMENT"
        }

    # Add neutral quality points.
    score += volatility[
        "score"
    ]

    score += session

    # Maximum theoretical score:
    # Trend 20
    # Momentum 20
    # Breakout 15
    # Daily 10
    # HTF 15
    # Volatility 10
    # Session 10
    # = 100

    if score < MIN_SCORE:

        return {
            "signal": "NONE",
            "score": score,
            "reason":
                "SCORE_BELOW_THRESHOLD"
        }

    # --------------------------------------------------------
    # ATR-based risk
    # --------------------------------------------------------

    atr_value = float(
        atr(
            data_15m
        ).iloc[-1]
    )

    if direction == "BUY":

        stop_loss = (
            price
            - atr_value
        )

        take_profit = (
            price
            + atr_value * 1.5
        )

    else:

        stop_loss = (
            price
            + atr_value
        )

        take_profit = (
            price
            - atr_value * 1.5
        )

    return {
        "signal": direction,

        "score": round(
            float(score),
            2
        ),

        "reason":
            "V8_MULTI_FACTOR",

        "entry": price,

        "stop_loss":
            float(stop_loss),

        "take_profit":
            float(take_profit),

        "atr":
            float(atr_value),

        "components": {
            "trend": trend,
            "momentum": momentum,
            "volatility": volatility,
            "breakout": breakout,
            "daily": daily,
            "htf": htf,
            "session": session
        }
    }
