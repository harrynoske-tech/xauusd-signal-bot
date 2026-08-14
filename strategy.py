import pandas as pd
import numpy as np

# ============================================================
# XAUUSD STRATEGY V3
# ============================================================
# Design:
#   1. Weekly + Daily regime
#   2. 4H confirmation / contradiction filter
#   3. Recent 3+ touch AOIs
#   4. Price must actually interact with the AOI
#   5. Rejection / engulfing confirmation at the AOI
#   6. ATR + structural stop
#   7. Minimum 2R target
#   8. No generic candle trades away from an AOI
#
# Public interfaces preserved for live.py:
#   generate_signal(data_15m, data_daily, current_price)
#   get_weekly_daily_areas(data_daily, current_price=None)
# ============================================================

MIN_TOUCHES = 3
MAX_ZONE_WIDTH = 30.0
AOI_TOLERANCE = 4.0
MAX_RELEVANT_AOI_DISTANCE = 300.0

DAILY_LOOKBACK = 180
WEEKLY_LOOKBACK = 52
DAILY_SWING = 3
WEEKLY_SWING = 2

ATR_PERIOD = 14
ATR_BUFFER = 0.30
MIN_STOP = 4.0
MAX_STOP = 40.0
RISK_REWARD = 2.0

CONFIRM_LOOKBACK = 6
MIN_BODY_FRACTION = 0.35
MIN_WICK_FRACTION = 0.20


# ============================================================
# BASIC DATA
# ============================================================

def _clean(data):
    if data is None or len(data) == 0:
        return pd.DataFrame()

    df = data.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            c[0] if isinstance(c, tuple) else c
            for c in df.columns
        ]

    required = ["Open", "High", "Low", "Close"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    for c in required:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    return (
        df.dropna(subset=required)
        .sort_index()
    )


def _completed(data):
    df = _clean(data)

    if len(df) <= 1:
        return df

    return df.iloc[:-1].copy()


def _atr(data):
    df = _clean(data)

    if len(df) < ATR_PERIOD + 1:
        return None

    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    value = (
        true_range
        .rolling(ATR_PERIOD)
        .mean()
        .iloc[-1]
    )

    return (
        float(value)
        if pd.notna(value)
        else None
    )


# ============================================================
# RESAMPLING
# ============================================================

def resample_data(data, timeframe):
    df = _clean(data)

    rules = {
        "4H": "4h",
        "1D": "1D",
        "1W": "1W",
    }

    if timeframe not in rules:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    if df.empty:
        return df

    result = df.resample(
        rules[timeframe]
    ).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )

    return result.dropna(
        subset=["Open", "High", "Low", "Close"]
    )


# ============================================================
# MARKET STRUCTURE
# ============================================================

def find_market_structure(
    data,
    swing_length=3
):
    df = _clean(data)

    if df.empty:
        return df

    df["swing_high"] = False
    df["swing_low"] = False
    df["structure"] = None

    if len(df) < swing_length * 2 + 5:
        return df

    previous_high = None
    previous_low = None

    for i in range(
        swing_length,
        len(df) - swing_length
    ):
        high = float(
            df["High"].iloc[i]
        )

        low = float(
            df["Low"].iloc[i]
        )

        left_high = df["High"].iloc[
            i - swing_length:i
        ]

        right_high = df["High"].iloc[
            i + 1:i + swing_length + 1
        ]

        left_low = df["Low"].iloc[
            i - swing_length:i
        ]

        right_low = df["Low"].iloc[
            i + 1:i + swing_length + 1
        ]

        if (
            high > left_high.max()
            and high >= right_high.max()
        ):
            df.iloc[
                i,
                df.columns.get_loc("swing_high")
            ] = True

            if previous_high is not None:
                df.iloc[
                    i,
                    df.columns.get_loc("structure")
                ] = (
                    "HH"
                    if high > previous_high
                    else "LH"
                )

            previous_high = high

        if (
            low < left_low.min()
            and low <= right_low.min()
        ):
            df.iloc[
                i,
                df.columns.get_loc("swing_low")
            ] = True

            if (
                previous_low is not None
                and pd.isna(
                    df["structure"].iloc[i]
                )
            ):
                df.iloc[
                    i,
                    df.columns.get_loc("structure")
                ] = (
                    "HL"
                    if low > previous_low
                    else "LL"
                )

            previous_low = low

    return df


def get_market_bias(data):
    structure = find_market_structure(data)

    if structure.empty:
        return "NEUTRAL"

    points = (
        structure[
            structure["structure"].notna()
        ]["structure"]
        .tolist()
    )

    if len(points) < 4:
        return "NEUTRAL"

    recent = points[-8:]

    bullish = sum(
        x in ("HH", "HL")
        for x in recent
    )

    bearish = sum(
        x in ("LH", "LL")
        for x in recent
    )

    if bullish >= bearish + 2:
        return "BULLISH"

    if bearish >= bullish + 2:
        return "BEARISH"

    return "NEUTRAL"


def get_higher_timeframe_bias(
    data_15m,
    data_daily
):
    d15 = _completed(data_15m)
    dd = _completed(data_daily)

    if d15.empty or dd.empty:
        return {
            "weekly": "NEUTRAL",
            "daily": "NEUTRAL",
            "4h": "NEUTRAL",
            "overall": "NEUTRAL",
        }

    weekly = resample_data(
        dd,
        "1W"
    )

    four_h = resample_data(
        d15,
        "4H"
    )

    weekly_bias = get_market_bias(
        weekly
    )

    daily_bias = get_market_bias(
        dd
    )

    four_h_bias = get_market_bias(
        four_h
    )

    # Weighted regime:
    # weekly and daily determine the main regime.
    # 4H confirms or weakens it.
    score = 0

    score += (
        2
        if weekly_bias == "BULLISH"
        else -2
        if weekly_bias == "BEARISH"
        else 0
    )

    score += (
        2
        if daily_bias == "BULLISH"
        else -2
        if daily_bias == "BEARISH"
        else 0
    )

    score += (
        1
        if four_h_bias == "BULLISH"
        else -1
        if four_h_bias == "BEARISH"
        else 0
    )

    if score >= 3:
        overall = "BULLISH"
    elif score <= -3:
        overall = "BEARISH"
    else:
        overall = "NEUTRAL"

    return {
        "weekly": weekly_bias,
        "daily": daily_bias,
        "4h": four_h_bias,
        "overall": overall,
        "score": score,
    }


# ============================================================
# AOI ENGINE
# ============================================================

def _cluster(values):
    values = sorted(
        float(v)
        for v in values
    )

    clusters = []

    for value in values:
        added = False

        for cluster in clusters:
            if (
                value - min(cluster)
                <= MAX_ZONE_WIDTH
            ):
                cluster.append(value)
                added = True
                break

        if not added:
            clusters.append([value])

    merged = True

    while merged:
        merged = False
        result = []

        for cluster in clusters:
            if not result:
                result.append(cluster)
                continue

            previous = result[-1]

            if (
                max(cluster)
                - min(previous)
                <= MAX_ZONE_WIDTH
            ):
                result[-1] = (
                    previous + cluster
                )
                merged = True
            else:
                result.append(cluster)

        clusters = result

    return clusters


def _touch_events(
    data,
    low,
    high
):
    df = _clean(data)

    events = []
    active = False

    for idx, row in df.iterrows():
        touched = (
            float(row["High"]) >= low
            and float(row["Low"]) <= high
        )

        if touched and not active:
            events.append(idx)

        active = touched

    return events


def _build_zones(
    data,
    swing_length,
    lookback
):
    df = _clean(data)

    if len(df) < swing_length * 2 + 10:
        return []

    structure = find_market_structure(
        df,
        swing_length
    )

    recent = df.tail(lookback)
    recent_start = recent.index[0]

    highs = structure[
        structure["swing_high"]
    ]

    lows = structure[
        structure["swing_low"]
    ]

    highs = highs[
        highs.index >= recent_start
    ]

    lows = lows[
        lows.index >= recent_start
    ]

    zones = []

    for zone_type, swings in (
        ("resistance", highs),
        ("support", lows)
    ):
        if swings.empty:
            continue

        prices = (
            swings["High"].tolist()
            if zone_type == "resistance"
            else swings["Low"].tolist()
        )

        for cluster in _cluster(prices):
            if len(cluster) < MIN_TOUCHES:
                continue

            low = float(min(cluster))
            high = float(max(cluster))

            if high - low > MAX_ZONE_WIDTH:
                continue

            touches = _touch_events(
                df.tail(lookback),
                low,
                high
            )

            if len(touches) < MIN_TOUCHES:
                continue

            latest_touch = touches[-1]

            zones.append(
                {
                    "type": zone_type,
                    "low": low,
                    "high": high,
                    "width": high - low,
                    "touches": len(touches),
                    "touch_times": touches,
                    "latest_touch": latest_touch,
                    "structure_bias":
                        get_market_bias(df),
                }
            )

    return zones


def _remove_duplicates(zones):
    zones = sorted(
        zones,
        key=lambda z: (
            -z["touches"],
            z["width"]
        )
    )

    result = []

    for zone in zones:
        duplicate = False

        for existing in result:
            if (
                zone["type"]
                == existing["type"]
                and zone["low"]
                <= existing["high"]
                and zone["high"]
                >= existing["low"]
            ):
                duplicate = True
                break

        if not duplicate:
            result.append(zone)

    return result


def find_area_of_interest(
    data,
    current_price=None,
    swing_length=3,
    touch_lookback=100
):
    df = _clean(data)

    if df.empty:
        return []

    zones = _build_zones(
        df,
        swing_length,
        min(
            len(df),
            touch_lookback
        )
    )

    zones = _remove_duplicates(
        zones
    )

    if current_price is None:
        return zones

    price = float(
        current_price
    )

    relevant = []

    for zone in zones:
        low = float(zone["low"])
        high = float(zone["high"])

        if low <= price <= high:
            item = dict(zone)
            item["distance"] = 0.0
            relevant.append(item)
            continue

        if (
            zone["type"] == "support"
            and high < price
        ):
            distance = price - high

        elif (
            zone["type"] == "resistance"
            and low > price
        ):
            distance = low - price

        else:
            continue

        if distance <= MAX_RELEVANT_AOI_DISTANCE:
            item = dict(zone)
            item["distance"] = float(
                distance
            )
            relevant.append(item)

    relevant.sort(
        key=lambda z: (
            z["distance"],
            -z["touches"]
        )
    )

    return relevant


def get_weekly_daily_areas(
    data_daily,
    current_price=None
):
    dd = _completed(
        data_daily
    )

    if dd.empty:
        return {
            "weekly": [],
            "daily": [],
        }

    weekly = resample_data(
        dd,
        "1W"
    )

    return {
        "weekly": find_area_of_interest(
            weekly,
            current_price,
            WEEKLY_SWING,
            WEEKLY_LOOKBACK
        ),
        "daily": find_area_of_interest(
            dd,
            current_price,
            DAILY_SWING,
            DAILY_LOOKBACK
        ),
    }


# ============================================================
# AOI SELECTION
# ============================================================

def _zone_quality(zone):
    score = 0

    touches = int(
        zone.get("touches", 0)
    )

    width = float(
        zone.get("width", 999)
    )

    if touches >= 3:
        score += 3

    if touches >= 4:
        score += 1

    if width <= 15:
        score += 2
    elif width <= 25:
        score += 1

    return score


def _select_aoi(
    price,
    areas,
    direction
):
    required = (
        "support"
        if direction == "BUY"
        else "resistance"
    )

    candidates = []

    for timeframe in (
        "daily",
        "weekly"
    ):
        for zone in areas.get(
            timeframe,
            []
        ):
            if zone["type"] != required:
                continue

            low = float(
                zone["low"]
            )
            high = float(
                zone["high"]
            )

            if price < low:
                distance = low - price
            elif price > high:
                distance = price - high
            else:
                distance = 0.0

            if (
                distance
                > MAX_RELEVANT_AOI_DISTANCE
            ):
                continue

            quality = _zone_quality(
                zone
            )

            timeframe_bonus = (
                2
                if timeframe == "daily"
                else 1
            )

            # Nearer + higher quality wins.
            rank = (
                quality * 10
                + timeframe_bonus
                - distance / 10
            )

            item = dict(zone)
            item["timeframe"] = timeframe
            item["distance"] = float(
                distance
            )
            item["quality"] = quality
            item["_rank"] = rank

            candidates.append(item)

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["_rank"],
        reverse=True
    )

    return candidates[0]


# ============================================================
# CONFIRMATION
# ============================================================

def _range(row):
    return max(
        float(row["High"])
        - float(row["Low"]),
        0.000001
    )


def _bullish_rejection(row, aoi):
    if aoi["type"] != "support":
        return False

    high = float(row["High"])
    low = float(row["Low"])
    open_ = float(row["Open"])
    close = float(row["Close"])

    body = abs(
        close - open_
    )

    candle_range = _range(row)

    lower_wick = (
        min(open_, close)
        - low
    )

    touched = (
        low
        <= float(aoi["high"])
        + AOI_TOLERANCE
    )

    closed_back_above = (
        close > float(aoi["high"])
    )

    return (
        touched
        and closed_back_above
        and close > open_
        and body / candle_range
        >= MIN_BODY_FRACTION
        and lower_wick / candle_range
        >= MIN_WICK_FRACTION
    )


def _bearish_rejection(row, aoi):
    if aoi["type"] != "resistance":
        return False

    high = float(row["High"])
    low = float(row["Low"])
    open_ = float(row["Open"])
    close = float(row["Close"])

    body = abs(
        close - open_
    )

    candle_range = _range(row)

    upper_wick = (
        high
        - max(open_, close)
    )

    touched = (
        high
        >= float(aoi["low"])
        - AOI_TOLERANCE
    )

    closed_back_below = (
        close < float(aoi["low"])
    )

    return (
        touched
        and closed_back_below
        and close < open_
        and body / candle_range
        >= MIN_BODY_FRACTION
        and upper_wick / candle_range
        >= MIN_WICK_FRACTION
    )


def _bullish_engulfing(
    previous,
    current,
    aoi
):
    if (
        float(current["Low"])
        > float(aoi["high"])
        + AOI_TOLERANCE
    ):
        return False

    return (
        float(previous["Close"])
        < float(previous["Open"])
        and float(current["Close"])
        > float(current["Open"])
        and float(current["Open"])
        <= float(previous["Close"])
        and float(current["Close"])
        >= float(previous["Open"])
    )


def _bearish_engulfing(
    previous,
    current,
    aoi
):
    if (
        float(current["High"])
        < float(aoi["low"])
        - AOI_TOLERANCE
    ):
        return False

    return (
        float(previous["Close"])
        > float(previous["Open"])
        and float(current["Close"])
        < float(current["Open"])
        and float(current["Open"])
        >= float(previous["Close"])
        and float(current["Close"])
        <= float(previous["Open"])
    )


def get_entry_confirmation(
    data,
    aoi=None
):
    if aoi is None:
        return "NONE"

    df = _completed(data)

    if len(df) < 2:
        return "NONE"

    recent = df.tail(
        CONFIRM_LOOKBACK
    )

    for _, row in recent.iterrows():
        if (
            aoi["type"] == "support"
            and _bullish_rejection(
                row,
                aoi
            )
        ):
            return "BUY"

        if (
            aoi["type"] == "resistance"
            and _bearish_rejection(
                row,
                aoi
            )
        ):
            return "SELL"

    previous = df.iloc[-2]
    current = df.iloc[-1]

    if (
        aoi["type"] == "support"
        and _bullish_engulfing(
            previous,
            current,
            aoi
        )
    ):
        return "BUY"

    if (
        aoi["type"] == "resistance"
        and _bearish_engulfing(
            previous,
            current,
            aoi
        )
    ):
        return "SELL"

    return "NONE"


# ============================================================
# RISK / TARGET
# ============================================================

def calculate_sl_tp(
    signal,
    entry,
    aoi,
    data_15m
):
    atr = _atr(
        data_15m
    )

    if atr is None:
        return None

    buffer = max(
        MIN_STOP,
        atr * ATR_BUFFER
    )

    entry = float(entry)

    if signal == "SELL":
        stop_loss = (
            float(aoi["high"])
            + buffer
        )

        risk = (
            stop_loss
            - entry
        )

        if risk <= 0:
            return None

        take_profit = (
            entry
            - risk * RISK_REWARD
        )

    elif signal == "BUY":
        stop_loss = (
            float(aoi["low"])
            - buffer
        )

        risk = (
            entry
            - stop_loss
        )

        if risk <= 0:
            return None

        take_profit = (
            entry
            + risk * RISK_REWARD
        )

    else:
        return None

    if (
        risk < MIN_STOP
        or risk > MAX_STOP
    ):
        return None

    return {
        "entry": entry,
        "stop_loss": float(
            stop_loss
        ),
        "take_profit": float(
            take_profit
        ),
        "risk": float(risk),
        "reward": float(
            abs(
                take_profit - entry
            )
        ),
        "risk_reward": RISK_REWARD,
        "atr": float(atr),
    }


def _target_space(
    entry,
    take_profit,
    areas,
    direction
):
    opposing = (
        "support"
        if direction == "SELL"
        else "resistance"
    )

    candidates = []

    for timeframe in (
        "daily",
        "weekly"
    ):
        for zone in areas.get(
            timeframe,
            []
        ):
            if zone["type"] != opposing:
                continue

            low = float(zone["low"])
            high = float(zone["high"])

            if (
                direction == "SELL"
                and high < entry
            ):
                candidates.append(
                    high
                )

            if (
                direction == "BUY"
                and low > entry
            ):
                candidates.append(
                    low
                )

    if not candidates:
        return True

    if direction == "SELL":
        nearest = max(candidates)
        return take_profit >= nearest

    nearest = min(candidates)
    return take_profit <= nearest


# ============================================================
# SIGNAL
# ============================================================

def generate_signal(
    data_15m,
    data_daily,
    current_price
):
    d15 = _clean(
        data_15m
    )

    dd = _clean(
        data_daily
    )

    price = float(
        current_price
    )

    neutral_bias = {
        "weekly": "NEUTRAL",
        "daily": "NEUTRAL",
        "4h": "NEUTRAL",
        "overall": "NEUTRAL",
        "score": 0,
    }

    if (
        d15.empty
        or dd.empty
    ):
        return {
            "signal": "NONE",
            "reason": "INSUFFICIENT_DATA",
            "bias": neutral_bias,
            "aoi": None,
        }

    bias = get_higher_timeframe_bias(
        d15,
        dd
    )

    areas = get_weekly_daily_areas(
        dd,
        price
    )

    overall = bias["overall"]

    if overall == "NEUTRAL":
        return {
            "signal": "NONE",
            "reason": "NEUTRAL_HIGHER_TIMEFRAME",
            "bias": bias,
            "aoi": None,
        }

    direction = (
        "BUY"
        if overall == "BULLISH"
        else "SELL"
    )

    # A directly opposing 4H trend is a warning, not an
    # automatic veto. This is deliberately less restrictive
    # than V2 so valid reversals are not discarded.
    if (
        direction == "BUY"
        and bias["4h"] == "BEARISH"
        and bias["score"] <= 3
    ):
        return {
            "signal": "NONE",
            "reason": "4H_TOO_BEARISH_FOR_BUY",
            "bias": bias,
            "aoi": None,
        }

    if (
        direction == "SELL"
        and bias["4h"] == "BULLISH"
        and bias["score"] >= -3
    ):
        return {
            "signal": "NONE",
            "reason": "4H_TOO_BULLISH_FOR_SELL",
            "bias": bias,
            "aoi": None,
        }

    aoi = _select_aoi(
        price,
        areas,
        direction
    )

    if aoi is None:
        return {
            "signal": "NONE",
            "reason": "WAITING_FOR_RELEVANT_AOI",
            "bias": bias,
            "aoi": None,
        }

    if (
        float(aoi["distance"])
        > AOI_TOLERANCE
    ):
        return {
            "signal": "NONE",
            "reason": "WAITING_FOR_AOI",
            "bias": bias,
            "aoi": aoi,
        }

    closed = _completed(
        d15
    )

    confirmation = (
        get_entry_confirmation(
            closed,
            aoi
        )
    )

    if confirmation != direction:
        return {
            "signal": "NONE",
            "reason": "WAITING_FOR_CONFIRMATION",
            "bias": bias,
            "aoi": aoi,
        }

    # Use the completed confirmation candle close rather than the
    # current live tick as the historical entry reference.
    entry = float(
        closed["Close"].iloc[-1]
    )

    levels = calculate_sl_tp(
        direction,
        entry,
        aoi,
        closed
    )

    if levels is None:
        return {
            "signal": "NONE",
            "reason": "INVALID_RISK",
            "bias": bias,
            "aoi": aoi,
        }

    if not _target_space(
        levels["entry"],
        levels["take_profit"],
        areas,
        direction
    ):
        return {
            "signal": "NONE",
            "reason": "INSUFFICIENT_TARGET_SPACE",
            "bias": bias,
            "aoi": aoi,
        }

    return {
        "signal": direction,
        "reason": "AOI_RETEST_CONFIRMED",
        "bias": bias,
        "aoi": aoi,
        "confirmation": confirmation,
        **levels,
    }
