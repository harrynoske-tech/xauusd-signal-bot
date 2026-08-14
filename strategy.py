import pandas as pd
import numpy as np


# ============================================================
# XAUUSD STRATEGY V7.2
# SDMC: Supply/Demand + Liquidity + BOS/CHOCH
#
# Optimised engine.
# Public interfaces preserved for live.py/backtest.py.
# ============================================================


MIN_TOUCHES = 2
MAX_ZONE_WIDTH = 28.0

DAILY_LOOKBACK = 220
WEEKLY_LOOKBACK = 60

DAILY_SWING = 3
WEEKLY_SWING = 2

ATR_PERIOD = 14

AOI_TOLERANCE = 8.0
MAX_RELEVANT_AOI_DISTANCE = 180.0

CONFIRM_LOOKBACK = 12
MAX_BARS_AFTER_SWEEP = 5

SWEEP_ATR_MIN = 0.05
SWEEP_ATR_MAX = 1.50

BREAK_BUFFER_ATR = 0.05
SL_ATR_BUFFER = 0.20

MIN_STOP_DISTANCE = 4.0
MAX_STOP_DISTANCE = 55.0

RISK_REWARD = 3.0

ACTIVE_START_UTC = 6
ACTIVE_END_UTC = 20


# ============================================================
# CLEANING
# ============================================================

def _clean(data):

    if data is None or len(data) == 0:
        return pd.DataFrame()

    df = data

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            c[0] if isinstance(c, tuple) else c
            for c in df.columns
        ]

    required = [
        "Open",
        "High",
        "Low",
        "Close"
    ]

    if any(
        c not in df.columns
        for c in required
    ):
        return pd.DataFrame()

    # Only copy if conversion is actually required.
    if not all(
        pd.api.types.is_numeric_dtype(df[c])
        for c in required
    ):
        df = df.copy()

        for c in required:
            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

    if "Volume" not in df.columns:
        df = df.copy()
        df["Volume"] = 1.0

    return df.dropna(
        subset=required
    )


# ============================================================
# COMPLETED DATA
# ============================================================

def _completed(data):

    if data is None or len(data) <= 1:
        return _clean(data)

    df = _clean(data)

    return df.iloc[:-1]


# ============================================================
# RESAMPLING
# ============================================================

def resample_data(data, timeframe):

    df = _clean(data)

    rules = {
        "4H": "4h",
        "1D": "1D",
        "1W": "1W"
    }

    if timeframe not in rules:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    if df.empty:
        return df

    return (
        df
        .resample(rules[timeframe])
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        })
        .dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close"
            ]
        )
    )


# ============================================================
# ATR
# ============================================================

def _atr(data, period=ATR_PERIOD):

    if data is None or len(data) < period + 1:
        return None

    df = _clean(data)

    if len(df) < period + 1:
        return None

    high = df["High"].to_numpy(
        dtype=float
    )

    low = df["Low"].to_numpy(
        dtype=float
    )

    close = df["Close"].to_numpy(
        dtype=float
    )

    previous_close = close[:-1]

    true_range = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(
                high[1:]
                - previous_close
            ),
            np.abs(
                low[1:]
                - previous_close
            )
        )
    )

    if len(true_range) < period:
        return None

    value = np.mean(
        true_range[-period:]
    )

    return (
        float(value)
        if np.isfinite(value)
        else None
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

    if len(df) < swing_length * 2 + 5:
        return df

    highs = df["High"].to_numpy(
        dtype=float
    )

    lows = df["Low"].to_numpy(
        dtype=float
    )

    n = len(df)

    swing_high = np.zeros(
        n,
        dtype=bool
    )

    swing_low = np.zeros(
        n,
        dtype=bool
    )

    # Vectorised swing detection.
    for shift in range(
        1,
        swing_length + 1
    ):

        swing_high[
            swing_length:
            n - swing_length
        ] &= (
            highs[
                swing_length:
                n - swing_length
            ]
            > highs[
                swing_length - shift:
                n - swing_length - shift
            ]
        )

        swing_low[
            swing_length:
            n - swing_length
        ] &= (
            lows[
                swing_length:
                n - swing_length
            ]
            < lows[
                swing_length - shift:
                n - swing_length - shift
            ]
        )

    # Rebuild correctly including the right side.
    swing_high[:] = False
    swing_low[:] = False

    for i in range(
        swing_length,
        n - swing_length
    ):

        current_high = highs[i]
        current_low = lows[i]

        left_high = highs[
            i - swing_length:i
        ]

        right_high = highs[
            i + 1:i + swing_length + 1
        ]

        left_low = lows[
            i - swing_length:i
        ]

        right_low = lows[
            i + 1:i + swing_length + 1
        ]

        if (
            current_high > left_high.max()
            and current_high >= right_high.max()
        ):
            swing_high[i] = True

        if (
            current_low < left_low.min()
            and current_low <= right_low.min()
        ):
            swing_low[i] = True

    structure = np.empty(
        n,
        dtype=object
    )

    structure[:] = None

    previous_high = None
    previous_low = None

    for i in range(n):

        if swing_high[i]:

            value = highs[i]

            if previous_high is not None:

                structure[i] = (
                    "HH"
                    if value > previous_high
                    else "LH"
                )

            previous_high = value

        if swing_low[i]:

            value = lows[i]

            if (
                previous_low is not None
                and structure[i] is None
            ):

                structure[i] = (
                    "HL"
                    if value > previous_low
                    else "LL"
                )

            previous_low = value

    result = df.copy()

    result["swing_high"] = swing_high
    result["swing_low"] = swing_low
    result["structure"] = structure

    return result


# ============================================================
# STRUCTURE BIAS
# ============================================================

def _structure_bias_from_swings(data):

    df = find_market_structure(
        data
    )

    if df.empty:
        return "NEUTRAL"

    values = df.loc[
        df["structure"].notna(),
        "structure"
    ].to_numpy()

    if len(values) < 4:
        return "NEUTRAL"

    recent = values[-8:]

    bullish = np.sum(
        np.isin(
            recent,
            ["HH", "HL"]
        )
    )

    bearish = np.sum(
        np.isin(
            recent,
            ["LH", "LL"]
        )
    )

    if bullish >= bearish + 2:
        return "BULLISH"

    if bearish >= bullish + 2:
        return "BEARISH"

    return "NEUTRAL"


def get_market_bias(data):

    return _structure_bias_from_swings(
        data
    )


# ============================================================
# HIGHER TIMEFRAME BIAS
# ============================================================

def get_higher_timeframe_bias(
    data_15m,
    data_daily
):

    d15 = _completed(
        data_15m
    )

    dd = _completed(
        data_daily
    )

    neutral = {
        "weekly": "NEUTRAL",
        "daily": "NEUTRAL",
        "4h": "NEUTRAL",
        "overall": "NEUTRAL",
        "score": 0
    }

    if d15.empty or dd.empty:
        return neutral

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

    if (
        daily_bias == "BULLISH"
        and four_h_bias != "BEARISH"
    ):

        overall = "BULLISH"

    elif (
        daily_bias == "BEARISH"
        and four_h_bias != "BULLISH"
    ):

        overall = "BEARISH"

    elif (
        weekly_bias == "BULLISH"
        and four_h_bias == "BULLISH"
    ):

        overall = "BULLISH"

    elif (
        weekly_bias == "BEARISH"
        and four_h_bias == "BEARISH"
    ):

        overall = "BEARISH"

    else:

        overall = "NEUTRAL"

    score = 0

    for value, weight in (
        (weekly_bias, 1),
        (daily_bias, 3),
        (four_h_bias, 2)
    ):

        if value == "BULLISH":
            score += weight

        elif value == "BEARISH":
            score -= weight

    return {
        "weekly": weekly_bias,
        "daily": daily_bias,
        "4h": four_h_bias,
        "overall": overall,
        "score": score
    }


# ============================================================
# CLUSTERING
# ============================================================

def _cluster(values):

    values = sorted(
        float(v)
        for v in values
    )

    clusters = []

    for value in values:

        target = None

        for cluster in clusters:

            if (
                value - min(cluster)
                <= MAX_ZONE_WIDTH
            ):

                target = cluster
                break

        if target is None:

            clusters.append(
                [value]
            )

        else:

            target.append(
                value
            )

    return clusters


# ============================================================
# TOUCH EVENTS
# ============================================================

def _touch_events(
    data,
    low,
    high
):

    if data.empty:
        return []

    highs = data["High"].to_numpy(
        dtype=float
    )

    lows = data["Low"].to_numpy(
        dtype=float
    )

    touched = (
        (highs >= low)
        & (lows <= high)
    )

    starts = (
        touched
        & ~np.concatenate(
            ([False], touched[:-1])
        )
    )

    positions = np.flatnonzero(
        starts
    )

    return [
        data.index[i]
        for i in positions
    ]


# ============================================================
# BUILD ZONES
# ============================================================

def _build_zones(
    data,
    swing_length,
    lookback
):

    df = _clean(data)

    if len(df) < (
        swing_length * 2 + 10
    ):
        return []

    recent = df.tail(
        lookback
    )

    structure = find_market_structure(
        recent,
        swing_length
    )

    if structure.empty:
        return []

    highs = structure[
        structure["swing_high"]
    ]

    lows = structure[
        structure["swing_low"]
    ]

    zones = []

    for zone_type, swings in (
        ("resistance", highs),
        ("support", lows)
    ):

        if swings.empty:
            continue

        prices = (
            swings["High"].to_numpy(
                dtype=float
            )
            if zone_type == "resistance"
            else
            swings["Low"].to_numpy(
                dtype=float
            )
        )

        for cluster in _cluster(
            prices
        ):

            if len(cluster) < MIN_TOUCHES:
                continue

            zone_low = float(
                min(cluster)
            )

            zone_high = float(
                max(cluster)
            )

            if (
                zone_high - zone_low
                > MAX_ZONE_WIDTH
            ):
                continue

            touches = _touch_events(
                recent,
                zone_low,
                zone_high
            )

            if len(touches) < MIN_TOUCHES:
                continue

            latest_touch = touches[-1]

            try:
                age_bars = len(
                    recent.loc[
                        latest_touch:
                    ]
                )

            except Exception:
                age_bars = lookback

            recency_score = max(
                0.0,
                1.0
                - age_bars
                / max(
                    lookback,
                    1
                )
            )

            zones.append({
                "type": zone_type,
                "low": zone_low,
                "high": zone_high,
                "width": (
                    zone_high
                    - zone_low
                ),
                "touches": len(touches),
                "touch_times": touches,
                "latest_touch": latest_touch,
                "recency_score":
                    recency_score
            })

    return zones


# ============================================================
# DEDUPE
# ============================================================

def _dedupe(zones):

    zones = sorted(
        zones,
        key=lambda z: (
            -z["touches"],
            -z.get(
                "recency_score",
                0
            ),
            z["width"]
        )
    )

    result = []

    for zone in zones:

        overlap = any(
            zone["type"] == x["type"]
            and zone["low"] <= x["high"]
            and zone["high"] >= x["low"]
            for x in result
        )

        if overlap:
            continue

        result.append(
            zone
        )

    return result


# ============================================================
# AREA OF INTEREST
# ============================================================

def find_area_of_interest(
    data,
    current_price=None,
    swing_length=3,
    touch_lookback=100
):

    df = _clean(data)

    if df.empty:
        return []

    zones = _dedupe(
        _build_zones(
            df,
            swing_length,
            min(
                len(df),
                touch_lookback
            )
        )
    )

    if current_price is None:
        return zones

    price = float(
        current_price
    )

    relevant = []

    for zone in zones:

        low = float(
            zone["low"]
        )

        high = float(
            zone["high"]
        )

        if (
            low <= price <= high
        ):

            distance = 0.0

        elif (
            zone["type"] == "support"
            and high < price
        ):

            distance = (
                price - high
            )

        elif (
            zone["type"] == "resistance"
            and low > price
        ):

            distance = (
                low - price
            )

        else:
            continue

        if (
            distance
            <= MAX_RELEVANT_AOI_DISTANCE
        ):

            item = dict(
                zone
            )

            item["distance"] = float(
                distance
            )

            relevant.append(
                item
            )

    relevant.sort(
        key=lambda z: (
            z["distance"],
            -z["touches"],
            -z.get(
                "recency_score",
                0
            )
        )
    )

    return relevant


# ============================================================
# WEEKLY / DAILY AREAS
# ============================================================

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
            "daily": []
        }

    weekly = resample_data(
        dd,
        "1W"
    )

    return {
        "weekly":
            find_area_of_interest(
                weekly,
                current_price,
                WEEKLY_SWING,
                WEEKLY_LOOKBACK
            ),

        "daily":
            find_area_of_interest(
                dd,
                current_price,
                DAILY_SWING,
                DAILY_LOOKBACK
            )
    }


# ============================================================
# CURRENT AOI
# ============================================================

def get_current_aoi(
    price,
    areas,
    tolerance=AOI_TOLERANCE
):

    matches = []

    for timeframe in (
        "daily",
        "weekly"
    ):

        for zone in areas.get(
            timeframe,
            []
        ):

            if (
                float(zone["low"])
                - tolerance
                <= float(price)
                <=
                float(zone["high"])
                + tolerance
            ):

                item = dict(
                    zone
                )

                item["timeframe"] = (
                    timeframe
                )

                matches.append(
                    item
                )

    matches.sort(
        key=lambda z: (
            -z["touches"],
            -z.get(
                "recency_score",
                0
            ),
            z["width"]
        )
    )

    return matches


def price_at_area_of_interest(
    price,
    areas,
    tolerance=AOI_TOLERANCE
):

    return get_current_aoi(
        price,
        areas,
        tolerance
    )


# ============================================================
# RECENT AOI TOUCH
# ============================================================

def find_recent_aoi_touch(
    data_15m,
    areas,
    lookback=CONFIRM_LOOKBACK,
    tolerance=AOI_TOLERANCE
):

    df = _clean(
        data_15m
    ).tail(
        lookback
    )

    out = []

    if df.empty:
        return out

    highs = df["High"].to_numpy(
        dtype=float
    )

    lows = df["Low"].to_numpy(
        dtype=float
    )

    for timeframe in (
        "daily",
        "weekly"
    ):

        for zone in areas.get(
            timeframe,
            []
        ):

            low = (
                float(zone["low"])
                - tolerance
            )

            high = (
                float(zone["high"])
                + tolerance
            )

            touched = (
                (highs >= low)
                & (lows <= high)
            )

            positions = np.flatnonzero(
                touched
            )

            if len(positions):

                item = dict(
                    zone
                )

                item["timeframe"] = (
                    timeframe
                )

                item["touch_time"] = (
                    df.index[
                        positions[-1]
                    ]
                )

                out.append(
                    item
                )

    return out


# ============================================================
# LIQUIDITY SWEEP
# ============================================================

def _find_liquidity_sweep(
    data,
    aoi,
    direction
):

    df = _completed(
        data
    )

    if len(df) < (
        ATR_PERIOD + 5
    ):
        return None

    atr = _atr(
        df
    )

    if atr is None:
        return None

    recent = df.tail(
        CONFIRM_LOOKBACK
    )

    zone_low = float(
        aoi["low"]
    )

    zone_high = float(
        aoi["high"]
    )

    opens = recent["Open"].to_numpy(
        dtype=float
    )

    highs = recent["High"].to_numpy(
        dtype=float
    )

    lows = recent["Low"].to_numpy(
        dtype=float
    )

    closes = recent["Close"].to_numpy(
        dtype=float
    )

    if direction == "BUY":

        penetration = (
            zone_low - lows
        )

        mask = (
            (lows < zone_low)
            & (closes > zone_low)
            & (
                penetration
                >= atr * SWEEP_ATR_MIN
            )
            & (
                penetration
                <= atr * SWEEP_ATR_MAX
            )
            & (closes > opens)
        )

    else:

        penetration = (
            highs - zone_high
        )

        mask = (
            (highs > zone_high)
            & (closes < zone_high)
            & (
                penetration
                >= atr * SWEEP_ATR_MIN
            )
            & (
                penetration
                <= atr * SWEEP_ATR_MAX
            )
            & (closes < opens)
        )

    positions = np.flatnonzero(
        mask
    )

    if len(positions) == 0:
        return None

    position = positions[-1]

    if direction == "BUY":

        extreme = lows[
            position
        ]

        level = zone_low

    else:

        extreme = highs[
            position
        ]

        level = zone_high

    return {
        "index":
            recent.index[position],

        "extreme":
            float(extreme),

        "level":
            float(level),

        "atr":
            float(atr)
    }


# ============================================================
# STRUCTURE BREAK
# ============================================================

def _confirm_structure_break(
    data,
    sweep,
    direction
):

    df = _completed(
        data
    )

    if (
        df.empty
        or sweep is None
    ):
        return None

    try:

        sweep_pos = (
            df.index
            .get_loc(
                sweep["index"]
            )
        )

        if isinstance(
            sweep_pos,
            slice
        ):
            sweep_pos = sweep_pos.stop - 1

    except KeyError:

        return None

    sweep_pos = int(
        sweep_pos
    )

    if (
        sweep_pos
        >= len(df) - 1
    ):
        return None

    atr = float(
        sweep.get(
            "atr",
            _atr(df) or 10.0
        )
    )

    end = min(
        len(df),
        sweep_pos
        + MAX_BARS_AFTER_SWEEP
        + 2
    )

    if end <= sweep_pos + 1:
        return None

    sweep_row = df.iloc[
        sweep_pos
    ]

    previous = (
        df.iloc[
            sweep_pos - 1
        ]
        if sweep_pos > 0
        else sweep_row
    )

    if direction == "BUY":

        break_level = (
            max(
                float(
                    sweep_row["High"]
                ),
                float(
                    previous["High"]
                )
            )
            + atr
            * BREAK_BUFFER_ATR
        )

        closes = df["Close"].to_numpy(
            dtype=float
        )

        positions = np.flatnonzero(
            closes[
                sweep_pos + 1:end
            ]
            > break_level
        )

    else:

        break_level = (
            min(
                float(
                    sweep_row["Low"]
                ),
                float(
                    previous["Low"]
                )
            )
            - atr
            * BREAK_BUFFER_ATR
        )

        closes = df["Close"].to_numpy(
            dtype=float
        )

        positions = np.flatnonzero(
            closes[
                sweep_pos + 1:end
            ]
            < break_level
        )

    if len(positions) == 0:
        return None

    position = (
        sweep_pos
        + 1
        + int(positions[0])
    )

    return {
        "index":
            df.index[position],

        "break_level":
            float(break_level),

        "sweep":
            sweep
    }


# ============================================================
# ENTRY CONFIRMATION
# ============================================================

def get_entry_confirmation(
    data,
    aoi=None
):

    if aoi is None:
        return "NONE"

    direction = (
        "BUY"
        if aoi["type"] == "support"
        else "SELL"
    )

    sweep = _find_liquidity_sweep(
        data,
        aoi,
        direction
    )

    if sweep is None:
        return "NONE"

    confirmation = (
        _confirm_structure_break(
            data,
            sweep,
            direction
        )
    )

    return (
        direction
        if confirmation is not None
        else "NONE"
    )


# ============================================================
# FVG
# ============================================================

def _find_recent_fvg(
    data,
    direction,
    lookback=8
):

    df = _completed(
        data
    ).tail(
        lookback + 2
    )

    if len(df) < 3:
        return None

    highs = df["High"].to_numpy(
        dtype=float
    )

    lows = df["Low"].to_numpy(
        dtype=float
    )

    for i in range(
        len(df) - 1,
        1,
        -1
    ):

        if (
            direction == "BUY"
            and lows[i] > highs[i - 2]
        ):

            return {
                "low":
                    float(
                        highs[i - 2]
                    ),

                "high":
                    float(
                        lows[i]
                    ),

                "index":
                    df.index[i]
            }

        if (
            direction == "SELL"
            and highs[i] < lows[i - 2]
        ):

            return {
                "low":
                    float(
                        highs[i]
                    ),

                "high":
                    float(
                        lows[i - 2]
                    ),

                "index":
                    df.index[i]
            }

    return None


# ============================================================
# ORDER BLOCK
# ============================================================

def _find_recent_order_block(
    data,
    direction,
    lookback=8
):

    df = _completed(
        data
    ).tail(
        lookback + 4
    )

    if len(df) < 4:
        return None

    atr = _atr(
        df
    )

    if atr is None:
        return None

    opens = df["Open"].to_numpy(
        dtype=float
    )

    closes = df["Close"].to_numpy(
        dtype=float
    )

    highs = df["High"].to_numpy(
        dtype=float
    )

    lows = df["Low"].to_numpy(
        dtype=float
    )

    for i in range(
        len(df) - 1,
        1,
        -1
    ):

        body = abs(
            closes[i]
            - opens[i]
        )

        if (
            direction == "BUY"
            and closes[i - 1]
            < opens[i - 1]
            and closes[i]
            > opens[i]
            and body >= atr * 0.8
        ):

            return {
                "low":
                    float(
                        lows[i - 1]
                    ),

                "high":
                    float(
                        highs[i - 1]
                    ),

                "index":
                    df.index[i - 1]
            }

        if (
            direction == "SELL"
            and closes[i - 1]
            > opens[i - 1]
            and closes[i]
            < opens[i]
            and body >= atr * 0.8
        ):

            return {
                "low":
                    float(
                        lows[i - 1]
                    ),

                "high":
                    float(
                        highs[i - 1]
                    ),

                "index":
                    df.index[i - 1]
            }

    return None


# ============================================================
# CONFLUENCE
# ============================================================

def _confirmation_confluence(
    data,
    aoi,
    direction
):

    fvg = _find_recent_fvg(
        data,
        direction
    )

    ob = _find_recent_order_block(
        data,
        direction
    )

    zone_low = float(
        aoi["low"]
    )

    zone_high = float(
        aoi["high"]
    )

    score = 0

    if (
        fvg is not None
        and not (
            fvg["high"] < zone_low
            or fvg["low"] > zone_high
        )
    ):
        score += 1

    if (
        ob is not None
        and not (
            ob["high"] < zone_low
            or ob["low"] > zone_high
        )
    ):
        score += 1

    return {
        "score": score,
        "fvg": fvg,
        "order_block": ob
    }


# ============================================================
# AOI SELECTION
# ============================================================

def _select_aoi(
    price,
    areas,
    direction
):

    desired_type = (
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

            if (
                zone["type"]
                != desired_type
            ):
                continue

            distance = float(
                zone.get(
                    "distance",
                    999999
                )
            )

            if (
                distance
                > MAX_RELEVANT_AOI_DISTANCE
            ):
                continue

            timeframe_bonus = (
                8.0
                if timeframe == "daily"
                else 5.0
            )

            score = (
                zone["touches"]
                * 8.0
                + zone.get(
                    "recency_score",
                    0.0
                ) * 12.0
                - zone["width"]
                * 0.20
                - distance
                * 0.08
                + timeframe_bonus
            )

            item = dict(
                zone
            )

            item["timeframe"] = (
                timeframe
            )

            item["selection_score"] = (
                score
            )

            candidates.append(
                item
            )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda z:
            z["selection_score"]
    )


# ============================================================
# SESSION
# ============================================================

def _active_session(index):

    try:
        hour = int(
            index.hour
        )

    except Exception:
        return True

    return (
        ACTIVE_START_UTC
        <= hour
        < ACTIVE_END_UTC
    )


# ============================================================
# SL / TP
# ============================================================

def calculate_sl_tp(
    signal,
    entry,
    aoi,
    data_15m=None
):

    if (
        signal not in (
            "BUY",
            "SELL"
        )
        or aoi is None
    ):
        return None

    entry = float(
        entry
    )

    atr = (
        _atr(data_15m)
        if data_15m is not None
        else None
    )

    if atr is None:
        atr = 10.0

    sweep = aoi.get(
        "sweep"
    )

    if signal == "SELL":

        structural_level = float(
            aoi["high"]
        )

        if sweep is not None:

            structural_level = max(
                structural_level,
                float(
                    sweep.get(
                        "extreme",
                        structural_level
                    )
                )
            )

        stop_loss = (
            structural_level
            + max(
                MIN_STOP_DISTANCE,
                atr
                * SL_ATR_BUFFER
            )
        )

        risk = (
            stop_loss
            - entry
        )

        take_profit = (
            entry
            - risk
            * RISK_REWARD
        )

    else:

        structural_level = float(
            aoi["low"]
        )

        if sweep is not None:

            structural_level = min(
                structural_level,
                float(
                    sweep.get(
                        "extreme",
                        structural_level
                    )
                )
            )

        stop_loss = (
            structural_level
            - max(
                MIN_STOP_DISTANCE,
                atr
                * SL_ATR_BUFFER
            )
        )

        risk = (
            entry
            - stop_loss
        )

        take_profit = (
            entry
            + risk
            * RISK_REWARD
        )

    if (
        risk <= 0
        or risk > MAX_STOP_DISTANCE
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
        "risk": float(
            risk
        ),
        "reward": float(
            abs(
                take_profit
                - entry
            )
        ),
        "risk_reward":
            RISK_REWARD,
        "atr":
            float(atr)
    }


# ============================================================
# TARGET SPACE
# ============================================================

def _target_space(
    entry,
    tp,
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

            if (
                zone["type"]
                != opposing
            ):
                continue

            low = float(
                zone["low"]
            )

            high = float(
                zone["high"]
            )

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

    nearest = (
        max(candidates)
        if direction == "SELL"
        else min(candidates)
    )

    return (
        tp > nearest
        if direction == "SELL"
        else tp < nearest
    )


# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_signal(
    data_15m,
    data_daily,
    current_price
):

    # Keep only the data actually needed.
    d15 = _completed(
        data_15m
    ).tail(
        1005
    )

    dd = _completed(
        data_daily
    ).tail(
        DAILY_LOOKBACK + 5
    )

    price = float(
        current_price
    )

    neutral_bias = {
        "weekly": "NEUTRAL",
        "daily": "NEUTRAL",
        "4h": "NEUTRAL",
        "overall": "NEUTRAL",
        "score": 0
    }

    if (
        d15.empty
        or dd.empty
    ):

        return {
            "signal": "NONE",
            "reason":
                "INSUFFICIENT_DATA",
            "bias":
                neutral_bias,
            "aoi": None
        }

    # --------------------------------------------------------
    # Higher timeframe bias
    # --------------------------------------------------------

    bias = get_higher_timeframe_bias(
        d15,
        dd
    )

    overall = bias[
        "overall"
    ]

    if overall == "NEUTRAL":

        return {
            "signal": "NONE",
            "reason":
                "NEUTRAL_HIGHER_TIMEFRAME",
            "bias": bias,
            "aoi": None
        }

    direction = (
        "BUY"
        if overall == "BULLISH"
        else "SELL"
    )

    if (
        direction == "BUY"
        and bias["4h"] == "BEARISH"
    ):

        return {
            "signal": "NONE",
            "reason":
                "4H_CONTRADICTS_BUY",
            "bias": bias,
            "aoi": None
        }

    if (
        direction == "SELL"
        and bias["4h"] == "BULLISH"
    ):

        return {
            "signal": "NONE",
            "reason":
                "4H_CONTRADICTS_SELL",
            "bias": bias,
            "aoi": None
        }

    # --------------------------------------------------------
    # AOI
    # --------------------------------------------------------

    areas = get_weekly_daily_areas(
        dd,
        current_price=price
    )

    aoi = _select_aoi(
        price,
        areas,
        direction
    )

    if aoi is None:

        return {
            "signal": "NONE",
            "reason":
                "WAITING_FOR_RELEVANT_AOI",
            "bias": bias,
            "aoi": None
        }

    if (
        float(
            aoi.get(
                "distance",
                999999
            )
        )
        > AOI_TOLERANCE
    ):

        return {
            "signal": "NONE",
            "reason":
                "WAITING_FOR_AOI",
            "bias": bias,
            "aoi": aoi
        }

    # --------------------------------------------------------
    # Session
    # --------------------------------------------------------

    if not _active_session(
        d15.index[-1]
    ):

        return {
            "signal": "NONE",
            "reason":
                "OUTSIDE_ACTIVE_SESSION",
            "bias": bias,
            "aoi": aoi
        }

    # --------------------------------------------------------
    # Liquidity sweep
    # --------------------------------------------------------

    sweep = _find_liquidity_sweep(
        d15,
        aoi,
        direction
    )

    if sweep is None:

        return {
            "signal": "NONE",
            "reason":
                "WAITING_FOR_LIQUIDITY_SWEEP",
            "bias": bias,
            "aoi": aoi
        }

    # --------------------------------------------------------
    # BOS / CHOCH
    # --------------------------------------------------------

    confirmation = (
        _confirm_structure_break(
            d15,
            sweep,
            direction
        )
    )

    if confirmation is None:

        return {
            "signal": "NONE",
            "reason":
                "WAITING_FOR_STRUCTURE_BREAK",
            "bias": bias,
            "aoi": aoi,
            "sweep": sweep
        }

    # --------------------------------------------------------
    # Confirmation
    # --------------------------------------------------------

    aoi_with_sweep = dict(
        aoi
    )

    aoi_with_sweep[
        "sweep"
    ] = sweep

    aoi_with_sweep[
        "confirmation"
    ] = confirmation

    aoi_with_sweep[
        "confluence"
    ] = _confirmation_confluence(
        d15,
        aoi_with_sweep,
        direction
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    levels = calculate_sl_tp(
        direction,
        price,
        aoi_with_sweep,
        d15
    )

    if levels is None:

        return {
            "signal": "NONE",
            "reason":
                "INVALID_RISK",
            "bias": bias,
            "aoi":
                aoi_with_sweep
        }

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if not _target_space(
        levels["entry"],
        levels["take_profit"],
        areas,
        direction
    ):

        return {
            "signal": "NONE",
            "reason":
                "INSUFFICIENT_TARGET_SPACE",
            "bias": bias,
            "aoi":
                aoi_with_sweep
        }

    return {
        "signal": direction,
        "reason":
            "SDMC_LIQUIDITY_SWEEP_BOS_CHOCH",
        "bias": bias,
        "aoi":
            aoi_with_sweep,
        **levels
    }
