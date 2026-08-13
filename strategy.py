import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PIP_SIZE = 0.1

MIN_TOUCHES = 2

MIN_ZONE_PIPS = 5
MAX_ZONE_PIPS = 300

MIN_ZONE_WIDTH = MIN_ZONE_PIPS * PIP_SIZE
MAX_ZONE_WIDTH = MAX_ZONE_PIPS * PIP_SIZE

AOI_TOLERANCE = 5.0

SL_BUFFER = 5.0

RISK_REWARD = 2.0

# Maximum distance from current price for an AOI
# to be considered relevant.
MAX_RELEVANT_AOI_DISTANCE = 300.0

# Number of recent candles used when counting
# actual interactions with an AOI.
RECENT_TOUCH_LOOKBACK_DAILY = 250
RECENT_TOUCH_LOOKBACK_WEEKLY = 52


# ============================================================
# MARKET STRUCTURE
# ============================================================

def find_market_structure(
    data: pd.DataFrame,
    swing_length: int = 3
):

    data = data.copy()

    data["swing_high"] = False
    data["swing_low"] = False
    data["structure"] = None

    previous_swing_high = None
    previous_swing_low = None

    for i in range(
        swing_length,
        len(data) - swing_length
    ):

        high = float(data["High"].iloc[i])
        low = float(data["Low"].iloc[i])

        left_highs = data["High"].iloc[
            i - swing_length:i
        ]

        right_highs = data["High"].iloc[
            i + 1:i + swing_length + 1
        ]

        left_lows = data["Low"].iloc[
            i - swing_length:i
        ]

        right_lows = data["Low"].iloc[
            i + 1:i + swing_length + 1
        ]

        # ----------------------------------------------------
        # SWING HIGH
        # ----------------------------------------------------

        if (
            high > left_highs.max()
            and high > right_highs.max()
        ):

            data.loc[
                data.index[i],
                "swing_high"
            ] = True

            if previous_swing_high is not None:

                if high > previous_swing_high:

                    data.loc[
                        data.index[i],
                        "structure"
                    ] = "HH"

                else:

                    data.loc[
                        data.index[i],
                        "structure"
                    ] = "LH"

            previous_swing_high = high

        # ----------------------------------------------------
        # SWING LOW
        # ----------------------------------------------------

        if (
            low < left_lows.min()
            and low < right_lows.min()
        ):

            data.loc[
                data.index[i],
                "swing_low"
            ] = True

            if previous_swing_low is not None:

                if low > previous_swing_low:

                    data.loc[
                        data.index[i],
                        "structure"
                    ] = "HL"

                else:

                    data.loc[
                        data.index[i],
                        "structure"
                    ] = "LL"

            previous_swing_low = low

    return data


# ============================================================
# MARKET BIAS
# ============================================================

def get_market_bias(
    data: pd.DataFrame
):

    structures = (
        data[
            data["structure"].notna()
        ]["structure"]
        .tolist()
    )

    if len(structures) < 2:
        return "NEUTRAL"

    recent = structures[-6:]

    bullish_points = sum(
        point in ("HH", "HL")
        for point in recent
    )

    bearish_points = sum(
        point in ("LH", "LL")
        for point in recent
    )

    if bullish_points > bearish_points:
        return "BULLISH"

    if bearish_points > bullish_points:
        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# TIMEFRAME RESAMPLING
# ============================================================

def resample_data(
    data: pd.DataFrame,
    timeframe: str
):

    rules = {
        "4H": "4h",
        "1D": "1D",
        "1W": "1W"
    }

    if timeframe not in rules:

        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    result = data.resample(
        rules[timeframe]
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    return result.dropna()


# ============================================================
# HIGHER TIMEFRAME BIAS
# ============================================================

def get_higher_timeframe_bias(
    data_15m: pd.DataFrame,
    data_daily: pd.DataFrame
):

    weekly_data = resample_data(
        data_daily,
        "1W"
    )

    weekly_structure = find_market_structure(
        weekly_data
    )

    weekly_bias = get_market_bias(
        weekly_structure
    )

    daily_structure = find_market_structure(
        data_daily
    )

    daily_bias = get_market_bias(
        daily_structure
    )

    data_4h = resample_data(
        data_15m,
        "4H"
    )

    structure_4h = find_market_structure(
        data_4h
    )

    bias_4h = get_market_bias(
        structure_4h
    )

    biases = [
        weekly_bias,
        daily_bias,
        bias_4h
    ]

    bullish = biases.count(
        "BULLISH"
    )

    bearish = biases.count(
        "BEARISH"
    )

    if bullish >= 2:

        overall = "BULLISH"

    elif bearish >= 2:

        overall = "BEARISH"

    else:

        overall = "NEUTRAL"

    return {
        "weekly": weekly_bias,
        "daily": daily_bias,
        "4h": bias_4h,
        "overall": overall
    }


# ============================================================
# STRUCTURE RANGE
# ============================================================

def get_valid_structure_range(
    structure: pd.DataFrame
):

    points = []

    for i in range(
        len(structure)
    ):

        row = structure.iloc[i]

        if pd.isna(
            row["structure"]
        ):
            continue

        points.append({
            "index": i,
            "high": float(
                row["High"]
            ),
            "low": float(
                row["Low"]
            ),
            "structure": row[
                "structure"
            ]
        })

    if len(points) < 2:
        return None

    recent = points[-8:]

    recent_hh = [
        p for p in recent
        if p["structure"] == "HH"
    ]

    recent_hl = [
        p for p in recent
        if p["structure"] == "HL"
    ]

    recent_lh = [
        p for p in recent
        if p["structure"] == "LH"
    ]

    recent_ll = [
        p for p in recent
        if p["structure"] == "LL"
    ]

    if recent_hh and recent_hl:

        hh = recent_hh[-1]
        hl = recent_hl[-1]

        if hh["index"] > hl["index"]:

            return {
                "bias": "BULLISH",
                "low": hl["low"],
                "high": hh["high"]
            }

    if recent_lh and recent_ll:

        lh = recent_lh[-1]
        ll = recent_ll[-1]

        if lh["index"] > ll["index"]:

            return {
                "bias": "BEARISH",
                "low": ll["low"],
                "high": lh["high"]
            }

    return None


# ============================================================
# COUNT ACTUAL AOI TOUCHES
# ============================================================

def count_zone_touches(
    data: pd.DataFrame,
    zone_low: float,
    zone_high: float,
    lookback: int
):

    if data.empty:
        return 0

    recent = data.tail(
        lookback
    )

    touches = 0

    currently_touching = False

    for _, candle in recent.iterrows():

        candle_high = float(
            candle["High"]
        )

        candle_low = float(
            candle["Low"]
        )

        candle_touches_zone = (
            candle_high >= zone_low
            and
            candle_low <= zone_high
        )

        # Count a touch event only when price
        # enters the zone from outside.
        if candle_touches_zone:

            if not currently_touching:

                touches += 1

            currently_touching = True

        else:

            currently_touching = False

    return touches


# ============================================================
# AREA OF INTEREST
# ============================================================

def find_area_of_interest(
    data: pd.DataFrame,
    current_price: float = None,
    swing_length: int = 3,
    touch_lookback: int = 100
):

    if data.empty:
        return []

    structure = find_market_structure(
        data,
        swing_length
    )

    bias = get_market_bias(
        structure
    )

    swing_highs = structure[
        structure["swing_high"]
    ]

    swing_lows = structure[
        structure["swing_low"]
    ]

    zones = []

    # ========================================================
    # SUPPORT
    # ========================================================

    support_prices = [
        float(price)
        for price in swing_lows["Low"].tolist()
    ]

    support_prices.sort()

    processed_support = set()

    for anchor in support_prices:

        nearby = [
            price
            for price in support_prices
            if abs(
                price - anchor
            ) <= MAX_ZONE_WIDTH
        ]

        if len(nearby) < MIN_TOUCHES:
            continue

        zone_low = min(
            nearby
        )

        zone_high = max(
            nearby
        )

        width = (
            zone_high
            - zone_low
        )

        if width < MIN_ZONE_WIDTH:
            continue

        if width > MAX_ZONE_WIDTH:
            continue

        key = (
            round(zone_low, 4),
            round(zone_high, 4)
        )

        if key in processed_support:
            continue

        processed_support.add(
            key
        )

        actual_touches = count_zone_touches(
            data,
            zone_low,
            zone_high,
            touch_lookback
        )

        if actual_touches < MIN_TOUCHES:
            continue

        zones.append({
            "type": "support",
            "low": zone_low,
            "high": zone_high,
            "width": width,
            "touches": actual_touches,
            "structure_bias": bias
        })

    # ========================================================
    # RESISTANCE
    # ========================================================

    resistance_prices = [
        float(price)
        for price in swing_highs["High"].tolist()
    ]

    resistance_prices.sort()

    processed_resistance = set()

    for anchor in resistance_prices:

        nearby = [
            price
            for price in resistance_prices
            if abs(
                price - anchor
            ) <= MAX_ZONE_WIDTH
        ]

        if len(nearby) < MIN_TOUCHES:
            continue

        zone_low = min(
            nearby
        )

        zone_high = max(
            nearby
        )

        width = (
            zone_high
            - zone_low
        )

        if width < MIN_ZONE_WIDTH:
            continue

        if width > MAX_ZONE_WIDTH:
            continue

        key = (
            round(zone_low, 4),
            round(zone_high, 4)
        )

        if key in processed_resistance:
            continue

        processed_resistance.add(
            key
        )

        actual_touches = count_zone_touches(
            data,
            zone_low,
            zone_high,
            touch_lookback
        )

        if actual_touches < MIN_TOUCHES:
            continue

        zones.append({
            "type": "resistance",
            "low": zone_low,
            "high": zone_high,
            "width": width,
            "touches": actual_touches,
            "structure_bias": bias
        })

    # ========================================================
    # REMOVE OVERLAPPING DUPLICATES
    # ========================================================

    zones.sort(
        key=lambda zone: (
            -zone["touches"],
            zone["width"]
        )
    )

    cleaned = []

    for zone in zones:

        duplicate = False

        for existing in cleaned:

            same_type = (
                zone["type"]
                == existing["type"]
            )

            overlaps = (
                zone["low"]
                <= existing["high"]
                and
                zone["high"]
                >= existing["low"]
            )

            if same_type and overlaps:

                duplicate = True
                break

        if not duplicate:

            cleaned.append(
                zone
            )

    # ========================================================
    # CURRENT PRICE FILTER
    # ========================================================

    if current_price is not None:

        relevant = []

        current_price = float(
            current_price
        )

        for zone in cleaned:

            zone_low = float(
                zone["low"]
            )

            zone_high = float(
                zone["high"]
            )

            # Price currently inside the zone.
            if (
                zone_low
                <= current_price
                <= zone_high
            ):

                relevant.append(
                    zone
                )

                continue

            # Support should be below price.
            if (
                zone["type"] == "support"
                and zone_high < current_price
            ):

                distance = (
                    current_price
                    - zone_high
                )

                if (
                    distance
                    <= MAX_RELEVANT_AOI_DISTANCE
                ):

                    relevant.append(
                        zone
                    )

                continue

            # Resistance should be above price.
            if (
                zone["type"] == "resistance"
                and zone_low > current_price
            ):

                distance = (
                    zone_low
                    - current_price
                )

                if (
                    distance
                    <= MAX_RELEVANT_AOI_DISTANCE
                ):

                    relevant.append(
                        zone
                    )

        cleaned = relevant

        # Sort by distance from current price.
        cleaned.sort(
            key=lambda zone: (
                min(
                    abs(
                        current_price
                        - float(zone["low"])
                    ),
                    abs(
                        current_price
                        - float(zone["high"])
                    )
                ),
                -zone["touches"]
            )
        )

    return cleaned


# ============================================================
# WEEKLY + DAILY AREAS
# ============================================================

def get_weekly_daily_areas(
    data_daily: pd.DataFrame,
    current_price: float = None
):

    weekly_data = resample_data(
        data_daily,
        "1W"
    )

    return {
        "weekly": find_area_of_interest(
            weekly_data,
            current_price=current_price,
            touch_lookback=RECENT_TOUCH_LOOKBACK_WEEKLY
        ),

        "daily": find_area_of_interest(
            data_daily,
            current_price=current_price,
            touch_lookback=RECENT_TOUCH_LOOKBACK_DAILY
        )
    }


# ============================================================
# CURRENT PRICE AT AOI
# ============================================================

def get_current_aoi(
    price: float,
    areas: dict,
    tolerance: float = AOI_TOLERANCE
):

    matches = []

    for timeframe in (
        "weekly",
        "daily"
    ):

        for zone in areas.get(
            timeframe,
            []
        ):

            expanded_low = (
                zone["low"]
                - tolerance
            )

            expanded_high = (
                zone["high"]
                + tolerance
            )

            if (
                expanded_low
                <= price
                <= expanded_high
            ):

                matches.append({
                    "timeframe": timeframe,
                    "type": zone["type"],
                    "low": zone["low"],
                    "high": zone["high"],
                    "touches": zone["touches"],
                    "structure_bias":
                        zone["structure_bias"]
                })

    return matches


# ============================================================
# PRICE AT AREA OF INTEREST
# ============================================================

def price_at_area_of_interest(
    price: float,
    areas: dict,
    tolerance: float = AOI_TOLERANCE
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
    data_15m: pd.DataFrame,
    areas: dict,
    lookback: int = 20,
    tolerance: float = AOI_TOLERANCE
):

    recent = data_15m.tail(
        lookback
    )

    matches = []

    for timeframe in (
        "weekly",
        "daily"
    ):

        for zone in areas.get(
            timeframe,
            []
        ):

            expanded_low = (
                zone["low"]
                - tolerance
            )

            expanded_high = (
                zone["high"]
                + tolerance
            )

            for _, candle in recent.iterrows():

                candle_high = float(
                    candle["High"]
                )

                candle_low = float(
                    candle["Low"]
                )

                touched = (
                    candle_high >= expanded_low
                    and
                    candle_low <= expanded_high
                )

                if touched:

                    matches.append({
                        "timeframe": timeframe,
                        "type": zone["type"],
                        "low": zone["low"],
                        "high": zone["high"],
                        "touches": zone["touches"],
                        "structure_bias":
                            zone["structure_bias"],
                        "touch_time":
                            candle.name
                    })

                    break

    return matches


# ============================================================
# CANDLE HELPERS
# ============================================================

def candle_body(row):

    return abs(
        float(row["Close"])
        - float(row["Open"])
    )


def candle_range(row):

    return (
        float(row["High"])
        - float(row["Low"])
    )


def is_bullish(row):

    return (
        float(row["Close"])
        > float(row["Open"])
    )


def is_bearish(row):

    return (
        float(row["Close"])
        < float(row["Open"])
    )


# ============================================================
# BULLISH ENGULFING
# ============================================================

def is_bullish_engulfing(
    data: pd.DataFrame
):

    if len(data) < 3:
        return False

    previous_two = data.iloc[-3:-1]
    current = data.iloc[-1]

    if not is_bullish(
        current
    ):
        return False

    current_body_low = min(
        float(current["Open"]),
        float(current["Close"])
    )

    current_body_high = max(
        float(current["Open"]),
        float(current["Close"])
    )

    for _, candle in previous_two.iterrows():

        if not is_bearish(
            candle
        ):
            return False

        previous_body_low = min(
            float(candle["Open"]),
            float(candle["Close"])
        )

        previous_body_high = max(
            float(candle["Open"]),
            float(candle["Close"])
        )

        if (
            current_body_low
            > previous_body_low
        ):

            return False

        if (
            current_body_high
            < previous_body_high
        ):

            return False

    return True


# ============================================================
# BEARISH ENGULFING
# ============================================================

def is_bearish_engulfing(
    data: pd.DataFrame
):

    if len(data) < 3:
        return False

    previous_two = data.iloc[-3:-1]
    current = data.iloc[-1]

    if not is_bearish(
        current
    ):
        return False

    current_body_low = min(
        float(current["Open"]),
        float(current["Close"])
    )

    current_body_high = max(
        float(current["Open"]),
        float(current["Close"])
    )

    for _, candle in previous_two.iterrows():

        if not is_bullish(
            candle
        ):
            return False

        previous_body_low = min(
            float(candle["Open"]),
            float(candle["Close"])
        )

        previous_body_high = max(
            float(candle["Open"]),
            float(candle["Close"])
        )

        if (
            current_body_low
            > previous_body_low
        ):

            return False

        if (
            current_body_high
            < previous_body_high
        ):

            return False

    return True


# ============================================================
# HAMMER
# ============================================================

def is_hammer(row):

    body = candle_body(
        row
    )

    total_range = candle_range(
        row
    )

    if total_range <= 0:
        return False

    upper_wick = (
        float(row["High"])
        - max(
            float(row["Open"]),
            float(row["Close"])
        )
    )

    lower_wick = (
        min(
            float(row["Open"]),
            float(row["Close"])
        )
        - float(row["Low"])
    )

    return (
        lower_wick >= body * 2
        and
        lower_wick > upper_wick
        and
        body / total_range <= 0.4
    )


# ============================================================
# SHOOTING STAR
# ============================================================

def is_shooting_star(row):

    body = candle_body(
        row
    )

    total_range = candle_range(
        row
    )

    if total_range <= 0:
        return False

    upper_wick = (
        float(row["High"])
        - max(
            float(row["Open"]),
            float(row["Close"])
        )
    )

    lower_wick = (
        min(
            float(row["Open"]),
            float(row["Close"])
        )
        - float(row["Low"])
    )

    return (
        upper_wick >= body * 2
        and
        upper_wick > lower_wick
        and
        body / total_range <= 0.4
    )


# ============================================================
# STRONG BULLISH AOI REJECTION
# ============================================================

def is_bullish_aoi_rejection(
    row,
    aoi
):

    if aoi is None:
        return False

    if aoi["type"] != "support":
        return False

    if not is_bullish(
        row
    ):
        return False

    body = candle_body(
        row
    )

    total_range = candle_range(
        row
    )

    if total_range <= 0:
        return False

    close = float(
        row["Close"]
    )

    low = float(
        row["Low"]
    )

    touched_aoi = (
        low
        <= aoi["high"]
    )

    if not touched_aoi:
        return False

    closed_above = (
        close
        >= aoi["high"]
    )

    if not closed_above:
        return False

    strong_body = (
        body / total_range
        >= 0.45
    )

    return strong_body


# ============================================================
# STRONG BEARISH AOI REJECTION
# ============================================================

def is_bearish_aoi_rejection(
    row,
    aoi
):

    if aoi is None:
        return False

    if aoi["type"] != "resistance":
        return False

    if not is_bearish(
        row
    ):
        return False

    body = candle_body(
        row
    )

    total_range = candle_range(
        row
    )

    if total_range <= 0:
        return False

    high = float(
        row["High"]
    )

    close = float(
        row["Close"]
    )

    touched_aoi = (
        high
        >= aoi["low"]
    )

    if not touched_aoi:
        return False

    closed_below = (
        close
        <= aoi["high"]
    )

    if not closed_below:
        return False

    strong_body = (
        body / total_range
        >= 0.45
    )

    return strong_body


# ============================================================
# MORNING STAR
# ============================================================

def is_morning_star(
    data: pd.DataFrame
):

    if len(data) < 3:
        return False

    first = data.iloc[-3]
    second = data.iloc[-2]
    third = data.iloc[-1]

    if not is_bearish(
        first
    ):
        return False

    if not is_bullish(
        third
    ):
        return False

    if not (
        is_hammer(second)
        or
        candle_body(second)
        <= candle_range(second) * 0.35
    ):

        return False

    first_body_high = max(
        float(first["Open"]),
        float(first["Close"])
    )

    third_body_low = min(
        float(third["Open"]),
        float(third["Close"])
    )

    return (
        third_body_low
        > first_body_high
    )


# ============================================================
# EVENING STAR
# ============================================================

def is_evening_star(
    data: pd.DataFrame
):

    if len(data) < 3:
        return False

    first = data.iloc[-3]
    second = data.iloc[-2]
    third = data.iloc[-1]

    if not is_bullish(
        first
    ):
        return False

    if not is_bearish(
        third
    ):
        return False

    if not (
        is_shooting_star(second)
        or
        candle_body(second)
        <= candle_range(second) * 0.35
    ):

        return False

    first_body_low = min(
        float(first["Open"]),
        float(first["Close"])
    )

    third_body_high = max(
        float(third["Open"]),
        float(third["Close"])
    )

    return (
        third_body_high
        < first_body_low
    )


# ============================================================
# ENTRY CONFIRMATION
# ============================================================

def get_entry_confirmation(
    data: pd.DataFrame,
    aoi=None
):

    if len(data) < 3:
        return "NONE"

    current = data.iloc[-1]

    # --------------------------------------------------------
    # AOI REJECTION
    # --------------------------------------------------------

    if is_bearish_aoi_rejection(
        current,
        aoi
    ):

        return "SELL"

    if is_bullish_aoi_rejection(
        current,
        aoi
    ):

        return "BUY"

    # --------------------------------------------------------
    # STANDARD CANDLE PATTERNS
    # --------------------------------------------------------

    if is_morning_star(
        data
    ):

        return "BUY"

    if is_bullish_engulfing(
        data
    ):

        return "BUY"

    if is_hammer(
        current
    ):

        return "BUY"

    if is_evening_star(
        data
    ):

        return "SELL"

    if is_bearish_engulfing(
        data
    ):

        return "SELL"

    if is_shooting_star(
        current
    ):

        return "SELL"

    return "NONE"


# ============================================================
# SL / TP
# ============================================================

def calculate_sl_tp(
    signal: str,
    entry: float,
    aoi: dict
):

    if signal == "SELL":

        stop_loss = (
            float(aoi["high"])
            + SL_BUFFER
        )

        risk = (
            stop_loss
            - entry
        )

        if risk <= 0:
            return None

        take_profit = (
            entry
            - (
                risk
                * RISK_REWARD
            )
        )

    elif signal == "BUY":

        stop_loss = (
            float(aoi["low"])
            - SL_BUFFER
        )

        risk = (
            entry
            - stop_loss
        )

        if risk <= 0:
            return None

        take_profit = (
            entry
            + (
                risk
                * RISK_REWARD
            )
        )

    else:

        return None

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk": risk,
        "reward": abs(
            take_profit
            - entry
        ),
        "risk_reward": RISK_REWARD
    }


# ============================================================
# COMPLETE SIGNAL
# ============================================================

def generate_signal(
    data_15m: pd.DataFrame,
    data_daily: pd.DataFrame,
    current_price: float
):

    # --------------------------------------------------------
    # HIGHER TIMEFRAME BIAS
    # --------------------------------------------------------

    bias = get_higher_timeframe_bias(
        data_15m,
        data_daily
    )

    # --------------------------------------------------------
    # AOIs
    # --------------------------------------------------------

    areas = get_weekly_daily_areas(
        data_daily,
        current_price=current_price
    )

    # --------------------------------------------------------
    # CURRENT PRICE AT AOI
    # --------------------------------------------------------

    current_aoi_matches = get_current_aoi(
        current_price,
        areas
    )

    if not current_aoi_matches:

        return {
            "signal": "NONE",
            "reason": "WAITING_FOR_AOI",
            "bias": bias,
            "aoi": None
        }

    # --------------------------------------------------------
    # SELECT AOI THAT MATCHES OVERALL BIAS
    # --------------------------------------------------------

    valid_aoi = None

    if bias["overall"] == "BEARISH":

        valid_aoi = next(
            (
                zone
                for zone in current_aoi_matches
                if zone["type"]
                == "resistance"
            ),
            None
        )

    elif bias["overall"] == "BULLISH":

        valid_aoi = next(
            (
                zone
                for zone in current_aoi_matches
                if zone["type"]
                == "support"
            ),
            None
        )

    if valid_aoi is None:

        return {
            "signal": "NONE",
            "reason": "AOI_DOES_NOT_MATCH_BIAS",
            "bias": bias,
            "aoi": current_aoi_matches
        }

    # --------------------------------------------------------
    # ENTRY CONFIRMATION
    # --------------------------------------------------------

    confirmation = get_entry_confirmation(
        data_15m,
        valid_aoi
    )

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    if (
        bias["overall"]
        == "BEARISH"
        and
        confirmation
        == "SELL"
    ):

        trade_levels = calculate_sl_tp(
            "SELL",
            current_price,
            valid_aoi
        )

        if trade_levels is not None:

            return {
                "signal": "SELL",
                "reason": "AOI_RETEST_CONFIRMED",
                "bias": bias,
                "aoi": valid_aoi,
                **trade_levels
            }

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    if (
        bias["overall"]
        == "BULLISH"
        and
        confirmation
        == "BUY"
    ):

        trade_levels = calculate_sl_tp(
            "BUY",
            current_price,
            valid_aoi
        )

        if trade_levels is not None:

            return {
                "signal": "BUY",
                "reason": "AOI_RETEST_CONFIRMED",
                "bias": bias,
                "aoi": valid_aoi,
                **trade_levels
            }

    # --------------------------------------------------------
    # AOI PRESENT BUT NO CONFIRMATION
    # --------------------------------------------------------

    return {
        "signal": "NONE",
        "reason": "WAITING_FOR_CONFIRMATION",
        "bias": bias,
        "aoi": [valid_aoi]
    }
