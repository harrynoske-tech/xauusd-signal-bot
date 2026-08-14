import pandas as pd
import numpy as np

# XAUUSD STRATEGY V6 - SDMC: Supply/Demand + Liquidity + BOS/CHOCH + Confirmation
# Public interfaces preserved for live.py/backtest.py.

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


def _clean(data):
    if data is None or len(data) == 0:
        return pd.DataFrame()
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    required = ["Open", "High", "Low", "Close"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Volume" not in df.columns:
        df["Volume"] = 1.0
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(1.0)
    return df.dropna(subset=required).sort_index()


def _completed(data):
    df = _clean(data)
    return df.iloc[:-1].copy() if len(df) > 1 else df


def resample_data(data, timeframe):
    df = _clean(data)
    rules = {"4H": "4h", "1D": "1D", "1W": "1W"}
    if timeframe not in rules:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    if df.empty:
        return df
    return df.resample(rules[timeframe]).agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    }).dropna(subset=["Open", "High", "Low", "Close"])


def _atr(data, period=ATR_PERIOD):
    df = _clean(data)
    if len(df) < period + 1:
        return None
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    value = tr.rolling(period).mean().iloc[-1]
    return float(value) if pd.notna(value) else None


def find_market_structure(data, swing_length=3):
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
    for i in range(swing_length, len(df) - swing_length):
        high = float(df["High"].iloc[i])
        low = float(df["Low"].iloc[i])
        if high > df["High"].iloc[i-swing_length:i].max() and high >= df["High"].iloc[i+1:i+swing_length+1].max():
            df.iloc[i, df.columns.get_loc("swing_high")] = True
            if previous_high is not None:
                df.iloc[i, df.columns.get_loc("structure")] = "HH" if high > previous_high else "LH"
            previous_high = high
        if low < df["Low"].iloc[i-swing_length:i].min() and low <= df["Low"].iloc[i+1:i+swing_length+1].min():
            df.iloc[i, df.columns.get_loc("swing_low")] = True
            if previous_low is not None and pd.isna(df["structure"].iloc[i]):
                df.iloc[i, df.columns.get_loc("structure")] = "HL" if low > previous_low else "LL"
            previous_low = low
    return df


def _structure_bias_from_swings(data):
    df = find_market_structure(data)
    if df.empty:
        return "NEUTRAL"
    points = df[df["structure"].notna()]["structure"].tolist()
    if len(points) < 4:
        return "NEUTRAL"
    recent = points[-8:]
    bullish = sum(p in ("HH", "HL") for p in recent)
    bearish = sum(p in ("LH", "LL") for p in recent)
    if bullish >= bearish + 2:
        return "BULLISH"
    if bearish >= bullish + 2:
        return "BEARISH"
    return "NEUTRAL"


def get_market_bias(data):
    return _structure_bias_from_swings(data)


def get_higher_timeframe_bias(data_15m, data_daily):
    d15 = _completed(data_15m)
    dd = _completed(data_daily)
    neutral = {"weekly":"NEUTRAL", "daily":"NEUTRAL", "4h":"NEUTRAL", "overall":"NEUTRAL", "score":0}
    if d15.empty or dd.empty:
        return neutral
    weekly = resample_data(dd, "1W")
    four_h = resample_data(d15, "4H")
    weekly_bias = get_market_bias(weekly)
    daily_bias = get_market_bias(dd)
    four_h_bias = get_market_bias(four_h)

    # Daily is primary direction; Weekly is context; 4H is execution filter.
    if daily_bias == "BULLISH" and four_h_bias != "BEARISH":
        overall = "BULLISH"
    elif daily_bias == "BEARISH" and four_h_bias != "BULLISH":
        overall = "BEARISH"
    elif weekly_bias == "BULLISH" and four_h_bias == "BULLISH":
        overall = "BULLISH"
    elif weekly_bias == "BEARISH" and four_h_bias == "BEARISH":
        overall = "BEARISH"
    else:
        overall = "NEUTRAL"

    score = 0
    for value, weight in ((weekly_bias, 1), (daily_bias, 3), (four_h_bias, 2)):
        if value == "BULLISH": score += weight
        elif value == "BEARISH": score -= weight

    return {"weekly":weekly_bias, "daily":daily_bias, "4h":four_h_bias, "overall":overall, "score":score}



def _cluster(values):
    values = sorted(float(v) for v in values)
    clusters = []
    for value in values:
        target = None
        for cluster in clusters:
            if value - min(cluster) <= MAX_ZONE_WIDTH:
                target = cluster
                break
        if target is None:
            clusters.append([value])
        else:
            target.append(value)
    return clusters


def _touch_events(data, low, high):
    df = _clean(data)
    events = []
    active = False
    for idx, row in df.iterrows():
        touched = float(row["High"]) >= low and float(row["Low"]) <= high
        if touched and not active:
            events.append(idx)
        active = touched
    return events


def _build_zones(data, swing_length, lookback):
    df = _clean(data)
    if len(df) < swing_length * 2 + 10:
        return []
    structure = find_market_structure(df, swing_length)
    recent = df.tail(lookback)
    if recent.empty:
        return []
    start = recent.index[0]
    highs = structure[structure["swing_high"] & (structure.index >= start)]
    lows = structure[structure["swing_low"] & (structure.index >= start)]
    zones = []
    for zone_type, swings in (("resistance", highs), ("support", lows)):
        if swings.empty:
            continue
        prices = swings["High"].tolist() if zone_type == "resistance" else swings["Low"].tolist()
        for cluster in _cluster(prices):
            if len(cluster) < MIN_TOUCHES:
                continue
            low, high = float(min(cluster)), float(max(cluster))
            if high - low > MAX_ZONE_WIDTH:
                continue
            touches = _touch_events(recent, low, high)
            if len(touches) < MIN_TOUCHES:
                continue
            latest_touch = touches[-1]
            try:
                age_bars = len(recent.loc[latest_touch:])
            except Exception:
                age_bars = lookback
            recency_score = max(0.0, 1.0 - age_bars / max(lookback, 1))
            zones.append({
                "type":zone_type, "low":low, "high":high,
                "width":high-low, "touches":len(touches),
                "touch_times":touches, "latest_touch":latest_touch,
                "recency_score":recency_score,
            })
    return zones


def _dedupe(zones):
    zones = sorted(zones, key=lambda z:(-z["touches"], -z.get("recency_score",0), z["width"]))
    result = []
    for zone in zones:
        if any(zone["type"] == x["type"] and zone["low"] <= x["high"] and zone["high"] >= x["low"] for x in result):
            continue
        result.append(zone)
    return result


def find_area_of_interest(data, current_price=None, swing_length=3, touch_lookback=100):
    df = _clean(data)
    if df.empty:
        return []
    zones = _dedupe(_build_zones(df, swing_length, min(len(df), touch_lookback)))
    if current_price is None:
        return zones
    price = float(current_price)
    relevant = []
    for zone in zones:
        low, high = float(zone["low"]), float(zone["high"])
        if low <= price <= high:
            distance = 0.0
        elif zone["type"] == "support" and high < price:
            distance = price - high
        elif zone["type"] == "resistance" and low > price:
            distance = low - price
        else:
            continue
        if distance <= MAX_RELEVANT_AOI_DISTANCE:
            item = dict(zone)
            item["distance"] = float(distance)
            relevant.append(item)
    relevant.sort(key=lambda z:(z["distance"], -z["touches"], -z.get("recency_score",0)))
    return relevant


def get_weekly_daily_areas(data_daily, current_price=None):
    dd = _completed(data_daily)
    if dd.empty:
        return {"weekly":[], "daily":[]}
    weekly = resample_data(dd, "1W")
    return {
        "weekly":find_area_of_interest(weekly, current_price, WEEKLY_SWING, WEEKLY_LOOKBACK),
        "daily":find_area_of_interest(dd, current_price, DAILY_SWING, DAILY_LOOKBACK),
    }


def get_current_aoi(price, areas, tolerance=AOI_TOLERANCE):
    matches = []
    for timeframe in ("daily", "weekly"):
        for zone in areas.get(timeframe, []):
            if float(zone["low"])-tolerance <= float(price) <= float(zone["high"])+tolerance:
                item = dict(zone)
                item["timeframe"] = timeframe
                matches.append(item)
    matches.sort(key=lambda z:(-z["touches"], -z.get("recency_score",0), z["width"]))
    return matches


def price_at_area_of_interest(price, areas, tolerance=AOI_TOLERANCE):
    return get_current_aoi(price, areas, tolerance)


def find_recent_aoi_touch(data_15m, areas, lookback=CONFIRM_LOOKBACK, tolerance=AOI_TOLERANCE):
    df = _clean(data_15m).tail(lookback)
    out = []
    for timeframe in ("daily", "weekly"):
        for zone in areas.get(timeframe, []):
            for idx, row in df.iterrows():
                if float(row["High"]) >= float(zone["low"])-tolerance and float(row["Low"]) <= float(zone["high"])+tolerance:
                    item = dict(zone)
                    item["timeframe"] = timeframe
                    item["touch_time"] = idx
                    out.append(item)
                    break
    return out


def _candle_range(row):
    return max(float(row["High"]) - float(row["Low"]), 1e-6)


def _find_liquidity_sweep(data, aoi, direction):
    df = _completed(data)
    if len(df) < ATR_PERIOD + 5:
        return None
    atr = _atr(df)
    if atr is None:
        return None
    recent = df.tail(CONFIRM_LOOKBACK)
    zone_low, zone_high = float(aoi["low"]), float(aoi["high"])
    for i in range(len(recent)-1, -1, -1):
        row = recent.iloc[i]
        high, low = float(row["High"]), float(row["Low"])
        close, open_ = float(row["Close"]), float(row["Open"])
        if direction == "BUY":
            penetration = zone_low - low
            if low < zone_low and close > zone_low and atr*SWEEP_ATR_MIN <= penetration <= atr*SWEEP_ATR_MAX and close > open_:
                return {"index":recent.index[i], "extreme":low, "level":zone_low, "atr":atr}
        else:
            penetration = high - zone_high
            if high > zone_high and close < zone_high and atr*SWEEP_ATR_MIN <= penetration <= atr*SWEEP_ATR_MAX and close < open_:
                return {"index":recent.index[i], "extreme":high, "level":zone_high, "atr":atr}
    return None


def _confirm_structure_break(data, sweep, direction):
    df = _completed(data)
    if df.empty or sweep is None:
        return None
    positions = np.where(df.index == sweep["index"])[0]
    if len(positions) == 0:
        return None
    sweep_pos = int(positions[-1])
    if sweep_pos >= len(df)-1:
        return None
    atr = float(sweep.get("atr", _atr(df) or 10.0))
    end = min(len(df), sweep_pos + MAX_BARS_AFTER_SWEEP + 2)
    after = df.iloc[sweep_pos+1:end]
    if after.empty:
        return None
    sweep_row = df.iloc[sweep_pos]
    previous = df.iloc[sweep_pos-1] if sweep_pos > 0 else sweep_row
    if direction == "BUY":
        break_level = max(float(sweep_row["High"]), float(previous["High"])) + atr*BREAK_BUFFER_ATR
        for idx, row in after.iterrows():
            if float(row["Close"]) > break_level:
                return {"index":idx, "break_level":break_level, "sweep":sweep}
    else:
        break_level = min(float(sweep_row["Low"]), float(previous["Low"])) - atr*BREAK_BUFFER_ATR
        for idx, row in after.iterrows():
            if float(row["Close"]) < break_level:
                return {"index":idx, "break_level":break_level, "sweep":sweep}
    return None


def get_entry_confirmation(data, aoi=None):
    if aoi is None:
        return "NONE"
    direction = "BUY" if aoi["type"] == "support" else "SELL"
    sweep = _find_liquidity_sweep(data, aoi, direction)
    if sweep is None:
        return "NONE"
    confirmation = _confirm_structure_break(data, sweep, direction)
    return direction if confirmation is not None else "NONE"


def _find_recent_fvg(data, direction, lookback=8):
    df = _completed(data).tail(lookback + 2)
    if len(df) < 3:
        return None
    for i in range(len(df) - 1, 1, -1):
        a = df.iloc[i-2]
        c = df.iloc[i]
        if direction == "BUY" and float(c["Low"]) > float(a["High"]):
            return {"low":float(a["High"]), "high":float(c["Low"]), "index":df.index[i]}
        if direction == "SELL" and float(c["High"]) < float(a["Low"]):
            return {"low":float(c["High"]), "high":float(a["Low"]), "index":df.index[i]}
    return None


def _find_recent_order_block(data, direction, lookback=8):
    df = _completed(data).tail(lookback + 4)
    atr = _atr(df)
    if len(df) < 4 or atr is None:
        return None
    for i in range(len(df) - 1, 1, -1):
        candle = df.iloc[i]
        previous = df.iloc[i-1]
        body = abs(float(candle["Close"]) - float(candle["Open"]))
        if direction == "BUY" and float(previous["Close"]) < float(previous["Open"]) and float(candle["Close"]) > float(candle["Open"]) and body >= atr * 0.8:
            return {"low":float(previous["Low"]), "high":float(previous["High"]), "index":df.index[i-1]}
        if direction == "SELL" and float(previous["Close"]) > float(previous["Open"]) and float(candle["Close"]) < float(candle["Open"]) and body >= atr * 0.8:
            return {"low":float(previous["Low"]), "high":float(previous["High"]), "index":df.index[i-1]}
    return None


def _confirmation_confluence(data, aoi, direction):
    fvg = _find_recent_fvg(data, direction)
    ob = _find_recent_order_block(data, direction)
    zl, zh = float(aoi["low"]), float(aoi["high"])
    score = 0
    if fvg is not None and not (fvg["high"] < zl or fvg["low"] > zh):
        score += 1
    if ob is not None and not (ob["high"] < zl or ob["low"] > zh):
        score += 1
    return {"score":score, "fvg":fvg, "order_block":ob}


def _select_aoi(price, areas, direction):
    desired_type = "support" if direction == "BUY" else "resistance"
    candidates = []
    for timeframe in ("daily", "weekly"):
        for zone in areas.get(timeframe, []):
            if zone["type"] != desired_type:
                continue
            distance = float(zone.get("distance", 999999))
            if distance > MAX_RELEVANT_AOI_DISTANCE:
                continue
            timeframe_bonus = 8.0 if timeframe == "daily" else 5.0
            score = (
                zone["touches"] * 8.0
                + zone.get("recency_score",0.0) * 12.0
                - zone["width"] * 0.20
                - distance * 0.08
                + timeframe_bonus
            )
            item = dict(zone)
            item["timeframe"] = timeframe
            item["selection_score"] = score
            candidates.append(item)
    return max(candidates, key=lambda z:z["selection_score"]) if candidates else None


def _active_session(index):
    try:
        hour = int(index.hour)
    except Exception:
        return True
    return ACTIVE_START_UTC <= hour < ACTIVE_END_UTC


def calculate_sl_tp(signal, entry, aoi, data_15m=None):
    if signal not in ("BUY", "SELL") or aoi is None:
        return None
    entry = float(entry)
    atr = _atr(data_15m) if data_15m is not None else None
    if atr is None:
        atr = 10.0
    sweep = aoi.get("sweep")
    if signal == "SELL":
        structural_level = float(aoi["high"])
        if sweep is not None:
            structural_level = max(structural_level, float(sweep.get("extreme", structural_level)))
        stop_loss = structural_level + max(MIN_STOP_DISTANCE, atr*SL_ATR_BUFFER)
        risk = stop_loss - entry
        take_profit = entry - risk*RISK_REWARD
    else:
        structural_level = float(aoi["low"])
        if sweep is not None:
            structural_level = min(structural_level, float(sweep.get("extreme", structural_level)))
        stop_loss = structural_level - max(MIN_STOP_DISTANCE, atr*SL_ATR_BUFFER)
        risk = entry - stop_loss
        take_profit = entry + risk*RISK_REWARD
    if risk <= 0 or risk > MAX_STOP_DISTANCE:
        return None
    return {"entry":entry, "stop_loss":float(stop_loss), "take_profit":float(take_profit), "risk":float(risk), "reward":float(abs(take_profit-entry)), "risk_reward":RISK_REWARD, "atr":float(atr)}


def _target_space(entry, tp, areas, direction):
    opposing = "support" if direction == "SELL" else "resistance"
    candidates = []
    for timeframe in ("daily", "weekly"):
        for zone in areas.get(timeframe, []):
            if zone["type"] != opposing:
                continue
            low, high = float(zone["low"]), float(zone["high"])
            if direction == "SELL" and high < entry:
                candidates.append(high)
            if direction == "BUY" and low > entry:
                candidates.append(low)
    if not candidates:
        return True
    nearest = max(candidates) if direction == "SELL" else min(candidates)
    return tp > nearest if direction == "SELL" else tp < nearest


def generate_signal(data_15m, data_daily, current_price):
    d15 = _completed(data_15m)
    dd = _completed(data_daily)
    price = float(current_price)
    if d15.empty or dd.empty:
        return {"signal":"NONE", "reason":"INSUFFICIENT_DATA", "bias":{"weekly":"NEUTRAL","daily":"NEUTRAL","4h":"NEUTRAL","overall":"NEUTRAL","score":0}, "aoi":None}
    bias = get_higher_timeframe_bias(d15, dd)
    areas = get_weekly_daily_areas(dd, current_price=price)
    overall = bias["overall"]
    if overall == "NEUTRAL":
        return {"signal":"NONE", "reason":"NEUTRAL_HIGHER_TIMEFRAME", "bias":bias, "aoi":None}
    direction = "BUY" if overall == "BULLISH" else "SELL"
    if direction == "BUY" and bias["4h"] == "BEARISH":
        return {"signal":"NONE", "reason":"4H_CONTRADICTS_BUY", "bias":bias, "aoi":None}
    if direction == "SELL" and bias["4h"] == "BULLISH":
        return {"signal":"NONE", "reason":"4H_CONTRADICTS_SELL", "bias":bias, "aoi":None}
    aoi = _select_aoi(price, areas, direction)
    if aoi is None:
        return {"signal":"NONE", "reason":"WAITING_FOR_RELEVANT_AOI", "bias":bias, "aoi":None}
    if float(aoi.get("distance",999999)) > AOI_TOLERANCE:
        return {"signal":"NONE", "reason":"WAITING_FOR_AOI", "bias":bias, "aoi":aoi}
    if not _active_session(d15.index[-1]):
        return {"signal":"NONE", "reason":"OUTSIDE_ACTIVE_SESSION", "bias":bias, "aoi":aoi}
    sweep = _find_liquidity_sweep(d15, aoi, direction)
    if sweep is None:
        return {"signal":"NONE", "reason":"WAITING_FOR_LIQUIDITY_SWEEP", "bias":bias, "aoi":aoi}
    confirmation = _confirm_structure_break(d15, sweep, direction)
    if confirmation is None:
        return {"signal":"NONE", "reason":"WAITING_FOR_STRUCTURE_BREAK", "bias":bias, "aoi":aoi, "sweep":sweep}
    aoi_with_sweep = dict(aoi)
    aoi_with_sweep["sweep"] = sweep
    aoi_with_sweep["confirmation"] = confirmation
    aoi_with_sweep["confluence"] = _confirmation_confluence(d15, aoi_with_sweep, direction)
    levels = calculate_sl_tp(direction, price, aoi_with_sweep, d15)
    if levels is None:
        return {"signal":"NONE", "reason":"INVALID_RISK", "bias":bias, "aoi":aoi_with_sweep}
    if not _target_space(levels["entry"], levels["take_profit"], areas, direction):
        return {"signal":"NONE", "reason":"INSUFFICIENT_TARGET_SPACE", "bias":bias, "aoi":aoi_with_sweep}
    return {"signal":direction, "reason":"SDMC_LIQUIDITY_SWEEP_BOS_CHOCH", "bias":bias, "aoi":aoi_with_sweep, **levels}
