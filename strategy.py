import pandas as pd

# ============================================================
# XAUUSD STRATEGY V2
# Public interfaces preserved for live.py/backtest.py.
# ============================================================
PIP_SIZE = 0.1
MIN_TOUCHES = 3
MAX_ZONE_WIDTH = 35.0
AOI_TOLERANCE = 2.0
MAX_RELEVANT_AOI_DISTANCE = 300.0
DAILY_LOOKBACK = 220
WEEKLY_LOOKBACK = 52
DAILY_SWING_LENGTH = 3
WEEKLY_SWING_LENGTH = 2
ATR_PERIOD = 14
ATR_BUFFER_MULTIPLIER = 0.35
MIN_STOP_DISTANCE = 3.0
MAX_STOP_DISTANCE = 45.0
RISK_REWARD = 2.0


def _clean(data):
    if data is None or len(data) == 0:
        return pd.DataFrame()
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    cols = ["Open", "High", "Low", "Close"]
    if any(c not in df.columns for c in cols):
        return pd.DataFrame()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=cols).sort_index()


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
    out = df.resample(rules[timeframe]).agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    })
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def find_market_structure(data, swing_length=3):
    df = _clean(data)
    df["swing_high"] = False
    df["swing_low"] = False
    df["structure"] = None
    if len(df) < swing_length * 2 + 5:
        return df
    previous_high = None
    previous_low = None
    for i in range(swing_length, len(df) - swing_length):
        h = float(df["High"].iloc[i]); l = float(df["Low"].iloc[i])
        if h > df["High"].iloc[i-swing_length:i].max() and h >= df["High"].iloc[i+1:i+swing_length+1].max():
            df.iloc[i, df.columns.get_loc("swing_high")] = True
            if previous_high is not None:
                df.iloc[i, df.columns.get_loc("structure")] = "HH" if h > previous_high else "LH"
            previous_high = h
        if l < df["Low"].iloc[i-swing_length:i].min() and l <= df["Low"].iloc[i+1:i+swing_length+1].min():
            df.iloc[i, df.columns.get_loc("swing_low")] = True
            if previous_low is not None and pd.isna(df["structure"].iloc[i]):
                df.iloc[i, df.columns.get_loc("structure")] = "HL" if l > previous_low else "LL"
            previous_low = l
    return df


def get_market_bias(data):
    s = find_market_structure(data)
    points = s[s["structure"].notna()]["structure"].tolist()
    if len(points) < 4:
        return "NEUTRAL"
    recent = points[-8:]
    bull = sum(x in ("HH", "HL") for x in recent)
    bear = sum(x in ("LH", "LL") for x in recent)
    if bull >= bear + 2:
        return "BULLISH"
    if bear >= bull + 2:
        return "BEARISH"
    return "NEUTRAL"


def get_higher_timeframe_bias(data_15m, data_daily):
    d15 = _completed(data_15m); dd = _completed(data_daily)
    if d15.empty or dd.empty:
        return {"weekly":"NEUTRAL","daily":"NEUTRAL","4h":"NEUTRAL","overall":"NEUTRAL"}
    weekly = get_market_bias(resample_data(dd, "1W"))
    daily = get_market_bias(dd)
    four = get_market_bias(resample_data(d15, "4H"))
    if weekly == daily == "BULLISH": overall = "BULLISH"
    elif weekly == daily == "BEARISH": overall = "BEARISH"
    elif [weekly, daily, four].count("BULLISH") >= 2: overall = "BULLISH"
    elif [weekly, daily, four].count("BEARISH") >= 2: overall = "BEARISH"
    else: overall = "NEUTRAL"
    return {"weekly":weekly,"daily":daily,"4h":four,"overall":overall}


def _touch_events(data, low, high, lookback):
    df = _clean(data).tail(lookback)
    events=[]; inside=False
    for idx, row in df.iterrows():
        touched=float(row.High)>=low and float(row.Low)<=high
        if touched and not inside: events.append(idx)
        inside=touched
    return events


def count_zone_touches(data, zone_low, zone_high, lookback):
    return len(_touch_events(data, zone_low, zone_high, lookback))


def _clusters(prices):
    clusters=[]
    for p in sorted(float(x) for x in prices):
        target=None
        for c in clusters:
            if p-min(c) <= MAX_ZONE_WIDTH and max(c)-p <= MAX_ZONE_WIDTH:
                target=c; break
        if target is None: clusters.append([p])
        else: target.append(p)
    return clusters


def _build_zones(data, swing_length, lookback):
    df=_clean(data)
    s=find_market_structure(df, swing_length)
    zones=[]
    for kind, col in (("support","Low"),("resistance","High")):
        swings=s[s["swing_low"] if kind=="support" else s["swing_high"]].tail(lookback)
        for cluster in _clusters(swings[col].tolist()):
            if len(cluster)<MIN_TOUCHES: continue
            low=min(cluster); high=max(cluster)
            if high-low>MAX_ZONE_WIDTH: continue
            events=_touch_events(df,low,high,lookback)
            if len(events)<MIN_TOUCHES: continue
            zones.append({"type":kind,"low":low,"high":high,"width":high-low,
                          "touches":len(events),"touch_times":events,
                          "structure_bias":get_market_bias(df),"latest_touch":events[-1]})
    zones.sort(key=lambda z:(-z["touches"],z["width"]))
    clean=[]
    for z in zones:
        if not any(z["type"]==x["type"] and z["low"]<=x["high"] and z["high"]>=x["low"] for x in clean):
            clean.append(z)
    return clean


def find_area_of_interest(data, current_price=None, swing_length=3, touch_lookback=100):
    zones=_build_zones(data,swing_length,min(len(data),touch_lookback))
    if current_price is None: return zones
    p=float(current_price); out=[]
    for z in zones:
        low=float(z["low"]); high=float(z["high"])
        if low<=p<=high: dist=0.0
        elif z["type"]=="support" and high<p: dist=p-high
        elif z["type"]=="resistance" and low>p: dist=low-p
        else: continue
        if dist<=MAX_RELEVANT_AOI_DISTANCE:
            item=dict(z); item["distance"]=dist; out.append(item)
    out.sort(key=lambda z:(z["distance"],-z["touches"],z["width"]))
    return out


def get_weekly_daily_areas(data_daily, current_price=None):
    dd=_completed(data_daily)
    if dd.empty: return {"weekly":[],"daily":[]}
    weekly=resample_data(dd,"1W")
    return {
        "weekly":find_area_of_interest(weekly,current_price,WEEKLY_SWING_LENGTH,min(len(weekly),WEEKLY_LOOKBACK)),
        "daily":find_area_of_interest(dd,current_price,DAILY_SWING_LENGTH,min(len(dd),DAILY_LOOKBACK)),
    }


def get_current_aoi(price, areas, tolerance=AOI_TOLERANCE):
    out=[]
    for tf in ("daily","weekly"):
        for z in areas.get(tf,[]):
            if float(z["low"])-tolerance <= float(price) <= float(z["high"])+tolerance:
                item=dict(z); item["timeframe"]=tf; out.append(item)
    return sorted(out,key=lambda z:(-z["touches"],z["width"]))


def price_at_area_of_interest(price, areas, tolerance=AOI_TOLERANCE):
    return get_current_aoi(price,areas,tolerance)


def find_recent_aoi_touch(data_15m, areas, lookback=20, tolerance=AOI_TOLERANCE):
    df=_clean(data_15m).tail(lookback); out=[]
    for tf in ("daily","weekly"):
        for z in areas.get(tf,[]):
            for idx,row in df.iterrows():
                if float(row.High)>=float(z["low"])-tolerance and float(row.Low)<=float(z["high"])+tolerance:
                    item=dict(z); item["timeframe"]=tf; item["touch_time"]=idx; out.append(item); break
    return out


def _atr(data, period=ATR_PERIOD):
    df=_clean(data)
    if len(df)<period+1:return None
    pc=df.Close.shift(1)
    tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1)
    v=tr.rolling(period).mean().iloc[-1]
    return float(v) if pd.notna(v) else None


def _rejection(row,aoi,direction):
    o=float(row.Open); c=float(row.Close); h=float(row.High); l=float(row.Low); r=max(h-l,1e-9); b=abs(c-o)
    if direction=="BUY" and aoi["type"]=="support":
        return l<=float(aoi["high"])+AOI_TOLERANCE and c>float(aoi["high"]) and c>o and b/r>=0.40 and (min(o,c)-l)/r>=0.20
    if direction=="SELL" and aoi["type"]=="resistance":
        return h>=float(aoi["low"])-AOI_TOLERANCE and c<float(aoi["low"]) and c<o and b/r>=0.40 and (h-max(o,c))/r>=0.20
    return False


def _engulfing(data,aoi,direction):
    if len(data)<2:return False
    p=data.iloc[-2]; c=data.iloc[-1]
    if direction=="BUY":
        if not (float(p.Close)<float(p.Open) and float(c.Close)>float(c.Open)):return False
        if float(c.Low)>float(aoi["high"])+AOI_TOLERANCE:return False
    else:
        if not (float(p.Close)>float(p.Open) and float(c.Close)<float(c.Open)):return False
        if float(c.High)<float(aoi["low"])-AOI_TOLERANCE:return False
    pl=min(float(p.Open),float(p.Close)); ph=max(float(p.Open),float(p.Close))
    cl=min(float(c.Open),float(c.Close)); ch=max(float(c.Open),float(c.Close))
    return cl<=pl and ch>=ph


def get_entry_confirmation(data,aoi=None):
    if aoi is None:return "NONE"
    df=_completed(data)
    if df.empty:return "NONE"
    direction="BUY" if aoi["type"]=="support" else "SELL"
    for _,row in df.tail(4).iterrows():
        if _rejection(row,aoi,direction): return direction
    if _engulfing(df,aoi,direction): return direction
    return "NONE"


def _select_aoi(price,areas,direction):
    typ="support" if direction=="BUY" else "resistance"; candidates=[]
    for tf in ("daily","weekly"):
        for z in areas.get(tf,[]):
            if z["type"]!=typ:continue
            d=float(z.get("distance",999999))
            if d>MAX_RELEVANT_AOI_DISTANCE:continue
            # Daily is preferred for execution; weekly gets a structural bonus.
            score=z["touches"]*10 - z["width"]*0.15 - d*0.5 + (3 if tf=="daily" else 1)
            item=dict(z); item["timeframe"]=tf; item["selection_score"]=score; candidates.append(item)
    if not candidates:return None
    return max(candidates,key=lambda x:x["selection_score"])


def calculate_sl_tp(signal,entry,aoi,data_15m=None):
    if signal not in ("BUY","SELL") or aoi is None:return None
    atr=_atr(data_15m) if data_15m is not None else None
    atr=10.0 if atr is None else atr
    buffer=max(MIN_STOP_DISTANCE,atr*ATR_BUFFER_MULTIPLIER)
    if signal=="SELL":
        sl=float(aoi["high"])+buffer; risk=sl-float(entry); tp=float(entry)-risk*RISK_REWARD
    else:
        sl=float(aoi["low"])-buffer; risk=float(entry)-sl; tp=float(entry)+risk*RISK_REWARD
    if risk<=0 or risk>MAX_STOP_DISTANCE:return None
    return {"entry":float(entry),"stop_loss":float(sl),"take_profit":float(tp),"risk":float(risk),"reward":float(abs(tp-float(entry))),"risk_reward":RISK_REWARD,"atr":float(atr)}


def _target_space(entry,tp,areas,direction):
    opposing="support" if direction=="SELL" else "resistance"; candidates=[]
    for tf in ("daily","weekly"):
        for z in areas.get(tf,[]):
            if z["type"]!=opposing:continue
            low=float(z["low"]); high=float(z["high"])
            if direction=="SELL" and high<float(entry):candidates.append(high)
            if direction=="BUY" and low>float(entry):candidates.append(low)
    if not candidates:return True
    nearest=max(candidates) if direction=="SELL" else min(candidates)
    return float(tp)>=nearest if direction=="SELL" else float(tp)<=nearest


def generate_signal(data_15m,data_daily,current_price):
    d15=_clean(data_15m); dd=_clean(data_daily); price=float(current_price)
    if d15.empty or dd.empty:
        return {"signal":"NONE","reason":"INSUFFICIENT_DATA","bias":{"weekly":"NEUTRAL","daily":"NEUTRAL","4h":"NEUTRAL","overall":"NEUTRAL"},"aoi":None}
    bias=get_higher_timeframe_bias(d15,dd)
    areas=get_weekly_daily_areas(dd,price)
    overall=bias["overall"]
    if overall=="NEUTRAL":return {"signal":"NONE","reason":"NEUTRAL_HIGHER_TIMEFRAME","bias":bias,"aoi":None}
    direction="SELL" if overall=="BEARISH" else "BUY"
    # Do not enter against a strongly opposing 4H structure.
    if direction=="SELL" and bias["4h"]=="BULLISH":return {"signal":"NONE","reason":"4H_CONTRADICTS_SELL","bias":bias,"aoi":None}
    if direction=="BUY" and bias["4h"]=="BEARISH":return {"signal":"NONE","reason":"4H_CONTRADICTS_BUY","bias":bias,"aoi":None}
    aoi=_select_aoi(price,areas,direction)
    if aoi is None:return {"signal":"NONE","reason":"WAITING_FOR_RELEVANT_AOI","bias":bias,"aoi":None}
    if float(aoi.get("distance",999999))>AOI_TOLERANCE:return {"signal":"NONE","reason":"WAITING_FOR_AOI","bias":bias,"aoi":aoi}
    closed=_completed(d15)
    confirmation=get_entry_confirmation(closed,aoi)
    if confirmation!=direction:return {"signal":"NONE","reason":"WAITING_FOR_CONFIRMATION","bias":bias,"aoi":aoi}
    levels=calculate_sl_tp(direction,price,aoi,closed)
    if levels is None:return {"signal":"NONE","reason":"INVALID_RISK","bias":bias,"aoi":aoi}
    if not _target_space(levels["entry"],levels["take_profit"],areas,direction):return {"signal":"NONE","reason":"INSUFFICIENT_TARGET_SPACE","bias":bias,"aoi":aoi}
    return {"signal":direction,"reason":"AOI_RETEST_CONFIRMED","bias":bias,"aoi":aoi,**levels}
