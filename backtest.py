import os
import time
import requests
import pandas as pd
from strategy import generate_signal

# ============================================================
# XAUUSD STRATEGY BACKTEST V6.2 - FCS API
# ============================================================

FCS_URL = "https://api-v4.fcsapi.com/forex/history"
API_KEY = os.getenv("FCS_API_KEY")
SYMBOL = "XAUUSD"
PAGE_SIZE = 300
MAX_15M_PAGES = 24
MAX_DAILY_PAGES = 5
MAX_15M_CANDLES = 1000
MAX_DAILY_CANDLES = 500
MIN_15M = 300
MIN_DAILY = 100
RESET_DISTANCE = 10.0
TIMEOUT = 30

if not API_KEY:
    raise RuntimeError("FCS_API_KEY is not available.")

def get_page(symbol, period, page):
    params = {
        "symbol": symbol,
        "period": period,
        "length": PAGE_SIZE,
        "page": page,
        "access_key": API_KEY,
    }
    if symbol == "XAUUSD":
        params["type"] = "commodity"

    r = requests.get(FCS_URL, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()

    if data.get("status") is not True:
        raise RuntimeError(f"FCS API error: {data}")

    return data.get("response", {})

def to_df(response):
    rows = []
    for key, c in response.items():
        if not isinstance(c, dict):
            continue
        try:
            t = int(float(c.get("t", key)))
            dt = pd.to_datetime(t, unit="s", utc=True).tz_convert(None)
            rows.append({
                "Datetime": dt,
                "Open": float(c["o"]),
                "High": float(c["h"]),
                "Low": float(c["l"]),
                "Close": float(c["c"]),
                "Volume": float(c.get("v", 0) or 0),
            })
        except (TypeError, ValueError, KeyError):
            continue

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .drop_duplicates("Datetime")
        .sort_values("Datetime")
        .set_index("Datetime")
    )

def download(symbol, period, pages):
    frames = []
    previous_oldest = None

    print(f"Downloading FCS {symbol} {period} history...", flush=True)

    for page in range(1, pages + 1):
        print(f"  Page {page}/{pages}", flush=True)
        frame = to_df(get_page(symbol, period, page))

        if frame.empty:
            print("  No more candles returned.", flush=True)
            break

        oldest = frame.index.min()
        newest = frame.index.max()

        print(
            f"  {len(frame)} candles | {oldest} -> {newest}",
            flush=True
        )

        frames.append(frame)

        if previous_oldest is not None and oldest >= previous_oldest:
            print("  Pagination stopped: page did not move backwards.", flush=True)
            break

        previous_oldest = oldest

        if len(frame) < PAGE_SIZE:
            print("  Final partial page reached.", flush=True)
            break

        time.sleep(0.25)

    if not frames:
        raise RuntimeError(f"No {symbol} {period} data returned.")

    df = (
        pd.concat(frames)
        .reset_index()
        .drop_duplicates("Datetime")
        .sort_values("Datetime")
        .set_index("Datetime")
    )
    return df

def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def get_aoi(signal):
    aoi = signal.get("aoi")
    return aoi if isinstance(aoi, dict) else None

def setup_key(signal):
    aoi = get_aoi(signal)
    if not aoi:
        return None
    low = safe_float(aoi.get("low"))
    high = safe_float(aoi.get("high"))
    if low is None or high is None:
        return None
    return (
        aoi.get("timeframe", "UNKNOWN"),
        aoi.get("type", "UNKNOWN"),
        round(low, 1),
        round(high, 1),
    )

def setup_text(signal):
    key = setup_key(signal)
    if not key:
        return "UNKNOWN_SETUP"
    tf, typ, low, high = key
    return f"{tf.upper()} {typ.upper()} {low}-{high}"

def aoi_reset(price, aoi):
    if not isinstance(aoi, dict):
        return True
    low = safe_float(aoi.get("low"))
    high = safe_float(aoi.get("high"))
    if low is None or high is None:
        return True
    if low <= price <= high:
        return False
    if price > high:
        return price - high >= RESET_DISTANCE
    return low - price >= RESET_DISTANCE

def result_for_trade(direction, entry, sl, tp, future):
    for ts, candle in future.iterrows():
        high = float(candle["High"])
        low = float(candle["Low"])

        if direction == "SELL":
            stop = high >= sl
            target = low <= tp
        else:
            stop = low <= sl
            target = high >= tp

        if stop and target:
            return "AMBIGUOUS", None, ts, 0.0
        if stop:
            return "SL", sl, ts, -1.0
        if target:
            r = abs(tp - entry) / abs(entry - sl)
            return "TP", tp, ts, r

    return "OPEN", None, None, 0.0

print()
print("=" * 60)
print("XAUUSD STRATEGY BACKTEST V6.2")
print("=" * 60)
print()
print("DATA SOURCE: FCS API")
print("SYMBOL: XAUUSD")
print("INSTRUMENT: GOLD SPOT / COMMODITY")
print("EXECUTION TIMEFRAME: 15M")
print()

data_15m = download(SYMBOL, "15m", MAX_15M_PAGES)
data_daily = download(SYMBOL, "1d", MAX_DAILY_PAGES)

print()
print("15m candles loaded:", len(data_15m))
print("Daily candles loaded:", len(data_daily))

if len(data_15m) < MIN_15M:
    raise RuntimeError("Not enough 15m candles for backtest.")
if len(data_daily) < MIN_DAILY:
    raise RuntimeError("Not enough daily candles for backtest.")

print("Checking every 15 minutes...")
print("One open trade at a time: YES")
print("Same-AOI re-entry lock: YES")
print()
print("Starting historical simulation...", flush=True)

signals = []
reasons = {}
active_trade = None
locked_key = None
locked_aoi = None
blocked_open = 0
blocked_duplicate = 0
blocked_lock = 0

for i in range(MIN_15M, len(data_15m)):
    ts = data_15m.index[i]

    if active_trade is not None:
        exit_ts = active_trade["exit_time"]
        if exit_ts is not None and ts <= exit_ts:
            blocked_open += 1
            continue
        active_trade = None

    hist15 = data_15m.iloc[max(0, i - MAX_15M_CANDLES):i + 1].copy()
    histdaily = data_daily[data_daily.index <= ts].tail(MAX_DAILY_CANDLES).copy()

    if len(histdaily) < MIN_DAILY:
        continue

    price = float(hist15["Close"].iloc[-1])

    if locked_key is not None and locked_aoi is not None:
        if aoi_reset(price, locked_aoi):
            locked_key = None
            locked_aoi = None
        else:
            blocked_lock += 1

    try:
        signal = generate_signal(hist15, histdaily, price)
    except Exception as e:
        reasons["STRATEGY_ERROR"] = reasons.get("STRATEGY_ERROR", 0) + 1
        print("STRATEGY ERROR:", ts, "|", e, flush=True)
        continue

    if not isinstance(signal, dict):
        reasons["INVALID_SIGNAL_OBJECT"] = reasons.get("INVALID_SIGNAL_OBJECT", 0) + 1
        continue

    direction = signal.get("signal", "NONE")
    reason = signal.get("reason", "UNKNOWN")

    if direction not in ("BUY", "SELL"):
        reasons[reason] = reasons.get(reason, 0) + 1
        continue

    entry = safe_float(signal.get("entry"))
    sl = safe_float(signal.get("stop_loss"))
    tp = safe_float(signal.get("take_profit"))

    if entry is None or sl is None or tp is None:
        reasons["INVALID_TRADE_LEVELS"] = reasons.get("INVALID_TRADE_LEVELS", 0) + 1
        continue

    key = setup_key(signal)
    aoi = get_aoi(signal)

    if locked_key is not None and key == locked_key and not aoi_reset(price, locked_aoi):
        blocked_duplicate += 1
        continue

    result, exit_price, exit_time, r = result_for_trade(
        direction, entry, sl, tp, data_15m.iloc[i + 1:]
    )

    bias = signal.get("bias", {})
    if not isinstance(bias, dict):
        bias = {}

    setup_id = (
        f"{ts.strftime('%Y%m%d-%H%M')}-{direction}-"
        f"{abs(hash(str(key))) % 100000:05d}"
    )

    trade = {
        "setup_id": setup_id,
        "time": ts,
        "signal": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "result": result,
        "r": r,
        "reason": reason,
        "setup": setup_text(signal),
        "weekly": bias.get("weekly", "UNKNOWN"),
        "daily": bias.get("daily", "UNKNOWN"),
        "4h": bias.get("4h", "UNKNOWN"),
        "overall": bias.get("overall", "UNKNOWN"),
        "exit_time": exit_time,
        "exit_price": exit_price,
    }

    signals.append(trade)
    locked_key = key
    locked_aoi = aoi
    active_trade = trade

    print(
        "SIGNAL FOUND:",
        ts,
        "|", direction,
        "| Entry:", round(entry, 2),
        "| Result:", result,
        "| R:", round(r, 2),
        "| Setup:", trade["setup"],
        flush=True,
    )

    if len(signals) % 10 == 0:
        print("Signals:", len(signals), flush=True)

print()
print("=" * 60)
print("BACKTEST COMPLETE")
print("=" * 60)
print()

start = data_15m.index[0]
end = data_15m.index[-1]
days = (end - start).total_seconds() / 86400
weeks = days / 7

total = len(signals)
buys = sum(x["signal"] == "BUY" for x in signals)
sells = sum(x["signal"] == "SELL" for x in signals)
tp_count = sum(x["result"] == "TP" for x in signals)
sl_count = sum(x["result"] == "SL" for x in signals)
amb = sum(x["result"] == "AMBIGUOUS" for x in signals)
open_count = sum(x["result"] == "OPEN" for x in signals)
resolved = tp_count + sl_count

win_rate = (tp_count / resolved * 100) if resolved else 0
total_r = sum(x["r"] for x in signals)
winning_r = sum(x["r"] for x in signals if x["r"] > 0)
losing_r = sum(x["r"] for x in signals if x["r"] < 0)
profit_factor = winning_r / abs(losing_r) if losing_r < 0 else None
expectancy = total_r / resolved if resolved else 0

equity = 0
peak = 0
max_dd = 0
for x in signals:
    equity += x["r"]
    peak = max(peak, equity)
    max_dd = max(max_dd, peak - equity)

if len(signals) >= 2:
    times = [x["time"] for x in signals]
    longest_gap = max(
        (times[j] - times[j-1]).total_seconds() / 86400
        for j in range(1, len(times))
    )
else:
    longest_gap = None

print("DATA SOURCE: FCS API")
print("SYMBOL: XAUUSD")
print("INSTRUMENT: GOLD SPOT")
print()
print("TEST PERIOD:")
print(start, "->", end)
print("DAYS TESTED:", round(days, 1))
print("WEEKS TESTED:", round(weeks, 1))
print()
print("15M CANDLES:", len(data_15m))
print("DAILY CANDLES:", len(data_daily))
print("15M EVALUATIONS:", max(0, len(data_15m) - MIN_15M))
print("CHECK INTERVAL: 15 minutes")
print("ONE OPEN TRADE AT A TIME: YES")
print("SAME-AOI RESET LOCK: YES")
print()
print("TOTAL INDEPENDENT SETUPS:", total)
print("BUY SETUPS:", buys)
print("SELL SETUPS:", sells)
print("SETUPS PER WEEK:", round(total / weeks, 2) if weeks else 0)
print()
print("TP:", tp_count)
print("SL:", sl_count)
print("AMBIGUOUS:", amb)
print("OPEN:", open_count)
print("RESOLVED TRADES:", resolved)
print("WIN RATE:", round(win_rate, 2), "%")
print()
print("TOTAL R:", round(total_r, 2))
print("EXPECTANCY:", round(expectancy, 3), "R/trade")
print("PROFIT FACTOR:", round(profit_factor, 2) if profit_factor is not None else "N/A")
print("MAX DRAWDOWN:", round(max_dd, 2), "R")
print(
    "LONGEST GAP BETWEEN INDEPENDENT SETUPS:",
    round(longest_gap, 2) if longest_gap is not None else "N/A",
    "days" if longest_gap is not None else "",
)
print()
print("=" * 60)
print("BACKTEST ACCOUNTING")
print("=" * 60)
print("Evaluations blocked while trade open:", blocked_open)
print("Repeated same-AOI signals blocked:", blocked_duplicate)
print("Evaluations inside locked AOI:", blocked_lock)

if reasons:
    print()
    print("=" * 60)
    print("TOP SIGNAL REJECTION REASONS")
    print("=" * 60)
    for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(reason, ":", count)

print()
print("=" * 60)
print("INDEPENDENT SETUPS")
print("=" * 60)

for n, x in enumerate(signals, 1):
    print()
    print(n, "|", x["setup_id"])
    print("Time:", x["time"])
    print("Setup:", x["setup"])
    print("Signal:", x["signal"])
    print("Entry:", round(x["entry"], 2))
    print("SL:", round(x["sl"], 2))
    print("TP:", round(x["tp"], 2))
    print("Result:", x["result"])
    print("R:", round(x["r"], 2))
    print("Reason:", x["reason"])
    print("Weekly Bias:", x["weekly"])
    print("Daily Bias:", x["daily"])
    print("4H Bias:", x["4h"])
    print("Overall Bias:", x["overall"])
    if x["exit_time"] is not None:
        print("Exit:", x["exit_time"])

if not signals:
    print("NO INDEPENDENT SETUPS FOUND.")

print()
print("=" * 60)
print("END OF BACKTEST")
print("=" * 60)
