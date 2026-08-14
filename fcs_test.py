import os
import requests

API_KEY = os.getenv("FCS_API_KEY")

if not API_KEY:
    raise RuntimeError("FCS_API_KEY is not set")

url = "https://api-v4.fcsapi.com/forex/history"

params = {
    "symbol": "XAU/USD",
    "period": "15m",
    "length": 300,
    "access_key": API_KEY,
}

response = requests.get(
    url,
    params=params,
    timeout=20
)

print("HTTP STATUS:", response.status_code)

data = response.json()

print("API STATUS:", data.get("status"))
print("API CODE:", data.get("code"))
print("MESSAGE:", data.get("msg"))

if data.get("status") is True:

    candles = data.get("response", {})

    print("CANDLES RECEIVED:", len(candles))

    print()
    print("LATEST 5 CANDLES:")

    for timestamp, candle in list(candles.items())[-5:]:

        print(
            timestamp,
            "|",
            "O:", candle.get("o"),
            "H:", candle.get("h"),
            "L:", candle.get("l"),
            "C:", candle.get("c"),
            "|",
            candle.get("tm")
        )

else:

    print()
    print("FCS API REQUEST FAILED")
    print(data)
