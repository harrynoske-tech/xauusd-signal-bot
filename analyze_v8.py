import os
import pandas as pd

FILE = "data/v8_trades.csv"

print("=" * 60)
print("V8 TRADE ANALYSIS")
print("=" * 60)

if not os.path.exists(FILE):
    print()
    print("ERROR: data/v8_trades.csv does not exist yet.")
    print()
    print("We need to export the trades from backtest.py first.")
    print("DO NOT run this yet.")
    raise SystemExit

df = pd.read_csv(FILE)

print()
print("TOTAL TRADES:", len(df))

if len(df) == 0:
    print("No trades found.")
    raise SystemExit

# ------------------------------------------------------------
# BASIC RESULTS
# ------------------------------------------------------------

wins = df[df["result"] == "TP"]
losses = df[df["result"] == "SL"]

print()
print("OVERALL")
print("-" * 60)
print("Wins:", len(wins))
print("Losses:", len(losses))
print(
    "Win rate:",
    round(
        len(wins) / len(df) * 100,
        2
    ),
    "%"
)

print(
    "Total R:",
    round(
        df["r"].sum(),
        2
    )
)

# ------------------------------------------------------------
# SCORE
# ------------------------------------------------------------

print()
print("SCORE RANGES")
print("-" * 60)

for low, high in [
    (70, 74),
    (75, 79),
    (80, 84),
    (85, 89),
    (90, 94),
    (95, 100)
]:

    subset = df[
        (df["score"] >= low)
        & (df["score"] <= high)
    ]

    if len(subset) == 0:
        continue

    subset_wins = (
        subset["result"] == "TP"
    ).sum()

    print(
        f"{low}-{high}: "
        f"{len(subset)} trades | "
        f"{subset_wins / len(subset) * 100:.2f}% win rate | "
        f"{subset['r'].sum():.2f}R"
    )

# ------------------------------------------------------------
# BUY / SELL
# ------------------------------------------------------------

print()
print("DIRECTION")
print("-" * 60)

for direction in [
    "BUY",
    "SELL"
]:

    subset = df[
        df["direction"] == direction
    ]

    if len(subset) == 0:
        continue

    wins_count = (
        subset["result"] == "TP"
    ).sum()

    print(
        direction,
        "|",
        len(subset),
        "trades |",
        round(
            wins_count / len(subset) * 100,
            2
        ),
        "% win rate |",
        round(
            subset["r"].sum(),
            2
        ),
        "R"
    )

# ------------------------------------------------------------
# TIME OF DAY
# ------------------------------------------------------------

print()
print("TIME OF DAY")
print("-" * 60)

df["time"] = pd.to_datetime(
    df["time"]
)

df["hour"] = (
    df["time"]
    .dt.hour
)

for hour in sorted(
    df["hour"].unique()
):

    subset = df[
        df["hour"] == hour
    ]

    wins_count = (
        subset["result"] == "TP"
    ).sum()

    print(
        f"{hour:02d}:00 | "
        f"{len(subset)} trades | "
        f"{wins_count / len(subset) * 100:.2f}% | "
        f"{subset['r'].sum():.2f}R"
    )

# ------------------------------------------------------------
# BEST HOURS
# ------------------------------------------------------------

print()
print("BEST HOURS")
print("-" * 60)

hours = []

for hour in sorted(
    df["hour"].unique()
):

    subset = df[
        df["hour"] == hour
    ]

    if len(subset) < 20:
        continue

    wins_count = (
        subset["result"] == "TP"
    ).sum()

    win_rate = (
        wins_count
        / len(subset)
        * 100
    )

    hours.append(
        (
            win_rate,
            len(subset),
            hour,
            subset["r"].sum()
        )
    )

for win_rate, count, hour, total_r in sorted(
    hours,
    reverse=True
)[:10]:

    print(
        f"{hour:02d}:00 | "
        f"{count} trades | "
        f"{win_rate:.2f}% | "
        f"{total_r:.2f}R"
    )

# ------------------------------------------------------------
# COMPONENTS
# ------------------------------------------------------------

component_columns = [
    "trend",
    "momentum",
    "breakout",
    "daily",
    "htf"
]

print()
print("COMPONENT ANALYSIS")
print("-" * 60)

for component in component_columns:

    if component not in df.columns:
        continue

    print()
    print(component.upper())

    for direction in [
        "BUY",
        "SELL",
        "NONE"
    ]:

        subset = df[
            df[component]
            .astype(str)
            .str.contains(
                f"'direction': '{direction}'",
                regex=False
            )
        ]

        if len(subset) < 10:
            continue

        wins_count = (
            subset["result"] == "TP"
        ).sum()

        print(
            f"{direction}: "
            f"{len(subset)} trades | "
            f"{wins_count / len(subset) * 100:.2f}% | "
            f"{subset['r'].sum():.2f}R"
        )

# ------------------------------------------------------------
# HIGH SCORE TEST
# ------------------------------------------------------------

print()
print("HIGH-CONFIDENCE TEST")
print("-" * 60)

for threshold in [
    80,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95
]:

    subset = df[
        df["score"] >= threshold
    ]

    if len(subset) < 10:
        continue

    wins_count = (
        subset["result"] == "TP"
    ).sum()

    win_rate = (
        wins_count
        / len(subset)
        * 100
    )

    print(
        f"{threshold}+ | "
        f"{len(subset)} trades | "
        f"{win_rate:.2f}% | "
        f"{subset['r'].sum():.2f}R"
    )

print()
print("=" * 60)
print("END ANALYSIS")
print("=" * 60)
