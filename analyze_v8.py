import pandas as pd


FILE = "data/v8_trades.csv"


print("=" * 60)
print("V8 WINNER vs LOSER ANALYSIS")
print("=" * 60)


df = pd.read_csv(FILE)

df["time"] = pd.to_datetime(df["time"])

df["hour"] = df["time"].dt.hour

df["win"] = (
    df["result"] == "TP"
)


wins = df[df["win"]].copy()
losses = df[~df["win"]].copy()


print()
print("TOTAL TRADES:", len(df))
print("WINS:", len(wins))
print("LOSSES:", len(losses))

print()
print(
    "WIN RATE:",
    round(
        len(wins) / len(df) * 100,
        2
    ),
    "%"
)


# ============================================================
# NUMERIC FACTOR ANALYSIS
# ============================================================

print()
print("=" * 60)
print("WINNERS vs LOSERS")
print("=" * 60)

numeric_columns = [
    "ema20",
    "ema50",
    "ema_separation",
    "ema20_slope",
    "ema50_slope",
    "bars_since_cross",
    "upper_wick_ratio",
    "body_ratio",
    "entry",
    "sl",
    "tp",
]


for column in numeric_columns:

    if column not in df.columns:
        continue

    win_values = pd.to_numeric(
        wins[column],
        errors="coerce"
    ).dropna()

    loss_values = pd.to_numeric(
        losses[column],
        errors="coerce"
    ).dropna()

    if len(win_values) == 0:
        continue

    print()
    print(column.upper())
    print("-" * 60)

    print(
        "WIN  |",
        round(win_values.mean(), 6),
        "| median:",
        round(win_values.median(), 6)
    )

    print(
        "LOSS |",
        round(loss_values.mean(), 6),
        "| median:",
        round(loss_values.median(), 6)
    )


# ============================================================
# HOUR ANALYSIS
# ============================================================

print()
print("=" * 60)
print("HOUR ANALYSIS")
print("=" * 60)

for hour in sorted(df["hour"].unique()):

    subset = df[
        df["hour"] == hour
    ]

    if len(subset) < 5:
        continue

    win_count = (
        subset["win"]
        .sum()
    )

    win_rate = (
        win_count
        / len(subset)
        * 100
    )

    total_r = subset["r"].sum()

    print(
        f"{hour:02d}:00 | "
        f"{len(subset)} trades | "
        f"{win_rate:.2f}% | "
        f"{total_r:.2f}R"
    )


# ============================================================
# UPPER WICK BUCKETS
# ============================================================

if "upper_wick_ratio" in df.columns:

    print()
    print("=" * 60)
    print("UPPER WICK ANALYSIS")
    print("=" * 60)

    df["upper_wick_ratio"] = pd.to_numeric(
        df["upper_wick_ratio"],
        errors="coerce"
    )

    buckets = [
        (0.25, 0.30),
        (0.30, 0.35),
        (0.35, 0.40),
        (0.40, 0.45),
        (0.45, 0.50),
        (0.50, 0.60),
        (0.60, 1.00),
    ]

    for low, high in buckets:

        subset = df[
            (df["upper_wick_ratio"] >= low)
            & (df["upper_wick_ratio"] < high)
        ]

        if len(subset) < 5:
            continue

        win_rate = (
            subset["win"].sum()
            / len(subset)
            * 100
        )

        print(
            f"{low:.2f}-{high:.2f} | "
            f"{len(subset)} trades | "
            f"{win_rate:.2f}% | "
            f"{subset['r'].sum():.2f}R"
        )


# ============================================================
# BODY RATIO
# ============================================================

if "body_ratio" in df.columns:

    print()
    print("=" * 60)
    print("BEARISH BODY ANALYSIS")
    print("=" * 60)

    df["body_ratio"] = pd.to_numeric(
        df["body_ratio"],
        errors="coerce"
    )

    buckets = [
        (0.20, 0.25),
        (0.25, 0.30),
        (0.30, 0.35),
        (0.35, 0.40),
        (0.40, 0.50),
        (0.50, 0.60),
        (0.60, 1.00),
    ]

    for low, high in buckets:

        subset = df[
            (df["body_ratio"] >= low)
            & (df["body_ratio"] < high)
        ]

        if len(subset) < 5:
            continue

        win_rate = (
            subset["win"].sum()
            / len(subset)
            * 100
        )

        print(
            f"{low:.2f}-{high:.2f} | "
            f"{len(subset)} trades | "
            f"{win_rate:.2f}% | "
            f"{subset['r'].sum():.2f}R"
        )


# ============================================================
# EMA SEPARATION
# ============================================================

if "ema_separation" in df.columns:

    print()
    print("=" * 60)
    print("EMA SEPARATION ANALYSIS")
    print("=" * 60)

    df["ema_separation"] = pd.to_numeric(
        df["ema_separation"],
        errors="coerce"
    )

    buckets = [
        (0.0008, 0.0010),
        (0.0010, 0.0015),
        (0.0015, 0.0020),
        (0.0020, 0.0030),
        (0.0030, 0.0050),
        (0.0050, 0.0100),
        (0.0100, 1.0000),
    ]

    for low, high in buckets:

        subset = df[
            (df["ema_separation"] >= low)
            & (df["ema_separation"] < high)
        ]

        if len(subset) < 5:
            continue

        win_rate = (
            subset["win"].sum()
            / len(subset)
            * 100
        )

        print(
            f"{low:.4f}-{high:.4f} | "
            f"{len(subset)} trades | "
            f"{win_rate:.2f}% | "
            f"{subset['r'].sum():.2f}R"
        )


# ============================================================
# BARS SINCE CROSS
# ============================================================

if "bars_since_cross" in df.columns:

    print()
    print("=" * 60)
    print("BARS SINCE CROSS")
    print("=" * 60)

    df["bars_since_cross"] = pd.to_numeric(
        df["bars_since_cross"],
        errors="coerce"
    )

    buckets = [
        (0, 5),
        (5, 10),
        (10, 15),
        (15, 20),
        (20, 30),
        (30, 40),
        (40, 60),
    ]

    for low, high in buckets:

        subset = df[
            (df["bars_since_cross"] >= low)
            & (df["bars_since_cross"] < high)
        ]

        if len(subset) < 5:
            continue

        win_rate = (
            subset["win"].sum()
            / len(subset)
            * 100
        )

        print(
            f"{low}-{high} bars | "
            f"{len(subset)} trades | "
            f"{win_rate:.2f}% | "
            f"{subset['r'].sum():.2f}R"
        )


# ============================================================
# COMBINATION TESTS
# ============================================================

print()
print("=" * 60)
print("HIGH-VALUE COMBINATIONS")
print("=" * 60)


tests = []


# Stronger wick
if "upper_wick_ratio" in df.columns:

    tests.append(
        (
            "Wick >= 0.50",
            df["upper_wick_ratio"] >= 0.50
        )
    )

    tests.append(
        (
            "Wick >= 0.55",
            df["upper_wick_ratio"] >= 0.55
        )
    )


# Stronger body
if "body_ratio" in df.columns:

    tests.append(
        (
            "Body >= 0.35",
            df["body_ratio"] >= 0.35
        )
    )

    tests.append(
        (
            "Body >= 0.40",
            df["body_ratio"] >= 0.40
        )
    )


# Stronger EMA separation
if "ema_separation" in df.columns:

    tests.append(
        (
            "EMA separation >= 0.0015",
            df["ema_separation"] >= 0.0015
        )
    )

    tests.append(
        (
            "EMA separation >= 0.0020",
            df["ema_separation"] >= 0.0020
        )
    )


# Early cross
if "bars_since_cross" in df.columns:

    tests.append(
        (
            "Cross <= 20 bars",
            df["bars_since_cross"] <= 20
        )
    )

    tests.append(
        (
            "Cross <= 15 bars",
            df["bars_since_cross"] <= 15
        )
    )


# Combinations
if (
    "upper_wick_ratio" in df.columns
    and "body_ratio" in df.columns
):

    tests.append(
        (
            "Wick >= .50 + Body >= .35",
            (
                (df["upper_wick_ratio"] >= 0.50)
                & (df["body_ratio"] >= 0.35)
            )
        )
    )

    tests.append(
        (
            "Wick >= .55 + Body >= .40",
            (
                (df["upper_wick_ratio"] >= 0.55)
                & (df["body_ratio"] >= 0.40)
            )
        )
    )


if (
    "upper_wick_ratio" in df.columns
    and "ema_separation" in df.columns
):

    tests.append(
        (
            "Wick >= .50 + EMA >= .0015",
            (
                (df["upper_wick_ratio"] >= 0.50)
                & (df["ema_separation"] >= 0.0015)
            )
        )
    )


if (
    "body_ratio" in df.columns
    and "ema_separation" in df.columns
):

    tests.append(
        (
            "Body >= .35 + EMA >= .0015",
            (
                (df["body_ratio"] >= 0.35)
                & (df["ema_separation"] >= 0.0015)
            )
        )
    )


results = []

for name, mask in tests:

    subset = df[mask]

    if len(subset) < 10:
        continue

    win_rate = (
        subset["win"].sum()
        / len(subset)
        * 100
    )

    total_r = subset["r"].sum()

    results.append(
        (
            win_rate,
            len(subset),
            total_r,
            name
        )
    )


print()

for win_rate, count, total_r, name in sorted(
    results,
    reverse=True
):

    print(
        f"{name} | "
        f"{count} trades | "
        f"{win_rate:.2f}% | "
        f"{total_r:.2f}R"
    )


print()
print("=" * 60)
print("END WINNER vs LOSER ANALYSIS")
print("=" * 60)
