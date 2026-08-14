# ============================================================
# EXPORT FULL TRADE DATA
# ============================================================

os.makedirs("data", exist_ok=True)

export_rows = []

for trade in trades:

    row = {
        "time": trade["time"],
        "direction": trade["direction"],
        "score": trade["score"],
        "entry": trade["entry"],
        "sl": trade["sl"],
        "tp": trade["tp"],
        "result": trade["result"],
        "r": trade["r"],
        "exit_time": trade["exit_time"],
    }

    components = trade.get(
        "components",
        {}
    )

    for name in [
        "trend",
        "momentum",
        "breakout",
        "daily",
        "htf",
        "volatility"
    ]:

        component = components.get(
            name,
            {}
        )

        if isinstance(component, dict):

            row[
                f"{name}_direction"
            ] = component.get(
                "direction",
                ""
            )

            row[
                f"{name}_score"
            ] = component.get(
                "score",
                0
            )

            if name == "volatility":

                row[
                    "volatility_regime"
                ] = component.get(
                    "regime",
                    ""
                )

        else:

            row[
                f"{name}_direction"
            ] = ""

            row[
                f"{name}_score"
            ] = 0

    export_rows.append(row)


trades_df = pd.DataFrame(
    export_rows
)

trades_df.to_csv(
    "data/v8_trades.csv",
    index=False
)

print()
print("=" * 60)
print("TRADE DATA EXPORTED")
print("=" * 60)
print(
    "File: data/v8_trades.csv"
)
print(
    "Trades exported:",
    len(trades_df)
)
