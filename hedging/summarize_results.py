"""Summarize the rerun hedge sweeps and make a cadence/cost chart."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "hedging" / "results"


def main() -> None:
    table = pd.read_csv(RESULTS / "transaction_cost_sweep_summary.csv")
    best = table.loc[table.groupby("slippage_points_per_side")[
        "net_pnl_rupees_one_lot_each"
    ].idxmax()].sort_values("slippage_points_per_side")
    best.to_csv(RESULTS / "best_cost_aware_parameters.csv", index=False)

    # Hold the signal cutoff and delta trigger fixed to make hedge cadence the
    # only changing strategy parameter in this comparison.
    fixed = table[
        table["relative_gap_cutoff"].eq(0.02)
        & table["delta_trigger"].eq(0.01)
    ].copy()
    fixed.to_csv(RESULTS / "fixed_signal_cadence_cost_sensitivity.csv", index=False)

    # Decompose the baseline strategy's option-only and hedged outcomes. This
    # is a cross-strike dispersion diagnostic, not a time-series VaR estimate.
    baseline = pd.read_csv(RESULTS / "intraday_delta_hedge_pnl_by_contract.csv")
    baseline = baseline[
        baseline["relative_gap_cutoff"].eq(0.04)
        & baseline["delta_trigger"].eq(0.05)
    ]
    risk_rows = []
    for interval, group in baseline.groupby("hedge_interval"):
        for label, column in (("option_only", "option_pnl"),
                              ("hedged_total", "gross_pnl")):
            risk_rows.append({
                "hedge_interval": interval,
                "pnl_component": label,
                "contracts": len(group),
                "sum_pnl_points": group[column].sum(),
                "mean_pnl_points_per_strike": group[column].mean(),
                "cross_strike_std_points": group[column].std(ddof=1),
                "positive_strike_fraction": (group[column] > 0).mean(),
            })
    pd.DataFrame(risk_rows).to_csv(
        RESULTS / "hedging_effectiveness_baseline.csv", index=False
    )

    order = ["1min", "5min", "30min", "1hour", "1day"]
    plt.figure(figsize=(9, 6))
    for slippage, group in fixed.groupby("slippage_points_per_side"):
        group = group.set_index("hedge_interval").reindex(order)
        plt.plot(order, group["net_pnl_rupees_one_lot_each"], marker="o",
                 label=f"Slippage {slippage:.2f} points / side")
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Hedge check interval")
    plt.ylabel("Net P&L (INR; 24 strikes × one 25-unit lot)")
    plt.title("Delta-Hedging Cadence and Estimated Costs\n"
              "2% RV–IV cutoff, 0.01 delta trigger")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "net_pnl_by_hedge_interval.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
