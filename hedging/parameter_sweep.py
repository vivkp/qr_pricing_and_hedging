"""Sweep RV-vs-IV entry gaps and delta-rehedging thresholds.

Defaults run relative IV/RV gaps of 2%, 4%, 6%, 8%, and 10%, with delta triggers from
0.01 through 0.10 in 0.01 steps, and 1-minute through daily hedge schedules.
The IV/delta observations are calculated once and reused for every setting.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hedging.intraday_delta_hedging import (
    DEFAULT_DIVIDEND_YIELD,
    DEFAULT_OPTIONS,
    DEFAULT_RESULTS,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_UNDERLYING,
    HEDGE_INTERVALS,
    calculate_iv_delta,
    align_to_common_entry,
    load_options,
    load_rv_forecasts,
    run_contract_strategy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path, default=DEFAULT_OPTIONS)
    parser.add_argument("--underlying", type=Path, default=DEFAULT_UNDERLYING)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--risk-free-rate", type=float, default=DEFAULT_RISK_FREE_RATE)
    parser.add_argument("--dividend-yield", type=float, default=DEFAULT_DIVIDEND_YIELD)
    parser.add_argument(
        "--relative-gaps", type=float, nargs="+", default=[0.02, 0.04, 0.06, 0.08, 0.10],
        help="Relative RV-vs-entry-IV cutoffs as decimals (default: 0.02 0.04 0.06 0.08 0.10).",
    )
    parser.add_argument(
        "--delta-triggers", type=float, nargs="+",
        default=[x / 100 for x in range(1, 11)],
        help="Absolute delta-change thresholds (default: 0.01 through 0.10 by 0.01).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(x < 0 for x in args.relative_gaps + args.delta_triggers):
        raise ValueError("Gap and delta trigger values must be non-negative.")

    options = load_options(args.options)
    if options.empty:
        raise ValueError("No option rows were loaded.")
    rv_map = load_rv_forecasts(args.underlying, sorted(options["quote_date"].unique()))
    options["rv_forecast_prior_close"] = options["quote_date"].map(rv_map)

    ivs, deltas = [], []
    for row in options.itertuples(index=False):
        iv, delta = calculate_iv_delta(row, args.risk_free_rate, args.dividend_yield)
        ivs.append(iv)
        deltas.append(delta)
    options["calculated_iv"] = ivs
    options["calculated_delta"] = deltas
    options, common_entry_timestamp, _ = align_to_common_entry(options)
    print(f"Synchronized common entry timestamp: {common_entry_timestamp}")

    contract_groups = list(options.groupby(["expiry", "strike_price", "option_type"], sort=True))
    detail_rows = []
    summary_rows = []
    for gap in args.relative_gaps:
        for trigger in args.delta_triggers:
            for interval in HEDGE_INTERVALS:
                run_rows = []
                for _, contract in contract_groups:
                    result, _ = run_contract_strategy(
                        contract,
                        interval,
                        gap,
                        trigger,
                        args.risk_free_rate,
                        args.dividend_yield,
                    )
                    result["relative_gap_cutoff"] = gap
                    result["delta_trigger"] = trigger
                    detail_rows.append(result)
                    run_rows.append(result)

                traded = [
                    row for row in run_rows
                    if row.get("signal") in {"long_option", "short_option"}
                ]
                pnls = np.asarray([row["gross_pnl"] for row in traded], dtype=float)
                summary_rows.append({
                    "relative_gap_cutoff": gap,
                    "delta_trigger": trigger,
                    "hedge_interval": interval,
                    "contracts_evaluated": len(run_rows),
                    "trades_triggered": len(traded),
                    "long_option_trades": sum(row["signal"] == "long_option" for row in traded),
                    "short_option_trades": sum(row["signal"] == "short_option" for row in traded),
                    "mean_gross_pnl_per_trade": float(np.mean(pnls)) if len(pnls) else np.nan,
                    "median_gross_pnl_per_trade": float(np.median(pnls)) if len(pnls) else np.nan,
                    "sum_gross_pnl_one_unit_each": float(np.sum(pnls)) if len(pnls) else 0.0,
                    "win_rate": float(np.mean(pnls > 0)) if len(pnls) else np.nan,
                    "mean_delta_rebalances": (
                        float(np.mean([row["delta_rebalance_count"] for row in traded]))
                        if traded else np.nan
                    ),
                })

    args.results_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.results_dir / "parameter_sweep_by_contract.csv"
    summary_path = args.results_dir / "parameter_sweep_summary.csv"
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)

    print(f"Option rows: {len(options):,}; contracts: {len(contract_groups):,}")
    print(f"Relative RV-IV gaps: {', '.join(f'{x:.0%}' for x in args.relative_gaps)}")
    print(f"Delta triggers: {', '.join(f'{x:.2f}' for x in args.delta_triggers)}")
    print("\nP&L summary (gross index points; one option unit per active contract):")
    display_cols = [
        "relative_gap_cutoff", "delta_trigger", "hedge_interval", "trades_triggered",
        "mean_gross_pnl_per_trade", "median_gross_pnl_per_trade",
        "sum_gross_pnl_one_unit_each", "win_rate", "mean_delta_rebalances",
    ]
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nPer-contract results: {detail_path}")
    print(f"Sweep summary: {summary_path}")


if __name__ == "__main__":
    main()
