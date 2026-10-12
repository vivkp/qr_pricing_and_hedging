# Hedge result notes

See the root README for the study design, best-row table, assumptions, and interpretation. This folder contains the complete generated data, not just the winning row.

Key files:
- `intraday_tte_alignment.csv`, `common_entry_alignment.csv`: timestamp/TTE checks.
- `intraday_iv_observations.csv`, `intraday_delta_hedge_events.csv`: calculated IVs and hedge events.
- `intraday_delta_hedge_pnl_by_contract.csv`, `intraday_delta_hedge_pnl_summary.csv`: baseline 4% relative gap / 0.05 delta threshold results across cadences.
- `parameter_sweep_summary.csv`, `parameter_sweep_by_contract.csv`: gross grid over signal cutoffs, delta thresholds, and hedge intervals.
- `transaction_cost_sweep_summary.csv`, `transaction_cost_sweep_by_contract_gross.csv`: modeled net grid at three slippage levels.
- `best_cost_aware_parameters.csv`: best in-sample row for each slippage assumption.
- `hedging_effectiveness_baseline.csv`: option-only versus hedged P&L and cross-strike dispersion for the baseline 4% / 0.05 setup; this is not a time-series risk statistic.
- `fixed_signal_cadence_cost_sensitivity.csv`, `net_pnl_by_hedge_interval.png`: same 2% / 0.01 signal settings across cadences and cost assumptions.
