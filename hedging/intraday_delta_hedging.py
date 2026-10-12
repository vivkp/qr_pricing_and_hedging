"""Backtest IV-versus-EWMA-RV option positions with intraday delta hedging.

The trade side is fixed at entry for each expiry/strike/type contract:
    relative_gap = (forecast_RV - entry_IV) / entry_IV
    long option if relative_gap >= +4%;
    short option if relative_gap <= -4%; otherwise stay flat.

RV is the user's 30-span EWMA annualized volatility, calculated from daily
close-to-close log returns. Each option-day uses only the prior trading day's
EWMA value. IV is recalculated from each observed LTP with Black-Scholes.

P&L is gross per one option unit, using NIFTY spot as the hedge proxy. It
excludes bid/ask, fees, financing and futures basis. LTP is used as the option
mark. The final quote is the last available quote, not an assumed expiry
settlement price.
"""

from __future__ import annotations

import argparse
import math
import sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from greeks.analytical import call_delta, put_delta
from pricing.implied_volatility import implied_volatility


DEFAULT_OPTIONS = PROJECT_ROOT / "data" / "nifty_calls_600pt_expiry_2024-10-03.csv"
DEFAULT_UNDERLYING = PROJECT_ROOT / "data" / "nifty_daily_close_through_2024-09-24.csv"
DEFAULT_RESULTS = PROJECT_ROOT / "hedging" / "results"
DEFAULT_RISK_FREE_RATE = 0.065
DEFAULT_DIVIDEND_YIELD = 0.0
DEFAULT_RELATIVE_GAP = 0.04
DEFAULT_DELTA_TRIGGER = 0.05
EWMA_SPAN = 30
TRADING_DAYS_PER_YEAR = 252
HEDGE_INTERVALS = {"1min": 1, "5min": 5, "30min": 30, "1hour": 60, "1day": None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path, default=DEFAULT_OPTIONS)
    parser.add_argument("--underlying", type=Path, default=DEFAULT_UNDERLYING)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--risk-free-rate", type=float, default=DEFAULT_RISK_FREE_RATE)
    parser.add_argument("--dividend-yield", type=float, default=DEFAULT_DIVIDEND_YIELD)
    parser.add_argument("--relative-gap", type=float, default=DEFAULT_RELATIVE_GAP)
    parser.add_argument("--delta-trigger", type=float, default=DEFAULT_DELTA_TRIGGER)
    return parser.parse_args()


def load_options(path: Path) -> pd.DataFrame:
    options = pd.read_csv(path)
    required = {
        "strike_price", "option_type", "expiry", "timestamp", "ltp",
        "underlying_spot_price", "tte_years",
    }
    missing = required.difference(options.columns)
    if missing:
        raise ValueError(f"Options data missing columns: {', '.join(sorted(missing))}")
    options["timestamp"] = pd.to_datetime(options["timestamp"], errors="coerce")
    options["expiry"] = pd.to_datetime(options["expiry"], errors="coerce")
    for col in ("strike_price", "ltp", "underlying_spot_price", "tte_years"):
        options[col] = pd.to_numeric(options[col], errors="coerce")
    options = options.dropna(
        subset=["strike_price", "option_type", "expiry", "timestamp", "ltp",
                "underlying_spot_price", "tte_years"]
    ).copy()
    options["option_type"] = options["option_type"].astype(str).str.upper()
    options["quote_date"] = options["timestamp"].dt.date
    return options.sort_values("timestamp").reset_index(drop=True)


def load_rv_forecasts(path: Path, option_dates: list) -> dict:
    """Recreate the notebook EWMA, then map each date to the prior close's RV."""
    underlying = pd.read_csv(path)
    if not {"datetime", "close"}.issubset(underlying.columns):
        raise ValueError("Underlying file needs 'datetime' and 'close' columns.")
    underlying["datetime"] = pd.to_datetime(underlying["datetime"], errors="coerce")
    underlying["close"] = pd.to_numeric(underlying["close"], errors="coerce")
    underlying = underlying.dropna(subset=["datetime", "close"])
    underlying = underlying.sort_values("datetime").set_index("datetime")

    daily_close = underlying["close"].resample("D").last().dropna()
    daily_log_returns = np.log(daily_close / daily_close.shift(1)).dropna()
    ewma_mean = daily_log_returns.ewm(span=EWMA_SPAN).mean()
    demeaned_returns = daily_log_returns - ewma_mean
    ewma_variance = demeaned_returns.ewm(span=EWMA_SPAN).var()
    annualized_rv = np.sqrt(ewma_variance) * math.sqrt(TRADING_DAYS_PER_YEAR)

    rv_dates = [index.date() for index in annualized_rv.index]
    rv_values = annualized_rv.to_numpy()
    result = {}
    for quote_date in option_dates:
        # Strictly earlier date prevents use of any same-day return.
        previous_index = bisect_left(rv_dates, quote_date) - 1
        result[quote_date] = (
            float(rv_values[previous_index])
            if previous_index >= 0 and math.isfinite(rv_values[previous_index])
            else math.nan
        )
    return result


def option_kind(option_type: str) -> str:
    if option_type in {"CE", "CALL", "C"}:
        return "call"
    if option_type in {"PE", "PUT", "P"}:
        return "put"
    raise ValueError(f"Unsupported option type: {option_type}")


def calculate_iv_delta(row, risk_free_rate: float, dividend_yield: float):
    kind = option_kind(row.option_type)
    values = (row.underlying_spot_price, row.strike_price, row.tte_years, row.ltp)
    if not all(math.isfinite(float(value)) for value in values):
        return math.nan, math.nan
    if row.underlying_spot_price <= 0 or row.strike_price <= 0:
        return math.nan, math.nan
    if row.tte_years <= 0 or row.ltp <= 0:
        return math.nan, math.nan

    iv = implied_volatility(
        float(row.underlying_spot_price),
        float(row.strike_price),
        float(row.tte_years),
        risk_free_rate,
        dividend_yield,
        float(row.ltp),
        option_type=kind,
    )
    if not math.isfinite(iv) or iv <= 0:
        return math.nan, math.nan
    delta_fn = call_delta if kind == "call" else put_delta
    delta = delta_fn(
        float(row.underlying_spot_price), float(row.strike_price),
        float(row.tte_years), risk_free_rate, iv, dividend_yield,
    )
    return iv, delta


def audit_tte(options: pd.DataFrame) -> pd.DataFrame:
    by_time = options.groupby(["expiry", "timestamp"], sort=True)
    tte_counts = by_time["tte_years"].nunique(dropna=False)
    strike_counts = by_time["strike_price"].nunique()
    by_expiry = options.groupby("expiry", sort=True)
    expected_strikes = by_expiry["strike_price"].nunique()
    audit = []
    for expiry, group in by_expiry:
        time_slice = tte_counts.loc[expiry]
        strike_slice = strike_counts.loc[expiry]
        audit.append({
            "expiry": expiry,
            "strikes_in_file": int(group["strike_price"].nunique()),
            "option_types_in_file": ",".join(sorted(group["option_type"].unique())),
            "unique_timestamps": int(group["timestamp"].nunique()),
            "timestamps_with_tte_disagreement": int((time_slice > 1).sum()),
            "timestamps_missing_some_strikes": int(
                (strike_slice < expected_strikes.loc[expiry]).sum()
            ),
            "first_quote": group["timestamp"].min(),
            "last_quote": group["timestamp"].max(),
            "expiry_date_has_quote": bool(
                group["timestamp"].max().date() == expiry.date()
            ),
        })
    return pd.DataFrame(audit)


def align_to_common_entry(options: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp, pd.DataFrame]:
    """Trim all contracts to the first timestamp with valid IV/delta for each."""
    keys = ["expiry", "strike_price", "option_type"]
    expected = options[keys].drop_duplicates().shape[0]
    valid = options[
        options["calculated_iv"].notna()
        & options["calculated_delta"].notna()
        & options["rv_forecast_prior_close"].notna()
    ]
    counts = valid.groupby("timestamp")[keys].apply(
        lambda frame: frame.drop_duplicates().shape[0]
    )
    common_times = counts[counts >= expected]
    if common_times.empty:
        raise ValueError(f"No timestamp has valid IV/delta for all {expected} contracts.")
    common_timestamp = pd.Timestamp(common_times.index.min())

    first_raw = options.sort_values("timestamp").groupby(keys, sort=True).first()
    first_valid = valid.sort_values("timestamp").groupby(keys, sort=True).first()
    alignment = first_raw[["timestamp"]].rename(columns={"timestamp": "first_raw_quote"})
    alignment["first_valid_iv_quote"] = first_valid["timestamp"]
    alignment = alignment.reset_index()
    alignment["common_entry_timestamp"] = common_timestamp
    common_keys = options.loc[options["timestamp"] == common_timestamp, keys].drop_duplicates()
    alignment["has_quote_at_common_entry"] = pd.MultiIndex.from_frame(alignment[keys]).isin(
        pd.MultiIndex.from_frame(common_keys)
    )
    valid_keys = valid.loc[valid["timestamp"] == common_timestamp, keys].drop_duplicates()
    alignment["has_valid_iv_at_common_entry"] = pd.MultiIndex.from_frame(alignment[keys]).isin(
        pd.MultiIndex.from_frame(valid_keys)
    )
    return options.loc[options["timestamp"] >= common_timestamp].copy(), common_timestamp, alignment


def determine_side(relative_gap: float, cutoff: float) -> str:
    if not math.isfinite(relative_gap):
        return "no_trade_missing_rv_or_iv"
    if relative_gap >= cutoff:
        return "long_option"
    if relative_gap <= -cutoff:
        return "short_option"
    return "no_trade_inside_gap"


def hedge_checkpoints(group: pd.DataFrame, interval_name: str) -> pd.DataFrame:
    if interval_name == "1day":
        # One check at the last observed quote of each trading day.
        return group.groupby(group["timestamp"].dt.date, sort=True).tail(1)
    interval_minutes = HEDGE_INTERVALS[interval_name]
    market_open = 9 * 60 + 15
    minute_of_day = group["timestamp"].dt.hour * 60 + group["timestamp"].dt.minute
    minutes_from_open = minute_of_day - market_open
    selected = group[(minutes_from_open >= 0) & (minutes_from_open % interval_minutes == 0)]
    return selected


def run_contract_strategy(
    contract: pd.DataFrame,
    interval_name: str,
    relative_gap_cutoff: float,
    delta_trigger: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> tuple[dict, list[dict]]:
    contract = contract.sort_values("timestamp").reset_index(drop=True)
    valid = contract[
        contract["calculated_iv"].notna()
        & contract["calculated_delta"].notna()
        & contract["rv_forecast_prior_close"].notna()
    ].copy()
    base = {
        "expiry": contract["expiry"].iloc[0],
        "strike": float(contract["strike_price"].iloc[0]),
        "option_type": contract["option_type"].iloc[0],
        "hedge_interval": interval_name,
        "source_quote_count": len(contract),
        "valid_iv_quote_count": int(contract["calculated_iv"].notna().sum()),
        "invalid_iv_quote_count": int(contract["calculated_iv"].isna().sum()),
        "delta_trigger": delta_trigger,
        "relative_gap_cutoff": relative_gap_cutoff,
        "risk_free_rate": risk_free_rate,
        "dividend_yield": dividend_yield,
        "pnl_units": "index points per one option unit, gross",
    }
    if valid.empty:
        return {**base, "signal": "no_trade_missing_rv_or_iv", "gross_pnl": 0.0}, []

    entry = valid.iloc[0]
    entry_iv = float(entry["calculated_iv"])
    entry_rv = float(entry["rv_forecast_prior_close"])
    relative_gap = (entry_rv - entry_iv) / entry_iv
    side = determine_side(relative_gap, relative_gap_cutoff)
    raw_last = contract.iloc[-1]
    exit_price = float(raw_last["ltp"])
    exit_spot = float(raw_last["underlying_spot_price"])

    result = {
        **base,
        "entry_timestamp": entry["timestamp"],
        "exit_timestamp": raw_last["timestamp"],
        "entry_ltp": float(entry["ltp"]),
        "exit_ltp": exit_price,
        "entry_iv": entry_iv,
        "entry_rv_forecast": entry_rv,
        "relative_rv_minus_iv": relative_gap,
        "signal": side,
        "entry_delta": float(entry["calculated_delta"]),
        "exit_spot": exit_spot,
    }
    if side not in {"long_option", "short_option"}:
        return {**result, "option_pnl": 0.0, "hedge_pnl": 0.0, "gross_pnl": 0.0,
                "delta_rebalance_count": 0, "scheduled_checkpoint_count": 0}, []

    side_sign = 1.0 if side == "long_option" else -1.0
    initial_delta = float(entry["calculated_delta"])
    hedge_units = -side_sign * initial_delta
    last_hedged_delta = initial_delta
    previous_hedge_row = entry
    cumulative_hedge_pnl = 0.0
    rebalance_count = 0
    events = [{
        "expiry": result["expiry"],
        "strike": result["strike"],
        "option_type": result["option_type"],
        "hedge_interval": interval_name,
        "timestamp": entry["timestamp"],
        "event": "entry_hedge",
        "option_side": side,
        "market_ltp": float(entry["ltp"]),
        "spot": float(entry["underlying_spot_price"]),
        "calculated_iv": entry_iv,
        "calculated_delta": initial_delta,
        "relative_rv_minus_iv": relative_gap,
        "delta_gap_from_last_hedge": 0.0,
        "hedge_rebalanced": True,
        "hedge_trade_units": hedge_units,
        "hedge_units_after": hedge_units,
        "cumulative_hedge_pnl": 0.0,
    }]

    checkpoints = hedge_checkpoints(valid, interval_name)
    checkpoints = checkpoints[checkpoints["timestamp"] > entry["timestamp"]]
    for _, current in checkpoints.iterrows():
        current_delta = float(current["calculated_delta"])
        current_spot = float(current["underlying_spot_price"])
        cumulative_hedge_pnl += hedge_units * (
            current_spot - float(previous_hedge_row["underlying_spot_price"])
        )
        delta_gap = abs(current_delta - last_hedged_delta)
        rebalance = delta_gap > delta_trigger
        trade_units = 0.0
        if rebalance:
            new_hedge_units = -side_sign * current_delta
            trade_units = new_hedge_units - hedge_units
            hedge_units = new_hedge_units
            last_hedged_delta = current_delta
            rebalance_count += 1
        events.append({
            "expiry": result["expiry"],
            "strike": result["strike"],
            "option_type": result["option_type"],
            "hedge_interval": interval_name,
            "timestamp": current["timestamp"],
            "event": "delta_rebalance" if rebalance else "check_no_rebalance",
            "option_side": side,
            "market_ltp": float(current["ltp"]),
            "spot": current_spot,
            "calculated_iv": float(current["calculated_iv"]),
            "calculated_delta": current_delta,
            "relative_rv_minus_iv": relative_gap,
            "delta_gap_from_last_hedge": delta_gap,
            "hedge_rebalanced": bool(rebalance),
            "hedge_trade_units": trade_units,
            "hedge_units_after": hedge_units,
            "cumulative_hedge_pnl": cumulative_hedge_pnl,
        })
        previous_hedge_row = current

    # Carry the last hedge to the final available option mark, even if the
    # final quote does not land exactly on the selected rehedge schedule.
    cumulative_hedge_pnl += hedge_units * (
        exit_spot - float(previous_hedge_row["underlying_spot_price"])
    )
    option_pnl = side_sign * (exit_price - float(entry["ltp"]))
    gross_pnl = option_pnl + cumulative_hedge_pnl
    result.update({
        "option_pnl": option_pnl,
        "hedge_pnl": cumulative_hedge_pnl,
        "gross_pnl": gross_pnl,
        "delta_rebalance_count": rebalance_count,
        "scheduled_checkpoint_count": len(checkpoints),
        "last_quote_is_expiry_day": raw_last["timestamp"].date() == raw_last["expiry"].date(),
    })
    return result, events


def write_method_note(
    path: Path,
    options: pd.DataFrame,
    options_path: Path,
    rv_path: Path,
    risk_free_rate: float,
    dividend_yield: float,
    relative_gap_cutoff: float,
    delta_trigger: float,
    common_entry_timestamp: pd.Timestamp,
) -> None:
    expiries = ", ".join(sorted(options["expiry"].dt.strftime("%Y-%m-%d").unique()))
    dates = sorted(options["timestamp"].dt.strftime("%Y-%m-%d").unique())
    note = f"""# Intraday IV versus realized volatility hedge study

Options input: `{options_path.name}`. Underlying history: `{rv_path.name}`.

- Expiry in this run: {expiries}
- Option quote dates: {dates[0]} through {dates[-1]} ({len(dates)} trading dates)
- RV forecast: reproduce the notebook's EWMA method (`span=30`, demeaned daily log returns, annualized by `sqrt(252)`). For each quote date, use the prior trading day's RV.
- Synchronized entry: every contract enters at the first timestamp with a valid IV/delta for all strikes: {common_entry_timestamp}.
- Entry signal: `(RV forecast - entry IV) / entry IV`. Buy if at least +{relative_gap_cutoff:.1%}; sell if at most -{relative_gap_cutoff:.1%}; otherwise no trade.
- IV: invert Black-Scholes using option LTP, `r={risk_free_rate:.2%}`, `q={dividend_yield:.2%}`, and source `tte_years`.
- Delta: Black-Scholes delta from that row's calculated IV. Start with the entry hedge and check at the requested frequency. Rehedge only when the absolute delta change from the last hedge is greater than {delta_trigger:.2f}.
- Frequencies: 5-minute, 30-minute, and 60-minute checks anchored at 09:15 India time; daily checks at each day's final available quote.
- Exit: mark the option and hedge at the last available quote for that contract. It may be before expiry if the contract has no later quote.
- P&L: option mark-to-market plus hedge price changes, per one option unit. Gross only. No bid/ask, fees, funding, lot multiplier, futures basis, or expiry settlement is included.
- LTP is a last-traded price and can be stale. Rows whose LTP cannot produce a positive Black-Scholes IV are excluded from IV/delta checks and counted in the outputs.

The summary's total is the simple sum across active one-unit contracts. It is a diagnostic, not a capital-weighted portfolio return.
"""
    path.write_text(note, encoding="utf-8")


def run() -> None:
    args = parse_args()
    if args.relative_gap < 0 or args.delta_trigger < 0:
        raise ValueError("Gap and delta trigger must be non-negative")
    options = load_options(args.options)
    if options.empty:
        raise ValueError("No option rows were loaded.")

    tte_audit = audit_tte(options)
    rv_map = load_rv_forecasts(args.underlying, sorted(options["quote_date"].unique()))
    options["rv_forecast_prior_close"] = options["quote_date"].map(rv_map)

    ivs = []
    deltas = []
    for row in options.itertuples(index=False):
        iv, delta = calculate_iv_delta(row, args.risk_free_rate, args.dividend_yield)
        ivs.append(iv)
        deltas.append(delta)
    options["calculated_iv"] = ivs
    options["calculated_delta"] = deltas
    options["relative_rv_minus_iv"] = (
        options["rv_forecast_prior_close"] - options["calculated_iv"]
    ) / options["calculated_iv"]
    options, common_entry_timestamp, alignment = align_to_common_entry(options)

    summary_rows = []
    event_rows = []
    contract_keys = ["expiry", "strike_price", "option_type"]
    for _, contract in options.groupby(contract_keys, sort=True, dropna=False):
        for interval_name in HEDGE_INTERVALS:
            summary, events = run_contract_strategy(
                contract,
                interval_name,
                args.relative_gap,
                args.delta_trigger,
                args.risk_free_rate,
                args.dividend_yield,
            )
            summary_rows.append(summary)
            event_rows.extend(events)

    summary = pd.DataFrame(summary_rows)
    active = summary[summary["signal"].isin(["long_option", "short_option"])]
    aggregate_rows = []
    for interval_name in HEDGE_INTERVALS:
        all_interval = summary[summary["hedge_interval"] == interval_name]
        traded = active[active["hedge_interval"] == interval_name]
        aggregate_rows.append({
            "hedge_interval": interval_name,
            "contracts_evaluated": len(all_interval),
            "trades_triggered": len(traded),
            "long_option_trades": int((traded["signal"] == "long_option").sum()),
            "short_option_trades": int((traded["signal"] == "short_option").sum()),
            "mean_gross_pnl_per_trade": traded["gross_pnl"].mean(),
            "median_gross_pnl_per_trade": traded["gross_pnl"].median(),
            "sum_gross_pnl_one_unit_each": traded["gross_pnl"].sum(),
            "win_rate": (traded["gross_pnl"] > 0).mean(),
            "mean_delta_rebalances": traded["delta_rebalance_count"].mean(),
        })

    args.results_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.results_dir / "intraday_tte_alignment.csv"
    iv_path = args.results_dir / "intraday_iv_observations.csv"
    summary_path = args.results_dir / "intraday_delta_hedge_pnl_by_contract.csv"
    aggregate_path = args.results_dir / "intraday_delta_hedge_pnl_summary.csv"
    events_path = args.results_dir / "intraday_delta_hedge_events.csv"
    method_path = args.results_dir / "intraday_iv_rv_method.md"
    alignment_path = args.results_dir / "common_entry_alignment.csv"

    tte_audit.to_csv(audit_path, index=False)
    alignment.to_csv(alignment_path, index=False)
    options.drop(columns=["quote_date"]).to_csv(iv_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(aggregate_rows).to_csv(aggregate_path, index=False)
    pd.DataFrame(event_rows).to_csv(events_path, index=False)
    write_method_note(
        method_path, options, args.options, args.underlying,
        args.risk_free_rate, args.dividend_yield,
        args.relative_gap, args.delta_trigger,
        common_entry_timestamp,
    )

    print(f"Option rows: {len(options):,}")
    print(f"Contracts: {len(options.groupby(contract_keys)):,}")
    print(f"Strikes: {options['strike_price'].nunique():,}")
    print(f"Option types: {', '.join(sorted(options['option_type'].unique()))}")
    print(f"Quote dates: {options['timestamp'].min()} through {options['timestamp'].max()}")
    print(f"Common entry timestamp: {common_entry_timestamp}")
    print(f"TTE disagreements at same expiry/timestamp: {int(tte_audit['timestamps_with_tte_disagreement'].sum())}")
    print(f"IV-valid rows: {int(options['calculated_iv'].notna().sum()):,} / {len(options):,}")
    print(f"Risk-free rate={args.risk_free_rate:.2%}, dividend yield={args.dividend_yield:.2%}")
    print(f"Signal cutoff: {args.relative_gap:.1%} relative to entry IV; delta trigger={args.delta_trigger:.2f}")
    print("\nP&L summary (gross, one option unit per active contract):")
    print(pd.DataFrame(aggregate_rows).to_string(index=False))
    print(f"\nTTE audit: {audit_path}")
    print(f"Common-entry alignment: {alignment_path}")
    print(f"IV observations: {iv_path}")
    print(f"Contract P&L: {summary_path}")
    print(f"P&L summary: {aggregate_path}")
    print(f"Hedge events: {events_path}")
    print(f"Method: {method_path}")


if __name__ == "__main__":
    run()
