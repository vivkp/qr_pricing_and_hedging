"""Cost-aware portfolio sweep for the filtered NIFTY call-option study.

Run this for one NIFTY option lot (25 units) per selected strike. Hedge changes
are netted across strikes at each timestamp. Because the source contains option
LTP and NIFTY spot only, not bid/ask or futures quotes, this uses explicit
slippage sensitivity and spot as the futures-price proxy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hedging.intraday_delta_hedging import (
    DEFAULT_DIVIDEND_YIELD, DEFAULT_OPTIONS, DEFAULT_RESULTS,
    DEFAULT_RISK_FREE_RATE, DEFAULT_UNDERLYING, HEDGE_INTERVALS,
    calculate_iv_delta, load_options, load_rv_forecasts, run_contract_strategy,
    align_to_common_entry,
)

LOT_SIZE = 25
BROKERAGE_PER_ORDER = 20.0
SLIPPAGE_SCENARIOS = (0.05, 0.25, 0.50)
GST_RATE = 0.18
SEBI_FEE_PER_RUPEE = 10.0 / 10_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path, default=DEFAULT_OPTIONS)
    parser.add_argument("--underlying", type=Path, default=DEFAULT_UNDERLYING)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--risk-free-rate", type=float, default=DEFAULT_RISK_FREE_RATE)
    parser.add_argument("--dividend-yield", type=float, default=DEFAULT_DIVIDEND_YIELD)
    parser.add_argument("--relative-gaps", type=float, nargs="+", default=[.02, .04, .06, .08, .10])
    parser.add_argument("--delta-triggers", type=float, nargs="+", default=[x/100 for x in range(1, 11)])
    parser.add_argument("--slippage-points", type=float, nargs="+", default=list(SLIPPAGE_SCENARIOS))
    return parser.parse_args()


def option_costs(results: list[dict], slip: float) -> dict:
    totals = defaultdict(float)
    active = [r for r in results if r.get("signal") in {"long_option", "short_option"}]
    for r in active:
        qty = LOT_SIZE
        entry_date = pd.Timestamp(r["entry_timestamp"]).date()
        exit_date = pd.Timestamp(r["exit_timestamp"]).date()
        entry_premium = float(r["entry_ltp"])*qty
        exit_premium = float(r["exit_ltp"])*qty
        entry_sell = r["signal"] == "short_option"
        exit_sell = not entry_sell
        premium_turnover = entry_premium+exit_premium
        # Pre-Oct-1 NSE option fees used broker monthly slabs; approximate with
        # the published marginal rate (not a broker-specific monthly allocation).
        entry_ex_rate = 35.03/100_000 if entry_date >= date(2024,10,1) else 49.50/100_000
        exit_ex_rate = 35.03/100_000 if exit_date >= date(2024,10,1) else 49.50/100_000
        totals["option_exchange_charges"] += entry_premium*entry_ex_rate+exit_premium*exit_ex_rate
        totals["option_sebi_fees"] += premium_turnover*SEBI_FEE_PER_RUPEE
        entry_stt = .001 if entry_date >= date(2024,10,1) else .000625
        exit_stt = .001 if exit_date >= date(2024,10,1) else .000625
        if entry_sell: totals["option_stt"] += entry_premium*entry_stt
        if exit_sell: totals["option_stt"] += exit_premium*exit_stt
        if not entry_sell: totals["option_stamp_duty"] += entry_premium*.00003
        if not exit_sell: totals["option_stamp_duty"] += exit_premium*.00003
        totals["option_brokerage"] += 2*BROKERAGE_PER_ORDER
        totals["option_slippage"] += 2*slip*qty
    totals["option_orders"] = 2*len(active)
    return totals


def portfolio_hedge_trades(pairs: list[tuple[dict,list[dict]]]) -> list[tuple[pd.Timestamp,float,float]]:
    by_time: dict[pd.Timestamp,list[tuple[float,float]]] = defaultdict(list)
    for result, events in pairs:
        if result.get("signal") not in {"long_option","short_option"} or not events:
            continue
        for e in events:
            units=float(e.get("hedge_trade_units",0.0))
            if e.get("hedge_rebalanced") and abs(units)>1e-12:
                by_time[pd.Timestamp(e["timestamp"])].append((units,float(e["spot"])))
        final_units=float(events[-1].get("hedge_units_after",0.0))
        by_time[pd.Timestamp(result["exit_timestamp"])].append((-final_units,float(result["exit_spot"])))
    trades=[]
    for ts, legs in sorted(by_time.items()):
        net_units=sum(units for units,_ in legs)
        if abs(net_units)<=1e-12: continue
        abs_units=sum(abs(units) for units,_ in legs)
        spot=sum(abs(units)*price for units,price in legs)/abs_units if abs_units else legs[0][1]
        trades.append((ts,net_units*LOT_SIZE,spot))
    return trades


def hedge_costs(trades: list[tuple[pd.Timestamp,float,float]], slip: float) -> dict:
    totals=defaultdict(float)
    for ts,qty,spot in trades:
        notional=abs(qty)*spot
        trade_date=ts.date()
        ex_rate=1.73/100_000 if trade_date>=date(2024,10,1) else 1.88/100_000
        stt_rate=.0002 if trade_date>=date(2024,10,1) else .000125
        totals["hedge_exchange_charges"]+=notional*ex_rate
        totals["hedge_sebi_fees"]+=notional*SEBI_FEE_PER_RUPEE
        if qty<0: totals["hedge_stt"]+=notional*stt_rate
        else: totals["hedge_stamp_duty"]+=notional*.00002
        totals["hedge_brokerage"]+=BROKERAGE_PER_ORDER
        totals["hedge_slippage"]+=abs(qty)*slip
    totals["hedge_orders"]=len(trades)
    return totals


def finish_costs(costs: dict) -> dict:
    costs=dict(costs)
    brokerage=costs.get("option_brokerage",0)+costs.get("hedge_brokerage",0)
    exchange=costs.get("option_exchange_charges",0)+costs.get("hedge_exchange_charges",0)
    sebi=costs.get("option_sebi_fees",0)+costs.get("hedge_sebi_fees",0)
    costs["gst"]=GST_RATE*(brokerage+exchange+sebi)
    cost_keys=[k for k in costs if k.endswith(("_charges","_fees","_stt","_stamp_duty","_brokerage","_slippage")) or k=="gst"]
    costs["total_cost_rupees"]=sum(float(costs[k]) for k in cost_keys)
    return costs


def main() -> None:
    args=parse_args()
    options=load_options(args.options)
    if options.empty: raise ValueError("No option rows loaded.")
    rv_map=load_rv_forecasts(args.underlying,sorted(options["quote_date"].unique()))
    options["rv_forecast_prior_close"]=options["quote_date"].map(rv_map)
    ivs=[]; deltas=[]
    for row in options.itertuples(index=False):
        iv,delta=calculate_iv_delta(row,args.risk_free_rate,args.dividend_yield)
        ivs.append(iv); deltas.append(delta)
    options["calculated_iv"]=ivs; options["calculated_delta"]=deltas
    options, common_entry_timestamp, alignment = align_to_common_entry(options)
    print(f"Synchronized common entry timestamp: {common_entry_timestamp}")
    groups=list(options.groupby(["expiry","strike_price","option_type"],sort=True))
    summary_rows=[]; contract_rows=[]
    nconfigs=len(args.relative_gaps)*len(args.delta_triggers)*len(HEDGE_INTERVALS)
    config_no=0
    for gap in args.relative_gaps:
        for trigger in args.delta_triggers:
            for interval in HEDGE_INTERVALS:
                config_no+=1; pairs=[]
                for _,contract in groups:
                    result,events=run_contract_strategy(contract,interval,gap,trigger,args.risk_free_rate,args.dividend_yield)
                    result["relative_gap_cutoff"]=gap; result["delta_trigger"]=trigger
                    pairs.append((result,events)); contract_rows.append(result)
                results=[r for r,_ in pairs]
                active=[r for r in results if r.get("signal") in {"long_option","short_option"}]
                gross_points=sum(float(r["gross_pnl"]) for r in active)
                gross_rupees=gross_points*LOT_SIZE
                hedge_trades=portfolio_hedge_trades(pairs)
                for slip in args.slippage_points:
                    costs=defaultdict(float)
                    for k,v in option_costs(results,slip).items(): costs[k]+=float(v)
                    for k,v in hedge_costs(hedge_trades,slip).items(): costs[k]+=float(v)
                    costs=finish_costs(costs)
                    net=gross_rupees-costs["total_cost_rupees"]
                    summary_rows.append({
                        "relative_gap_cutoff":gap,"delta_trigger":trigger,"hedge_interval":interval,
                        "slippage_points_per_side":slip,"contracts_evaluated":len(results),"trades_triggered":len(active),
                        "gross_pnl_points_one_unit_each":gross_points,"gross_pnl_rupees_one_lot_each":gross_rupees,
                        "option_exchange_charges":costs.get("option_exchange_charges",0),
                        "hedge_exchange_charges":costs.get("hedge_exchange_charges",0),
                        "option_stt":costs.get("option_stt",0),"hedge_stt":costs.get("hedge_stt",0),
                        "option_stamp_duty":costs.get("option_stamp_duty",0),"hedge_stamp_duty":costs.get("hedge_stamp_duty",0),
                        "sebi_fees":costs.get("option_sebi_fees",0)+costs.get("hedge_sebi_fees",0),
                        "brokerage":costs.get("option_brokerage",0)+costs.get("hedge_brokerage",0),
                        "gst":costs.get("gst",0),"option_slippage":costs.get("option_slippage",0),
                        "hedge_slippage":costs.get("hedge_slippage",0),"hedge_orders_after_netting":costs.get("hedge_orders",0),
                        "total_cost_rupees":costs["total_cost_rupees"],"net_pnl_rupees_one_lot_each":net,
                        "net_pnl_points_equivalent":net/LOT_SIZE,
                        "mean_net_rupees_per_strike":net/len(active) if active else np.nan,
                    })
                if config_no%25==0: print(f"Completed {config_no}/{nconfigs} configurations")
    args.results_dir.mkdir(parents=True,exist_ok=True)
    summary_path=args.results_dir/"transaction_cost_sweep_summary.csv"
    contracts_path=args.results_dir/"transaction_cost_sweep_by_contract_gross.csv"
    alignment_path=args.results_dir/"common_entry_alignment.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path,index=False)
    pd.DataFrame(contract_rows).to_csv(contracts_path,index=False)
    alignment.to_csv(alignment_path,index=False)
    summary=pd.DataFrame(summary_rows)
    baseline=summary[summary.slippage_points_per_side==min(args.slippage_points)]
    best=baseline.loc[baseline.net_pnl_rupees_one_lot_each.idxmax()]
    print("\nLowest-slippage net P&L top 10 (INR; one option lot per strike)")
    cols=["relative_gap_cutoff","delta_trigger","hedge_interval","gross_pnl_rupees_one_lot_each","total_cost_rupees","net_pnl_rupees_one_lot_each","net_pnl_points_equivalent","hedge_orders_after_netting"]
    print(baseline[cols].sort_values("net_pnl_rupees_one_lot_each",ascending=False).head(10).to_string(index=False,float_format=lambda x:f"{x:,.2f}"))
    print("\nBest setting:",best[cols].to_dict())
    print(f"\nSaved {summary_path}\nSaved {contracts_path}")

if __name__=="__main__": main()
