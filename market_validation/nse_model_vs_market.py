"""Build an NSE NIFTY option-chain IV comparison from downloaded CSVs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pricing.black_scholes import call_price, put_price
from pricing.implied_volatility import implied_volatility
from greeks.analytical import all_greeks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTION_CHAIN_FILE = PROJECT_ROOT / "data" / "nifty50" / "niftyoc.csv"
UNDERLYING_HISTORY_FILE = PROJECT_ROOT / "data" / "nifty50" / "niftystock.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The downloaded option-chain CSV does not contain its valuation date or
# selected expiry. Set these to the values supplied with the downloaded file.
VALUATION_DATE = pd.Timestamp("2026-10-05")
EXPIRY_DATE = pd.Timestamp("2026-11-03")

# Use the 91-day T-bill cut-off yield from the latest auction before the
# valuation date (30 Sep 2026) as a short-rate proxy. NSE's displayed IV uses
# its own fixed 10% convention, retained in the CSV for reference only.
RISK_FREE_RATE = 0.055199
DIVIDEND_YIELD = 0.0
TRADING_DAYS_PER_YEAR = 252
MAX_RELATIVE_SPREAD = 0.25


def historical_volatility(history: pd.DataFrame) -> float:
    """Annualized close-to-close log-return volatility from available data."""
    prices = history["Close"].astype(float).dropna().sort_index()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) < 2:
        raise ValueError("Not enough underlying history to estimate volatility.")
    return float(log_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def load_underlying_history() -> pd.DataFrame:
    history = pd.read_csv(UNDERLYING_HISTORY_FILE)
    history.columns = history.columns.str.strip()
    history["Date"] = pd.to_datetime(
        history["Date"].astype(str).str.strip(), format="%d-%b-%Y"
    )
    history["Close"] = pd.to_numeric(history["Close"], errors="coerce")
    return history.dropna(subset=["Date", "Close"]).set_index("Date").sort_index()


def load_option_chain() -> pd.DataFrame:
    # The downloaded NSE file currently has a single header row. Repeated
    # option-side labels are disambiguated by pandas (BID, BID.1, etc.).
    raw = pd.read_csv(OPTION_CHAIN_FILE, header=0, thousands=",")
    raw.columns = raw.columns.str.strip()
    common = pd.DataFrame({
        "strike": pd.to_numeric(raw["STRIKE"], errors="coerce"),
    })

    call = common.assign(
        option_type="call",
        bid=pd.to_numeric(raw["BID"], errors="coerce"),
        ask=pd.to_numeric(raw["ASK"], errors="coerce"),
        bid_size=pd.to_numeric(raw["BID QTY"], errors="coerce"),
        ask_size=pd.to_numeric(raw["ASK QTY"], errors="coerce"),
        last_price=pd.to_numeric(raw["LTP"], errors="coerce"),
        nse_reference_iv_percent=pd.to_numeric(raw["IV"], errors="coerce"),
        volume=pd.to_numeric(raw["VOLUME"], errors="coerce"),
        open_interest=pd.to_numeric(raw["OI"], errors="coerce"),
    )
    put = common.assign(
        option_type="put",
        bid=pd.to_numeric(raw["BID.1"], errors="coerce"),
        ask=pd.to_numeric(raw["ASK.1"], errors="coerce"),
        bid_size=pd.to_numeric(raw["BID QTY.1"], errors="coerce"),
        ask_size=pd.to_numeric(raw["ASK QTY.1"], errors="coerce"),
        last_price=pd.to_numeric(raw["LTP.1"], errors="coerce"),
        nse_reference_iv_percent=pd.to_numeric(raw["IV.1"], errors="coerce"),
        volume=pd.to_numeric(raw["VOLUME.1"], errors="coerce"),
        open_interest=pd.to_numeric(raw["OI.1"], errors="coerce"),
    )
    return pd.concat([call, put], ignore_index=True).dropna(subset=["strike"])


def enrich_option_chain(chain: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    if VALUATION_DATE not in history.index:
        raise ValueError(f"No NIFTY close found for {VALUATION_DATE.date()}.")
    if EXPIRY_DATE <= VALUATION_DATE:
        raise ValueError("Expiry date must be after the valuation date.")

    spot = float(history.loc[VALUATION_DATE, "Close"])
    hist_vol = historical_volatility(history.loc[:VALUATION_DATE])
    time_to_expiry = (EXPIRY_DATE - VALUATION_DATE).days / 365.0

    data = chain.copy()
    data["valuation_date"] = VALUATION_DATE.date().isoformat()
    data["expiry"] = EXPIRY_DATE.date().isoformat()
    data["spot"] = spot
    data["time_to_expiry"] = time_to_expiry
    data["risk_free_rate"] = RISK_FREE_RATE
    data["dividend_yield"] = DIVIDEND_YIELD
    data["historical_volatility"] = hist_vol
    data["market_price"] = np.where(
        data["bid"].gt(0) & data["ask"].gt(0) & data["ask"].ge(data["bid"]),
        (data["bid"] + data["ask"]) / 2.0,
        np.nan,
    )
    data["relative_spread"] = (
        (data["ask"] - data["bid"]) / data["market_price"]
    )
    data["quote_usable"] = (
        data["market_price"].notna()
        & data["bid_size"].gt(0)
        & data["ask_size"].gt(0)
        & data["relative_spread"].le(MAX_RELATIVE_SPREAD)
    )

    def model_price(row):
        price_fn = call_price if row["option_type"] == "call" else put_price
        return price_fn(
            row["spot"], row["strike"], row["time_to_expiry"],
            row["risk_free_rate"], row["historical_volatility"],
            row["dividend_yield"],
        )

    data["bs_price_hist_vol"] = data.apply(model_price, axis=1)
    data["pricing_error"] = data["market_price"] - data["bs_price_hist_vol"]

    def solve_iv(row):
        if not row["quote_usable"]:
            return np.nan
        return implied_volatility(
            row["spot"], row["strike"], row["time_to_expiry"],
            row["risk_free_rate"], row["dividend_yield"],
            row["market_price"], row["option_type"],
        )

    data["implied_volatility"] = data.apply(solve_iv, axis=1)

    def greek_values(row, volatility):
        if not np.isfinite(volatility) or volatility <= 0:
            return pd.Series({
                "delta": np.nan, "gamma": np.nan, "vega_per_vol_point": np.nan,
                "theta_per_calendar_day": np.nan, "rho_per_rate_point": np.nan,
            })
        values = all_greeks(
            row["spot"], row["strike"], row["time_to_expiry"],
            row["risk_free_rate"], volatility, row["dividend_yield"],
            row["option_type"],
        )
        return pd.Series({
            "delta": values["delta"],
            "gamma": values["gamma"],
            "vega_per_vol_point": values["vega"] / 100.0,
            "theta_per_calendar_day": values["theta"] / 365.0,
            "rho_per_rate_point": values["rho"] / 100.0,
        })

    hist_greeks = data.apply(
        lambda row: greek_values(row, row["historical_volatility"]), axis=1
    ).add_prefix("bs_hist_vol_")
    iv_greeks = data.apply(
        lambda row: greek_values(row, row["implied_volatility"]), axis=1
    ).add_prefix("market_iv_")
    data = pd.concat([data, hist_greeks, iv_greeks], axis=1)
    return data


def run():
    history = load_underlying_history()
    chain = load_option_chain()
    data = enrich_option_chain(chain, history)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = RESULTS_DIR / "nifty_model_vs_market.csv"
    data.to_csv(output_csv, index=False)

    metric_rows = []
    usable = data[data["quote_usable"] & data["market_price"].notna()]
    for option_type, group in usable.groupby("option_type"):
        errors = group["pricing_error"].dropna()
        metric_rows.append({
            "option_type": option_type,
            "quote_count": len(errors),
            "mean_error_market_minus_model": errors.mean(),
            "mae": errors.abs().mean(),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mean_absolute_relative_error": (
                errors.abs() / group.loc[errors.index, "market_price"]
            ).mean(),
        })
    metrics_path = RESULTS_DIR / "nifty_model_validation_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)

    # Compare observed option prices with Black–Scholes prices at the same
    # strikes, using only quotes that passed the quality filter.
    price_chart_paths = {}
    for option_type in ("call", "put"):
        quoted = data[
            (data["option_type"] == option_type) & data["quote_usable"]
        ].sort_values("strike")
        if quoted.empty:
            continue
        plt.figure(figsize=(9, 6))
        plt.plot(
            quoted["strike"], quoted["market_price"],
            marker="o", label="Market bid–ask midpoint",
        )
        plt.plot(
            quoted["strike"], quoted["bs_price_hist_vol"],
            marker="o", label="Black–Scholes using historical volatility",
        )
        spot = float(quoted["spot"].iloc[0])
        plt.axvline(
            spot, color="black", linestyle="--", alpha=0.7,
            label=f"Spot = {spot:,.2f}",
        )
        plt.xlabel("Strike")
        plt.ylabel("Option price (index points)")
        plt.title(
            f"NIFTY {option_type.capitalize()}: Market vs Black–Scholes\n"
            f"BS inputs: r={RISK_FREE_RATE:.2%}, q={DIVIDEND_YIELD:.1%}, "
            f"historical volatility={float(quoted['historical_volatility'].iloc[0]):.2%}"
        )
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        path = RESULTS_DIR / f"nifty_{option_type}_market_vs_bs.png"
        plt.savefig(path, dpi=160)
        plt.close()
        price_chart_paths[option_type] = path

    # Plot near-spot calls with usable two-sided markets only.
    calls = data[
        (data["option_type"] == "call")
        & data["quote_usable"]
        & data["implied_volatility"].notna()
        & data["implied_volatility"].gt(0)
        & data["strike"].between(0.85 * data["spot"], 1.15 * data["spot"])
    ].sort_values("strike")

    if not calls.empty:
        plt.figure(figsize=(9, 6))
        plt.plot(
            calls["strike"], calls["implied_volatility"] * 100,
            marker="o", label="IV from bid–ask midpoint",
        )
        spot = float(calls["spot"].iloc[0])
        plt.axvline(
            spot, color="black", linestyle="--", alpha=0.7,
            label=f"Spot = {spot:,.2f}",
        )
        hist_vol = float(calls["historical_volatility"].iloc[0])
        plt.axhline(
            hist_vol * 100, linestyle=":",
            label=f"Historical volatility = {hist_vol:.2%}",
        )
        plt.xlabel("Strike")
        plt.ylabel("Volatility (%)")
        plt.title("NIFTY Call Implied Volatility by Strike")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        chart_path = RESULTS_DIR / "nifty_call_iv_smile.png"
        plt.savefig(chart_path, dpi=160)
        plt.close()
    else:
        chart_path = None

    # Compare call and put IVs only at strikes where both sides have usable
    # quotes. This avoids comparing different strikes or mismatched samples.
    quoted_options = data[
        data["quote_usable"]
        & data["implied_volatility"].notna()
        & data["implied_volatility"].gt(0)
    ]
    iv_comparison = quoted_options.pivot(
        index="strike", columns="option_type", values="implied_volatility"
    )
    if {"call", "put"}.issubset(iv_comparison.columns):
        iv_comparison = iv_comparison.dropna(subset=["call", "put"]).reset_index()
        iv_comparison["historical_volatility"] = historical_volatility(
            history.loc[:VALUATION_DATE]
        )
        iv_comparison["call_minus_put_iv"] = (
            iv_comparison["call"] - iv_comparison["put"]
        )
        iv_comparison["call_minus_historical_volatility"] = (
            iv_comparison["call"] - iv_comparison["historical_volatility"]
        )
        iv_comparison["put_minus_historical_volatility"] = (
            iv_comparison["put"] - iv_comparison["historical_volatility"]
        )
        comparison_csv = RESULTS_DIR / "nifty_call_put_iv_comparison.csv"
        iv_comparison.to_csv(comparison_csv, index=False)

        plt.figure(figsize=(9, 6))
        plt.plot(
            iv_comparison["strike"], iv_comparison["call"] * 100,
            marker="o", label="Call implied volatility",
        )
        plt.plot(
            iv_comparison["strike"], iv_comparison["put"] * 100,
            marker="s", label="Put implied volatility",
        )
        spot = float(data["spot"].iloc[0])
        plt.axvline(
            spot, color="black", linestyle="--", alpha=0.7,
            label=f"Spot = {spot:,.2f}",
        )
        plt.axhline(
            float(iv_comparison["historical_volatility"].iloc[0]) * 100,
            linestyle=":", label="Historical volatility",
        )
        plt.xlabel("Strike")
        plt.ylabel("Volatility (%)")
        plt.title("NIFTY Call and Put IV at Matched Strikes")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        comparison_chart = RESULTS_DIR / "nifty_call_put_iv_comparison.png"
        plt.savefig(comparison_chart, dpi=160)
        plt.close()
    else:
        iv_comparison = pd.DataFrame()
        comparison_csv = None
        comparison_chart = None

    print(f"Valuation date      : {VALUATION_DATE.date()}")
    print(f"Expiry date         : {EXPIRY_DATE.date()}")
    print(f"NIFTY close         : {float(history.loc[VALUATION_DATE, 'Close']):.2f}")
    print(f"Historical vol      : {historical_volatility(history.loc[:VALUATION_DATE]):.2%}")
    print(f"Chain rows          : {len(data)}")
    print(f"Usable call quotes  : {int(data.loc[data.option_type == 'call', 'quote_usable'].sum())}")
    print(f"Usable put quotes   : {int(data.loc[data.option_type == 'put', 'quote_usable'].sum())}")
    print(f"Matched call/put IVs: {len(iv_comparison)}")
    print(f"CSV                 : {output_csv}")
    print(f"Price-error metrics : {metrics_path}")
    for option_type, path in price_chart_paths.items():
        print(f"{option_type.capitalize()} price chart : {path}")
    print(f"IV chart            : {chart_path if chart_path else 'not generated (no usable call quotes)'}")
    print(f"Call/put IV CSV     : {comparison_csv if comparison_csv else 'not generated (no matched quotes)'}")
    print(f"Call/put IV chart   : {comparison_chart if comparison_chart else 'not generated (no matched quotes)'}")


if __name__ == "__main__":
    run()
