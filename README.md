# NIFTY Derivatives Pricing and Delta Hedging

This combines the pricing/numerical-method work from `quant-derivatives-engine 3` with the synchronized NIFTY intraday option study from `project1-quant`.

## Research question

Can we (1) price European options with Black–Scholes and numerical methods, (2) compare model prices and implied volatility with NIFTY option quotes, and (3) observe how a dynamic delta hedge changes option P&L over a short historical sample?

The project demonstrates pricing and hedging mechanics. The IV-versus-realized-volatility entry rule is an exploratory signal test; it is not evidence of a durable trading edge and belongs in a later alpha-research project.

## Project map

```text
qrproject1/
├── pricing/                       # Black–Scholes, Monte Carlo, finite differences, IV solver
├── convergence/                   # MC and finite-difference error/convergence
├── greeks/                        # Analytical Greeks
├── market_validation/             # NIFTY quote comparison, IV and smile
├── hedging/                       # synchronized entry, hedge schedules, cost sweeps
├── data/
│   ├── inputs.py                  # controlled toy example inputs
│   ├── nifty50/                   # NIFTY chain and underlying snapshot for 2026-10-05
│   ├── nifty_calls_600pt_expiry_2024-10-03.csv
│   └── nifty_daily_close_through_2024-09-24.csv
├── run_all.py
└── results are kept beside each phase's scripts
```

The large merged source file is not included. The included filtered option file is enough to rerun the hedge study. `data/extract_nifty_calls.awk` documents how the filtered CE sample can be streamed from the original merged file if it is available locally.

## Reproduce

From this directory, create/activate a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python run_all.py
```

The full run regenerates the pricing output, convergence CSVs/plots, NIFTY market CSVs/plots, synchronized hedge outputs, 2/4/6/8/10% RV–IV and 0.01–0.10 delta-threshold sweep, cost sweep, best-parameter table, and net-P&L chart. To run just one phase, use the commands below from this directory:

```bash
python main.py
python -m convergence.run_convergence
python -m market_validation.nse_model_vs_market
python hedging/intraday_delta_hedging.py
python hedging/parameter_sweep.py
python hedging/transaction_cost_sweep.py
python hedging/summarize_results.py
```

### Inputs and settings

- Toy pricing: `S0=100`, `K=100`, `T=1 year`, `r=5%`, `sigma=20%`. These pedagogic defaults were restored because the earlier engine's imported `data/inputs.py` was missing.
- NIFTY snapshot validation: valuation date 2026-10-05, expiry 2026-11-03, spot from the included underlying file, continuously compounded `r=5.5199%`, `q=0`; historical volatility uses close-to-close daily log returns and sample standard deviation annualized by `sqrt(252)`. Market prices are bid/ask midpoints; quotes without usable two-sided size or with relative spread above 25% are excluded from IV analysis. The snapshot is dated 2026-10-05 even though this project is being rerun later.
- Intraday hedge: NIFTY CE options, expiry 2024-10-03, 24 strikes within the supplied ±600-point sample, observations 2024-09-25 to 2024-10-03. The common valid entry is 2024-09-25 09:18. RV is the prior-day EWMA estimate (span 30) from demeaned daily close-to-close log returns; option IV is inverted from LTP using `r=6.5%`, `q=0`. One call lot is modeled as 25 units per selected strike.
- Hedge sweeps: RV-vs-entry-IV relative cutoffs 2/4/6/8/10%; delta triggers 0.01–0.10; check intervals 1m/5m/30m/1h/1day. Cost sensitivity uses fixed slippage 0.05/0.25/0.50 index points per execution side plus modeled brokerage and 2024-era exchange, STT, stamp duty, SEBI and GST charges. See `hedging/transaction_cost_sweep.py` for rates and mechanics.

## Results from this rerun

### Pricing and numerical validation

For the toy ATM call, Black–Scholes gives **10.450584**. Monte Carlo gives **10.466688** (standard error 0.020799; 95% interval 10.425923–10.507453). Finite differences on the 200-step grid give **10.455021**. The finite-difference grid error falls from 0.2692 at 25 steps to 0.000276 at 800 steps. Monte Carlo error generally tightens as paths increase, with sampling noise so individual points need not improve monotonically.

### NIFTY market snapshot

The 2026-10-05 snapshot has spot **22,555.75**, 208 chain rows, 45 usable calls and 43 usable puts. The historical-volatility estimate is **13.50%**. Against Black–Scholes priced at that historical volatility, the mean market-minus-model error is +8.74 points for calls (MAE 13.43) and +16.72 for puts (MAE 22.12). These are quote-snapshot errors under a simplified constant-rate/zero-dividend setup, not a backtest or evidence of mispricing. The generated IV smile uses filtered bid/ask midpoints and marks spot.

### Delta-hedging experiment and best in-sample setting

The current sweep has 250 unique signal/hedge configurations and 3 slippage scenarios (750 rows total). At the lowest modeled slippage (0.05 points per side), the maximum net result is:

| Parameter | Best in-sample setting |
|---|---:|
| RV–entry-IV relative gap cutoff | 2% |
| Delta rebalance threshold | 0.01 |
| Hedge check interval | 1 hour |
| Contracts | 24 CE strikes, one 25-unit lot each |
| Entry/exit window | 2024-09-25 09:18 to each strike's last quote, no later than 2024-10-03 15:29 |
| Gross P&L | ₹26,917.26 |
| Modeled costs | ₹6,026.17 |
| Net P&L | **₹20,891.09** (₹835.64 per option-point equivalent) |
| Net hedge orders after cross-strike netting | 43 |

For the same setting, net P&L is ₹20,414.91 at 0.25 points/side and ₹19,819.68 at 0.50 points/side. 180 of 250 configurations are positive at each of those three cost assumptions. Daily-check configurations are negative across the sweep; the intraday schedules are positive in the modeled results. The gap cutoff does not alter this best row's trade set: 2%, 4%, 6%, and 8% give the same result because they all select the same 24 contracts in this sample. A 10% cutoff selects fewer contracts.

At entry, those 24 calls had mean calculated IV **12.12%** (range 10.87–16.16%) versus the same prior-close EWMA RV forecast **9.95%**. Every selected leg was therefore a short call. Across 24 contracts, option-price decay contributed about **₹168,772** for one 25-unit lot per strike; the delta hedge lost about **₹141,855**; gross portfolio P&L was ₹26,917. The modeled ₹6,026 costs reduce it to ₹20,891. The underlying proxy moved from 25,928.25 at the common entry to 25,266 at the final marks. So the observed net gain came from this specific short-call/hedge/path combination, not just from “hedging making money.”

The modeled best-row cost estimate includes about ₹1,820 brokerage (₹20 per order), ₹119 slippage total at 0.05 index points per side, plus modeled exchange charges, STT, stamp duty, SEBI fees, and GST. The largest components are hedge STT (~₹2,554) and brokerage. The rates are the script’s historical assumptions, not a broker contract note.

This “best” row is selected by maximizing P&L on this very same short sample. It is an in-sample optimum, not a validated parameter choice. P&L is an absolute modeled amount, not a percentage return: no agreed capital or margin denominator is supplied, and short calls can have large/unbounded losses. Do not describe this as a proven profitable strategy.

A separate baseline decomposition (4% RV–IV cutoff, 0.05 delta trigger) compares the option-only position with the delta-hedged result. The option-only cross-strike P&L standard deviation is 215.51 index points; after hedging it is 29.31 (1-minute), 32.12 (5-minute), 37.11 (30-minute), and 39.81 (1-hour). Daily checks leave 60.68 points and produce negative aggregate gross P&L. This suggests the intraday hedge reduced dispersion across the 24 strike outcomes in this sample. It is **not** a time-series VaR or proof of future risk reduction: the 24 strikes are related outcomes from one short market path. See `hedging/results/hedging_effectiveness_baseline.csv`.

## Interpretation and limits

- The hedge file has option **LTP**, not bid/ask. LTP records the last trade; it does not reveal whether a hypothetical order would have filled at that value or whether it was buyer/seller initiated. The fixed slippage scenarios do not reconstruct the missing spread. Therefore the reported net result remains an approximation, despite modeled fees.
- NIFTY spot proxies for futures. We do not have futures bid/ask, futures basis, expiry settlement mechanics, financing, margin, impact, or integer lot rounding of the hedge.
- The option side is determined by the relative RV–IV rule. All selected positions in the best row are short options, so its outcome combines the volatility signal with the market path and hedge implementation. This is the alpha-selection question, not a clean isolated test of hedging quality.
- The sample is one expiry and about seven calendar days, with 24 strikes from the same underlying. Strikes are not independent observations; parameter optimization across them can overfit.
- The 2026 snapshot experiment and 2024 intraday experiment are separate datasets and should not be stitched into a single chronological performance series.

### Verdict

**Successful as a reproducible implementation and exploratory research exercise; not successful as evidence that the trading rule is profitable.** The code establishes the pricing and numerical methods, extracts market IV/smile information from usable quotes, and shows the mechanics and cost sensitivity of dynamic delta hedging. But the simulated edge has not been validated out of sample, actual execution cannot be reconstructed without option and futures bid/ask quotes, and the apparent optimum is selected from the same data used to evaluate it. The next rigorous step for Project 1 is a hedge-risk study with a pre-specified option position and synchronized quote/execution data; move IV/RV trade selection and alpha claims into Project 2.

## Result files

- `convergence/results/`: MC/FD errors and plots.
- `market_validation/results/`: per-strike model prices, IV, Greeks, pricing errors, metrics, and spot-marked IV smile / price charts.
- `hedging/results/`: entry alignment, IV and hedge events, per-contract P&L, full parameter sweep, full cost sweep, best rows, and cadence cost-sensitivity plot.
