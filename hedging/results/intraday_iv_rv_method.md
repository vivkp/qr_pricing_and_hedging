# Intraday IV versus realized volatility hedge study

Options input: `nifty_calls_600pt_expiry_2024-10-03.csv`. Underlying history: `nifty_daily_close_through_2024-09-24.csv`.

- Expiry in this run: 2024-10-03
- Option quote dates: 2024-09-25 through 2024-10-03 (6 trading dates)
- RV forecast: reproduce the notebook's EWMA method (`span=30`, demeaned daily log returns, annualized by `sqrt(252)`). For each quote date, use the prior trading day's RV.
- Synchronized entry: every contract enters at the first timestamp with a valid IV/delta for all strikes: 2024-09-25 09:18:00.
- Entry signal: `(RV forecast - entry IV) / entry IV`. Buy if at least +4.0%; sell if at most -4.0%; otherwise no trade.
- IV: invert Black-Scholes using option LTP, `r=6.50%`, `q=0.00%`, and source `tte_years`.
- Delta: Black-Scholes delta from that row's calculated IV. Start with the entry hedge and check at the requested frequency. Rehedge only when the absolute delta change from the last hedge is greater than 0.05.
- Frequencies: 5-minute, 30-minute, and 60-minute checks anchored at 09:15 India time; daily checks at each day's final available quote.
- Exit: mark the option and hedge at the last available quote for that contract. It may be before expiry if the contract has no later quote.
- P&L: option mark-to-market plus hedge price changes, per one option unit. Gross only. No bid/ask, fees, funding, lot multiplier, futures basis, or expiry settlement is included.
- LTP is a last-traded price and can be stale. Rows whose LTP cannot produce a positive Black-Scholes IV are excluded from IV/delta checks and counted in the outputs.

The summary's total is the simple sum across active one-unit contracts. It is a diagnostic, not a capital-weighted portfolio return.
