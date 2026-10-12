"""Reproduce every pricing, market-validation, and hedge result."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    [sys.executable, "main.py"],
    [sys.executable, "-m", "convergence.run_convergence"],
    [sys.executable, "-m", "market_validation.nse_model_vs_market"],
    [sys.executable, "hedging/intraday_delta_hedging.py"],
    [sys.executable, "hedging/parameter_sweep.py"],
    [sys.executable, "hedging/transaction_cost_sweep.py"],
    [sys.executable, "hedging/summarize_results.py"],
]

for step in STEPS:
    print("\n$", " ".join(step), flush=True)
    subprocess.run(step, cwd=ROOT, check=True)
