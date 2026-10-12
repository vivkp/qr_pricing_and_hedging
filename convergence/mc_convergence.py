import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data.inputs import S0, K, T, r, sigma, MC_TIME_STEPS
from pricing.black_scholes import call_price
from pricing.monte_carlo import monte_carlo_call_price


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_mc_convergence():
    simulation_counts = np.array(
        [1_000, 5_000, 10_000, 50_000, 100_000, 500_000],
        dtype=int,
    )

    benchmark = call_price(S0, K, T, r, sigma)

    rows = []
    for n in simulation_counts:
        price, se, lower, upper = monte_carlo_call_price(
            S0, K, T, r, sigma,
            n_simulations=n,
            time_steps=MC_TIME_STEPS,
            seed=42,
        )
        rows.append({
            "simulations": int(n),
            "mc_price": price,
            "standard_error": se,
            "lower_95": lower,
            "upper_95": upper,
            "absolute_error": abs(price - benchmark),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / "mc_convergence.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["simulations"] for row in rows])
    y = np.array([row["mc_price"] for row in rows])
    lower = np.array([row["lower_95"] for row in rows])
    upper = np.array([row["upper_95"] for row in rows])

    plt.figure(figsize=(9, 6))
    plt.axhline(
        benchmark,
        linestyle="--",
        linewidth=2,
        label=f"Black-Scholes = {benchmark:.4f}",
    )
    plt.plot(x, y, marker="o", label="Monte Carlo")
    plt.fill_between(
        x, lower, upper,
        alpha=0.2,
        label="95% confidence interval",
    )
    plt.xscale("log")
    plt.xlabel("Number of Monte Carlo simulations")
    plt.ylabel("Call option price")
    plt.title("Monte Carlo Convergence")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    png_path = RESULTS_DIR / "mc_convergence.png"
    plt.savefig(png_path, dpi=160)
    plt.close()

    return png_path, csv_path


if __name__ == "__main__":
    png, csv = run_mc_convergence()
    print(f"Graph: {png}")
    print(f"Data : {csv}")
