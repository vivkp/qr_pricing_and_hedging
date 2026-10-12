import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data.inputs import S0, K, T, r, sigma, FD_SMAX_MULTIPLIER
from pricing.black_scholes import call_price
from pricing.finite_difference import finite_difference_call_price


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_fd_convergence():
    grid_sizes = np.array([25, 50, 100, 200, 400, 800], dtype=int)
    benchmark = call_price(S0, K, T, r, sigma)

    rows = []
    for grid in grid_sizes:
        price, _, _ = finite_difference_call_price(
            S0, K, T, r, sigma,
            space_steps=grid,
            smax_multiplier=FD_SMAX_MULTIPLIER,
        )
        rows.append({
            "grid_steps": int(grid),
            "fd_price": price,
            "absolute_error": abs(price - benchmark),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / "fd_convergence.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["grid_steps"] for row in rows])
    y = np.array([row["fd_price"] for row in rows])

    plt.figure(figsize=(9, 6))
    plt.axhline(
        benchmark,
        linestyle="--",
        linewidth=2,
        label=f"Black-Scholes = {benchmark:.4f}",
    )
    plt.plot(x, y, marker="o", label="Finite Difference")
    plt.xlabel("Number of spatial grid steps")
    plt.ylabel("Call option price")
    plt.title("Finite Difference Convergence")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    png_path = RESULTS_DIR / "fd_convergence.png"
    plt.savefig(png_path, dpi=160)
    plt.close()

    return png_path, csv_path


if __name__ == "__main__":
    png, csv = run_fd_convergence()
    print(f"Graph: {png}")
    print(f"Data : {csv}")
