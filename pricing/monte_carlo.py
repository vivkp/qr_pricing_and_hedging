import numpy as np


def monte_carlo_call_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_simulations: int = 100_000,
    time_steps: int = 22,
    q: float = 0.0,
    seed: int = 42,
):
    """European call price using Monte Carlo under GBM."""
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if n_simulations <= 0 or time_steps <= 0:
        raise ValueError("Simulation settings must be positive.")

    rng = np.random.default_rng(seed)
    dt = T / time_steps
    S = np.full(n_simulations, S0, dtype=float)

    # Risk-neutral GBM.
    for _ in range(time_steps):
        Z = rng.standard_normal(n_simulations)
        S *= np.exp(
            (r - q - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * Z
        )

    payoffs = np.maximum(S - K, 0.0)
    discounted_payoffs = np.exp(-r * T) * payoffs

    price = float(np.mean(discounted_payoffs))
    standard_error = float(
        np.std(discounted_payoffs, ddof=1) / np.sqrt(n_simulations)
    )

    lower = price - 1.96 * standard_error
    upper = price + 1.96 * standard_error

    return price, standard_error, lower, upper


def simulate_gbm_path(
    S0: float,
    T: float,
    mu: float,
    sigma: float,
    time_steps: int,
    seed: int | None = None,
):
    """Generate one GBM price path."""
    if S0 <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("Invalid GBM parameters.")
    if time_steps <= 0:
        raise ValueError("time_steps must be positive.")

    rng = np.random.default_rng(seed)
    dt = T / time_steps
    times = np.linspace(0.0, T, time_steps + 1)

    prices = np.empty(time_steps + 1)
    prices[0] = S0

    for t in range(time_steps):
        Z = rng.standard_normal()
        prices[t + 1] = prices[t] * np.exp(
            (mu - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * Z
        )

    return times, prices


if __name__ == "__main__":
    result = monte_carlo_call_price(
        100, 105, 30 / 365, 0.05, 0.20
    )
    print(f"Price: {result[0]:.6f}")
