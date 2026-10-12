import numpy as np


def finite_difference_call_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    space_steps: int = 400,
    smax_multiplier: float = 3.0,
    q: float = 0.0,
):
    """
    European call price using an explicit finite-difference
    solution of the Black-Scholes-Merton PDE.
    """
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if space_steps < 10:
        raise ValueError("space_steps must be at least 10.")

    S_max = smax_multiplier * K
    dS = S_max / space_steps
    S = np.linspace(0.0, S_max, space_steps + 1)

    # Conservative explicit-scheme stability condition.
    stability_factor = sigma**2 * S_max**2 / dS**2 + abs(r - q)
    dt_max = 0.9 / stability_factor
    time_steps = max(1, int(np.ceil(T / dt_max)))
    dt = T / time_steps

    # Expiry payoff.
    V = np.maximum(S - K, 0.0)

    # March backward from T to 0.
    for n in range(time_steps):
        tau = (n + 1) * dt
        V_old = V.copy()

        # European call boundary conditions.
        V[0] = 0.0
        V[-1] = (
            S_max * np.exp(-q * tau)
            - K * np.exp(-r * tau)
        )

        i = np.arange(1, space_steps)
        S_i = S[i]

        a = (
            0.5 * sigma**2 * S_i**2 * dt / dS**2
            - 0.5 * (r - q) * S_i * dt / dS
        )
        b = (
            1.0
            - sigma**2 * S_i**2 * dt / dS**2
            - r * dt
        )
        c = (
            0.5 * sigma**2 * S_i**2 * dt / dS**2
            + 0.5 * (r - q) * S_i * dt / dS
        )

        V[i] = (
            a * V_old[i - 1]
            + b * V_old[i]
            + c * V_old[i + 1]
        )

    price = float(np.interp(S0, S, V))
    return price, S, V


if __name__ == "__main__":
    price, _, _ = finite_difference_call_price(
        100, 105, 30 / 365, 0.05, 0.20
    )
    print(f"Price: {price:.6f}")
