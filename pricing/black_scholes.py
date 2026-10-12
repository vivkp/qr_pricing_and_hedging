import math


def normal_cdf(x: float) -> float:
    """Standard normal CDF N(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def d1_d2(S0: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """Calculate d1 and d2 for Black-Scholes-Merton with continuous yield q."""
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    d1 = (
        math.log(S0 / K)
        + (r - q + 0.5 * sigma**2) * T
    ) / (sigma * math.sqrt(T))

    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def call_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    d1, d2 = d1_d2(S0, K, T, r, sigma, q)
    return (
        S0 * math.exp(-q * T) * normal_cdf(d1)
        - K * math.exp(-r * T) * normal_cdf(d2)
    )


def put_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    d1, d2 = d1_d2(S0, K, T, r, sigma, q)
    return (
        K * math.exp(-r * T) * normal_cdf(-d2)
        - S0 * math.exp(-q * T) * normal_cdf(-d1)
    )


if __name__ == "__main__":
    print(
        call_price(
            S0=100,
            K=105,
            T=30 / 365,
            r=0.05,
            sigma=0.20,
        )
    )
