"""Implied-volatility inversion for European options."""

import math

from pricing.black_scholes import call_price, put_price


def implied_volatility(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    market_option_price: float,
    option_type: str = "call",
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> float:
    """Solve for volatility matching a European option's market price.

    Uses bisection, with a growing upper volatility bracket. Returns NaN
    when the market price is outside European no-arbitrage bounds or a
    finite volatility solution cannot be bracketed.
    """
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if not all(math.isfinite(x) for x in (S0, K, T, r, q, market_option_price)):
        return float("nan")
    if S0 <= 0 or K <= 0 or T <= 0:
        raise ValueError("S0, K, and T must be positive.")
    if tolerance <= 0 or max_iterations <= 0:
        raise ValueError("tolerance and max_iterations must be positive.")
    if market_option_price < 0:
        return float("nan")

    discounted_spot = S0 * math.exp(-q * T)
    discounted_strike = K * math.exp(-r * T)
    if kind == "call":
        lower_bound = max(discounted_spot - discounted_strike, 0.0)
        upper_bound = discounted_spot
        price_fn = call_price
    else:
        lower_bound = max(discounted_strike - discounted_spot, 0.0)
        upper_bound = discounted_strike
        price_fn = put_price

    price_tolerance = max(tolerance, tolerance * max(1.0, market_option_price))
    if market_option_price < lower_bound - price_tolerance:
        return float("nan")
    if market_option_price > upper_bound + price_tolerance:
        return float("nan")
    if market_option_price <= lower_bound + price_tolerance:
        return 0.0

    def price_at(volatility: float) -> float:
        return price_fn(S0, K, T, r, volatility, q)

    low, high = 1e-12, 1.0
    while price_at(high) < market_option_price and high < 16.0:
        high *= 2.0
    if price_at(high) < market_option_price:
        return float("nan")

    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        model_price = price_at(mid)
        if abs(model_price - market_option_price) <= price_tolerance:
            return mid
        if model_price < market_option_price:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0
