import math

from pricing.black_scholes import d1_d2, normal_cdf


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def call_delta(S0, K, T, r, sigma, q=0.0):
    d1, _ = d1_d2(S0, K, T, r, sigma, q)
    return math.exp(-q * T) * normal_cdf(d1)


def put_delta(S0, K, T, r, sigma, q=0.0):
    d1, _ = d1_d2(S0, K, T, r, sigma, q)
    return math.exp(-q * T) * (normal_cdf(d1) - 1.0)


def gamma(S0, K, T, r, sigma, q=0.0):
    d1, _ = d1_d2(S0, K, T, r, sigma, q)
    return (
        math.exp(-q * T)
        * normal_pdf(d1)
        / (S0 * sigma * math.sqrt(T))
    )


def vega(S0, K, T, r, sigma, q=0.0):
    d1, _ = d1_d2(S0, K, T, r, sigma, q)
    return S0 * math.exp(-q * T) * normal_pdf(d1) * math.sqrt(T)


def theta_call(S0, K, T, r, sigma, q=0.0):
    d1, d2 = d1_d2(S0, K, T, r, sigma, q)
    return (
        -S0 * math.exp(-q * T) * normal_pdf(d1) * sigma / (2.0 * math.sqrt(T))
        - r * K * math.exp(-r * T) * normal_cdf(d2)
        + q * S0 * math.exp(-q * T) * normal_cdf(d1)
    )


def theta_put(S0, K, T, r, sigma, q=0.0):
    """Put theta per year (calendar time passing), before contract multiplier."""
    d1, d2 = d1_d2(S0, K, T, r, sigma, q)
    return (
        -S0 * math.exp(-q * T) * normal_pdf(d1) * sigma / (2.0 * math.sqrt(T))
        + r * K * math.exp(-r * T) * normal_cdf(-d2)
        - q * S0 * math.exp(-q * T) * normal_cdf(-d1)
    )


def rho_call(S0, K, T, r, sigma, q=0.0):
    _, d2 = d1_d2(S0, K, T, r, sigma, q)
    return K * T * math.exp(-r * T) * normal_cdf(d2)


def rho_put(S0, K, T, r, sigma, q=0.0):
    _, d2 = d1_d2(S0, K, T, r, sigma, q)
    return -K * T * math.exp(-r * T) * normal_cdf(-d2)


def all_put_greeks(S0, K, T, r, sigma, q=0.0):
    return {
        "delta": put_delta(S0, K, T, r, sigma, q),
        "gamma": gamma(S0, K, T, r, sigma, q),
        "vega": vega(S0, K, T, r, sigma, q),
        "theta": theta_put(S0, K, T, r, sigma, q),
        "rho": rho_put(S0, K, T, r, sigma, q),
    }


def all_call_greeks(S0, K, T, r, sigma, q=0.0):
    return {
        "delta": call_delta(S0, K, T, r, sigma, q),
        "gamma": gamma(S0, K, T, r, sigma, q),
        "vega": vega(S0, K, T, r, sigma, q),
        "theta": theta_call(S0, K, T, r, sigma, q),
        "rho": rho_call(S0, K, T, r, sigma, q),
    }


def all_greeks(S0, K, T, r, sigma, q=0.0, option_type="call"):
    """Return analytical BSM Greeks for a European call or put."""
    kind = option_type.lower()
    if kind == "call":
        return all_call_greeks(S0, K, T, r, sigma, q)
    if kind == "put":
        return all_put_greeks(S0, K, T, r, sigma, q)
    raise ValueError("option_type must be 'call' or 'put'.")
