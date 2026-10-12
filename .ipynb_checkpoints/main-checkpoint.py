from data.inputs import (
    S0,
    K,
    T,
    r,
    sigma,
    N_SIMULATIONS,
    MC_TIME_STEPS,
    FD_SPACE_STEPS,
    FD_SMAX_MULTIPLIER,
)
from pricing.black_scholes import call_price, put_price
from pricing.monte_carlo import monte_carlo_call_price
from pricing.finite_difference import finite_difference_call_price


def main():
    print("=" * 64)
    print("QUANTITATIVE DERIVATIVES PRICING ENGINE")
    print("=" * 64)

    print("\nTOY INPUTS")
    print("-" * 64)
    print(f"Spot price       : {S0}")
    print(f"Strike           : {K}")
    print(f"Time to expiry   : {T:.6f} years")
    print(f"Risk-free rate   : {r:.2%}")
    print(f"Volatility       : {sigma:.2%}")

    bs_call = call_price(S0, K, T, r, sigma)
    bs_put = put_price(S0, K, T, r, sigma)

    print("\n1. BLACK-SCHOLES ANALYTICAL")
    print("-" * 64)
    print(f"Call price : {bs_call:.6f}")
    print(f"Put price  : {bs_put:.6f}")

    mc_call, mc_se, mc_lower, mc_upper = monte_carlo_call_price(
        S0, K, T, r, sigma,
        n_simulations=N_SIMULATIONS,
        time_steps=MC_TIME_STEPS,
    )

    print("\n2. MONTE CARLO")
    print("-" * 64)
    print(f"Call price     : {mc_call:.6f}")
    print(f"Standard error : {mc_se:.6f}")
    print(f"95% CI         : [{mc_lower:.6f}, {mc_upper:.6f}]")

    fd_call, _, _ = finite_difference_call_price(
        S0, K, T, r, sigma,
        space_steps=FD_SPACE_STEPS,
        smax_multiplier=FD_SMAX_MULTIPLIER,
    )

    print("\n3. FINITE DIFFERENCE")
    print("-" * 64)
    print(f"Call price : {fd_call:.6f}")

    print("\n4. PRICING COMPARISON")
    print("-" * 64)
    print(f"Black-Scholes     : {bs_call:.6f}")
    print(f"Monte Carlo       : {mc_call:.6f}")
    print(f"Finite Difference : {fd_call:.6f}")
    print(f"MC absolute error : {abs(mc_call - bs_call):.6f}")
    print(f"FD absolute error : {abs(fd_call - bs_call):.6f}")


if __name__ == "__main__":
    main()
