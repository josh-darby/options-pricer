import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares

from black_scholes import vega as bs_vega


def _heston_char_func(u, S0, T, r, q, v0, kappa, theta, sigma_v, rho, j):
    i = 1j

    if j == 1:
        u_j, b_j = 0.5, kappa - rho * sigma_v
    else:
        u_j, b_j = -0.5, kappa

    x = np.log(S0)
    a = kappa * theta

    d = np.sqrt(
        (rho * sigma_v * i * u - b_j) ** 2
        - sigma_v ** 2 * (2 * u_j * i * u - u ** 2)
    )

    g = (b_j - rho * sigma_v * i * u + d) / (
        b_j - rho * sigma_v * i * u - d
    )

    c = 1.0 / g
    exp_dT = np.exp(-d * T)

    C = (r - q) * i * u * T + (a / sigma_v ** 2) * (
        (b_j - rho * sigma_v * i * u - d) * T
        - 2 * np.log((1 - c * exp_dT) / (1 - c))
    )

    D = (
        (b_j - rho * sigma_v * i * u - d)
        / sigma_v ** 2
        * (1 - exp_dT)
        / (1 - c * exp_dT)
    )

    return np.exp(C + D * v0 + i * u * x)


def _heston_Pj(S0, K, T, r, q, v0, kappa, theta, sigma_v, rho, j):
    def integrand(u):
        cf = _heston_char_func(
            u, S0, T, r, q, v0, kappa, theta, sigma_v, rho, j
        )
        return (
            np.exp(-1j * u * np.log(K)) * cf / (1j * u)
        ).real

    integral, _ = quad(integrand, 1e-8, 200, limit=200)
    return 0.5 + integral / np.pi


def heston_call_price(S0, K, T, r, q, v0, kappa, theta, sigma_v, rho):
    P1 = _heston_Pj(
        S0, K, T, r, q, v0, kappa, theta, sigma_v, rho, j=1
    )
    P2 = _heston_Pj(
        S0, K, T, r, q, v0, kappa, theta, sigma_v, rho, j=2
    )

    return S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2


def calibrate_heston(surface, r=0.05, q=0.0, x0=None, weight_by_vega=True):
    S = surface["spot"].values
    K = surface["strike"].values
    T = surface["T"].values
    market_price = surface["market_price"].values
    market_iv = surface["implied_vol"].values

    if weight_by_vega:
        vegas = np.array([
            bs_vega(s, k, t, r, iv, q)
            for s, k, t, iv in zip(S, K, T, market_iv)
        ])
        w = 1.0 / np.maximum(vegas, 1e-6)
    else:
        w = np.ones_like(market_price)

    def residuals(x):
        v0, kappa, theta, sigma_v, rho = x

        model_price = np.array([
            heston_call_price(
                s, k, t, r, q, v0, kappa, theta, sigma_v, rho
            )
            for s, k, t in zip(S, K, T)
        ])

        return w * (model_price - market_price)

    if x0 is None:
        atm_var = float(np.median(market_iv) ** 2)
        x0 = [atm_var, 1.0, atm_var, 0.5, -0.5]

    lower = [1e-4, 1e-3, 1e-4, 1e-3, -0.999]
    upper = [2.0, 30.0, 2.0, 3.0, 0.999]

    result = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        verbose=2
    )

    v0, kappa, theta, sigma_v, rho = result.x

    return {
        "v0": v0,
        "kappa": kappa,
        "theta": theta,
        "sigma_v": sigma_v,
        "rho": rho,
        "feller_satisfied": bool(2 * kappa * theta > sigma_v ** 2),
        "cost": result.cost,
        "success": result.success,
    }


if __name__ == "__main__":
    from black_scholes import call_price as bs_call_price

    S0, K, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.0
    sigma = 0.2

    heston_degenerate = heston_call_price(
        S0, K, T, r, q,
        v0=sigma**2,
        kappa=2.0,
        theta=sigma**2,
        sigma_v=1e-4,
        rho=0.0
    )

    bs_price = bs_call_price(S0, K, T, r, sigma, q)

    print(
        f"Heston (sigma_v->0): {heston_degenerate:.4f}   "
        f"BS: {bs_price:.4f}   "
        f"diff: {abs(heston_degenerate - bs_price):.6f}"
    )

    import pandas as pd

    rng = np.random.default_rng(0)

    true = {
        "v0": 0.04,
        "kappa": 1.5,
        "theta": 0.045,
        "sigma_v": 0.6,
        "rho": -0.65
    }

    strikes = np.array([80, 90, 95, 100, 105, 110, 120] * 3, dtype=float)
    maturities = np.repeat([0.25, 0.5, 1.0], 7)
    spot = 100.0

    synthetic_prices = np.array([
        heston_call_price(spot, k, t, r, q, **true)
        for k, t in zip(strikes, maturities)
    ])

    synthetic_prices *= (
        1 + rng.normal(0, 0.002, size=synthetic_prices.shape)
    )

    from black_scholes import implied_vol, implied_vol_brent

    ivs = []

    for p, k, t in zip(synthetic_prices, strikes, maturities):
        try:
            ivs.append(implied_vol(p, spot, k, t, r, q))
        except ValueError:
            ivs.append(implied_vol_brent(p, spot, k, t, r, q))

    synth_surface = pd.DataFrame({
        "spot": spot,
        "strike": strikes,
        "T": maturities,
        "market_price": synthetic_prices,
        "implied_vol": ivs,
    })

    calibrated = calibrate_heston(synth_surface, r=r, q=q)

    print(
        "\ntrue params:     ",
        {k: round(v, 4) for k, v in true.items()}
    )
    print(
        "calibrated params:",
        {k: round(calibrated[k], 4) for k in true}
    )
    print(
        "feller satisfied:",
        calibrated["feller_satisfied"],
        " success:",
        calibrated["success"]
    )