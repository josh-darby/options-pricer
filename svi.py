import numpy as np
from scipy.optimize import least_squares


def raw_svi(k, a, b, rho, m, sigma):
    return a + b * (
        rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2)
    )


def svi_vol(k, T, params):
    a, b, rho, m, sigma = params
    w = raw_svi(k, a, b, rho, m, sigma)
    return np.sqrt(np.maximum(w, 0.0) / T)


def _initial_guess(k, w):
    a0 = max(w.min() * 0.9, 1e-6)
    m0 = k[np.argmin(w)]
    sigma0 = max(np.std(k), 0.05)
    b0 = 0.1
    rho0 = -0.5

    return np.array([a0, b0, rho0, m0, sigma0])


def fit_svi_slice(k, w, weights=None):
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    weights = (
        np.ones_like(w)
        if weights is None
        else np.asarray(weights, dtype=float)
    )

    def residuals(params):
        a, b, rho, m, sigma = params
        model = raw_svi(k, a, b, rho, m, sigma)
        return weights * (model - w)

    x0 = _initial_guess(k, w)

    lower = [-np.inf, 0.0, -0.95, -np.inf, 1e-4]
    upper = [np.inf, 3.0, 0.95, np.inf, np.inf]

    result = least_squares(
        residuals,
        x0,
        bounds=(lower, upper)
    )

    return tuple(result.x), result


def butterfly_arbitrage_ok(params):
    a, b, rho, m, sigma = params
    return b * sigma * (1 + abs(rho)) <= 4 + 1e-9


def fit_svi_surface(surface):
    fits = {}

    for T, group in surface.groupby("T"):
        k = group["log_moneyness"].values
        iv = group["implied_vol"].values

        median_iv = np.median(iv)
        mask = iv < 3 * median_iv

        k, iv = k[mask], iv[mask]
        n_dropped = int((~mask).sum())

        w = iv ** 2 * T
        params, result = fit_svi_slice(k, w)

        model_vol = svi_vol(k, T, params)
        rmse_vol = float(np.sqrt(np.mean((model_vol - iv) ** 2)))

        fits[T] = {
            "params": params,
            "n_points": len(k),
            "n_dropped": n_dropped,
            "arbitrage_free": butterfly_arbitrage_ok(params),
            "rmse_vol": rmse_vol,
        }

    return fits


def surface_implied_vol(K, T, forward, fits):
    k = np.log(K / forward)
    Ts = np.array(sorted(fits.keys()))

    if T <= Ts[0]:
        return svi_vol(k, Ts[0], fits[Ts[0]]["params"])

    if T >= Ts[-1]:
        return svi_vol(k, Ts[-1], fits[Ts[-1]]["params"])

    i = np.searchsorted(Ts, T)

    T_lo, T_hi = Ts[i - 1], Ts[i]

    w_lo = raw_svi(k, *fits[T_lo]["params"])
    w_hi = raw_svi(k, *fits[T_hi]["params"])

    frac = (T - T_lo) / (T_hi - T_lo)
    w = w_lo + frac * (w_hi - w_lo)

    return np.sqrt(max(w, 0.0) / T)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    true_params = (0.02, 0.15, -0.6, 0.0, 0.1)
    k_test = np.linspace(-0.4, 0.4, 25)

    w_true = raw_svi(k_test, *true_params)
    w_noisy = w_true * (
        1 + rng.normal(0, 0.01, size=k_test.shape)
    )

    fitted_params, res = fit_svi_slice(k_test, w_noisy)

    print("true params:  ", np.round(true_params, 4))
    print("fitted params:", np.round(fitted_params, 4))
    print(
        "arbitrage check passes:",
        butterfly_arbitrage_ok(fitted_params)
    )