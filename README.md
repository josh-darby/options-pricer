# Options Pricer

A from-scratch options pricing and implied volatility modelling toolkit in Python, built to explore the full pipeline from a raw options chain to calibrated stochastic volatility models. No options-pricing libraries are used; pricing formulas and calibration logic are implemented directly.

## What's here

- **`black_scholes.py`** — Black-Scholes-Merton pricing, the five standard Greeks, and an implied volatility solver using Newton-Raphson with automatic fallback to Brent's method when vega becomes too small near deep OTM/ITM strikes.
- **`vol_surface.py`** — pulls multi-expiry option chains via `yfinance`, applies spread and liquidity filters, and builds a tidy implied volatility surface in log-moneyness / time-to-maturity space.
- **`svi.py`** — fits the raw SVI (Stochastic Volatility Inspired) parameterisation to each expiry slice, with an automatic outlier filter and butterfly-arbitrage sanity check.
- **`heston.py`** — semi-closed-form Heston (1993) pricing via the "little trap" characteristic-function formulation, calibrated to the full surface using vega-weighted least squares.
- **`fit_surface.py`** — end-to-end script that builds the surface, fits SVI to each expiry slice, and calibrates Heston across expiries.

## Example: SPY implied volatility surface

Built from SPY option chain data, filtered to expiries beyond 1 week to reduce the impact of noisy short-dated quotes and to `|log-moneyness| < 0.4` to remove thinly traded far-wing options:

- 4 expiries, T = 0.047 to 0.622 years
- SVI fit RMSE per slice: 0.009 – 0.023 in volatility terms
- SVI skew (`ρ`) becomes substantially less negative with maturity, from approximately -0.95 at 17 days to -0.07 at 6 months, consistent with the stronger downside skew typically observed in short-dated equity-index options.

## Finding: weak identifiability of κ and σᵥ in Heston calibration

Calibrating Heston to this surface (4 expiries, approximately 12 strikes per expiry after subsampling) repeatedly drove the mean-reversion speed κ to the upper bound, while σᵥ and ρ adjusted to maintain a similar fitted price surface:

| κ upper bound | Fitted κ | Fitted σᵥ | Fitted ρ | Cost |
|---|---|---|---|---|
| 15 | 15.000 (wall) | 0.481 | -0.999 (wall) | 0.02328 |
| 30 | 30.000 (wall) | 1.357 | -0.648 | 0.02103 |

Both parameters continued moving as the bound widened, while the fit quality changed only slightly. This suggests a **weakly identified ridge** in parameter space rather than a well-defined optimum.

This is a known calibration issue in the Heston model: different combinations of mean-reversion speed and vol-of-vol can generate very similar option prices, making the parameters difficult to identify from a relatively small option surface.

Natural next steps would be to fix κ using an independent estimate and calibrate the remaining parameters, or use a larger and denser panel of expiries to provide more information about the term structure.

## Usage

```python
from vol_surface import build_vol_surface
from svi import fit_svi_surface
from heston import calibrate_heston

surface = build_vol_surface("SPY", min_T=7/365, spread_threshold=0.15)
svi_fits = fit_svi_surface(surface)
heston_fit = calibrate_heston(surface)
```

## Validation

Key pricing/calibration components have a self-test in their `if __name__ == "__main__"` block:

- `heston.py`: confirms Heston collapses exactly onto Black-Scholes as σᵥ → 0 (diff = 0.000000), and recovers known parameters from a synthetic calibration round-trip.
- `svi.py`: recovers known SVI parameters from a synthetic noisy slice.

## Requirements

`numpy`, `scipy`, `pandas`, `yfinance`, `matplotlib`
