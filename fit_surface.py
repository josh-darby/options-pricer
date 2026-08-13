import yfinance as yf

from vol_surface import build_vol_surface
from svi import fit_svi_surface
from heston import calibrate_heston


ticker = yf.Ticker("SPY")
all_expiries = ticker.options
chosen = list(all_expiries[::max(1, len(all_expiries) // 6)][:6])

surface = build_vol_surface(
    "SPY",
    max_expiries=None,
    min_T=7 / 365,
    spread_threshold=0.15
)

surface = surface[surface["expiry"].isin(chosen)]
surface = surface[surface["log_moneyness"].abs() < 0.4]

for T, group in surface.groupby("T"):
    k = group["log_moneyness"]
    print(
        f"T={T:.3f}  n={len(group)}  "
        f"k range=[{k.min():.3f}, {k.max():.3f}]  "
        f"k mean={k.mean():.3f}  "
        f"vol range=[{group['implied_vol'].min():.3f}, "
        f"{group['implied_vol'].max():.3f}]"
    )

svi_fits = fit_svi_surface(surface)

for T, fit in svi_fits.items():
    print(
        f"T={T:.3f}  n={fit['n_points']}  "
        f"dropped={fit['n_dropped']}  "
        f"rmse_vol={fit['rmse_vol']:.4f}  "
        f"arb_ok={fit['arbitrage_free']}  "
        f"params={tuple(round(p, 4) for p in fit['params'])}"
    )

calib_surface = surface.groupby("expiry", group_keys=False).apply(
    lambda g: g.iloc[::max(1, len(g) // 12)]
)

heston_fit = calibrate_heston(calib_surface)

print(heston_fit)