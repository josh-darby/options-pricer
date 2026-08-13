import numpy as np
import pandas as pd
import yfinance as yf

from black_scholes import clean_chain, implied_vol, implied_vol_brent


def build_vol_surface(
    symbol,
    r=0.05,
    q=0.0,
    max_expiries=None,
    min_T=1 / 365,
    spread_threshold=0.5,
):
    ticker = yf.Ticker(symbol)
    spot = ticker.history(period="1d")["Close"].iloc[-1]

    expiries = ticker.options

    if max_expiries is not None:
        expiries = expiries[:max_expiries]

    rows = []
    today = pd.Timestamp.today()

    for expiry_str in expiries:
        expiry_date = pd.Timestamp(expiry_str)
        T = (expiry_date - today).days / 365

        if T < min_T:
            continue

        chain = ticker.option_chain(expiry_str)
        calls = clean_chain(
            chain.calls,
            spread_threshold=spread_threshold
        )

        if calls.empty:
            continue

        forward = spot * np.exp((r - q) * T)

        for _, row in calls.iterrows():
            K = row["strike"]
            market_price = (row["bid"] + row["ask"]) / 2

            if market_price <= 0:
                continue

            intrinsic = max(
                spot - K * np.exp(-r * T),
                0.0
            )

            if market_price <= intrinsic:
                continue

            try:
                iv = implied_vol(
                    market_price,
                    spot,
                    K,
                    T,
                    r,
                    q
                )
            except ValueError:
                try:
                    iv = implied_vol_brent(
                        market_price,
                        spot,
                        K,
                        T,
                        r,
                        q
                    )
                except ValueError:
                    continue

            rows.append({
                "expiry": expiry_str,
                "T": T,
                "strike": K,
                "spot": spot,
                "forward": forward,
                "log_moneyness": np.log(K / forward),
                "market_price": market_price,
                "implied_vol": iv,
            })

    surface = pd.DataFrame(rows)

    return surface.sort_values(
        ["T", "strike"]
    ).reset_index(drop=True)


def plot_vol_surface(surface, ax=None):
    import matplotlib.pyplot as plt

    fig = None

    if ax is None:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")

    ax.plot_trisurf(
        surface["log_moneyness"],
        surface["T"],
        surface["implied_vol"],
        cmap="viridis",
        edgecolor="none",
        alpha=0.9,
    )

    ax.set_xlabel("log-moneyness  k = log(K/F)")
    ax.set_ylabel("T (years)")
    ax.set_zlabel("implied vol")
    ax.set_title("Implied volatility surface")

    return fig, ax


if __name__ == "__main__":
    ticker = yf.Ticker("SPY")
    all_expiries = ticker.options
    chosen = list(
        all_expiries[
            ::max(1, len(all_expiries) // 6)
        ][:6]
    )

    surface = build_vol_surface(
        "SPY",
        max_expiries=None,
        min_T=7 / 365,
        spread_threshold=0.15,
    )

    missing = set(chosen) - set(surface["expiry"].unique())
    print(
        f"chosen expiries with zero surviving rows: {missing}"
    )

    surface = surface[
        surface["expiry"].isin(chosen)
    ]

    surface = surface[
        surface["log_moneyness"].abs() < 0.4
    ]

    print(
        surface.groupby("expiry")["implied_vol"].describe()
    )

    plot_vol_surface(surface)

    import matplotlib.pyplot as plt

    plt.savefig("vol_surface.png", dpi=150)
    print("Saved vol_surface.png")