import numpy as np
import yfinance as yf
from scipy.stats import norm
import pandas as pd
from scipy.optimize import brentq


def compute_d1_d2(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + (1 / 2) * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_price(S, K, T, r, sigma, q=0.0):
    d1, d2 = compute_d1_d2(S, K, T, r, sigma, q)
    C = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return C


def put_price(S, K, T, r, sigma, q=0.0):
    d1, d2 = compute_d1_d2(S, K, T, r, sigma, q)
    P = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return P


def call_delta(S, K, T, r, sigma, q=0.0):
    d1, _ = compute_d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.cdf(d1)


def put_delta(S, K, T, r, sigma, q=0.0):
    d1, _ = compute_d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * (norm.cdf(d1) - 1)


def gamma(S, K, T, r, sigma, q=0.0):
    d1, _ = compute_d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    d1, _ = compute_d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def call_theta(S, K, T, r, sigma, q=0.0):
    d1, d2 = compute_d1_d2(S, K, T, r, sigma, q)
    a = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    b = -r * K * np.exp(-r * T) * norm.cdf(d2)
    c = q * S * np.exp(-q * T) * norm.cdf(d1)
    return a + b + c


def put_theta(S, K, T, r, sigma, q=0.0):
    d1, d2 = compute_d1_d2(S, K, T, r, sigma, q)
    a = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    b = r * K * np.exp(-r * T) * norm.cdf(-d2)
    c = -q * S * np.exp(-q * T) * norm.cdf(-d1)
    return a + b + c


def call_rho(S, K, T, r, sigma, q=0.0):
    _, d2 = compute_d1_d2(S, K, T, r, sigma, q)
    return K * T * np.exp(-r * T) * norm.cdf(d2)


def put_rho(S, K, T, r, sigma, q=0.0):
    _, d2 = compute_d1_d2(S, K, T, r, sigma, q)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)


def implied_vol(market_price, S, K, T, r, q=0.0, guess_sigma=0.2, max_iteration=200):
    for _ in range(max_iteration):
        model_price = call_price(S, K, T, r, guess_sigma, q)
        diff = model_price - market_price

        if abs(diff) < 0.0001:
            return guess_sigma

        v = vega(S, K, T, r, guess_sigma, q)

        if v < 1e-8:
            raise ValueError("Vega too small for Newton-Raphson to be stable")

        guess_sigma = guess_sigma - diff / v

        if guess_sigma <= 0:
            raise ValueError("Newton-Raphson stepped into non-positive sigma")

    raise ValueError("Newton-Raphson did not converge within max_iteration")


def clean_chain(calls, spread_threshold=0.5):
    calls = calls.copy()
    calls["relative_spread"] = (calls["ask"] - calls["bid"]) / calls["bid"]

    calls = calls[
        (calls["volume"] > 0)
        & (calls["bid"] > 0)
        & (calls["ask"] > 0)
        & (calls["relative_spread"] < spread_threshold)
    ]

    return calls


def implied_vol_brent(market_price, S, K, T, r, q=0.0, lo=0.01, hi=5.0):
    f = lambda sigma: call_price(S, K, T, r, sigma, q) - market_price
    return brentq(f, lo, hi)


if __name__ == "__main__":
    ticker = yf.Ticker("SPY")
    chain = ticker.option_chain("2026-08-21")
    calls = chain.calls
    clean = clean_chain(calls)

    spot = ticker.history(period="1d")["Close"].iloc[-1]
    expiry_date = pd.Timestamp("2026-08-21")
    T = (expiry_date - pd.Timestamp.today()).days / 365
    r = 0.05
    q = 0.013

    implied_vols = []
    fallback_count = 0

    for _, row in clean.iterrows():
        K = row["strike"]
        market_price = (row["bid"] + row["ask"]) / 2

        try:
            iv = implied_vol(market_price, spot, K, T, r, q)
        except ValueError:
            iv = implied_vol_brent(market_price, spot, K, T, r, q)
            fallback_count += 1

        implied_vols.append(iv)

    clean["implied_vol"] = implied_vols

    print(f"Rows that needed Brent fallback: {fallback_count} / {len(clean)}")
    print(clean[["strike", "bid", "ask", "implied_vol"]])