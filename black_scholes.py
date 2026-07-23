import numpy as np
from scipy.stats import norm

def compute_d1_d2(S, K, T, r, sigma, q = 0.0):
    d1 = (np.log(S/K) + (r - q + (1/2)*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - (sigma*np.sqrt(T))
    return d1, d2

def call_price(S, K, T, r, sigma, q = 0.0):
    d1, d2 = compute_d1_d2(S, K, T, r, sigma, q)
    C = S*np.exp(-q*T) * (norm.cdf(d1)) - K*np.exp(-r*T) * (norm.cdf(d2))
    return C

def put_price(S, K, T, r, sigma, q = 0.0):
    d1, d2 = compute_d1_d2(S, K, T, r, sigma, q)
    P = K*np.exp(-r*T) * (norm.cdf(-d2)) - S*np.exp(-q*T) * (norm.cdf(-d1))
    return P

def call_delta(S, K, T, r, sigma, q = 0.0):
    d1, _ = compute_d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q*T) * norm.cdf(d1)

def put_delta(S, K, T, r, sigma, q = 0.0):
    d1, _ = compute_d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q*T) * (norm.cdf(d1) - 1)

def gamma(S, K, T, r, sigma, q = 0.0):
    d1, _ = compute_d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q*T)*norm.pdf(d1) / (S*sigma*np.sqrt(T))

def vega(S, K, T, r, sigma, q = 0.0):
    d1, _ = compute_d1_d2(S, K, T, r, sigma, q)
    return S*np.exp(-q*T)*norm.pdf(d1)*(np.sqrt(T))

def call_theta(S, K, T, r, sigma, q = 0.0):
    d1, d2 = compute_d1_d2(S, K, T, r, sigma, q)
    a = -(S*np.exp(-q*T)*norm.pdf(d1)*sigma) / (2*np.sqrt(T))
    b = -r*K*np.exp(-r*T)*norm.cdf(d2)
    c = q*S*np.exp(-q*T)*norm.cdf(d1)
    return a + b + c

print((call_price(100, 100, 1 + 0.05, 0.05, 0.2) - call_price(100, 100, 1 - 0.05, 0.05, 0.2)) / (-2*0.05))
print(call_theta(100, 100, 1, 0.05, 0.2))