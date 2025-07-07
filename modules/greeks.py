# modules/greeks.py

import math
from scipy.stats import norm

def calculate_d1(S, K, T, r, sigma):
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def calculate_d2(d1, sigma, T):
    return d1 - sigma * math.sqrt(T)

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(d1, sigma, T)

    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) 
                 - r * K * math.exp(-r * T) * norm.cdf(d2))
        rho = K * T * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) 
                 + r * K * math.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T)

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega / 100,   # Per 1% change in IV
        'theta': theta / 365, # Per day
        'rho': rho / 100      # Per 1% change in r
    }

def implied_volatility(price, S, K, T, r, option_type='call', tol=1e-5, max_iter=100):
    sigma = 0.2  # initial guess
    for i in range(max_iter):
        greeks = black_scholes_greeks(S, K, T, r, sigma, option_type)
        d1 = calculate_d1(S, K, T, r, sigma)
        d2 = calculate_d2(d1, sigma, T)
        
        if option_type == 'call':
            model_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            model_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        vega = greeks['vega'] * 100  # bring back to actual scale

        diff = model_price - price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega

    return None  # if it doesn't converge

if __name__ == "__main__":
    greeks = black_scholes_greeks(
        S=100,
        K=100,
        T=7/365,
        r=0.05,
        sigma=0.30,
        option_type='call'
    )
    print(greeks)
