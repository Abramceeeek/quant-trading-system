# modules/signal_generator.py

from datetime import datetime

def should_enter_straddle(spot, call, put, expiry_date, debug=False):
    if debug:
        print(f"IVs — Call: {call['iv']}, Put: {put['iv']}")
        print(f"Volumes — Call: {call['volume']}, Put: {put['volume']}")
        print(f"Total cost: {call['price'] + put['price']}")
        print(f"Spot price: {spot}")
        print(f"Expiry date: {expiry_date}")
        print(f"Max allowed cost: {spot * 0.02}")

    # --- Params ---
    min_iv = 0.00001
    max_iv = 1.0
    min_volume = 1000
    max_cost_ratio = 0.02  # max 2% of spot

    print(f"Max allowed cost: {spot * max_cost_ratio}")

    # --- IV filter ---
    if not (min_iv <= call['iv'] <= max_iv and min_iv <= put['iv'] <= max_iv):
        print("Filtered out due to IV")
        return None
    
    # --- Liquidity filter ---
    if call['volume'] < min_volume or put['volume'] < min_volume:
        return None
    
    # --- Cost filter ---
    total_cost = call['price'] + put['price']
    if total_cost > spot * max_cost_ratio:
        return None
    
    # --- Time filter ---
    expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
    days_to_expiry = (expiry - datetime.today()).days
    if days_to_expiry > 3:
        return None
    
    # --- Signal generated ---
    return {
        'signal': 'STRADDLE ENTRY',
        'strike': call['strike'],
        'call_symbol': call['symbol'],
        'put_symbol': put['symbol'],
        'total_cost': round(total_cost, 2),
        'days_to_expiry': days_to_expiry
    }

if __name__ == "__main__":
    from options_data import get_stock_price, get_expiry_dates, get_option_chain, find_atm_option

    ticker = "MSFT"
    spot = get_stock_price(ticker)
    expiries = get_expiry_dates(ticker)
    expiry = expiries[0]

    calls, puts = get_option_chain(ticker, expiry)
    atm = find_atm_option(calls, puts, spot)

    signal = should_enter_straddle(spot, atm['call'], atm['put'], expiry, debug=True)
    print("Trade Signal:", signal)
