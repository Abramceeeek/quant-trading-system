# modules/options_data.py

import yfinance as yf

# modules/options_data.py

def get_stock_price(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1d')

    if data.empty:
        data = stock.history(period='5d')
    
    if data.empty:
        raise ValueError(f"No price data found for {ticker}. Try again later or check ticker symbol.")
    
    return data['Close'].iloc[-1]


def get_expiry_dates(ticker):
    stock = yf.Ticker(ticker)
    return stock.options

def get_option_chain(ticker, expiry):
    stock = yf.Ticker(ticker)
    chain = stock.option_chain(expiry)
    return chain.calls, chain.puts

def find_atm_option(calls, puts, spot_price, tolerance=2.5):
    # Find closest strike to spot price
    calls['diff'] = abs(calls['strike'] - spot_price)
    puts['diff'] = abs(puts['strike'] - spot_price)

    atm_call = calls.sort_values('diff').iloc[0]
    atm_put = puts.sort_values('diff').iloc[0]

    # Filter out illiquid contracts (optional)
    if atm_call['impliedVolatility'] > 0 and atm_put['impliedVolatility'] > 0:
        print(f"Selected Call: {atm_call['contractSymbol']} | IV: {atm_call['impliedVolatility']}")
        print(f"Selected Put: {atm_put['contractSymbol']} | IV: {atm_put['impliedVolatility']}")

        return {
            'call': {
                'strike': atm_call['strike'],
                'price': atm_call['lastPrice'],
                'iv': atm_call['impliedVolatility'],
                'volume': atm_call['volume'],
                'symbol': atm_call['contractSymbol']
            },
            'put': {
                'strike': atm_put['strike'],
                'price': atm_put['lastPrice'],
                'iv': atm_put['impliedVolatility'],
                'volume': atm_put['volume'],
                'symbol': atm_put['contractSymbol']
            }
        }
    else:
        return None

