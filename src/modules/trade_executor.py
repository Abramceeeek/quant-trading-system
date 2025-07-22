from ib_insync import IB, Option, MarketOrder
import sys
from modules.options_data import (
    get_stock_price, get_expiry_dates, get_option_chain, find_atm_option
)

def connect_ibkr():
    ib = IB()
    print("➡ Attempting to connect to TWS on 127.0.0.1:7497 (clientId=2)...")
    try:
        ib.connect('127.0.0.1', 7497, clientId=2, timeout=15)
        print("✅ Connected to IBKR TWS API.")
    except Exception as e:
        print("❌ API connection failed:", e)
        sys.exit(1)
    return ib

def create_option_contract(symbol, expiry, strike, right):
    return Option(
        symbol=symbol,
        lastTradeDateOrContractMonth=expiry.replace('-', ''),  # FIXED
        strike=float(strike),
        right=right,
        exchange='SMART',
        currency='USD',
        multiplier='100'
    )

def place_order(ib, contract, action='BUY', quantity=1):
    order = MarketOrder(action, quantity)
    trade = ib.placeOrder(contract, order)
    ib.sleep(2)  # wait for order status
    return trade

if __name__ == "__main__":
    ib = connect_ibkr()

    symbol = "MSFT"
    spot_price = get_stock_price(symbol)
    expiry = get_expiry_dates(symbol)[0]  # Nearest expiry

    print(f"📈 Spot price of {symbol}: {spot_price:.2f}")
    print(f"📅 Selected expiry: {expiry}")

    calls, puts = get_option_chain(symbol, expiry)

    if calls.empty or puts.empty:
        print("❌ No options chain data available.")
        ib.disconnect()
        sys.exit(1)

    option_data = find_atm_option(calls, puts, spot_price)
    if option_data is None:
        print("❌ Could not find liquid ATM options.")
        ib.disconnect()
        sys.exit(1)

    atm_call = option_data['call']
    atm_put = option_data['put']

    print(f"🔎 Selected Call: {atm_call['symbol']} | Strike: {atm_call['strike']} | IV: {atm_call['iv']:.4f}")
    print(f"🔎 Selected Put:  {atm_put['symbol']} | Strike: {atm_put['strike']} | IV: {atm_put['iv']:.4f}")

    call_contract = create_option_contract(symbol, expiry, atm_call['strike'], "C")
    put_contract  = create_option_contract(symbol, expiry, atm_put['strike'], "P")

    ib.qualifyContracts(call_contract, put_contract)

    call_trade = place_order(ib, call_contract, action="BUY", quantity=1)
    put_trade  = place_order(ib, put_contract, action="BUY", quantity=1)

    print(f"📤 CALL TRADE STATUS: {call_trade.orderStatus.status}")
    print(f"📤 PUT TRADE STATUS:  {put_trade.orderStatus.status}")

    ib.disconnect()
