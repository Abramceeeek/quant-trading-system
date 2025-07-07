from ib_insync import IB

ib = IB()

print("➡ Attempting to connect to TWS (localhost:7497)...")

try:
    ib.connect('127.0.0.1', 7497, clientId=9, timeout=10)
    print("✅ Connected to IBKR!")
    ib.disconnect()
except Exception as e:
    print("❌ Failed to connect:", e)
