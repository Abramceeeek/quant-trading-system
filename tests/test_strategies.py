import unittest
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategies.advanced_strategies import AdvancedTradingStrategies

class TestAdvancedTradingStrategies(unittest.TestCase):

    def setUp(self):
        self.strategies = AdvancedTradingStrategies()
        self.spot_price = 100
        self.expiry_date = (datetime.today() + timedelta(days=30)).strftime("%Y-%m-%d")

        calls_data = {
            'strike': [105, 110, 115],
            'lastPrice': [2.0, 1.0, 0.5],
            'volume': [100, 100, 100],
            'contractSymbol': ['C1', 'C2', 'C3']
        }
        self.calls = pd.DataFrame(calls_data)

        puts_data = {
            'strike': [95, 90, 85],
            'lastPrice': [2.0, 1.0, 0.5],
            'volume': [100, 100, 100],
            'impliedVolatility': [0.2, 0.2, 0.2],
            'contractSymbol': ['P1', 'P2', 'P3']
        }
        self.puts = pd.DataFrame(puts_data)

    def test_calculate_rsi(self):
        data = {'Close': [10, 12, 11, 13, 14, 15, 14, 16, 17, 18, 19, 20, 19, 18]}
        prices = pd.Series(data['Close'])
        rsi = self.strategies._calculate_rsi(prices, period=4)
        self.assertAlmostEqual(rsi.iloc[-1], 53.83, places=2)

    def test_iron_condor_signal(self):
        signal = self.strategies.iron_condor_signal(self.calls, self.puts, self.spot_price, self.expiry_date)
        self.assertIsNotNone(signal)
        self.assertEqual(signal['signal'], 'IRON_CONDOR')

    def test_wheel_strategy_signal(self):
        signal = self.strategies.wheel_strategy_signal(self.puts, self.spot_price, self.expiry_date, 10000)
        self.assertIsNotNone(signal)
        self.assertEqual(signal['signal'], 'WHEEL_PUT')

if __name__ == '__main__':
    unittest.main()