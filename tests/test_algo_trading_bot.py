import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from algo_trading_bot import AlgoTradingBot

class TestAlgoTradingBot(unittest.TestCase):

    def setUp(self):
        self.bot = AlgoTradingBot(config_path='src/config/settings.yaml')

    def test_analyze_backtest_results(self):
        backtest_results = {
            'iron_condor': {
                'SPY': {'total_return': 10, 'total_trades': 5, 'win_rate': 60},
                'QQQ': {'total_return': 5, 'total_trades': 5, 'win_rate': 80}
            },
            'wheel': {
                'SPY': {'total_return': 15, 'total_trades': 10, 'win_rate': 70},
                'QQQ': {'total_return': 20, 'total_trades': 10, 'win_rate': 90}
            }
        }

        summary = self.bot.analyze_backtest_results(backtest_results)

        self.assertEqual(summary['best_strategy'], 'wheel')
        self.assertAlmostEqual(summary['strategy_rankings']['wheel']['avg_return'], 17.5)
        self.assertAlmostEqual(summary['strategy_rankings']['wheel']['avg_win_rate'], 80.0)

if __name__ == '__main__':
    unittest.main()