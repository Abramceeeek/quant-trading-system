import os
import json
from datetime import datetime, timedelta
import sys

#Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from algo_trading_bot import AlgoTradingBot

def main():
    """Main execution function"""
    print("Algorithmic Trading Bot Starting...")

    # Initialize bot
    bot = AlgoTradingBot(config_path='src/config/settings.yaml')

    # Run comprehensive backtest
    print("\nRunning comprehensive backtest...")
    backtest_results = bot.run_comprehensive_backtest(start_date='2022-01-01')

    # Generate backtest reports
    for strategy in ['iron_condor', 'wheel', 'momentum']:
        for symbol in ['SPY', 'QQQ']:
            if symbol in backtest_results.get(strategy, {}) and backtest_results[strategy][symbol]:
                report = bot.backtest_engine.generate_backtest_report(
                    backtest_results[strategy][symbol],
                    f"{strategy.upper()} - {symbol}"
                )
                print(report)

    # Run 1-year simulation
    print("\nRunning 1-year simulation starting with $1000...")
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    simulation_results = bot.run_simulation(one_year_ago, 365)

    print(f"\nSIMULATION RESULTS:")
    print(f"Start Date: {simulation_results['start_date']}")
    print(f"End Date: {simulation_results['end_date']}")
    print(f"Initial Capital: ${simulation_results['initial_capital']:,}")
    print(f"Final Capital: ${simulation_results['final_capital']:,.2f}")
    print(f"Total Return: {simulation_results['total_return']:.1f}%")

    # Generate final report
    print("\nGenerating performance report...")
    performance_report = bot.generate_performance_report()
    print(performance_report)

    # Save comprehensive results
    final_results = {
        'backtest_results': backtest_results,
        'simulation_results': simulation_results,
        'bot_config': {
            'initial_capital': bot.initial_capital,
            'watchlist': bot.watchlist,
            'enabled_strategies': bot.enabled_strategies
        }
    }

    os.makedirs('src/data', exist_ok=True)
    with open('src/data/final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print("\nAlgorithmic Trading Bot Analysis Complete!")
    print("Results saved to 'src/data/final_results.json'")

if __name__ == "__main__":
    main()
