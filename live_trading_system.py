# live_trading_system.py

import sys
import os
import json
import time
import schedule
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'strategies'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'risk_management'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ibkr'))

from algo_trading_bot import AlgoTradingBot

class LiveTradingSystem:
    def __init__(self):
        self.bot = AlgoTradingBot(initial_capital=10000, live_trading=True)
        self.is_running = False
        self.daily_logs = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/live_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def pre_market_routine(self):
        """Pre-market preparation routine"""
        self.logger.info("Starting pre-market routine...")
        
        # Check IBKR connection
        if self.bot.ibkr_manager:
            if not self.bot.ibkr_manager.connect():
                self.logger.error("Failed to connect to IBKR - aborting trading day")
                return False
        
        # Update account information
        account_balance = self.bot.ibkr_manager.get_account_balance() if self.bot.ibkr_manager else 10000
        self.logger.info(f"Account balance: ${account_balance:,.2f}")
        
        # Check market conditions
        market_conditions = self.bot.assess_market_conditions()
        self.logger.info(f"Market conditions: {market_conditions}")
        
        if not market_conditions.get('favorable_for_trading', True):
            self.logger.warning("Unfavorable market conditions - trading may be limited today")
        
        return True
    
    def market_open_scan(self):
        """Market open scanning routine"""
        self.logger.info("Running market open scan...")
        
        try:
            # Run daily scan
            daily_results = self.bot.run_daily_scan()
            
            # Log results
            self.daily_logs.append({
                'date': datetime.now().isoformat(),
                'type': 'market_scan',
                'results': daily_results
            })
            
            self.logger.info(f"Market scan complete - {len(daily_results.get('signals_generated', []))} signals, "
                           f"{len(daily_results.get('trades_executed', []))} trades executed")
            
        except Exception as e:
            self.logger.error(f"Error in market scan: {e}")
    
    def midday_check(self):
        """Midday position monitoring"""
        self.logger.info("Running midday position check...")
        
        if self.bot.ibkr_manager and self.bot.ibkr_manager.connected:
            try:
                # Get current positions
                positions = self.bot.ibkr_manager.get_positions()
                self.logger.info(f"Current positions: {len(positions)}")
                
                # Check for profit-taking or stop-loss conditions
                for position in positions:
                    if position['contract_type'] == 'OPT':
                        pnl_percent = position['pnl'] / (abs(position['avgCost']) * abs(position['position']) * 100)
                        
                        if pnl_percent > 0.50:
                            self.logger.info(f"Position {position['symbol']} up {pnl_percent:.1%} - consider profit taking")
                        elif pnl_percent < -1.0:
                            self.logger.warning(f"Position {position['symbol']} down {pnl_percent:.1%} - monitor closely")
                
            except Exception as e:
                self.logger.error(f"Error in midday check: {e}")
    
    def end_of_day_routine(self):
        """End of day summary and cleanup"""
        self.logger.info("Running end of day routine...")
        
        # Generate daily summary
        daily_summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'account_balance': self.bot.ibkr_manager.get_account_balance() if self.bot.ibkr_manager else 0,
            'positions': len(self.bot.ibkr_manager.get_positions()) if self.bot.ibkr_manager else 0,
            'daily_pnl': 0,  # Would calculate from positions
            'logs': self.daily_logs
        }
        
        # Save daily summary
        os.makedirs('data/daily_summaries', exist_ok=True)
        filename = f"data/daily_summaries/summary_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, 'w') as f:
            json.dump(daily_summary, f, indent=2, default=str)
        
        self.logger.info(f"Daily summary saved to {filename}")
        
        # Clear daily logs
        self.daily_logs = []
    
    def setup_schedule(self):
        """Setup trading schedule"""
        # Pre-market routine (8:30 AM)
        schedule.every().day.at("08:30").do(self.pre_market_routine)
        
        # Market open scan (9:35 AM - 5 minutes after open)
        schedule.every().day.at("09:35").do(self.market_open_scan)
        
        # Midday check (12:30 PM)
        schedule.every().day.at("12:30").do(self.midday_check)
        
        # End of day routine (4:05 PM - 5 minutes after close)
        schedule.every().day.at("16:05").do(self.end_of_day_routine)
        
        self.logger.info("Trading schedule configured")
    
    def run_live_trading(self, duration_days: int = 30):
        """
        Run live trading system for specified duration
        """
        self.logger.info(f"Starting live trading system for {duration_days} days")
        
        self.setup_schedule()
        self.is_running = True
        
        start_date = datetime.now()
        end_date = start_date + timedelta(days=duration_days)
        
        try:
            while self.is_running and datetime.now() < end_date:
                # Run scheduled tasks
                schedule.run_pending()
                
                # Sleep for 1 minute
                time.sleep(60)
                
                # Check if it's a weekend
                if datetime.now().weekday() >= 5:  # Saturday or Sunday
                    self.logger.debug("Weekend - no trading")
                    continue
        
        except KeyboardInterrupt:
            self.logger.info("Live trading interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in live trading loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        
        if self.bot.ibkr_manager:
            self.bot.ibkr_manager.disconnect()
        
        self.logger.info("Live trading system stopped")
    
    def generate_monthly_report(self) -> str:
        """Generate monthly performance report"""
        report = f"\n{'='*80}\n"
        report += f"MONTHLY LIVE TRADING REPORT\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"{'='*80}\n\n"
        
        # Load daily summaries
        summaries_dir = 'data/daily_summaries'
        if os.path.exists(summaries_dir):
            summary_files = [f for f in os.listdir(summaries_dir) if f.endswith('.json')]
            summary_files.sort()
            
            total_trades = 0
            start_balance = 0
            end_balance = 0
            
            for i, filename in enumerate(summary_files):
                with open(os.path.join(summaries_dir, filename), 'r') as f:
                    daily_data = json.load(f)
                
                if i == 0:
                    start_balance = daily_data.get('account_balance', 0)
                if i == len(summary_files) - 1:
                    end_balance = daily_data.get('account_balance', 0)
                
                # Count trades from logs
                for log_entry in daily_data.get('logs', []):
                    if log_entry.get('type') == 'market_scan':
                        results = log_entry.get('results', {})
                        total_trades += len(results.get('trades_executed', []))
            
            # Calculate performance
            if start_balance > 0:
                total_return = (end_balance - start_balance) / start_balance * 100
            else:
                total_return = 0
            
            report += f"PERFORMANCE SUMMARY:\n"
            report += f"Starting Balance: ${start_balance:,.2f}\n"
            report += f"Ending Balance: ${end_balance:,.2f}\n"
            report += f"Total Return: {total_return:.2f}%\n"
            report += f"Total Trades: {total_trades}\n"
            report += f"Trading Days: {len(summary_files)}\n"
            
        else:
            report += "No daily summaries found.\n"
        
        return report

def main():
    """Main function for live trading"""
    print("Live Trading System Starting...")
    
    # Check if IBKR TWS/Gateway is available
    print("\nIMPORTANT: Make sure IBKR TWS or Gateway is running!")
    print("Default connection: localhost:7497")
    print("Press Enter to continue or Ctrl+C to abort...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("Aborted by user")
        return
    
    # Initialize and run live trading
    live_system = LiveTradingSystem()
    
    try:
        # Run for 30 days (1 month)
        live_system.run_live_trading(duration_days=30)
        
        # Generate final report
        monthly_report = live_system.generate_monthly_report()
        print(monthly_report)
        
        # Save report
        with open('data/monthly_report.txt', 'w') as f:
            f.write(monthly_report)
        
    except Exception as e:
        print(f"Error in live trading: {e}")
    finally:
        live_system.cleanup()

if __name__ == "__main__":
    main()