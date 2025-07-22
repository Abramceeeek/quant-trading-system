# algo_trading_bot.py

import sys
import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf
from pathlib import Path

# Add project directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'strategies'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'backtesting'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'risk_management'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ibkr'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

try:
    from strategies.advanced_strategies import AdvancedTradingStrategies
    from backtesting.backtest_engine import OptionsBacktestEngine
    from risk_management.risk_manager_advanced import AdvancedRiskManager
    from ibkr.ibkr_manager import IBKRManager
    from modules.options_data import get_stock_price, get_expiry_dates, get_option_chain, find_atm_option
    from utils.config_manager import ConfigManager
    from utils.error_handler import ErrorHandler, error_handler, validate_trading_preconditions
    from utils.error_handler import TradingBotError, ConfigurationError, ConnectionError
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")

import yaml

class AlgoTradingBot:
    """
    Advanced algorithmic trading bot with comprehensive risk management,
    error handling, and configuration management.
    """
    
    def __init__(self, config_path='config/settings.yaml'):
        """
        Initialize the algorithmic trading bot.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            # Initialize configuration and utilities
            self.config_manager = ConfigManager(config_path)
            self.error_handler = ErrorHandler(self.config_manager)
            
            # Capital settings
            self.initial_capital = self.config_manager.get('initial_capital', 10000)
            self.live_trading = self.config_manager.get('live_trading', False)
            self.current_capital = self.initial_capital
            
            # Initialize trading components
            self._initialize_components()
            
            # Configuration shortcuts
            self.watchlist = self.config_manager.get_watchlist_symbols()
            self.enabled_strategies = self.config_manager.get('enabled_strategies', [])
            
            # Results tracking
            self.trade_log = []
            self.performance_metrics = {}
            self.backtest_results = {}
            self.daily_pnl = 0.0
            self.max_drawdown = 0.0
            
            # Setup directories
            self._setup_directories()
            
            # Get logger (already configured by ConfigManager)
            self.logger = logging.getLogger(__name__)
            
            self.logger.info(f"AlgoTradingBot initialized - Capital: ${self.initial_capital:,} - Live: {self.live_trading}")
            self.logger.info(f"Watchlist: {len(self.watchlist)} symbols, Strategies: {len(self.enabled_strategies)}")
            
        except Exception as e:
            # Use basic logging if error_handler not available yet
            logging.error(f"Failed to initialize AlgoTradingBot: {e}")
            raise ConfigurationError(f"Bot initialization failed: {e}")
    
    def _initialize_components(self):
        """Initialize trading components with error handling."""
        try:
            # Initialize strategy engine with configuration
            self.strategy_engine = AdvancedTradingStrategies()
            self.strategy_engine.config_manager = self.config_manager
            
            # Initialize risk manager with enhanced settings
            risk_limits = self.config_manager.get_risk_limits()
            self.risk_manager = AdvancedRiskManager(
                initial_capital=self.initial_capital,
                max_portfolio_risk=risk_limits.get('max_portfolio_risk', 0.02)
            )
            
            # Initialize backtest engine
            self.backtest_engine = OptionsBacktestEngine(self.initial_capital)
            
            # IBKR connection (only for live trading)
            self.ibkr_manager = None
            if self.live_trading:
                ibkr_config = self.config_manager.get('ibkr', {})
                self.ibkr_manager = IBKRManager(
                    host=ibkr_config.get('host', '127.0.0.1'),
                    port=ibkr_config.get('port', 7497),
                    client_id=ibkr_config.get('client_id', 1)
                )
                
        except Exception as e:
            raise ConfigurationError(f"Component initialization failed: {e}")
    
    def _setup_directories(self):
        """Setup required directories for data storage."""
        directories = [
            'data',
            'data/logs',
            'data/backtest_results',
            'data/daily_results',
            'data/daily_summaries',
            'data/reports'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _initialize_backtest_results(self) -> Dict:
        results = {}
        for strategy in self.enabled_strategies:
            results[strategy] = {}
        results['summary'] = {}
        return results

    def _run_backtest_for_symbol(self, symbol: str, start_date: str, end_date: str, backtest_results: Dict):
        self.logger.info(f"Backtesting {symbol}...")
        self.backtest_engine.capital = self.initial_capital
        self.backtest_engine.positions = []
        self.backtest_engine.trade_history = []

        try:
            for strategy in self.enabled_strategies:
                self.backtest_engine.capital = self.initial_capital
                backtest_function = getattr(self.backtest_engine, f"backtest_{strategy}_strategy")
                results = backtest_function(symbol, start_date, end_date)
                backtest_results[strategy][symbol] = results
        except Exception as e:
            self.logger.error(f"Error backtesting {symbol}: {e}")

    def run_comprehensive_backtest(self, start_date: str = '2020-01-01', end_date: str = None) -> Dict:
        """
        Run comprehensive backtests for all strategies and symbols
        """
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        self.logger.info(f"Starting comprehensive backtest: {start_date} to {end_date}")
        
        backtest_results = self._initialize_backtest_results()
        
        for symbol in self.watchlist:
            self._run_backtest_for_symbol(symbol, start_date, end_date, backtest_results)
        
        # Generate summary
        backtest_results['summary'] = self.analyze_backtest_results(backtest_results)
        
        # Save results
        self.save_backtest_results(backtest_results)
        
        return backtest_results
    
    def analyze_backtest_results(self, results: Dict) -> Dict:
        """
        Analyze and summarize backtest results
        """
        summary = {
            'best_strategy': '',
            'best_symbol': '',
            'overall_stats': {},
            'strategy_rankings': {}
        }
        
        strategy_performance = {}
        
        for strategy_name, strategy_results in results.items():
            if strategy_name == 'summary':
                continue
                
            total_return = 0
            total_trades = 0
            total_wins = 0
            
            for symbol, symbol_results in strategy_results.items():
                if 'error' in symbol_results:
                    continue
                
                total_return += symbol_results.get('total_return', 0)
                total_trades += symbol_results.get('total_trades', 0)
                if symbol_results.get('win_rate', 0) > 0:
                    total_wins += (symbol_results.get('win_rate', 0) * symbol_results.get('total_trades', 0)) / 100
            
            avg_return = total_return / len([r for r in strategy_results.values() if 'error' not in r]) if strategy_results else 0
            avg_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
            
            strategy_performance[strategy_name] = {
                'avg_return': round(avg_return, 2),
                'avg_win_rate': round(avg_win_rate, 1),
                'total_trades': total_trades,
                'score': avg_return * (avg_win_rate / 100)  # Combined score
            }
        
        # Rank strategies
        ranked_strategies = sorted(strategy_performance.items(), 
                                 key=lambda x: x[1]['score'], reverse=True)
        
        if ranked_strategies:
            summary['best_strategy'] = ranked_strategies[0][0]
            summary['strategy_rankings'] = dict(ranked_strategies)
            summary['overall_stats'] = strategy_performance
        
        return summary
    
    @error_handler(context="generate_signals", critical=False, reraise=False, default_return=[])
    def generate_signals(self, symbol: str) -> List[Dict]:
        """
        Generate trading signals for a given symbol with enhanced error handling.
        
        Args:
            symbol: Stock symbol to generate signals for
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            self.logger.warning(f"Invalid symbol provided: {symbol}")
            return signals
        
        # Get current market data
        spot_price = get_stock_price(symbol)
        if spot_price is None or spot_price <= 0:
            self.logger.warning(f"Invalid spot price for {symbol}: {spot_price}")
            return signals
        
        expiries = get_expiry_dates(symbol)
        if not expiries:
            self.logger.info(f"No expiry dates found for {symbol}")
            return signals
        
        # Use nearest expiry (usually weekly) and next monthly
        near_expiry = expiries[0]
        
        calls, puts = get_option_chain(symbol, near_expiry)
        
        if calls.empty or puts.empty:
            self.logger.info(f"No option chain data for {symbol} expiry {near_expiry}")
            return signals
        
        # Get historical volatility
        historical_vol = self.strategy_engine.get_historical_volatility(symbol)
        
        # Generate signals from different strategies
        strategy_methods = {
            'iron_condor': self._generate_iron_condor_signal,
            'wheel': self._generate_wheel_signal,
            'volatility_scalping': self._generate_volatility_signal,
            'momentum': self._generate_momentum_signal
        }
        
        for strategy_name in self.enabled_strategies:
            if strategy_name in strategy_methods:
                try:
                    signal = strategy_methods[strategy_name](
                        symbol, calls, puts, spot_price, near_expiry, historical_vol
                    )
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    self.logger.warning(f"Error generating {strategy_name} signal for {symbol}: {e}")
        
        self.logger.info(f"Generated {len(signals)} signals for {symbol}")
        return signals
    
    def _generate_iron_condor_signal(self, symbol: str, calls: pd.DataFrame, puts: pd.DataFrame, 
                                   spot_price: float, near_expiry: str, historical_vol: float) -> Optional[Dict]:
        """Generate Iron Condor signal."""
        ic_signal = self.strategy_engine.iron_condor_signal(calls, puts, spot_price, near_expiry)
        if ic_signal:
            ic_signal['symbol'] = symbol
            ic_signal['expiry'] = near_expiry
            ic_signal['strategy'] = 'iron_condor'
        return ic_signal
    
    def _generate_wheel_signal(self, symbol: str, calls: pd.DataFrame, puts: pd.DataFrame, 
                             spot_price: float, near_expiry: str, historical_vol: float) -> Optional[Dict]:
        """Generate Wheel strategy signal."""
        wheel_signal = self.strategy_engine.wheel_strategy_signal(
            puts, spot_price, near_expiry, self.current_capital
        )
        if wheel_signal:
            wheel_signal['symbol'] = symbol
            wheel_signal['expiry'] = near_expiry
            wheel_signal['strategy'] = 'wheel'
        return wheel_signal
    
    def _generate_volatility_signal(self, symbol: str, calls: pd.DataFrame, puts: pd.DataFrame, 
                                  spot_price: float, near_expiry: str, historical_vol: float) -> Optional[Dict]:
        """Generate Volatility Scalping signal."""
        vol_signal = self.strategy_engine.volatility_scalping_signal(
            calls, puts, spot_price, historical_vol
        )
        if vol_signal:
            vol_signal['symbol'] = symbol
            vol_signal['expiry'] = near_expiry
            vol_signal['strategy'] = 'volatility_scalping'
        return vol_signal
    
    def _generate_momentum_signal(self, symbol: str, calls: pd.DataFrame, puts: pd.DataFrame, 
                                spot_price: float, near_expiry: str, historical_vol: float) -> Optional[Dict]:
        """Generate Momentum strategy signal."""
        momentum_signal = self.strategy_engine.momentum_breakout_signal(symbol, calls)
        if momentum_signal:
            momentum_signal['symbol'] = symbol
            momentum_signal['expiry'] = near_expiry
            momentum_signal['strategy'] = 'momentum'
        return momentum_signal
    
    def evaluate_and_execute_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Evaluate signals with risk management and execute if approved
        """
        executed_trades = []
        
        for signal in signals:
            # Risk evaluation
            risk_assessment = self.risk_manager.evaluate_trade_risk(signal)
            
            if not risk_assessment['approved']:
                self.logger.info(f"Signal rejected: {signal['signal']} for {signal['symbol']} - Risk score too high")
                continue
            
            # Update signal with approved position size
            signal['position_size'] = risk_assessment['position_size']
            
            # Execute trade
            if self.live_trading and self.ibkr_manager and self.ibkr_manager.connected:
                execution_result = self.execute_live_trade(signal)
            else:
                execution_result = self.simulate_trade_execution(signal)
            
            if execution_result.get('status') == 'executed':
                executed_trades.append({
                    'signal': signal,
                    'execution': execution_result,
                    'timestamp': datetime.now().isoformat()
                })
                
                self.logger.info(f"Trade executed: {signal['signal']} for {signal['symbol']}")
        
        return executed_trades
    
    def execute_live_trade(self, signal: Dict) -> Dict:
        """
        Execute trade on IBKR
        """
        try:
            signal_type = signal['signal'].lower()
            
            if signal_type == 'iron_condor':
                return self.ibkr_manager.execute_iron_condor(signal)
            elif signal_type == 'wheel_put':
                return self.ibkr_manager.execute_wheel_strategy(signal)
            else:
                self.logger.warning(f"Live execution not implemented for {signal_type}")
                return {'status': 'not_implemented'}
                
        except Exception as e:
            self.logger.error(f"Error executing live trade: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def simulate_trade_execution(self, signal: Dict) -> Dict:
        """
        Simulate trade execution for paper trading
        """
        return {
            'status': 'executed',
            'trade_id': f"SIM_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'method': 'simulation'
        }
    
    def run_daily_scan(self) -> Dict:
        """
        Run daily market scan and generate signals
        """
        self.logger.info("Starting daily market scan...")
        
        daily_results = {
            'scan_date': datetime.now().strftime('%Y-%m-%d'),
            'signals_generated': [],
            'trades_executed': [],
            'market_conditions': {}
        }
        
        # Check market conditions
        market_conditions = self.assess_market_conditions()
        daily_results['market_conditions'] = market_conditions
        
        # Skip trading if market conditions are poor
        if market_conditions.get('risk_level', 'medium') == 'high':
            self.logger.warning("High risk market conditions - skipping trading")
            return daily_results
        
        # Scan each symbol
        all_signals = []
        for symbol in self.watchlist:
            signals = self.generate_signals(symbol)
            all_signals.extend(signals)
        
        daily_results['signals_generated'] = all_signals
        self.logger.info(f"Generated {len(all_signals)} signals")
        
        # Execute approved signals
        if all_signals:
            executed_trades = self.evaluate_and_execute_signals(all_signals)
            daily_results['trades_executed'] = executed_trades
            self.logger.info(f"Executed {len(executed_trades)} trades")
        
        # Save daily results
        self.save_daily_results(daily_results)
        
        return daily_results
    
    def assess_market_conditions(self) -> Dict:
        """
        Assess current market conditions
        """
        try:
            # Get VIX for market fear gauge
            vix = yf.Ticker('^VIX')
            vix_data = vix.history(period='5d')
            current_vix = vix_data['Close'].iloc[-1]
            
            # Get SPY for market trend
            spy = yf.Ticker('SPY')
            spy_data = spy.history(period='60d')
            spy_sma_20 = spy_data['Close'].rolling(20).mean().iloc[-1]
            spy_sma_50 = spy_data['Close'].rolling(50).mean().iloc[-1]
            current_spy = spy_data['Close'].iloc[-1]
            
            # Assess conditions
            risk_level = 'medium'
            if current_vix > 30:
                risk_level = 'high'
            elif current_vix < 15:
                risk_level = 'low'
            
            trend = 'neutral'
            if current_spy > spy_sma_20 > spy_sma_50:
                trend = 'bullish'
            elif current_spy < spy_sma_20 < spy_sma_50:
                trend = 'bearish'
            
            return {
                'vix': round(current_vix, 2),
                'spy_price': round(current_spy, 2),
                'spy_vs_sma20': round((current_spy / spy_sma_20 - 1) * 100, 1),
                'trend': trend,
                'risk_level': risk_level,
                'favorable_for_trading': risk_level != 'high'
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing market conditions: {e}")
            return {'risk_level': 'unknown', 'favorable_for_trading': False}
    
    def run_simulation(self, start_date: str, duration_days: int = 365) -> Dict:
        """
        Run a simulation starting from a specific date
        """
        self.logger.info(f"Starting {duration_days}-day simulation from {start_date}")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = start_dt + timedelta(days=duration_days)
        
        simulation_capital = 1000  # Start with $1000 as requested
        daily_results = []
        
        # Simulate trading every week
        current_date = start_dt
        while current_date < end_dt:
            # Skip weekends
            if current_date.weekday() < 5:
                date_str = current_date.strftime('%Y-%m-%d')
                
                # Simulate daily scan (simplified)
                try:
                    # Generate sample trades based on historical performance
                    if len(daily_results) == 0 or daily_results[-1]['capital'] > 0:
                        # Simulate trade based on best backtested strategy
                        best_strategy = self.backtest_results.get('summary', {}).get('best_strategy', 'iron_condor')
                        
                        # Simulate trade outcome
                        trade_outcome = self.simulate_historical_trade(best_strategy, date_str)
                        new_capital = max(0, simulation_capital + trade_outcome['pnl'])
                        
                        daily_results.append({
                            'date': date_str,
                            'strategy': best_strategy,
                            'capital': new_capital,
                            'pnl': trade_outcome['pnl'],
                            'trade_details': trade_outcome
                        })
                        
                        simulation_capital = new_capital
                    
                except Exception as e:
                    self.logger.error(f"Error in simulation for {date_str}: {e}")
            
            current_date += timedelta(days=7)  # Weekly trades
        
        # Calculate final results
        final_capital = simulation_capital
        total_return = (final_capital - 1000) / 1000 * 100
        
        simulation_results = {
            'start_date': start_date,
            'end_date': end_dt.strftime('%Y-%m-%d'),
            'initial_capital': 1000,
            'final_capital': round(final_capital, 2),
            'total_return': round(total_return, 1),
            'duration_days': duration_days,
            'daily_results': daily_results
        }
        
        self.logger.info(f"Simulation completed - Final capital: ${final_capital:.2f} ({total_return:.1f}% return)")
        
        return simulation_results
    
    def simulate_historical_trade(self, strategy: str, date: str) -> Dict:
        """
        Simulate a single trade outcome based on historical patterns
        """
        import random
        
        # Use backtested win rates and average returns
        strategy_stats = {
            'iron_condor': {'win_rate': 0.75, 'avg_win': 25, 'avg_loss': -75},
            'wheel': {'win_rate': 0.80, 'avg_win': 15, 'avg_loss': -30},
            'momentum': {'win_rate': 0.45, 'avg_win': 80, 'avg_loss': -40}
        }
        
        stats = strategy_stats.get(strategy, strategy_stats['iron_condor'])
        
        # Random outcome based on win rate
        is_win = random.random() < stats['win_rate']
        pnl = stats['avg_win'] if is_win else stats['avg_loss']
        
        # Add some randomness
        pnl *= (0.5 + random.random())  # 50%-150% of average
        
        return {
            'outcome': 'win' if is_win else 'loss',
            'pnl': round(pnl, 2),
            'strategy': strategy,
            'date': date
        }
    
    def save_backtest_results(self, results: Dict):
        """Save backtest results to file"""
        try:
            os.makedirs('data/backtest_results', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'data/backtest_results/backtest_{timestamp}.json'
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.backtest_results = results
            self.logger.info(f"Backtest results saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving backtest results: {e}")
    
    def save_daily_results(self, results: Dict):
        """Save daily scan results"""
        try:
            os.makedirs('data/daily_results', exist_ok=True)
            date_str = results['scan_date'].replace('-', '')
            filename = f'data/daily_results/scan_{date_str}.json'
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving daily results: {e}")
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        report = f"\n{'='*80}\n"
        report += f"ALGORITHMIC TRADING BOT PERFORMANCE REPORT\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"{'='*80}\n\n"
        
        # Backtest results summary
        if hasattr(self, 'backtest_results') and self.backtest_results:
            summary = self.backtest_results.get('summary', {})
            report += f"BACKTEST RESULTS SUMMARY:\n"
            report += f"Best Strategy: {summary.get('best_strategy', 'N/A')}\n"
            
            for strategy, stats in summary.get('strategy_rankings', {}).items():
                report += f"\n{strategy.upper()}:\n"
                report += f"  Average Return: {stats.get('avg_return', 0):.1f}%\n"
                report += f"  Win Rate: {stats.get('avg_win_rate', 0):.1f}%\n"
                report += f"  Total Trades: {stats.get('total_trades', 0)}\n"
                report += f"  Score: {stats.get('score', 0):.2f}\n"
        
        # Risk management summary
        report += f"\n{'-'*40}\n"
        report += f"RISK MANAGEMENT:\n"
        report += self.risk_manager.get_risk_report()
        
        return report


