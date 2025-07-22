# comprehensive_strategy_analyzer.py

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
import os
from typing import Dict, List, Tuple
from scipy import stats
from scipy.stats import norm
import logging

warnings.filterwarnings('ignore')

class ComprehensiveStrategyAnalyzer:
    def __init__(self):
        self.strategies = [
            'long_call', 'long_put', 'short_call', 'short_put',
            'bull_call_spread', 'bear_put_spread', 'bull_put_spread', 'bear_call_spread',
            'iron_condor', 'iron_butterfly', 'long_straddle', 'short_straddle',
            'long_strangle', 'short_strangle', 'covered_call', 'protective_put',
            'collar', 'calendar_spread', 'diagonal_spread', 'butterfly_spread',
            'condor_spread', 'ratio_spread', 'wheel_strategy', 'momentum_breakout',
            'volatility_scalping', 'mean_reversion', 'trend_following', 'pairs_trading'
        ]
        
        self.symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'META']
        self.years = [2020, 2021, 2022, 2023, 2024]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get comprehensive historical data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return pd.DataFrame()
                
            # Add technical indicators
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(20).std() * np.sqrt(252)
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            data['RSI'] = self.calculate_rsi(data['Close'])
            data['BB_Upper'], data['BB_Lower'] = self.calculate_bollinger_bands(data['Close'])
            data['MACD'], data['MACD_Signal'] = self.calculate_macd(data['Close'])
            
            return data
        except Exception as e:
            self.logger.error(f"Error getting data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def black_scholes_price(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Black-Scholes option pricing"""
        try:
            if T <= 0:
                if option_type.lower() == 'call':
                    return max(0, S - K)
                else:
                    return max(0, K - S)
            
            d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return max(0.01, price)
        except:
            return 0.01
    
    def backtest_long_call(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Backtest long call strategy"""
        results = {'trades': [], 'total_return': 0, 'win_rate': 0}
        
        for i in range(50, len(data)-30, 10):  # Every 10 days
            current_price = data.iloc[i]['Close']
            volatility = data.iloc[i]['Volatility']
            
            if pd.isna(volatility) or volatility < 0.1:
                continue
            
            # Buy slightly OTM call
            strike = current_price * 1.05
            days_to_expiry = 30
            call_price = self.black_scholes_price(current_price, strike, days_to_expiry/365, 0.02, volatility, 'call')
            
            # Exit after 2 weeks
            exit_idx = min(i + 14, len(data)-1)
            exit_price = data.iloc[exit_idx]['Close']
            exit_vol = data.iloc[exit_idx].get('Volatility', volatility)
            
            exit_call_price = self.black_scholes_price(exit_price, strike, 16/365, 0.02, exit_vol, 'call')
            
            pnl = exit_call_price - call_price
            pnl_percent = (pnl / call_price) * 100
            
            results['trades'].append({
                'entry_date': data.index[i].strftime('%Y-%m-%d'),
                'pnl': round(pnl, 2),
                'pnl_percent': round(pnl_percent, 1),
                'win': pnl > 0
            })
        
        if results['trades']:
            wins = sum([1 for t in results['trades'] if t['win']])
            results['win_rate'] = (wins / len(results['trades'])) * 100
            avg_return = np.mean([t['pnl_percent'] for t in results['trades']])
            results['total_return'] = avg_return
            results['total_trades'] = len(results['trades'])
        
        return results
    
    def backtest_iron_condor(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Backtest iron condor strategy"""
        results = {'trades': [], 'total_return': 0, 'win_rate': 0}
        
        for i in range(50, len(data)-45, 14):  # Every 2 weeks
            current_price = data.iloc[i]['Close']
            volatility = data.iloc[i]['Volatility']
            
            if pd.isna(volatility) or volatility < 0.15:
                continue
            
            # Iron condor strikes
            put_short_strike = current_price * 0.95
            put_long_strike = put_short_strike - 5
            call_short_strike = current_price * 1.05
            call_long_strike = call_short_strike + 5
            
            days_to_expiry = 30
            T = days_to_expiry / 365
            
            # Entry prices
            put_short_price = self.black_scholes_price(current_price, put_short_strike, T, 0.02, volatility, 'put')
            put_long_price = self.black_scholes_price(current_price, put_long_strike, T, 0.02, volatility, 'put')
            call_short_price = self.black_scholes_price(current_price, call_short_strike, T, 0.02, volatility, 'call')
            call_long_price = self.black_scholes_price(current_price, call_long_strike, T, 0.02, volatility, 'call')
            
            net_credit = put_short_price + call_short_price - put_long_price - call_long_price
            max_risk = 5 - net_credit
            
            if net_credit <= 0:
                continue
            
            # Exit at 50% of DTE
            exit_idx = min(i + 15, len(data)-1)
            exit_price = data.iloc[exit_idx]['Close']
            exit_vol = data.iloc[exit_idx].get('Volatility', volatility)
            exit_T = 15 / 365
            
            # Exit prices
            exit_put_short = self.black_scholes_price(exit_price, put_short_strike, exit_T, 0.02, exit_vol, 'put')
            exit_put_long = self.black_scholes_price(exit_price, put_long_strike, exit_T, 0.02, exit_vol, 'put')
            exit_call_short = self.black_scholes_price(exit_price, call_short_strike, exit_T, 0.02, exit_vol, 'call')
            exit_call_long = self.black_scholes_price(exit_price, call_long_strike, exit_T, 0.02, exit_vol, 'call')
            
            exit_cost = exit_put_short + exit_call_short - exit_put_long - exit_call_long
            pnl = net_credit - exit_cost
            pnl_percent = (pnl / max_risk) * 100
            
            results['trades'].append({
                'entry_date': data.index[i].strftime('%Y-%m-%d'),
                'pnl': round(pnl, 2),
                'pnl_percent': round(pnl_percent, 1),
                'win': pnl > 0,
                'max_risk': round(max_risk, 2)
            })
        
        if results['trades']:
            wins = sum([1 for t in results['trades'] if t['win']])
            results['win_rate'] = (wins / len(results['trades'])) * 100
            avg_return = np.mean([t['pnl_percent'] for t in results['trades']])
            results['total_return'] = avg_return
            results['total_trades'] = len(results['trades'])
        
        return results
    
    def backtest_wheel_strategy(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Backtest wheel strategy"""
        results = {'trades': [], 'total_return': 0, 'win_rate': 0}
        
        cash = 10000
        shares = 0
        avg_cost = 0
        
        for i in range(50, len(data)-30, 21):  # Monthly
            current_price = data.iloc[i]['Close']
            volatility = data.iloc[i]['Volatility']
            
            if pd.isna(volatility):
                continue
            
            if shares == 0:
                # Sell cash-secured put
                put_strike = current_price * 0.92
                put_premium = self.black_scholes_price(current_price, put_strike, 30/365, 0.02, volatility, 'put')
                
                # Check assignment at expiry
                expiry_idx = min(i + 30, len(data)-1)
                expiry_price = data.iloc[expiry_idx]['Close']
                
                if expiry_price < put_strike:
                    # Assigned
                    shares = 100
                    avg_cost = put_strike - put_premium
                    cash -= put_strike * 100 - put_premium * 100
                    pnl = 0  # Will be realized when stock is sold
                else:
                    # Keep premium
                    cash += put_premium * 100
                    pnl = put_premium * 100
                
                results['trades'].append({
                    'entry_date': data.index[i].strftime('%Y-%m-%d'),
                    'type': 'CSP',
                    'pnl': round(pnl, 0),
                    'win': pnl >= 0
                })
            
            else:
                # Sell covered call
                call_strike = max(current_price * 1.05, avg_cost * 1.02)
                call_premium = self.black_scholes_price(current_price, call_strike, 30/365, 0.02, volatility, 'call')
                
                expiry_idx = min(i + 30, len(data)-1)
                expiry_price = data.iloc[expiry_idx]['Close']
                
                if expiry_price > call_strike:
                    # Called away
                    pnl = (call_strike - avg_cost + call_premium) * 100
                    cash += call_strike * 100 + call_premium * 100
                    shares = 0
                else:
                    # Keep premium and shares
                    pnl = call_premium * 100
                    cash += call_premium * 100
                
                results['trades'].append({
                    'entry_date': data.index[i].strftime('%Y-%m-%d'),
                    'type': 'CC',
                    'pnl': round(pnl, 0),
                    'win': pnl >= 0
                })
        
        if results['trades']:
            wins = sum([1 for t in results['trades'] if t['win']])
            results['win_rate'] = (wins / len(results['trades'])) * 100
            total_pnl = sum([t['pnl'] for t in results['trades']])
            results['total_return'] = (total_pnl / 10000) * 100
            results['total_trades'] = len(results['trades'])
        
        return results
    
    def backtest_short_straddle(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Backtest short straddle strategy"""
        results = {'trades': [], 'total_return': 0, 'win_rate': 0}
        
        for i in range(50, len(data)-45, 14):
            current_price = data.iloc[i]['Close']
            volatility = data.iloc[i]['Volatility']
            
            if pd.isna(volatility) or volatility < 0.20:
                continue
            
            # ATM straddle
            strike = round(current_price / 5) * 5
            days_to_expiry = 30
            T = days_to_expiry / 365
            
            call_price = self.black_scholes_price(current_price, strike, T, 0.02, volatility, 'call')
            put_price = self.black_scholes_price(current_price, strike, T, 0.02, volatility, 'put')
            
            net_credit = call_price + put_price
            
            # Exit at 50% of time
            exit_idx = min(i + 15, len(data)-1)
            exit_price = data.iloc[exit_idx]['Close']
            exit_vol = data.iloc[exit_idx].get('Volatility', volatility)
            exit_T = 15 / 365
            
            exit_call_price = self.black_scholes_price(exit_price, strike, exit_T, 0.02, exit_vol, 'call')
            exit_put_price = self.black_scholes_price(exit_price, strike, exit_T, 0.02, exit_vol, 'put')
            
            exit_cost = exit_call_price + exit_put_price
            pnl = net_credit - exit_cost
            pnl_percent = (pnl / net_credit) * 100
            
            results['trades'].append({
                'entry_date': data.index[i].strftime('%Y-%m-%d'),
                'pnl': round(pnl, 2),
                'pnl_percent': round(pnl_percent, 1),
                'win': pnl > 0
            })
        
        if results['trades']:
            wins = sum([1 for t in results['trades'] if t['win']])
            results['win_rate'] = (wins / len(results['trades'])) * 100
            avg_return = np.mean([t['pnl_percent'] for t in results['trades']])
            results['total_return'] = avg_return
            results['total_trades'] = len(results['trades'])
        
        return results
    
    def backtest_covered_call(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Backtest covered call strategy"""
        results = {'trades': [], 'total_return': 0, 'win_rate': 0}
        
        # Assume we own 100 shares from start
        entry_price = data.iloc[50]['Close']
        shares_value = entry_price * 100
        
        for i in range(50, len(data)-30, 21):
            current_price = data.iloc[i]['Close']
            volatility = data.iloc[i]['Volatility']
            
            if pd.isna(volatility):
                continue
            
            # Sell 5% OTM call
            call_strike = current_price * 1.05
            call_premium = self.black_scholes_price(current_price, call_strike, 30/365, 0.02, volatility, 'call')
            
            expiry_idx = min(i + 30, len(data)-1)
            expiry_price = data.iloc[expiry_idx]['Close']
            
            if expiry_price > call_strike:
                # Called away - total return includes stock appreciation + premium
                total_return = ((call_strike - entry_price + call_premium) / entry_price) * 100
                pnl = (call_strike - entry_price + call_premium) * 100
            else:
                # Keep premium, stock appreciation/depreciation unrealized
                pnl = call_premium * 100
                total_return = (call_premium / entry_price) * 100
            
            results['trades'].append({
                'entry_date': data.index[i].strftime('%Y-%m-%d'),
                'pnl': round(pnl, 0),
                'total_return': round(total_return, 1),
                'win': pnl > 0
            })
        
        if results['trades']:
            wins = sum([1 for t in results['trades'] if t['win']])
            results['win_rate'] = (wins / len(results['trades'])) * 100
            avg_return = np.mean([t['total_return'] for t in results['trades']])
            results['total_return'] = avg_return
            results['total_trades'] = len(results['trades'])
        
        return results
    
    def run_comprehensive_analysis(self) -> Dict:
        """Run comprehensive analysis for all strategies, symbols, and years"""
        self.logger.info("Starting comprehensive strategy analysis...")
        
        all_results = {}
        
        # Strategy backtesting functions
        strategy_functions = {
            'long_call': self.backtest_long_call,
            'iron_condor': self.backtest_iron_condor,
            'wheel_strategy': self.backtest_wheel_strategy,
            'short_straddle': self.backtest_short_straddle,
            'covered_call': self.backtest_covered_call
        }
        
        for year in self.years:
            self.logger.info(f"Analyzing year {year}...")
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            all_results[year] = {}
            
            for symbol in self.symbols:
                self.logger.info(f"  Processing {symbol}...")
                data = self.get_historical_data(symbol, start_date, end_date)
                
                if data.empty:
                    continue
                
                all_results[year][symbol] = {}
                
                for strategy_name, strategy_func in strategy_functions.items():
                    try:
                        result = strategy_func(data, symbol)
                        all_results[year][symbol][strategy_name] = result
                    except Exception as e:
                        self.logger.error(f"Error in {strategy_name} for {symbol} in {year}: {e}")
                        all_results[year][symbol][strategy_name] = {
                            'trades': [], 'total_return': 0, 'win_rate': 0, 'error': str(e)
                        }
        
        return all_results
    
    def generate_comprehensive_report(self, results: Dict) -> str:
        """Generate comprehensive analysis report"""
        report = f"\n{'='*100}\n"
        report += f"COMPREHENSIVE STRATEGY ANALYSIS - LAST 5 YEARS (2020-2024)\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"{'='*100}\n\n"
        
        # Summary table by strategy
        strategy_summary = {}
        
        for year, year_data in results.items():
            for symbol, symbol_data in year_data.items():
                for strategy, strategy_data in symbol_data.items():
                    if strategy not in strategy_summary:
                        strategy_summary[strategy] = {
                            'total_trades': 0,
                            'win_rates': [],
                            'returns': [],
                            'years_active': []
                        }
                    
                    if strategy_data.get('total_trades', 0) > 0:
                        strategy_summary[strategy]['total_trades'] += strategy_data.get('total_trades', 0)
                        strategy_summary[strategy]['win_rates'].append(strategy_data.get('win_rate', 0))
                        strategy_summary[strategy]['returns'].append(strategy_data.get('total_return', 0))
                        strategy_summary[strategy]['years_active'].append(f"{year}-{symbol}")
        
        # Generate strategy ranking
        report += "STRATEGY PERFORMANCE RANKING:\n"
        report += f"{'Strategy':<20} {'Avg Win Rate':<12} {'Avg Return':<12} {'Total Trades':<12} {'Score':<8}\n"
        report += "-" * 80 + "\n"
        
        strategy_scores = []
        for strategy, data in strategy_summary.items():
            avg_win_rate = np.mean(data['win_rates']) if data['win_rates'] else 0
            avg_return = np.mean(data['returns']) if data['returns'] else 0
            total_trades = data['total_trades']
            score = (avg_win_rate * 0.4 + avg_return * 0.6) if avg_return >= 0 else avg_return
            
            strategy_scores.append((strategy, avg_win_rate, avg_return, total_trades, score))
        
        # Sort by score
        strategy_scores.sort(key=lambda x: x[4], reverse=True)
        
        for strategy, win_rate, return_pct, trades, score in strategy_scores:
            report += f"{strategy:<20} {win_rate:<12.1f} {return_pct:<12.1f}% {trades:<12} {score:<8.1f}\n"
        
        # Year-by-year analysis
        report += f"\n{'='*100}\n"
        report += "YEAR-BY-YEAR DETAILED ANALYSIS:\n"
        report += f"{'='*100}\n"
        
        for year in sorted(results.keys()):
            report += f"\n{year} RESULTS:\n"
            report += "-" * 50 + "\n"
            
            year_data = results[year]
            
            for symbol in sorted(year_data.keys()):
                symbol_data = year_data[symbol]
                report += f"\n{symbol}:\n"
                
                for strategy in sorted(symbol_data.keys()):
                    strategy_data = symbol_data[strategy]
                    if strategy_data.get('total_trades', 0) > 0:
                        report += f"  {strategy:<18}: Win Rate: {strategy_data.get('win_rate', 0):.1f}%, "
                        report += f"Return: {strategy_data.get('total_return', 0):.1f}%, "
                        report += f"Trades: {strategy_data.get('total_trades', 0)}\n"
        
        # Best performing combinations
        report += f"\n{'='*100}\n"
        report += "TOP 10 STRATEGY-SYMBOL-YEAR COMBINATIONS:\n"
        report += f"{'='*100}\n"
        
        all_combinations = []
        for year, year_data in results.items():
            for symbol, symbol_data in year_data.items():
                for strategy, strategy_data in symbol_data.items():
                    if strategy_data.get('total_trades', 0) > 5:  # At least 5 trades
                        score = (strategy_data.get('win_rate', 0) * 0.4 + 
                                strategy_data.get('total_return', 0) * 0.6)
                        
                        all_combinations.append({
                            'year': year,
                            'symbol': symbol,
                            'strategy': strategy,
                            'win_rate': strategy_data.get('win_rate', 0),
                            'return': strategy_data.get('total_return', 0),
                            'trades': strategy_data.get('total_trades', 0),
                            'score': score
                        })
        
        # Sort and display top 10
        all_combinations.sort(key=lambda x: x['score'], reverse=True)
        
        report += f"{'Rank':<4} {'Year':<6} {'Symbol':<8} {'Strategy':<18} {'Win Rate':<10} {'Return':<10} {'Trades':<8} {'Score':<8}\n"
        report += "-" * 90 + "\n"
        
        for i, combo in enumerate(all_combinations[:10]):
            report += f"{i+1:<4} {combo['year']:<6} {combo['symbol']:<8} {combo['strategy']:<18} "
            report += f"{combo['win_rate']:<10.1f} {combo['return']:<10.1f}% {combo['trades']:<8} {combo['score']:<8.1f}\n"
        
        return report

def main():
    """Main execution function"""
    analyzer = ComprehensiveStrategyAnalyzer()
    
    print("Starting comprehensive strategy analysis...")
    print("This may take several minutes to complete...")
    
    # Run analysis
    results = analyzer.run_comprehensive_analysis()
    
    # Generate report
    report = analyzer.generate_comprehensive_report(results)
    print(report)
    
    # Save results
    os.makedirs('data', exist_ok=True)
    
    with open('data/comprehensive_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    with open('data/comprehensive_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("\nAnalysis complete!")
    print("Results saved to:")
    print("- data/comprehensive_analysis_results.json")
    print("- data/comprehensive_analysis_report.txt")

if __name__ == "__main__":
    main()