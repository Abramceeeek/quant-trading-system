# backtesting/backtest_engine.py

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple
import json

class OptionsBacktestEngine:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.trade_history = []
        self.daily_pnl = []
        self.max_drawdown = 0
        self.peak_capital = initial_capital
        
    def load_historical_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical stock price data"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(20).std() * np.sqrt(252)
            return data
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
            return pd.DataFrame()
    
    def simulate_option_prices(self, spot_price: float, strike: float, 
                              days_to_expiry: int, volatility: float,
                              option_type: str = 'call', risk_free_rate: float = 0.02) -> float:
        """
        Simplified Black-Scholes option pricing for backtesting
        """
        try:
            if days_to_expiry <= 0:
                if option_type.lower() == 'call':
                    return max(0, spot_price - strike)
                else:
                    return max(0, strike - spot_price)
            
            # Black-Scholes components
            T = days_to_expiry / 365
            d1 = (np.log(spot_price/strike) + (risk_free_rate + volatility**2/2)*T) / (volatility*np.sqrt(T))
            d2 = d1 - volatility*np.sqrt(T)
            
            from scipy.stats import norm
            
            if option_type.lower() == 'call':
                price = spot_price * norm.cdf(d1) - strike * np.exp(-risk_free_rate*T) * norm.cdf(d2)
            else:
                price = strike * np.exp(-risk_free_rate*T) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
            
            return max(0.01, price)  # Minimum bid of $0.01
        except:
            # Fallback to intrinsic value
            if option_type.lower() == 'call':
                return max(0.01, spot_price - strike)
            else:
                return max(0.01, strike - spot_price)
    
    def backtest_iron_condor_strategy(self, ticker: str, start_date: str, 
                                    end_date: str, dte_entry: int = 30) -> Dict:
        """
        Backtest Iron Condor strategy
        """
        print(f"Backtesting Iron Condor strategy for {ticker}...")
        
        data = self.load_historical_data(ticker, start_date, end_date)
        if data.empty:
            return {"error": "No data available"}
        
        results = {
            'trades': [],
            'total_return': 0,
            'win_rate': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
        
        # Trade every 2 weeks
        trade_dates = pd.date_range(start=start_date, end=end_date, freq='14D')
        
        for trade_date in trade_dates:
            if trade_date not in data.index:
                continue
                
            spot_price = data.loc[trade_date, 'Close']
            volatility = data.loc[trade_date, 'Volatility']
            
            if pd.isna(volatility) or volatility < 0.1:
                continue
            
            # Iron Condor strikes
            put_short_strike = round(spot_price * 0.95, 0)
            put_long_strike = put_short_strike - 5
            call_short_strike = round(spot_price * 1.05, 0)
            call_long_strike = call_short_strike + 5
            
            # Entry prices
            put_short_price = self.simulate_option_prices(spot_price, put_short_strike, dte_entry, volatility, 'put')
            put_long_price = self.simulate_option_prices(spot_price, put_long_strike, dte_entry, volatility, 'put')
            call_short_price = self.simulate_option_prices(spot_price, call_short_strike, dte_entry, volatility, 'call')
            call_long_price = self.simulate_option_prices(spot_price, call_long_strike, dte_entry, volatility, 'call')
            
            net_credit = put_short_price + call_short_price - put_long_price - call_long_price
            max_risk = 5 - net_credit  # 5-point spread
            
            if net_credit <= 0 or max_risk <= 0:
                continue
            
            # Exit date (target 50% of DTE)
            exit_date = trade_date + timedelta(days=int(dte_entry * 0.5))
            if exit_date not in data.index:
                exit_date = min(data.index[data.index > exit_date])
            
            if exit_date >= data.index[-1]:
                continue
            
            exit_spot = data.loc[exit_date, 'Close']
            days_remaining = int(dte_entry * 0.5)
            exit_vol = data.loc[exit_date, 'Volatility']
            
            if pd.isna(exit_vol):
                exit_vol = volatility
            
            # Exit prices
            exit_put_short = self.simulate_option_prices(exit_spot, put_short_strike, days_remaining, exit_vol, 'put')
            exit_put_long = self.simulate_option_prices(exit_spot, put_long_strike, days_remaining, exit_vol, 'put')
            exit_call_short = self.simulate_option_prices(exit_spot, call_short_strike, days_remaining, exit_vol, 'call')
            exit_call_long = self.simulate_option_prices(exit_spot, call_long_strike, days_remaining, exit_vol, 'call')
            
            exit_cost = exit_put_short + exit_call_short - exit_put_long - exit_call_long
            
            # P&L calculation
            pnl = net_credit - exit_cost
            pnl_percent = (pnl / max_risk) * 100
            
            trade = {
                'entry_date': trade_date.strftime('%Y-%m-%d'),
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'spot_entry': round(spot_price, 2),
                'spot_exit': round(exit_spot, 2),
                'net_credit': round(net_credit, 2),
                'exit_cost': round(exit_cost, 2),
                'pnl': round(pnl, 2),
                'pnl_percent': round(pnl_percent, 1),
                'max_risk': round(max_risk, 2),
                'win': pnl > 0
            }
            
            results['trades'].append(trade)
            self.capital += (pnl * 100)  # Assuming 1 contract
            
            # Update drawdown
            if self.capital > self.peak_capital:
                self.peak_capital = self.capital
            
            current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
        
        if results['trades']:
            total_pnl = sum([t['pnl'] for t in results['trades']])
            wins = sum([1 for t in results['trades'] if t['win']])
            
            results['total_return'] = round((self.capital - self.initial_capital) / self.initial_capital * 100, 2)
            results['win_rate'] = round(wins / len(results['trades']) * 100, 1)
            results['max_drawdown'] = round(self.max_drawdown * 100, 2)
            results['avg_pnl'] = round(total_pnl / len(results['trades']), 2)
            results['total_trades'] = len(results['trades'])
        
        return results
    
    def backtest_wheel_strategy(self, ticker: str, start_date: str, end_date: str) -> Dict:
        """
        Backtest Wheel strategy (cash-secured puts + covered calls)
        """
        print(f"Backtesting Wheel strategy for {ticker}...")
        
        data = self.load_historical_data(ticker, start_date, end_date)
        if data.empty:
            return {"error": "No data available"}
        
        results = {
            'trades': [],
            'total_return': 0,
            'win_rate': 0,
            'assignments': 0
        }
        
        cash = self.initial_capital
        shares_owned = 0
        avg_cost_basis = 0
        
        trade_dates = pd.date_range(start=start_date, end=end_date, freq='21D')  # Monthly
        
        for trade_date in trade_dates:
            if trade_date not in data.index:
                continue
            
            spot_price = data.loc[trade_date, 'Close']
            volatility = data.loc[trade_date, 'Volatility']
            
            if pd.isna(volatility):
                continue
            
            if shares_owned == 0:
                # Sell cash-secured put
                put_strike = round(spot_price * 0.92, 0)  # 8% OTM
                put_premium = self.simulate_option_prices(spot_price, put_strike, 30, volatility, 'put')
                
                required_cash = put_strike * 100
                if cash >= required_cash:
                    # Assume assignment if spot goes below strike by expiry
                    expiry_date = trade_date + timedelta(days=30)
                    if expiry_date in data.index:
                        expiry_spot = data.loc[expiry_date, 'Close']
                        
                        if expiry_spot < put_strike:
                            # Assigned - buy shares
                            shares_owned = 100
                            avg_cost_basis = put_strike - put_premium
                            cash -= put_strike * 100
                            cash += put_premium * 100
                            results['assignments'] += 1
                        else:
                            # Keep premium
                            cash += put_premium * 100
                        
                        trade = {
                            'date': trade_date.strftime('%Y-%m-%d'),
                            'type': 'CSP',
                            'strike': put_strike,
                            'premium': round(put_premium, 2),
                            'assigned': expiry_spot < put_strike if expiry_date in data.index else False,
                            'pnl': round(put_premium * 100, 0)
                        }
                        results['trades'].append(trade)
            
            else:
                # Own shares - sell covered call
                call_strike = round(max(spot_price * 1.05, avg_cost_basis * 1.02), 0)
                call_premium = self.simulate_option_prices(spot_price, call_strike, 30, volatility, 'call')
                
                expiry_date = trade_date + timedelta(days=30)
                if expiry_date in data.index:
                    expiry_spot = data.loc[expiry_date, 'Close']
                    
                    if expiry_spot > call_strike:
                        # Called away
                        cash += call_strike * 100 + call_premium * 100
                        pnl = (call_strike - avg_cost_basis + call_premium) * 100
                        shares_owned = 0
                        avg_cost_basis = 0
                    else:
                        # Keep premium and shares
                        cash += call_premium * 100
                        pnl = call_premium * 100
                    
                    trade = {
                        'date': trade_date.strftime('%Y-%m-%d'),
                        'type': 'CC',
                        'strike': call_strike,
                        'premium': round(call_premium, 2),
                        'called_away': expiry_spot > call_strike if expiry_date in data.index else False,
                        'pnl': round(pnl, 0)
                    }
                    results['trades'].append(trade)
        
        # Calculate final results
        total_value = cash + (shares_owned * data.iloc[-1]['Close'])
        
        if results['trades']:
            total_pnl = sum([t['pnl'] for t in results['trades']])
            wins = sum([1 for t in results['trades'] if t['pnl'] > 0])
            
            results['total_return'] = round((total_value - self.initial_capital) / self.initial_capital * 100, 2)
            results['win_rate'] = round(wins / len(results['trades']) * 100, 1)
            results['final_cash'] = round(cash, 0)
            results['shares_owned'] = shares_owned
            results['total_trades'] = len(results['trades'])
        
        return results
    
    def backtest_momentum_strategy(self, ticker: str, start_date: str, end_date: str) -> Dict:
        """
        Backtest momentum call buying strategy
        """
        print(f"Backtesting Momentum strategy for {ticker}...")
        
        data = self.load_historical_data(ticker, start_date, end_date)
        if data.empty:
            return {"error": "No data available"}
        
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        
        results = {'trades': [], 'total_return': 0, 'win_rate': 0}
        
        for i in range(50, len(data)-30):  # Need historical data for indicators
            current_date = data.index[i]
            current_price = data.iloc[i]['Close']
            sma_20 = data.iloc[i]['SMA_20']
            sma_50 = data.iloc[i]['SMA_50']
            rsi = data.iloc[i]['RSI']
            vol = data.iloc[i]['Volatility']
            
            if pd.isna(vol) or pd.isna(rsi):
                continue
            
            # Momentum conditions
            if (current_price > sma_20 * 1.02 and sma_20 > sma_50 and 
                50 < rsi < 70 and vol < 0.4):
                
                # Buy call option
                strike = round(current_price * 1.05, 0)  # 5% OTM
                call_premium = self.simulate_option_prices(current_price, strike, 30, vol, 'call')
                
                # Hold for 2 weeks
                exit_idx = min(i + 14, len(data)-1)
                exit_price = data.iloc[exit_idx]['Close']
                exit_vol = data.iloc[exit_idx]['Volatility']
                
                if pd.isna(exit_vol):
                    exit_vol = vol
                
                exit_call_price = self.simulate_option_prices(exit_price, strike, 16, exit_vol, 'call')
                
                pnl = exit_call_price - call_premium
                pnl_percent = (pnl / call_premium) * 100
                
                trade = {
                    'entry_date': current_date.strftime('%Y-%m-%d'),
                    'exit_date': data.index[exit_idx].strftime('%Y-%m-%d'),
                    'entry_premium': round(call_premium, 2),
                    'exit_premium': round(exit_call_price, 2),
                    'pnl': round(pnl, 2),
                    'pnl_percent': round(pnl_percent, 1),
                    'win': pnl > 0
                }
                
                results['trades'].append(trade)
                self.capital += (pnl * 100)
        
        if results['trades']:
            wins = sum([1 for t in results['trades'] if t['win']])
            results['total_return'] = round((self.capital - self.initial_capital) / self.initial_capital * 100, 2)
            results['win_rate'] = round(wins / len(results['trades']) * 100, 1)
            results['total_trades'] = len(results['trades'])
        
        return results
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def generate_backtest_report(self, results: Dict, strategy_name: str) -> str:
        """Generate a comprehensive backtest report"""
        if 'error' in results:
            return f"Error in {strategy_name}: {results['error']}"
        
        report = f"\n{'='*50}\n"
        report += f"{strategy_name.upper()} BACKTEST RESULTS\n"
        report += f"{'='*50}\n"
        report += f"Total Trades: {results.get('total_trades', 0)}\n"
        report += f"Win Rate: {results.get('win_rate', 0)}%\n"
        report += f"Total Return: {results.get('total_return', 0)}%\n"
        
        if 'max_drawdown' in results:
            report += f"Max Drawdown: {results['max_drawdown']}%\n"
        if 'avg_pnl' in results:
            report += f"Average P&L per trade: ${results['avg_pnl']}\n"
        if 'assignments' in results:
            report += f"Put Assignments: {results['assignments']}\n"
        
        report += f"\nFirst 10 trades:\n"
        for i, trade in enumerate(results.get('trades', [])[:10]):
            report += f"{i+1}: {trade}\n"
        
        return report