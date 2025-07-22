# strategies/advanced_strategies.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedTradingStrategies:
    def __init__(self):
        self.min_iv = 0.15
        self.max_iv = 0.60
        self.min_volume = 100
        self.max_cost_ratio = 0.03
        self.profit_target = 0.50
        self.stop_loss = -0.75
        
    def iron_condor_signal(self, calls: pd.DataFrame, puts: pd.DataFrame, 
                          spot_price: float, expiry_date: str) -> Optional[Dict]:
        """
        Iron Condor strategy: Sell OTM put spread + OTM call spread
        Profits from low volatility and time decay
        """
        try:
            expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
            days_to_expiry = (expiry - datetime.today()).days
            
            if days_to_expiry < 7 or days_to_expiry > 45:
                return None
                
            # Find strikes for iron condor
            otm_put_strike = spot_price * 0.95  # 5% OTM put
            otm_call_strike = spot_price * 1.05  # 5% OTM call
            
            # Find closest strikes
            put_short = puts.loc[(puts['strike'] - otm_put_strike).abs().idxmin()]
            call_short = calls.loc[(calls['strike'] - otm_call_strike).abs().idxmin()]
            
            # Wing strikes (further OTM)
            put_long_strike = put_short['strike'] - 5  # $5 wide
            call_long_strike = call_short['strike'] + 5
            
            put_long = puts.loc[(puts['strike'] - put_long_strike).abs().idxmin()]
            call_long = calls.loc[(calls['strike'] - call_long_strike).abs().idxmin()]
            
            # Check liquidity
            min_volume_check = all([
                put_short['volume'] >= self.min_volume,
                call_short['volume'] >= self.min_volume,
                put_long['volume'] >= self.min_volume/2,
                call_long['volume'] >= self.min_volume/2
            ])
            
            if not min_volume_check:
                return None
                
            # Calculate net credit
            net_credit = (put_short['lastPrice'] + call_short['lastPrice'] - 
                         put_long['lastPrice'] - call_long['lastPrice'])
            
            max_risk = 5 - net_credit  # Wing width - net credit
            
            if net_credit <= 0 or max_risk <= 0:
                return None
                
            return {
                'signal': 'IRON_CONDOR',
                'net_credit': round(net_credit, 2),
                'max_risk': round(max_risk, 2),
                'profit_prob': self._calculate_iron_condor_prob(spot_price, put_short['strike'], call_short['strike']),
                'days_to_expiry': days_to_expiry,
                'legs': {
                    'put_short': {'strike': put_short['strike'], 'symbol': put_short['contractSymbol']},
                    'put_long': {'strike': put_long['strike'], 'symbol': put_long['contractSymbol']},
                    'call_short': {'strike': call_short['strike'], 'symbol': call_short['contractSymbol']},
                    'call_long': {'strike': call_long['strike'], 'symbol': call_long['contractSymbol']}
                }
            }
        except Exception as e:
            return None
    
    def wheel_strategy_signal(self, puts: pd.DataFrame, spot_price: float, 
                             expiry_date: str, cash_available: float) -> Optional[Dict]:
        """
        Wheel strategy: Sell cash-secured puts, if assigned sell covered calls
        """
        try:
            expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
            days_to_expiry = (expiry - datetime.today()).days
            
            if days_to_expiry < 7 or days_to_expiry > 45:
                return None
            
            # Target strike: 5-10% OTM
            target_strike = spot_price * 0.92
            put_option = puts.loc[(puts['strike'] - target_strike).abs().idxmin()]
            
            # Check if we have enough cash
            required_cash = put_option['strike'] * 100
            if cash_available < required_cash:
                return None
                
            # Liquidity and IV checks
            if (put_option['volume'] < self.min_volume or 
                put_option['impliedVolatility'] < self.min_iv):
                return None
                
            # Calculate annualized return
            premium = put_option['lastPrice']
            annual_return = (premium / put_option['strike']) * (365 / days_to_expiry)
            
            if annual_return < 0.12:  # Minimum 12% annualized
                return None
                
            return {
                'signal': 'WHEEL_PUT',
                'strike': put_option['strike'],
                'premium': round(premium, 2),
                'symbol': put_option['contractSymbol'],
                'annual_return': round(annual_return * 100, 1),
                'days_to_expiry': days_to_expiry,
                'required_cash': required_cash
            }
        except Exception as e:
            return None
    
    def volatility_scalping_signal(self, calls: pd.DataFrame, puts: pd.DataFrame,
                                  spot_price: float, historical_vol: float) -> Optional[Dict]:
        """
        Volatility scalping: Buy/sell options based on IV vs realized volatility
        """
        try:
            # Get ATM options
            atm_strike = min(calls['strike'], key=lambda x: abs(x - spot_price))
            atm_call = calls[calls['strike'] == atm_strike].iloc[0]
            atm_put = puts[puts['strike'] == atm_strike].iloc[0]
            
            avg_iv = (atm_call['impliedVolatility'] + atm_put['impliedVolatility']) / 2
            
            # Compare IV to 20-day realized volatility
            vol_ratio = avg_iv / historical_vol
            
            # Signal generation
            if vol_ratio > 1.2:  # IV > 20% above realized vol
                return {
                    'signal': 'SELL_VOLATILITY',
                    'strategy': 'SHORT_STRADDLE',
                    'iv': round(avg_iv, 4),
                    'realized_vol': round(historical_vol, 4),
                    'vol_ratio': round(vol_ratio, 2),
                    'call_symbol': atm_call['contractSymbol'],
                    'put_symbol': atm_put['contractSymbol'],
                    'strike': atm_strike
                }
            elif vol_ratio < 0.8:  # IV < 20% below realized vol
                return {
                    'signal': 'BUY_VOLATILITY',
                    'strategy': 'LONG_STRADDLE',
                    'iv': round(avg_iv, 4),
                    'realized_vol': round(historical_vol, 4),
                    'vol_ratio': round(vol_ratio, 2),
                    'call_symbol': atm_call['contractSymbol'],
                    'put_symbol': atm_put['contractSymbol'],
                    'strike': atm_strike
                }
            return None
        except Exception as e:
            return None
    
    def momentum_breakout_signal(self, ticker: str, calls: pd.DataFrame) -> Optional[Dict]:
        """
        Momentum breakout: Buy calls on strong momentum with technical confirmation
        """
        try:
            # Get recent price data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="60d")
            
            if len(hist) < 50:
                return None
            
            current_price = hist['Close'].iloc[-1]
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
            rsi = self._calculate_rsi(hist['Close'], 14).iloc[-1]
            
            # Momentum conditions
            price_above_sma20 = current_price > sma_20 * 1.02
            sma20_above_sma50 = sma_20 > sma_50
            rsi_bullish = 50 < rsi < 70
            volume_spike = hist['Volume'].iloc[-1] > hist['Volume'].rolling(10).mean().iloc[-1] * 1.5
            
            if not all([price_above_sma20, sma20_above_sma50, rsi_bullish, volume_spike]):
                return None
            
            # Find suitable call option
            target_strike = current_price * 1.05  # 5% OTM
            call_option = calls.loc[(calls['strike'] - target_strike).abs().idxmin()]
            
            if (call_option['volume'] < self.min_volume or 
                call_option['impliedVolatility'] > 0.5):
                return None
            
            return {
                'signal': 'MOMENTUM_CALL',
                'strike': call_option['strike'],
                'symbol': call_option['contractSymbol'],
                'current_price': round(current_price, 2),
                'rsi': round(rsi, 1),
                'iv': round(call_option['impliedVolatility'], 4),
                'premium': call_option['lastPrice']
            }
        except Exception as e:
            return None
    
    def calendar_spread_signal(self, calls: pd.DataFrame, spot_price: float,
                              near_expiry: str, far_expiry: str) -> Optional[Dict]:
        """
        Calendar spread: Sell near-term, buy far-term at same strike
        """
        try:
            # Target ATM strike
            atm_strike = min(calls['strike'], key=lambda x: abs(x - spot_price))
            
            # This would need additional data for far expiry options
            # Simplified implementation
            near_call = calls[calls['strike'] == atm_strike].iloc[0]
            
            near_expiry_dt = datetime.strptime(near_expiry, "%Y-%m-%d")
            days_to_near = (near_expiry_dt - datetime.today()).days
            
            if days_to_near < 7 or days_to_near > 30:
                return None
            
            return {
                'signal': 'CALENDAR_SPREAD',
                'strike': atm_strike,
                'near_symbol': near_call['contractSymbol'],
                'near_premium': near_call['lastPrice'],
                'days_to_near_expiry': days_to_near
            }
        except Exception as e:
            return None
    
    def _calculate_iron_condor_prob(self, spot: float, put_strike: float, call_strike: float) -> float:
        """Calculate probability of profit for iron condor"""
        # Simplified: assume normal distribution
        range_width = call_strike - put_strike
        return min(85.0, max(15.0, (range_width / spot) * 100))
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def get_historical_volatility(self, ticker: str, days: int = 20) -> float:
        """Calculate historical volatility"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{days+10}d")
            returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            return returns.std() * np.sqrt(252)
        except:
            return 0.25  # Default volatility