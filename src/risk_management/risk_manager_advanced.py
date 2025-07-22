# risk_management/risk_manager_advanced.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedRiskManager:
    def __init__(self, initial_capital: float = 10000, max_portfolio_risk: float = 0.02):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_portfolio_risk = max_portfolio_risk  # 2% max risk per trade
        self.max_sector_exposure = 0.25  # 25% max in any sector
        self.max_single_position = 0.10   # 10% max in single position
        self.stop_loss_threshold = -0.05  # 5% portfolio stop loss
        
        self.positions = []
        self.sector_exposures = {}
        self.daily_pnl_history = []
        self.trade_history = []
        
    def calculate_position_size(self, trade_type: str, max_loss: float, 
                              confidence: float = 0.75) -> int:
        """
        Calculate optimal position size using Kelly Criterion and risk management
        """
        # Kelly Criterion: f = (bp - q) / b
        # where b = odds, p = win probability, q = loss probability
        
        if trade_type.lower() in ['iron_condor', 'credit_spread']:
            # For credit strategies
            win_prob = min(0.85, max(0.60, confidence))
            avg_win = 0.25  # 25% of max profit
            avg_loss = 0.75  # 75% of max loss
            
        elif trade_type.lower() in ['long_call', 'long_put', 'momentum']:
            # For debit strategies  
            win_prob = min(0.65, max(0.35, confidence))
            avg_win = 1.0   # 100% gain potential
            avg_loss = 1.0  # 100% loss potential
            
        else:
            # Conservative default
            win_prob = 0.50
            avg_win = 0.50
            avg_loss = 0.50
        
        # Kelly fraction
        kelly_f = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        
        # Conservative Kelly (use 25% of full Kelly)
        conservative_f = max(0.01, min(0.25, kelly_f * 0.25))
        
        # Risk-based position sizing
        risk_capital = self.current_capital * self.max_portfolio_risk
        position_size = int(risk_capital / max_loss)
        
        # Kelly-adjusted size
        kelly_size = int(self.current_capital * conservative_f / max_loss)
        
        # Return the smaller of the two
        return min(position_size, kelly_size, 10)  # Max 10 contracts
    
    def evaluate_trade_risk(self, signal: Dict) -> Dict:
        """
        Comprehensive risk evaluation for a trade signal
        """
        risk_assessment = {
            'approved': True,
            'position_size': 1,
            'risk_score': 0,
            'warnings': [],
            'adjustments': []
        }
        
        signal_type = signal.get('signal', '').lower()
        
        # Calculate basic risk metrics
        if 'max_risk' in signal:
            max_loss = signal['max_risk']
        elif 'premium' in signal:
            max_loss = signal['premium']
        else:
            max_loss = 100  # Default conservative estimate
        
        # Position sizing
        confidence = signal.get('profit_prob', 65) / 100
        position_size = self.calculate_position_size(signal_type, max_loss, confidence)
        risk_assessment['position_size'] = position_size
        
        # Portfolio risk check
        total_risk = max_loss * position_size
        portfolio_risk_pct = total_risk / self.current_capital
        
        if portfolio_risk_pct > self.max_portfolio_risk:
            risk_assessment['position_size'] = max(1, int(self.max_portfolio_risk * self.current_capital / max_loss))
            risk_assessment['adjustments'].append(f"Position size reduced due to portfolio risk limit")
        
        # Volatility risk assessment
        if 'iv' in signal:
            iv = signal['iv']
            if iv > 0.50:
                risk_assessment['risk_score'] += 15
                risk_assessment['warnings'].append(f"High IV: {iv:.2%}")
            elif iv < 0.15:
                risk_assessment['risk_score'] += 10
                risk_assessment['warnings'].append(f"Low IV may indicate low premium: {iv:.2%}")
        
        # Time decay risk
        if 'days_to_expiry' in signal:
            dte = signal['days_to_expiry']
            if dte < 7:
                risk_assessment['risk_score'] += 20
                risk_assessment['warnings'].append(f"High gamma risk - only {dte} days to expiry")
            elif dte > 60:
                risk_assessment['risk_score'] += 10
                risk_assessment['warnings'].append(f"Long time to expiry - {dte} days")
        
        # Liquidity risk
        if 'volume' in signal and signal['volume'] < 100:
            risk_assessment['risk_score'] += 25
            risk_assessment['warnings'].append(f"Low volume: {signal['volume']}")
        
        # Strategy-specific risk checks
        if signal_type == 'iron_condor':
            profit_prob = signal.get('profit_prob', 0)
            if profit_prob < 60:
                risk_assessment['risk_score'] += 15
                risk_assessment['warnings'].append(f"Low profit probability: {profit_prob}%")
        
        elif signal_type in ['momentum_call', 'long_call']:
            if 'rsi' in signal and signal['rsi'] > 80:
                risk_assessment['risk_score'] += 20
                risk_assessment['warnings'].append(f"Overbought RSI: {signal['rsi']}")
        
        # Final approval decision
        if risk_assessment['risk_score'] > 50:
            risk_assessment['approved'] = False
            risk_assessment['warnings'].append("Trade rejected - excessive risk score")
        
        return risk_assessment
    
    def monitor_portfolio_risk(self) -> Dict:
        """
        Monitor overall portfolio risk metrics
        """
        portfolio_metrics = {
            'total_exposure': 0,
            'sector_concentrations': {},
            'var_95': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'alerts': []
        }
        
        if not self.positions:
            return portfolio_metrics
        
        # Calculate total exposure
        total_exposure = sum([pos.get('max_risk', 0) * pos.get('quantity', 1) for pos in self.positions])
        portfolio_metrics['total_exposure'] = total_exposure
        
        # Check sector concentrations
        for sector, exposure in self.sector_exposures.items():
            sector_pct = exposure / self.current_capital
            portfolio_metrics['sector_concentrations'][sector] = sector_pct
            
            if sector_pct > self.max_sector_exposure:
                portfolio_metrics['alerts'].append(f"High {sector} exposure: {sector_pct:.1%}")
        
        # Calculate VaR (95% confidence)
        if len(self.daily_pnl_history) >= 20:
            daily_returns = np.array(self.daily_pnl_history[-252:])  # Last year
            portfolio_metrics['var_95'] = np.percentile(daily_returns, 5) * self.current_capital
            
            # Sharpe ratio
            if np.std(daily_returns) > 0:
                portfolio_metrics['sharpe_ratio'] = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        
        # Maximum drawdown
        if len(self.daily_pnl_history) > 0:
            cumulative_returns = np.cumsum(self.daily_pnl_history)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns) / self.current_capital
            portfolio_metrics['max_drawdown'] = np.max(drawdown)
        
        return portfolio_metrics
    
    def dynamic_hedge_signals(self, current_positions: List[Dict]) -> List[Dict]:
        """
        Generate dynamic hedging signals based on portfolio Greeks and risk
        """
        hedge_signals = []
        
        if not current_positions:
            return hedge_signals
        
        # Calculate portfolio Greeks
        portfolio_delta = sum([pos.get('delta', 0) * pos.get('quantity', 1) for pos in current_positions])
        portfolio_gamma = sum([pos.get('gamma', 0) * pos.get('quantity', 1) for pos in current_positions])
        portfolio_vega = sum([pos.get('vega', 0) * pos.get('quantity', 1) for pos in current_positions])
        
        # Delta hedging
        if abs(portfolio_delta) > 10:  # Delta threshold
            hedge_signals.append({
                'type': 'DELTA_HEDGE',
                'action': 'SELL' if portfolio_delta > 0 else 'BUY',
                'quantity': abs(int(portfolio_delta / 100)),
                'reason': f"Portfolio delta: {portfolio_delta:.1f}",
                'priority': 'HIGH' if abs(portfolio_delta) > 25 else 'MEDIUM'
            })
        
        # Vega hedging for high volatility exposure
        if abs(portfolio_vega) > 50:
            hedge_signals.append({
                'type': 'VEGA_HEDGE',
                'action': 'SELL_VOL' if portfolio_vega > 0 else 'BUY_VOL',
                'exposure': portfolio_vega,
                'reason': f"High vega exposure: {portfolio_vega:.1f}",
                'priority': 'MEDIUM'
            })
        
        # Gamma hedging for high convexity
        if abs(portfolio_gamma) > 5:
            hedge_signals.append({
                'type': 'GAMMA_HEDGE',
                'exposure': portfolio_gamma,
                'reason': f"High gamma exposure: {portfolio_gamma:.2f}",
                'priority': 'LOW'
            })
        
        return hedge_signals
    
    def stress_test_portfolio(self, scenarios: List[Dict]) -> Dict:
        """
        Stress test the portfolio under various market scenarios
        """
        stress_results = {
            'scenarios': [],
            'worst_case_loss': 0,
            'probability_of_ruin': 0,
            'recommendations': []
        }
        
        default_scenarios = [
            {'name': 'Market Crash -20%', 'price_change': -0.20, 'vol_change': 0.50},
            {'name': 'Flash Crash -10%', 'price_change': -0.10, 'vol_change': 0.30},
            {'name': 'Low Vol Grind +5%', 'price_change': 0.05, 'vol_change': -0.30},
            {'name': 'High Vol Rally +15%', 'price_change': 0.15, 'vol_change': 0.20}
        ]
        
        scenarios = scenarios or default_scenarios
        
        for scenario in scenarios:
            scenario_pnl = 0
            
            for position in self.positions:
                # Simplified P&L calculation based on scenario
                pos_pnl = self.calculate_scenario_pnl(position, scenario)
                scenario_pnl += pos_pnl
            
            scenario_result = {
                'name': scenario['name'],
                'pnl': round(scenario_pnl, 0),
                'pnl_pct': round(scenario_pnl / self.current_capital * 100, 1)
            }
            stress_results['scenarios'].append(scenario_result)
            
            if scenario_pnl < stress_results['worst_case_loss']:
                stress_results['worst_case_loss'] = scenario_pnl
        
        # Probability of ruin (simplified)
        worst_case_pct = stress_results['worst_case_loss'] / self.current_capital
        if worst_case_pct < -0.15:  # >15% loss
            stress_results['probability_of_ruin'] = min(25.0, abs(worst_case_pct) * 100)
            stress_results['recommendations'].append("Consider reducing position sizes")
            
        if worst_case_pct < -0.25:  # >25% loss
            stress_results['recommendations'].append("URGENT: Portfolio at risk of significant loss")
        
        return stress_results
    
    def calculate_scenario_pnl(self, position: Dict, scenario: Dict) -> float:
        """
        Calculate P&L for a position under a specific scenario
        """
        # Simplified scenario P&L - would need actual Greeks in practice
        position_type = position.get('type', '')
        max_risk = position.get('max_risk', 100)
        quantity = position.get('quantity', 1)
        
        price_change = scenario.get('price_change', 0)
        vol_change = scenario.get('vol_change', 0)
        
        if 'short' in position_type.lower():
            # Short positions benefit from time and vol crush
            pnl = max_risk * 0.5 * quantity  # Simplified
            if abs(price_change) > 0.10:  # Large moves hurt short positions
                pnl -= max_risk * quantity
        else:
            # Long positions
            if price_change > 0 and 'call' in position_type.lower():
                pnl = max_risk * price_change * 3 * quantity  # Simplified delta
            elif price_change < 0 and 'put' in position_type.lower():
                pnl = max_risk * abs(price_change) * 3 * quantity
            else:
                pnl = -max_risk * 0.5 * quantity  # Time decay
        
        return pnl
    
    def get_risk_report(self) -> str:
        """
        Generate comprehensive risk report
        """
        portfolio_metrics = self.monitor_portfolio_risk()
        
        report = f"\n{'='*60}\n"
        report += f"PORTFOLIO RISK REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        report += f"{'='*60}\n"
        
        report += f"Portfolio Value: ${self.current_capital:,.0f}\n"
        report += f"Total Exposure: ${portfolio_metrics['total_exposure']:,.0f}\n"
        report += f"Exposure Ratio: {portfolio_metrics['total_exposure']/self.current_capital:.1%}\n\n"
        
        if portfolio_metrics['var_95'] != 0:
            report += f"1-Day VaR (95%): ${portfolio_metrics['var_95']:,.0f}\n"
            report += f"Max Drawdown: {portfolio_metrics['max_drawdown']:.1%}\n"
            report += f"Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}\n\n"
        
        # Sector exposures
        if portfolio_metrics['sector_concentrations']:
            report += "Sector Exposures:\n"
            for sector, exposure in portfolio_metrics['sector_concentrations'].items():
                report += f"  {sector}: {exposure:.1%}\n"
            report += "\n"
        
        # Alerts
        if portfolio_metrics['alerts']:
            report += "⚠️  RISK ALERTS:\n"
            for alert in portfolio_metrics['alerts']:
                report += f"  • {alert}\n"
            report += "\n"
        
        # Active positions
        report += f"Active Positions: {len(self.positions)}\n"
        
        return report