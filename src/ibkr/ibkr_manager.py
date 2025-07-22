# ibkr/ibkr_manager.py

from ib_insync import IB, Option, Stock, MarketOrder, LimitOrder, util
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class IBKRManager:
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self.connected = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Position tracking
        self.positions = {}
        self.orders = {}
        self.account_info = {}
        
    def connect(self, timeout: int = 10) -> bool:
        """Connect to IBKR TWS/Gateway"""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=timeout)
            self.connected = True
            self.logger.info(f"✅ Connected to IBKR at {self.host}:{self.port}")
            
            # Get account information
            self.update_account_info()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to IBKR: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from IBKR"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            self.logger.info("Disconnected from IBKR")
    
    def update_account_info(self):
        """Update account information"""
        try:
            account_values = self.ib.accountValues()
            for av in account_values:
                self.account_info[av.tag] = av.value
            
            self.logger.info(f"Account updated: {self.account_info.get('NetLiquidation', 'N/A')}")
        except Exception as e:
            self.logger.error(f"Failed to update account info: {e}")
    
    def get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            self.update_account_info()
            return float(self.account_info.get('NetLiquidation', '0'))
        except:
            return 0.0
    
    def get_buying_power(self) -> float:
        """Get available buying power"""
        try:
            self.update_account_info()
            return float(self.account_info.get('BuyingPower', '0'))
        except:
            return 0.0
    
    def create_option_contract(self, symbol: str, expiry: str, strike: float, 
                              option_type: str, exchange: str = 'SMART') -> Option:
        """Create option contract"""
        return Option(
            symbol=symbol.upper(),
            lastTradeDateOrContractMonth=expiry.replace('-', ''),
            strike=float(strike),
            right=option_type.upper(),
            exchange=exchange,
            currency='USD',
            multiplier='100'
        )
    
    def create_stock_contract(self, symbol: str, exchange: str = 'SMART') -> Stock:
        """Create stock contract"""
        return Stock(
            symbol=symbol.upper(),
            exchange=exchange,
            currency='USD'
        )
    
    def get_option_chain(self, symbol: str, expiry: str = None) -> Dict:
        """Get real-time option chain from IBKR"""
        try:
            stock = self.create_stock_contract(symbol)
            self.ib.qualifyContracts(stock)
            
            # Get market data for underlying
            self.ib.reqMktData(stock, '', False, False)
            self.ib.sleep(2)
            
            ticker = self.ib.ticker(stock)
            spot_price = ticker.marketPrice()
            
            if not expiry:
                # Get nearest expiry
                chains = self.ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
                if chains:
                    expiry = min(chains[0].expirations)
            
            # Request option chain
            strikes = []
            option_contracts = []
            
            # Generate strikes around current price
            base_strike = round(spot_price / 5) * 5  # Round to nearest $5
            for i in range(-10, 11):  # 20 strikes around current price
                strike = base_strike + (i * 5)
                if strike > 0:
                    strikes.append(strike)
            
            # Create contracts for calls and puts
            calls = []
            puts = []
            
            for strike in strikes:
                call_contract = self.create_option_contract(symbol, expiry, strike, 'C')
                put_contract = self.create_option_contract(symbol, expiry, strike, 'P')
                
                calls.append(call_contract)
                puts.append(put_contract)
            
            # Qualify contracts
            all_contracts = calls + puts
            self.ib.qualifyContracts(*all_contracts)
            
            # Request market data
            for contract in all_contracts:
                self.ib.reqMktData(contract, '', False, False)
            
            self.ib.sleep(3)  # Wait for data
            
            # Collect data
            call_data = []
            put_data = []
            
            for call in calls:
                ticker = self.ib.ticker(call)
                if ticker.bid and ticker.ask:
                    call_data.append({
                        'strike': call.strike,
                        'contractSymbol': f"{symbol}{expiry.replace('-', '')}{call.right}{int(call.strike):05d}000",
                        'lastPrice': (ticker.bid + ticker.ask) / 2,
                        'bid': ticker.bid,
                        'ask': ticker.ask,
                        'volume': ticker.volume or 0,
                        'impliedVolatility': getattr(ticker, 'impliedVolatility', 0) or 0.25
                    })
            
            for put in puts:
                ticker = self.ib.ticker(put)
                if ticker.bid and ticker.ask:
                    put_data.append({
                        'strike': put.strike,
                        'contractSymbol': f"{symbol}{expiry.replace('-', '')}{put.right}{int(put.strike):05d}000",
                        'lastPrice': (ticker.bid + ticker.ask) / 2,
                        'bid': ticker.bid,
                        'ask': ticker.ask,
                        'volume': ticker.volume or 0,
                        'impliedVolatility': getattr(ticker, 'impliedVolatility', 0) or 0.25
                    })
            
            return {
                'calls': pd.DataFrame(call_data),
                'puts': pd.DataFrame(put_data),
                'spot_price': spot_price,
                'expiry': expiry
            }
            
        except Exception as e:
            self.logger.error(f"Error getting option chain: {e}")
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'spot_price': 0, 'expiry': ''}
    
    def place_market_order(self, contract, action: str, quantity: int) -> Optional[object]:
        """Place market order"""
        try:
            order = MarketOrder(action.upper(), abs(quantity))
            trade = self.ib.placeOrder(contract, order)
            
            # Wait for order to be submitted
            self.ib.sleep(2)
            
            self.logger.info(f"Market order placed: {action} {quantity} {contract.symbol}")
            return trade
            
        except Exception as e:
            self.logger.error(f"Error placing market order: {e}")
            return None
    
    def place_limit_order(self, contract, action: str, quantity: int, 
                         limit_price: float) -> Optional[object]:
        """Place limit order"""
        try:
            order = LimitOrder(action.upper(), abs(quantity), limit_price)
            trade = self.ib.placeOrder(contract, order)
            
            self.ib.sleep(2)
            
            self.logger.info(f"Limit order placed: {action} {quantity} {contract.symbol} @ ${limit_price}")
            return trade
            
        except Exception as e:
            self.logger.error(f"Error placing limit order: {e}")
            return None
    
    def execute_iron_condor(self, signal: Dict) -> Dict:
        """Execute Iron Condor strategy"""
        try:
            symbol = signal.get('symbol', 'SPY')
            expiry = signal.get('expiry', '')
            legs = signal.get('legs', {})
            
            results = {'status': 'pending', 'trades': [], 'errors': []}
            
            # Create contracts
            put_short_contract = self.create_option_contract(
                symbol, expiry, legs['put_short']['strike'], 'P')
            put_long_contract = self.create_option_contract(
                symbol, expiry, legs['put_long']['strike'], 'P')
            call_short_contract = self.create_option_contract(
                symbol, expiry, legs['call_short']['strike'], 'C')
            call_long_contract = self.create_option_contract(
                symbol, expiry, legs['call_long']['strike'], 'C')
            
            # Qualify all contracts
            contracts = [put_short_contract, put_long_contract, call_short_contract, call_long_contract]
            self.ib.qualifyContracts(*contracts)
            
            # Execute trades
            put_short_trade = self.place_limit_order(put_short_contract, 'SELL', 1, legs['put_short'].get('price', 1.0))
            put_long_trade = self.place_limit_order(put_long_contract, 'BUY', 1, legs['put_long'].get('price', 0.5))
            call_short_trade = self.place_limit_order(call_short_contract, 'SELL', 1, legs['call_short'].get('price', 1.0))
            call_long_trade = self.place_limit_order(call_long_contract, 'BUY', 1, legs['call_long'].get('price', 0.5))
            
            trades = [put_short_trade, put_long_trade, call_short_trade, call_long_trade]
            results['trades'] = [t.order.orderId if t else None for t in trades]
            
            # Check execution status
            successful_trades = sum([1 for t in trades if t is not None])
            if successful_trades == 4:
                results['status'] = 'executed'
                self.logger.info(f"Iron Condor executed successfully for {symbol}")
            else:
                results['status'] = 'partial'
                results['errors'].append(f"Only {successful_trades}/4 legs executed")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error executing Iron Condor: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def execute_wheel_strategy(self, signal: Dict) -> Dict:
        """Execute Wheel strategy (cash-secured put or covered call)"""
        try:
            if signal['signal'] == 'WHEEL_PUT':
                # Sell cash-secured put
                contract = self.create_option_contract(
                    signal.get('symbol', 'SPY'),
                    signal.get('expiry', ''),
                    signal['strike'],
                    'P'
                )
                
                self.ib.qualifyContracts(contract)
                trade = self.place_limit_order(contract, 'SELL', 1, signal['premium'])
                
                return {
                    'status': 'executed' if trade else 'failed',
                    'trade_id': trade.order.orderId if trade else None,
                    'strategy': 'cash_secured_put'
                }
            
        except Exception as e:
            self.logger.error(f"Error executing Wheel strategy: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            positions = self.ib.positions()
            position_list = []
            
            for pos in positions:
                position_list.append({
                    'symbol': pos.contract.symbol,
                    'position': pos.position,
                    'avgCost': pos.avgCost,
                    'marketPrice': pos.marketPrice,
                    'marketValue': pos.marketValue,
                    'pnl': pos.unrealizedPNL,
                    'contract_type': pos.contract.secType
                })
            
            return position_list
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    def monitor_positions(self, check_interval: int = 300) -> None:
        """Monitor positions for exit conditions"""
        self.logger.info("Starting position monitoring...")
        
        while self.connected:
            try:
                positions = self.get_positions()
                
                for pos in positions:
                    # Check for profit/loss thresholds
                    if pos['contract_type'] == 'OPT':
                        pnl_percent = pos['pnl'] / (abs(pos['avgCost']) * abs(pos['position']) * 100)
                        
                        # Take profit at 50%
                        if pnl_percent > 0.50:
                            self.logger.info(f"Taking profit on {pos['symbol']} - P&L: {pnl_percent:.1%}")
                            # Add closing order logic here
                        
                        # Stop loss at -200%
                        elif pnl_percent < -2.0:
                            self.logger.warning(f"Stop loss triggered on {pos['symbol']} - P&L: {pnl_percent:.1%}")
                            # Add closing order logic here
                
                time.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in position monitoring: {e}")
                time.sleep(60)
    
    def get_market_hours(self) -> Dict:
        """Check if market is open"""
        try:
            now = datetime.now()
            # Simplified market hours check (Eastern Time)
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            is_weekday = now.weekday() < 5
            is_market_hours = market_open <= now <= market_close
            
            return {
                'is_open': is_weekday and is_market_hours,
                'next_open': market_open + timedelta(days=1) if not (is_weekday and is_market_hours) else market_open,
                'next_close': market_close if is_weekday and is_market_hours else None
            }
            
        except Exception as e:
            self.logger.error(f"Error checking market hours: {e}")
            return {'is_open': False}
    
    def health_check(self) -> Dict:
        """Perform health check of IBKR connection"""
        health_status = {
            'connected': self.connected,
            'account_balance': self.get_account_balance(),
            'buying_power': self.get_buying_power(),
            'positions_count': len(self.get_positions()),
            'timestamp': datetime.now().isoformat()
        }
        
        return health_status
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()