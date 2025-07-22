# tests/test_config_manager.py

import unittest
import tempfile
import os
import yaml
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config_manager import ConfigManager

class TestConfigManager(unittest.TestCase):
    """Test suite for ConfigManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            'environment': 'testing',
            'initial_capital': 5000,
            'live_trading': False,
            'risk_management': {
                'max_portfolio_risk': 0.02,
                'max_position_size': 0.10
            },
            'trading': {
                'min_volume': 100,
                'trading_hours': {
                    'start': '09:30',
                    'end': '16:00'
                }
            },
            'watchlist': {
                'technology': ['AAPL', 'MSFT'],
                'etfs': ['SPY', 'QQQ']
            },
            'enabled_strategies': ['iron_condor', 'wheel'],
            'logging': {
                'level': 'INFO',
                'log_to_file': True,
                'log_file_path': 'data/logs/test.log'
            }
        }
        
        # Create temporary config file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.test_config, self.temp_file, default_flow_style=False)
        self.temp_file.close()
        
        # Create directories for logging
        os.makedirs('data/logs', exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass
    
    def test_config_loading(self):
        """Test basic configuration loading."""
        config_manager = ConfigManager(self.temp_file.name)
        
        self.assertEqual(config_manager.get('environment'), 'testing')
        self.assertEqual(config_manager.get('initial_capital'), 5000)
        self.assertFalse(config_manager.get('live_trading'))
    
    def test_nested_config_access(self):
        """Test accessing nested configuration values."""
        config_manager = ConfigManager(self.temp_file.name)
        
        self.assertEqual(config_manager.get('risk_management.max_portfolio_risk'), 0.02)
        self.assertEqual(config_manager.get('trading.trading_hours.start'), '09:30')
    
    def test_default_values(self):
        """Test default value handling."""
        config_manager = ConfigManager(self.temp_file.name)
        
        self.assertEqual(config_manager.get('nonexistent_key', 'default'), 'default')
        self.assertIsNone(config_manager.get('nonexistent_key'))
    
    def test_watchlist_symbols(self):
        """Test watchlist symbol extraction."""
        config_manager = ConfigManager(self.temp_file.name)
        
        symbols = config_manager.get_watchlist_symbols()
        expected_symbols = ['AAPL', 'MSFT', 'SPY', 'QQQ']
        
        self.assertEqual(set(symbols), set(expected_symbols))
    
    def test_strategy_config(self):
        """Test strategy-specific configuration."""
        config_manager = ConfigManager(self.temp_file.name)
        
        # Test with non-existent strategy
        strategy_config = config_manager.get_strategy_config('nonexistent')
        self.assertEqual(strategy_config, {})
    
    def test_environment_checks(self):
        """Test environment-specific checks."""
        config_manager = ConfigManager(self.temp_file.name)
        
        self.assertFalse(config_manager.is_production())
        self.assertFalse(config_manager.is_live_trading_enabled())
    
    def test_risk_limits(self):
        """Test risk limit retrieval."""
        config_manager = ConfigManager(self.temp_file.name)
        
        risk_limits = config_manager.get_risk_limits()
        self.assertEqual(risk_limits['max_portfolio_risk'], 0.02)
        self.assertEqual(risk_limits['max_position_size'], 0.10)
    
    def test_trading_hours(self):
        """Test trading hours configuration."""
        config_manager = ConfigManager(self.temp_file.name)
        
        trading_hours = config_manager.get_trading_hours()
        self.assertEqual(trading_hours['start'], '09:30')
        self.assertEqual(trading_hours['end'], '16:00')
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test with invalid config
        invalid_config = {
            'initial_capital': 500,  # Too low
            'risk_management': {
                'max_portfolio_risk': 0.15  # Too high
            }
        }
        
        temp_invalid = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(invalid_config, temp_invalid, default_flow_style=False)
        temp_invalid.close()
        
        with self.assertRaises(ValueError):
            ConfigManager(temp_invalid.name)
        
        os.unlink(temp_invalid.name)
    
    def test_config_update(self):
        """Test configuration updates."""
        config_manager = ConfigManager(self.temp_file.name)
        
        # Update a value
        config_manager.update_setting('initial_capital', 15000)
        self.assertEqual(config_manager.get('initial_capital'), 15000)
        
        # Update nested value
        config_manager.update_setting('risk_management.max_portfolio_risk', 0.03)
        self.assertEqual(config_manager.get('risk_management.max_portfolio_risk'), 0.03)
    
    def test_file_not_found(self):
        """Test handling of missing configuration file."""
        with self.assertRaises(FileNotFoundError):
            ConfigManager('nonexistent_file.yaml')
    
    def test_config_reload(self):
        """Test configuration reloading."""
        config_manager = ConfigManager(self.temp_file.name)
        
        # Modify the file
        modified_config = self.test_config.copy()
        modified_config['initial_capital'] = 20000
        
        with open(self.temp_file.name, 'w') as f:
            yaml.dump(modified_config, f, default_flow_style=False)
        
        # Reload and check
        config_manager.reload_config()
        self.assertEqual(config_manager.get('initial_capital'), 20000)

if __name__ == '__main__':
    unittest.main()