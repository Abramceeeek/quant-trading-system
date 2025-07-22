# utils/config_manager.py

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """
    Centralized configuration management for the algorithmic trading bot.
    Handles loading, validation, and environment-specific settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default path.
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Load environment-specific overrides if they exist
            env = config.get('environment', 'development')
            env_config_path = self.config_path.parent / f'settings_{env}.yaml'
            
            if env_config_path.exists():
                with open(env_config_path, 'r') as file:
                    env_config = yaml.safe_load(file)
                    config = self._merge_configs(config, env_config)
            
            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
    
    def _merge_configs(self, base_config: Dict, env_config: Dict) -> Dict:
        """Recursively merge environment-specific configuration with base configuration."""
        for key, value in env_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                base_config[key] = self._merge_configs(base_config[key], value)
            else:
                base_config[key] = value
        return base_config
    
    def _validate_config(self):
        """Validate configuration settings."""
        required_keys = [
            'initial_capital', 'enabled_strategies', 'watchlist',
            'risk_management', 'trading'
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Validate risk management settings
        risk_config = self.config['risk_management']
        if risk_config['max_portfolio_risk'] > 0.10:  # 10% max
            raise ValueError("max_portfolio_risk cannot exceed 10%")
        
        if risk_config['max_position_size'] > 0.20:  # 20% max
            raise ValueError("max_position_size cannot exceed 20%")
        
        # Validate initial capital
        if self.config['initial_capital'] < 1000:
            raise ValueError("initial_capital must be at least $1,000")
    
    def _setup_logging(self):
        """Setup logging based on configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        # Create logs directory if it doesn't exist
        log_file_path = log_config.get('log_file_path', 'data/logs/algo_bot.log')
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path),
                logging.StreamHandler()
            ]
        )
    
    def get(self, key: str, default=None) -> Any:
        """Get configuration value by key using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_watchlist_symbols(self) -> list:
        """Get flattened list of all watchlist symbols."""
        watchlist = self.config.get('watchlist', {})
        symbols = []
        
        if isinstance(watchlist, dict):
            for sector, sector_symbols in watchlist.items():
                if isinstance(sector_symbols, list):
                    symbols.extend(sector_symbols)
        elif isinstance(watchlist, list):
            symbols = watchlist
        
        return list(set(symbols))  # Remove duplicates
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for a specific strategy."""
        strategy_settings = self.config.get('strategy_settings', {})
        return strategy_settings.get(strategy_name, {})
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.config.get('environment') == 'production'
    
    def is_live_trading_enabled(self) -> bool:
        """Check if live trading is enabled."""
        return self.config.get('live_trading', False) and self.is_production()
    
    def get_risk_limits(self) -> Dict[str, float]:
        """Get all risk management limits."""
        return self.config.get('risk_management', {})
    
    def get_trading_hours(self) -> Dict[str, str]:
        """Get trading hours configuration."""
        return self.config.get('trading', {}).get('trading_hours', {})
    
    def reload_config(self):
        """Reload configuration from file."""
        self.config = self._load_config()
        self._validate_config()
    
    def save_config(self, config_path: Optional[str] = None):
        """Save current configuration to file."""
        if config_path is None:
            config_path = self.config_path
        
        with open(config_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
    
    def update_setting(self, key: str, value: Any):
        """Update a configuration setting using dot notation."""
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        # Validate after update
        self._validate_config()
    
    def __str__(self) -> str:
        """String representation of current configuration."""
        return f"ConfigManager(environment={self.config.get('environment')}, " \
               f"live_trading={self.config.get('live_trading')}, " \
               f"strategies={len(self.config.get('enabled_strategies', []))})"