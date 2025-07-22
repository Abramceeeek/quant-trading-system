# 🚀 Advanced Algorithmic Trading Bot

A comprehensive, production-ready algorithmic trading system with advanced risk management, error handling, and automated execution capabilities.

## ✨ Key Features

### 🔧 **Enhanced Infrastructure**
- **Centralized Configuration Management**: YAML-based configuration with environment-specific overrides
- **Comprehensive Error Handling**: Custom exceptions, automatic error recovery, and notification system  
- **Advanced Logging**: Structured logging with file rotation and multiple output formats
- **Robust Testing Suite**: Comprehensive unit tests with mocking and fixtures

### 📊 **Trading Strategies**
- **Iron Condor**: Market-neutral strategy for range-bound markets
- **Wheel Strategy**: Income generation through cash-secured puts and covered calls
- **Volatility Scalping**: Trade volatility discrepancies between IV and realized volatility
- **Momentum Trading**: Trend-following strategy with technical indicators

### 🛡️ **Risk Management**
- **Kelly Criterion Position Sizing**: Mathematically optimal position sizing
- **Dynamic Risk Limits**: Real-time portfolio risk monitoring and adjustment
- **Stop-Loss Mechanisms**: Automatic position closure on adverse moves
- **Sector Exposure Limits**: Prevent over-concentration in single sectors
- **Daily Loss Limits**: Maximum daily loss thresholds with emergency shutdown

### 🔗 **Broker Integration**
- **Interactive Brokers (IBKR)**: Full API integration with TWS/Gateway
- **Real-time Data**: Live market data and option chains
- **Order Management**: Advanced order types and execution algorithms
- **Position Monitoring**: Real-time P&L tracking and risk metrics

### 📈 **Analytics & Reporting**
- **Comprehensive Backtesting**: Historical strategy performance analysis
- **Performance Metrics**: Sharpe ratio, maximum drawdown, win rates
- **Real-time Monitoring**: Live dashboard with key metrics
- **Automated Reporting**: Daily, weekly, and monthly performance reports

## 🏗️ **Architecture Overview**

```
algorithmic_trading_bot/
├── src/
│   ├── algo_trading_bot.py          # Main bot orchestrator
│   ├── config/
│   │   └── settings.yaml            # Configuration file
│   ├── strategies/
│   │   └── advanced_strategies.py   # Trading strategy implementations
│   ├── risk_management/
│   │   └── risk_manager_advanced.py # Risk management system
│   ├── ibkr/
│   │   └── ibkr_manager.py         # IBKR API integration
│   ├── modules/
│   │   ├── options_data.py         # Market data utilities
│   │   ├── signal_generator.py     # Signal generation
│   │   └── greeks.py              # Options Greeks calculations
│   ├── utils/
│   │   ├── config_manager.py       # Configuration management
│   │   └── error_handler.py        # Error handling system
│   └── backtesting/
│       └── backtest_engine.py      # Backtesting framework
├── tests/                          # Comprehensive test suite
├── data/                          # Data storage and logs
├── live_trading_system.py         # Live trading execution
├── main.py                        # Main entry point
└── requirements.txt               # Dependencies
```

## 🚀 **Quick Start**

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd algorithmic-trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `src/config/settings.yaml` to customize your trading parameters:

```yaml
# Capital Settings
initial_capital: 10000
live_trading: false  # Set to true for live trading

# Risk Management
risk_management:
  max_portfolio_risk: 0.02    # 2% max risk per trade
  max_position_size: 0.10     # 10% max position size
  max_daily_loss: 0.05        # 5% daily loss limit

# Watchlist
watchlist:
  technology: [AAPL, MSFT, NVDA, GOOGL, META]
  etfs: [SPY, QQQ, IWM]
  
# Strategies
enabled_strategies:
  - iron_condor
  - wheel
  - volatility_scalping
  - momentum
```

### 3. Running the Bot

#### **Backtesting Mode** (Recommended First)
```bash
python main.py
```

#### **Paper Trading**
```bash
# Set paper_trading: true in config
python live_trading_system.py
```

#### **Live Trading** (Production)
```bash
# Requirements:
# 1. IBKR TWS or Gateway running
# 2. API permissions enabled
# 3. Set live_trading: true in config
# 4. Set environment: production in config

python live_trading_system.py
```

## 📊 **Configuration Guide**

### **Environment Settings**
```yaml
environment: development  # development | production | testing
initial_capital: 10000
live_trading: false
paper_trading: true
```

### **Risk Management**
```yaml
risk_management:
  max_portfolio_risk: 0.02      # 2% max risk per trade
  max_position_size: 0.10       # 10% max single position
  max_daily_loss: 0.05          # 5% daily loss limit
  max_sector_exposure: 0.25     # 25% max sector exposure
  stop_loss_threshold: -0.05    # 5% portfolio stop loss
  profit_target_multiplier: 2.0 # Risk:reward ratio
```

### **Trading Hours**
```yaml
trading:
  trading_hours:
    start: "09:30"
    end: "16:00"
    timezone: "America/New_York"
  max_daily_trades: 10
```

### **Strategy Configuration**
```yaml
strategy_settings:
  iron_condor:
    min_days_to_expiry: 7
    max_days_to_expiry: 45
    wing_width: 5
    otm_percentage: 0.05
  wheel:
    put_delta_target: 0.30
    call_delta_target: 0.30
    min_premium: 0.5
```

## 🧪 **Testing**

### **Run All Tests**
```bash
python -m pytest tests/ -v --cov=src
```

### **Run Specific Test Categories**
```bash
# Configuration tests
python -m pytest tests/test_config_manager.py -v

# Strategy tests  
python -m pytest tests/test_strategies.py -v

# Bot integration tests
python -m pytest tests/test_algo_trading_bot.py -v
```

## 📈 **Monitoring & Alerts**

### **Email Notifications**
```yaml
notifications:
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    username: "your_email@gmail.com"
    password: "your_app_password"
    to_addresses: ["alerts@yourcompany.com"]
```

### **Slack Integration**
```yaml
notifications:
  slack:
    enabled: true
    webhook_url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
```

## 🛡️ **Security Best Practices**

### **Environment Variables**
Create a `.env` file for sensitive information:
```
IBKR_USERNAME=your_username
IBKR_PASSWORD=your_password
EMAIL_PASSWORD=your_email_app_password
SLACK_WEBHOOK=your_slack_webhook_url
```

### **API Security**
- Use paper trading account for testing
- Implement IP whitelisting in IBKR
- Use read-only API keys when possible
- Regular password rotation

## 📊 **Performance Metrics**

The bot tracks comprehensive performance metrics:

- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Average Trade**: Mean profit/loss per trade
- **Volatility**: Standard deviation of returns

## 🔧 **Troubleshooting**

### **Common Issues**

1. **IBKR Connection Failed**
   ```
   Error: Failed to connect to IBKR
   Solution: Ensure TWS/Gateway is running and API is enabled
   ```

2. **Missing Market Data**
   ```
   Error: No option chain data
   Solution: Check market hours and data subscriptions
   ```

3. **Configuration Errors**
   ```
   Error: Invalid configuration
   Solution: Validate YAML syntax and required fields
   ```

### **Debug Mode**
Enable detailed logging:
```yaml
logging:
  level: DEBUG
  log_to_file: true
```

## 📋 **Roadmap**

### **Upcoming Features**
- [ ] Machine Learning signal generation
- [ ] Multi-broker support (TD Ameritrade, E*TRADE)
- [ ] Real-time web dashboard
- [ ] Mobile app for monitoring
- [ ] Cloud deployment options
- [ ] Advanced portfolio optimization
- [ ] Social sentiment analysis integration

## ⚠️ **Risk Disclaimer**

**IMPORTANT**: This software is for educational and research purposes only. Algorithmic trading involves substantial risk and may not be suitable for all investors. Past performance does not guarantee future results.

- Options trading carries significant risk of loss
- Automated systems can malfunction
- Market conditions can change rapidly
- Always test thoroughly in paper trading first
- Never risk more than you can afford to lose
- Consult with a financial advisor before live trading

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 **Support**

- 📧 Email: support@tradingbot.com
- 💬 Discord: [Trading Bot Community](https://discord.gg/tradingbot)
- 📖 Documentation: [Wiki](https://github.com/your-repo/wiki)
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)

---

**⚡ Ready to start algorithmic trading? Follow the Quick Start guide above!**