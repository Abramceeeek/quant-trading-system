#!/usr/bin/env python3
# validate_improvements.py

"""
Validation script to test all improvements made to the algorithmic trading bot.
This script verifies that all new features and enhancements work correctly.
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all new modules can be imported successfully."""
    print("🔍 Testing imports...")
    
    try:
        from utils.config_manager import ConfigManager
        from utils.error_handler import ErrorHandler, TradingBotError
        print("✅ Utils modules imported successfully")
    except ImportError as e:
        print(f"❌ Error importing utils: {e}")
        return False
    
    try:
        from algo_trading_bot import AlgoTradingBot
        print("✅ Enhanced AlgoTradingBot imported successfully")
    except ImportError as e:
        print(f"❌ Error importing AlgoTradingBot: {e}")
        return False
    
    return True

def test_config_manager():
    """Test the configuration manager functionality."""
    print("\n🔧 Testing ConfigManager...")
    
    try:
        from utils.config_manager import ConfigManager
        
        # Test loading configuration
        config_manager = ConfigManager('src/config/settings.yaml')
        
        # Test basic configuration access
        initial_capital = config_manager.get('initial_capital')
        print(f"✅ Initial capital: ${initial_capital:,}")
        
        # Test nested configuration access
        max_risk = config_manager.get('risk_management.max_portfolio_risk')
        print(f"✅ Max portfolio risk: {max_risk * 100}%")
        
        # Test watchlist symbols
        symbols = config_manager.get_watchlist_symbols()
        print(f"✅ Watchlist symbols: {len(symbols)} symbols")
        
        # Test environment checks
        is_prod = config_manager.is_production()
        print(f"✅ Production mode: {is_prod}")
        
        return True
        
    except Exception as e:
        print(f"❌ ConfigManager test failed: {e}")
        return False

def test_error_handler():
    """Test the error handling system."""
    print("\n🛡️ Testing ErrorHandler...")
    
    try:
        from utils.error_handler import ErrorHandler, TradingBotError
        from utils.config_manager import ConfigManager
        
        config_manager = ConfigManager('src/config/settings.yaml')
        error_handler = ErrorHandler(config_manager)
        
        # Test error handling
        test_error = TradingBotError("Test error for validation")
        error_handler.handle_error(test_error, "validation_test", critical=False, notify=False)
        
        # Test error summary
        summary = error_handler.get_error_summary()
        print(f"✅ Error summary: {summary['total_errors']} total errors")
        
        return True
        
    except Exception as e:
        print(f"❌ ErrorHandler test failed: {e}")
        return False

def test_enhanced_bot():
    """Test the enhanced AlgoTradingBot."""
    print("\n🤖 Testing Enhanced AlgoTradingBot...")
    
    try:
        from algo_trading_bot import AlgoTradingBot
        
        # Initialize bot with enhanced features
        bot = AlgoTradingBot('src/config/settings.yaml')
        
        # Test configuration access
        print(f"✅ Bot initialized with capital: ${bot.initial_capital:,}")
        print(f"✅ Watchlist size: {len(bot.watchlist)} symbols")
        print(f"✅ Enabled strategies: {len(bot.enabled_strategies)}")
        
        # Test error handler integration
        if hasattr(bot, 'error_handler'):
            print("✅ Error handler integrated")
        else:
            print("❌ Error handler not integrated")
            return False
        
        # Test configuration manager integration
        if hasattr(bot, 'config_manager'):
            print("✅ Configuration manager integrated")
        else:
            print("❌ Configuration manager not integrated")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced bot test failed: {e}")
        return False

def test_directory_structure():
    """Test that all required directories are created."""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        'data',
        'data/logs',
        'data/backtest_results',
        'data/daily_results',
        'data/daily_summaries',
        'data/reports'
    ]
    
    all_exist = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ Directory exists: {directory}")
        else:
            print(f"❌ Directory missing: {directory}")
            all_exist = False
    
    return all_exist

def test_configuration_validation():
    """Test configuration validation."""
    print("\n⚠️ Testing configuration validation...")
    
    try:
        from utils.config_manager import ConfigManager
        
        # Test valid configuration
        config_manager = ConfigManager('src/config/settings.yaml')
        print("✅ Valid configuration loaded successfully")
        
        # Test configuration update
        original_capital = config_manager.get('initial_capital')
        config_manager.update_setting('initial_capital', 15000)
        updated_capital = config_manager.get('initial_capital')
        
        if updated_capital == 15000:
            print("✅ Configuration update works")
            # Restore original value
            config_manager.update_setting('initial_capital', original_capital)
        else:
            print("❌ Configuration update failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def test_requirements():
    """Test that key requirements are available."""
    print("\n📦 Testing requirements...")
    
    required_packages = [
        'pandas',
        'numpy', 
        'yfinance',
        'pyyaml',
        'schedule'
    ]
    
    all_available = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} not available")
            all_available = False
    
    return all_available

def generate_validation_report():
    """Generate a comprehensive validation report."""
    print("\n📊 Generating validation report...")
    
    results = {
        'timestamp': str(os.path.getctime(__file__)),
        'tests': {}
    }
    
    # Run all tests
    test_functions = [
        ('imports', test_imports),
        ('config_manager', test_config_manager),
        ('error_handler', test_error_handler),
        ('enhanced_bot', test_enhanced_bot),
        ('directory_structure', test_directory_structure),
        ('configuration_validation', test_configuration_validation),
        ('requirements', test_requirements)
    ]
    
    total_tests = len(test_functions)
    passed_tests = 0
    
    for test_name, test_func in test_functions:
        try:
            success = test_func()
            results['tests'][test_name] = {
                'status': 'PASSED' if success else 'FAILED',
                'success': success
            }
            if success:
                passed_tests += 1
        except Exception as e:
            results['tests'][test_name] = {
                'status': 'ERROR',
                'success': False,
                'error': str(e)
            }
    
    # Summary
    results['summary'] = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': total_tests - passed_tests,
        'success_rate': (passed_tests / total_tests) * 100
    }
    
    # Save report
    os.makedirs('data', exist_ok=True)
    with open('data/validation_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    """Main validation function."""
    print("🚀 Algorithmic Trading Bot - Improvement Validation")
    print("=" * 60)
    
    # Generate validation report
    results = generate_validation_report()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    summary = results['summary']
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    
    if summary['success_rate'] == 100:
        print("\n🎉 ALL TESTS PASSED! The trading bot improvements are working correctly.")
        print("✅ System is ready for use")
    elif summary['success_rate'] >= 80:
        print("\n⚠️ Most tests passed, but some issues detected.")
        print("🔧 Review failed tests and fix issues before production use")
    else:
        print("\n❌ Multiple test failures detected.")
        print("🚨 System needs attention before use")
    
    print(f"\n📄 Detailed report saved to: data/validation_report.json")
    
    # Return exit code based on success rate
    return 0 if summary['success_rate'] == 100 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)