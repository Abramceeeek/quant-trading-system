# utils/error_handler.py

import logging
import traceback
import functools
from typing import Any, Callable, Optional, Dict
from datetime import datetime
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

class TradingBotError(Exception):
    """Base exception for trading bot errors."""
    pass

class ConfigurationError(TradingBotError):
    """Raised when there's a configuration error."""
    pass

class ConnectionError(TradingBotError):
    """Raised when there's a connection error to broker or data source."""
    pass

class InsufficientFundsError(TradingBotError):
    """Raised when there are insufficient funds for a trade."""
    pass

class RiskLimitExceededError(TradingBotError):
    """Raised when a trade would exceed risk limits."""
    pass

class DataError(TradingBotError):
    """Raised when there's an error with market data."""
    pass

class OrderExecutionError(TradingBotError):
    """Raised when there's an error executing an order."""
    pass

class ErrorHandler:
    """
    Centralized error handling and notification system.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the error handler.
        
        Args:
            config_manager: Configuration manager instance for notification settings
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.error_count = 0
        self.critical_errors = []
        
    def handle_error(self, error: Exception, context: str = "", 
                    critical: bool = False, notify: bool = True) -> None:
        """
        Handle an error with logging and optional notifications.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            critical: Whether this is a critical error
            notify: Whether to send notifications
        """
        self.error_count += 1
        
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'critical': critical,
            'traceback': traceback.format_exc()
        }
        
        # Log the error
        log_message = f"Error in {context}: {error_info['error_type']} - {error_info['error_message']}"
        
        if critical:
            self.logger.critical(log_message)
            self.critical_errors.append(error_info)
            
            # Stop trading if critical error
            if self.config_manager:
                self.config_manager.update_setting('live_trading', False)
                self.logger.critical("Live trading disabled due to critical error")
        else:
            self.logger.error(log_message)
        
        # Send notifications if enabled
        if notify and self.config_manager:
            self._send_notifications(error_info)
    
    def _send_notifications(self, error_info: Dict) -> None:
        """Send error notifications via configured channels."""
        notifications_config = self.config_manager.get('notifications', {})
        
        if not notifications_config.get('enabled', False):
            return
        
        # Email notification
        if notifications_config.get('email', {}).get('enabled', False):
            self._send_email_notification(error_info, notifications_config['email'])
        
        # Slack notification
        if notifications_config.get('slack', {}).get('enabled', False):
            self._send_slack_notification(error_info, notifications_config['slack'])
    
    def _send_email_notification(self, error_info: Dict, email_config: Dict) -> None:
        """Send email notification for error."""
        if not EMAIL_AVAILABLE:
            self.logger.warning("Email notifications not available - missing email dependencies")
            return
            
        try:
            msg = MimeMultipart()
            msg['From'] = email_config['username']
            msg['To'] = ', '.join(email_config['to_addresses'])
            msg['Subject'] = f"Trading Bot Error: {error_info['error_type']}"
            
            body = f"""
            An error occurred in the algorithmic trading bot:
            
            Error Type: {error_info['error_type']}
            Message: {error_info['error_message']}
            Context: {error_info['context']}
            Critical: {error_info['critical']}
            Timestamp: {error_info['timestamp']}
            
            Traceback:
            {error_info['traceback']}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info("Error notification sent via email")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
    
    def _send_slack_notification(self, error_info: Dict, slack_config: Dict) -> None:
        """Send Slack notification for error."""
        try:
            import requests
            
            message = {
                "text": f"🚨 Trading Bot Error: {error_info['error_type']}",
                "attachments": [
                    {
                        "color": "danger" if error_info['critical'] else "warning",
                        "fields": [
                            {
                                "title": "Error Message",
                                "value": error_info['error_message'],
                                "short": False
                            },
                            {
                                "title": "Context",
                                "value": error_info['context'],
                                "short": True
                            },
                            {
                                "title": "Critical",
                                "value": str(error_info['critical']),
                                "short": True
                            },
                            {
                                "title": "Timestamp",
                                "value": error_info['timestamp'],
                                "short": True
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(slack_config['webhook_url'], json=message)
            response.raise_for_status()
            
            self.logger.info("Error notification sent via Slack")
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
    
    def get_error_summary(self) -> Dict:
        """Get summary of all errors."""
        return {
            'total_errors': self.error_count,
            'critical_errors': len(self.critical_errors),
            'recent_critical': self.critical_errors[-5:] if self.critical_errors else []
        }
    
    def reset_error_count(self) -> None:
        """Reset error counters."""
        self.error_count = 0
        self.critical_errors = []

def error_handler(context: str = "", critical: bool = False, 
                 reraise: bool = True, default_return: Any = None):
    """
    Decorator for automatic error handling.
    
    Args:
        context: Context description for the error
        critical: Whether errors in this function are critical
        reraise: Whether to reraise the exception after handling
        default_return: Value to return if error occurs and reraise=False
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Try to get error handler from first argument (usually self)
                error_handler_instance = None
                if args and hasattr(args[0], 'error_handler'):
                    error_handler_instance = args[0].error_handler
                elif args and hasattr(args[0], 'config_manager'):
                    error_handler_instance = ErrorHandler(args[0].config_manager)
                else:
                    error_handler_instance = ErrorHandler()
                
                error_handler_instance.handle_error(
                    e, 
                    context or f"{func.__module__}.{func.__name__}",
                    critical
                )
                
                if reraise:
                    raise
                else:
                    return default_return
        
        return wrapper
    return decorator

def validate_trading_preconditions(func: Callable) -> Callable:
    """
    Decorator to validate trading preconditions before executing trades.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Assume first argument is self with config_manager
        if args and hasattr(args[0], 'config_manager'):
            config_manager = args[0].config_manager
            
            # Check if trading is enabled
            if not config_manager.get('live_trading', False):
                raise TradingBotError("Live trading is disabled")
            
            # Check trading hours
            trading_hours = config_manager.get_trading_hours()
            if trading_hours:
                current_time = datetime.now().strftime('%H:%M')
                if not (trading_hours['start'] <= current_time <= trading_hours['end']):
                    raise TradingBotError("Outside trading hours")
        
        return func(*args, **kwargs)
    
    return wrapper