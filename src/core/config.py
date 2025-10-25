"""
Core configuration management using Pydantic Settings.
Reads from environment variables and .env file.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Global configuration for the quant trading system."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # IBKR Connection
    ibkr_host: str = Field(default="127.0.0.1", description="IBKR Gateway/TWS host")
    ibkr_port: int = Field(default=7497, description="IBKR port (7497=TWS paper, 7496=live)")
    ibkr_client_id: int = Field(default=9001, description="IBKR client ID")
    ibkr_account: str = Field(default="DU1234567", description="IBKR account number")

    # Trading Mode
    live_trading: bool = Field(default=False, description="Enable live trading")
    paper_trading: bool = Field(default=True, description="Enable paper trading")

    # Data & Caching
    data_dir: Path = Field(default=Path("./data"), description="Data directory")
    cache_enabled: bool = Field(default=True, description="Enable data caching")
    cache_expiry_days: int = Field(default=7, description="Cache expiry in days")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    log_to_file: bool = Field(default=True, description="Log to file")

    # Risk Limits (Production)
    max_daily_loss_pct: float = Field(
        default=0.05, description="Max daily loss as fraction of starting equity"
    )
    max_position_size_pct: float = Field(
        default=0.10, description="Max single position size as fraction of portfolio"
    )
    max_portfolio_risk_pct: float = Field(
        default=0.02, description="Max portfolio risk per trade"
    )

    # Notifications (Optional)
    slack_webhook_url: str | None = Field(default=None, description="Slack webhook URL")
    email_alerts_to: str | None = Field(default=None, description="Email for alerts")

    @property
    def cache_dir(self) -> Path:
        """Return cache directory path."""
        return self.data_dir / "cache"

    @property
    def universes_dir(self) -> Path:
        """Return universes directory path."""
        return self.data_dir / "universes"

    @property
    def runtime_dir(self) -> Path:
        """Return runtime directory path."""
        return Path("./runtime")

    @property
    def reports_dir(self) -> Path:
        """Return reports directory path."""
        return Path("./reports")

    def validate_live_trading(self) -> None:
        """Validate that live trading is properly configured."""
        if self.live_trading and self.paper_trading:
            raise ValueError("Cannot enable both live_trading and paper_trading simultaneously")
        if self.live_trading and self.ibkr_port == 7497:
            raise ValueError(
                "Live trading enabled but IBKR_PORT=7497 (paper). Use 7496 for live TWS."
            )


# Global config instance
config = Config()
