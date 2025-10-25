"""
IBKR client wrapper for order execution using ib_insync.
Handles connection, order placement, and status tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ib_insync import IB, LimitOrder, MarketOrder, Stock

from src.core.config import Config

logger = logging.getLogger(__name__)


@dataclass
class IBKRConnection:
    """Wrapper for IBKR connection."""

    ib: IB
    account: str


def connect(cfg: Config, timeout: float = 10.0) -> IBKRConnection:
    """
    Connect to IBKR Gateway/TWS using configuration.

    Args:
        cfg: Configuration object with IBKR connection details
        timeout: Connection timeout in seconds

    Returns:
        IBKRConnection object

    Raises:
        RuntimeError: If connection fails
    """
    ib = IB()

    try:
        ib.connect(
            host=cfg.ibkr_host,
            port=cfg.ibkr_port,
            clientId=cfg.ibkr_client_id,
            timeout=timeout,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to connect to IBKR: {e}")

    if not ib.isConnected():
        raise RuntimeError("IBKR connection failed - not connected")

    logger.info(
        f"Connected to IBKR: {cfg.ibkr_host}:{cfg.ibkr_port} "
        f"(clientId={cfg.ibkr_client_id})"
    )

    return IBKRConnection(ib=ib, account=cfg.ibkr_account)


def ensure_connected(conn: IBKRConnection) -> None:
    """
    Verify connection is still active.

    Args:
        conn: IBKR connection

    Raises:
        RuntimeError: If connection is lost
    """
    if not conn.ib.isConnected():
        raise RuntimeError("IBKR connection lost")


def stock_contract(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Stock:
    """
    Create a stock contract for IBKR.

    Args:
        symbol: Stock ticker symbol
        exchange: Exchange (default: SMART for best execution)
        currency: Currency (default: USD)

    Returns:
        Stock contract
    """
    return Stock(symbol=symbol, exchange=exchange, currency=currency)


def place_market_order(
    conn: IBKRConnection,
    symbol: str,
    qty: int,
    account: str | None = None,
) -> tuple[int, str]:
    """
    Place a market order for a stock.

    Args:
        conn: IBKR connection
        symbol: Stock ticker symbol
        qty: Quantity (positive for BUY, negative for SELL)
        account: Optional account override

    Returns:
        Tuple of (orderId, status_text)

    Raises:
        RuntimeError: If contract qualification or order placement fails
    """
    ensure_connected(conn)

    # Determine side
    side = "BUY" if qty > 0 else "SELL"
    abs_qty = abs(int(qty))

    logger.info(f"Placing market order: {side} {abs_qty} {symbol}")

    # Create and qualify contract
    contract = stock_contract(symbol)
    qualified = conn.ib.qualifyContracts(contract)

    if not qualified:
        raise RuntimeError(f"Could not qualify contract for {symbol}")

    contract = qualified[0]

    # Create market order
    order = MarketOrder(action=side, totalQuantity=abs_qty)

    # Set account if specified
    if account:
        order.account = account
    elif conn.account:
        order.account = conn.account

    # Place order
    try:
        trade = conn.ib.placeOrder(contract, order)

        # Wait briefly for initial status
        conn.ib.sleep(0.5)

        # Get order ID and status
        order_id = trade.order.orderId
        status = trade.orderStatus.status if trade.orderStatus else "Submitted"

        logger.info(f"Order placed: {symbol} {side} {abs_qty} - OrderID={order_id}, Status={status}")

        return order_id, status

    except Exception as e:
        logger.error(f"Failed to place order for {symbol}: {e}")
        raise RuntimeError(f"Order placement failed for {symbol}: {e}")


def place_limit_order(
    conn: IBKRConnection,
    symbol: str,
    qty: int,
    limit_price: float,
    account: str | None = None,
) -> tuple[int, str]:
    """
    Place a limit order for a stock.

    Args:
        conn: IBKR connection
        symbol: Stock ticker symbol
        qty: Quantity (positive for BUY, negative for SELL)
        limit_price: Limit price
        account: Optional account override

    Returns:
        Tuple of (orderId, status_text)

    Raises:
        RuntimeError: If contract qualification or order placement fails
    """
    ensure_connected(conn)

    # Determine side
    side = "BUY" if qty > 0 else "SELL"
    abs_qty = abs(int(qty))

    logger.info(f"Placing limit order: {side} {abs_qty} {symbol} @ ${limit_price:.2f}")

    # Create and qualify contract
    contract = stock_contract(symbol)
    qualified = conn.ib.qualifyContracts(contract)

    if not qualified:
        raise RuntimeError(f"Could not qualify contract for {symbol}")

    contract = qualified[0]

    # Create limit order
    order = LimitOrder(action=side, totalQuantity=abs_qty, lmtPrice=limit_price)

    # Set account if specified
    if account:
        order.account = account
    elif conn.account:
        order.account = conn.account

    # Place order
    try:
        trade = conn.ib.placeOrder(contract, order)

        # Wait briefly for initial status
        conn.ib.sleep(0.5)

        # Get order ID and status
        order_id = trade.order.orderId
        status = trade.orderStatus.status if trade.orderStatus else "Submitted"

        logger.info(
            f"Order placed: {symbol} {side} {abs_qty} @ ${limit_price:.2f} - "
            f"OrderID={order_id}, Status={status}"
        )

        return order_id, status

    except Exception as e:
        logger.error(f"Failed to place order for {symbol}: {e}")
        raise RuntimeError(f"Order placement failed for {symbol}: {e}")


def disconnect(conn: IBKRConnection) -> None:
    """
    Disconnect from IBKR.

    Args:
        conn: IBKR connection
    """
    try:
        if conn.ib.isConnected():
            conn.ib.disconnect()
            logger.info("Disconnected from IBKR")
    except Exception as e:
        logger.warning(f"Error during disconnect: {e}")
