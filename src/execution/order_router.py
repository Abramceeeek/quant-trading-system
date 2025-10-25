"""
Order routing logic for sending orders to IBKR.
Handles order execution, error handling, and result tracking.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.config import config
from src.execution.ibkr_client import connect, disconnect, place_market_order

logger = logging.getLogger(__name__)


def route_orders(
    orders: dict[str, int],
    dry_run: bool = True,
) -> list[dict[str, Any]]:
    """
    Route orders to IBKR paper/live account.

    Args:
        orders: Dict of {symbol: signed_qty} where qty > 0 = BUY, qty < 0 = SELL
        dry_run: If True, just echo what would be sent without placing orders

    Returns:
        List of result dicts per symbol with keys:
        - symbol: Stock symbol
        - qty: Signed quantity
        - status: Order status or error message
        - orderId: IBKR order ID (None if dry_run or error)

    Example:
        >>> orders = {"AAPL": 100, "MSFT": -50, "SPY": 200}
        >>> results = route_orders(orders, dry_run=False)
        >>> for r in results:
        ...     print(f"{r['symbol']}: {r['status']} (ID={r['orderId']})")
    """
    results: list[dict[str, Any]] = []

    if dry_run:
        logger.info("DRY RUN mode - no orders will be placed")
        # Just echo what would be sent
        for symbol, qty in orders.items():
            action = "BUY" if qty > 0 else "SELL"
            logger.info(f"[DRY RUN] {action} {abs(qty)} {symbol}")
            results.append(
                {
                    "symbol": symbol,
                    "qty": int(qty),
                    "status": "DRY_RUN",
                    "orderId": None,
                }
            )
        return results

    # LIVE ROUTING - Connect to IBKR
    logger.info(f"Connecting to IBKR at {config.ibkr_host}:{config.ibkr_port}")

    try:
        conn = connect(config)
    except Exception as e:
        logger.error(f"Failed to connect to IBKR: {e}")
        # Return error for all orders
        for symbol, qty in orders.items():
            results.append(
                {
                    "symbol": symbol,
                    "qty": int(qty),
                    "status": f"CONNECTION_ERROR: {e}",
                    "orderId": None,
                }
            )
        return results

    # Place orders
    try:
        for symbol, qty in orders.items():
            try:
                # Skip zero quantities
                if qty == 0:
                    logger.info(f"Skipping {symbol} (qty=0)")
                    results.append(
                        {
                            "symbol": symbol,
                            "qty": 0,
                            "status": "SKIPPED",
                            "orderId": None,
                        }
                    )
                    continue

                # Place market order
                order_id, status = place_market_order(
                    conn=conn,
                    symbol=symbol,
                    qty=qty,
                    account=config.ibkr_account,
                )

                results.append(
                    {
                        "symbol": symbol,
                        "qty": int(qty),
                        "status": status,
                        "orderId": order_id,
                    }
                )

                logger.info(f"Order placed: {symbol} qty={qty} orderId={order_id} status={status}")

            except Exception as e:
                logger.error(f"Failed to place order for {symbol}: {e}")
                results.append(
                    {
                        "symbol": symbol,
                        "qty": int(qty),
                        "status": f"ERROR: {e}",
                        "orderId": None,
                    }
                )

    finally:
        # Always disconnect
        disconnect(conn)

    return results
