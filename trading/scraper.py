"""
Quant Challenge 2025

Algorithmic strategy template
"""

from enum import Enum
from typing import Optional, Dict
import json

class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    # TEAM_A (home team)
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    """Place a market order.
    
    Parameters
    ----------
    side
        Side of order to place
    ticker
        Ticker of order to place
    quantity
        Quantity of order to place
    """
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    """Place a limit order.
    
    Parameters
    ----------
    side
        Side of order to place
    ticker
        Ticker of order to place
    quantity
        Quantity of order to place
    price
        Price of order to place
    ioc
        Immediate or cancel flag (FOK)

    Returns
    -------
    order_id
        Order ID of order placed
    """
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    """Cancel an order.
    
    Parameters
    ----------
    ticker
        Ticker of order to cancel
    order_id
        Order ID of order to cancel

    Returns
    -------
    success
        True if order was cancelled, False otherwise
    """
    return 0

class Strategy:
    """Template for a strategy."""

    def reset_state(self) -> None:
        """Reset the state of the strategy to the start of game position.
        
        Since the sandbox execution can start mid-game, we recommend creating a
        function which can be called from __init__ and on_game_event_update (END_GAME).

        Note: In production execution, the game will start from the beginning
        and will not be replayed.
        """
        # Market state
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.best_bids: Optional[float] = None
        self.best_ask: Optional[float] = None
        self.last_trade_price: Optional[float] = None
        
        # Account state
        self.position: float = 0.0
        self.capital_remaining: Optional[float] = None

        # Game state
        self.game_active: bool = False

    def __init__(self) -> None:
        """Your initialization code goes here."""
        self.reset_state()

    # ---------------------------- Helpers ---------------------------- #
    def _recompute_bbo(self) -> None:
        self.best_bid = max(self.bids.keys()) if self.bids else None
        self.best_ask = min(self.asks.keys()) if self.asks else None
    
    def _mid(self) -> None:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None
    
    def _price_for_timestamp(self) -> Optional[float]:
        m = self._mid()
        if m is not None:
            return m
        return self.last_trade_price

    # ---------------------------- Exchange callbacks ---------------------------- #
    def on_trade_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever two orders match. Could be one of your orders, or two other people's orders.
        Parameters
        ----------
        ticker
            Ticker of orders that were matched
        side:
            Side of orders that were matched
        quantity
            Volume traded
        price
            Price that trade was executed at
        """
        self.last_trade_price = price

    def on_orderbook_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever the orderbook changes. This could be because of a trade, or because of a new order, or both.
        Parameters
        ----------
        ticker
            Ticker that has an orderbook update
        side
            Which orderbook was updated
        price
            Price of orderbook that has an update
        quantity
            Volume placed into orderbook
        """
        book = self.bids if side == Side.BUY else self.asks
        if quantity <= 0.0:
            if price in book:
                del book[price]
        else:
            book[price] = quantity
        self._recompute_bbo()

    def on_account_update(
        self,
        ticker: Ticker,
        side: Side,
        price: float,
        quantity: float,
        capital_remaining: float,
    ) -> None:
        """Called whenever one of your orders is filled.
        Parameters
        ----------
        ticker
            Ticker of order that was fulfilled
        side
            Side of order that was fulfilled
        price
            Price that order was fulfilled at
        quantity
            Volume of order that was fulfilled
        capital_remaining
            Amount of capital after fulfilling order
        """
        if side == Side.BUY:
            self.position += quantity
        else:
            self.position -= quantity
        self.capital_remaining = capital_remaining

    def on_game_event_update(self,
                           event_type: str,
                           home_away: str,
                           home_score: int,
                           away_score: int,
                           player_name: Optional[str],
                           substituted_player_name: Optional[str],
                           shot_type: Optional[str],
                           assist_player: Optional[str],
                           rebound_type: Optional[str],
                           coordinate_x: Optional[float],
                           coordinate_y: Optional[float],
                           time_seconds: Optional[float]
        ) -> None:
        """Called whenever a basketball game event occurs.
        Parameters
        ----------
        event_type
            Type of event that occurred
        home_score
            Home team score after event
        away_score
            Away team score after event
        player_name (Optional)
            Player involved in event
        substituted_player_name (Optional)
            Player being substituted out
        shot_type (Optional)
            Type of shot
        assist_player (Optional)
            Player who made the assist
        rebound_type (Optional)
            Type of rebound
        coordinate_x (Optional)
            X coordinate of shot location in feet
        coordinate_y (Optional)
            Y coordinate of shot location in feet
        time_seconds (Optional)
            Game time remaining in seconds
        """
        if event_type and event_type != "END_GAME":
            self.game_active = True

        tick = {
            # Base event data
            "home_away": home_away if home_away is not None else "unknown",
            "home_score": home_score,
            "away_score": away_score,
            "event_type": event_type,
            "player_name": player_name,
            "substituted_player_name": substituted_player_name,
            "shot_type": shot_type,
            "assist_player": assist_player,
            "rebound_type": rebound_type,
            "coordinate_x": coordinate_x,
            "coordinate_y": coordinate_y,
            "time_seconds": time_seconds,

            # Market snapshot
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self._mid(),
            "spread": self._spread(),
            "last_trade_price": self.last_trade_price,
            "price": self._price_for_timestamp(),

            # Account snapshot
            "float": self.capital_remaining,
            "position": self.position,
        }

        print(json.dumps(tick, separators=(",", ":")))

        if event_type == "END_GAME":
            # IMPORTANT: Highly recommended to call reset_state() when the
            # game ends. See reset_state() for more details.
            self.reset_state()
            return

