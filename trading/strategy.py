"""
Quant Challenge 2025

Algorithmic strategy template
"""

from enum import Enum
from typing import Optional, List, Dict

from scraper import GameScraper
import math
import time

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
    def _make_out_path(self) -> str:
        return f"game_debug_{self._game_index:03d}.json"

    def reset_state(self) -> None:
        """Reset the state of the strategy to the start of game position.
        
        Since the sandbox execution can start mid-game, we recommend creating a
        function which can be called from __init__ and on_game_event_update (END_GAME).

        Note: In production execution, the game will start from the beginning
        and will not be replayed.
        """
        if not hasattr(self, "_game_index"):
            self._game_index = 1
        
        if not hasattr(self, "scraper"):
            self.scraper = GameScraper(self._make_out_path())
        self.scraper.start_new_game(self._make_out_path())

        self.home_score = 0
        self.away_score = 0
        self._last_time_seconds = None

        pass

    def __init__(self) -> None:
        """Your initialization code goes here."""
        self._game_index = 1
        self.scraper = GameScraper(self._make_out_path())
        self.reset_state()

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
        print(f"Python Trade update: {ticker} {side} {quantity} shares @ {price}")
        # Log via scraper
        self.scraper.record_generic(
            event_type=f"TRADE_UPDATE {ticker.name} {side.name} q={quantity} p={price}"
        )

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
        # Log orderbook changes
        self.scraper.record_generic(
            event_type=f"ORDERBOOK_UPDATE {ticker.name} {side.name} q={quantity} p={price}"
        )
        pass

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
        # Log fills
        self.scraper.record_generic(
            event_type=(
                f"ACCOUNT_UPDATE {ticker.name} {side.name}"
                f"fill_q={quantity} fill_p={price} cap={capital_remaining}"
            )
        ),
        price=price
        capital_remaining=capital_remaining

        pass

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

        print(f"{event_type} {home_score} - {away_score}")

        # Update local state
        self._home_score = home_score
        self._away_score = away_score
        self._last_time_seconds = time_seconds

        # Record event
        self.scraper.record_game_event(
            home_away=home_away if home_away is not None else "unknown",
            home_score=home_score,
            event_type=event_type,
            player_name=player_name,
            substituted_player_name=substituted_player_name,
            shot_type=shot_type,
            assist_player=assist_player,
            rebound_type=rebound_type,
            coordinate_x=coordinate_x,
            coordinate_y=coordinate_y,
            time_seconds=time_seconds,
        )

        if event_type == "END_GAME":
            # IMPORTANT: Highly recommended to call reset_state() when the
            # game ends. See reset_state() for more details.
            self.scraper.finalize()
            self._game_index += 1
            self.reset_state()
            return
