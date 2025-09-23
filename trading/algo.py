from typing import Dict, Optional, List
from dataclasses import dataclass, field
import math

class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    """"Place a market order."""
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    """Place a limit order."""
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    """Cancel an order."""
    return False

@dataclass
class CurrPlayerState:
    player_id: str

    real_offense: float
    real_defense: float

    minutes_on_floor: float
    on_floor: bool = False

    three_pt_attempts: int = 0
    three_pt_makes: int = 0
    free_throw_attempts: int = 0
    free_throws_makes: int = 0
    two_pt_attempts: int
    two_pt_makes: int
    offensive_rebounds: int = 0
    turnovers: int = 0
    assists: int = 0
    offensive_fouls: int = 0

    defensive_rebounds: int = 0
    steals: int = 0
    blocks: int = 0
    defensive_fouls: int = 0

    real_offensive_rating: float = 0.0
    real_defensive_rating: float = 0.0
    real_overall_rating: float = 0.0

    blended_offense_rating: float = 0.0
    blended_defense_rating: float = 50
    blended_overall_rating: float = 0

@dataclass
class CurrTeamState:
    roster: List[CurrPlayerState] = []
    active_lineup: List[CurrPlayerState] = []
    team_offensive_rating: float = 0.0
    team_defensive_rating: float = 0.0
    team_overall_rating: float = 0.0
    curr_total_points: int = 0
    expected_total_points: int = 0
    num_possessions: int = 0

@dataclass
class CurrGameState:
    team_a_score: int = 0
    team_b_score: int = 0
    time_remeaning: int = 2000
    num_possessions_completed: int = 0
    average_possession_length: float = 0.0

@dataclass
class KellyCriterion:
    fraction: float = 0.33
    cap_percentage_total_capital: float = 0.3
    min_edge: float = 0.05

    def kelly_fraction(self, p_hat: float, trade_price: float) -> float:

        s = trade_price / 100.00
        s = min(max(s, 1e-9), 1 - 1e-9)
        edge = p_hat - s

        if abs(edge) < self.min_edge:
            return 0.0
        
        if edge > 0.0:
            f_full = edge / (1.0 - s)
        else:
            f_full = edge / s

        f = self.fraction * f_full

        cap = self.cap_percentage_total_capital
        return min(f, cap) if f >= 0.0 else max(f, -cap)
    
    def targe_units(self, wealth: float, trade_price: float, f: Optional[float] = None) -> float:
        S = min(max(trade_price, 1e-6), 100.0 - 1e-6)
        if f >= 0.0:
            return (f * wealth) / S
        else:
            return (abs(f) * wealth) / (100.0 - S)


class Strategy:
    """Basketball Trading Strategy with Real-time Analysis"""

    def reset_state(self) -> None:
        """Reset all state variables"""

        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.best_bid: Optional[float] = None
        self.best_ask: Optional[float] = None
        self.last_trade_price: Optional[float] = None

        self.position: float = 0.0
        self.capital_remaining: float = 100000.0

        self.home_team = CurrTeamState()
        self.away_team = CurrTeamState()
        self.game_state = CurrGameState()
        self.kelly = KellyCriterion()

        self.game_active: bool = False
        self.last_trade_time: float = 0.0
        self.min_trade_interval: float = 5.0

    def __init__(self) -> None:
            """Initialize strategy"""
            self.reset_state()
        
    def _recompute_bbo(self) -> None:
            """Recompute best bid and offer"""
            self.best_bid = max(self.bids.keys()) if self.bids else None
            self.best_ask = min(self.asks.keys()) if self.asks else None

    def _mid(self) -> Optional[float]:
            """Calculate mid price"""
            if self.best_bid is not None and self.best_ask is not None:
                return (self.best_bid + self.best_ask) / 2.0
            return None

    def _get_tradeable_price(self, side: Side) -> Optional[float]:
            """Get price we can actually trade at"""
            if side == Side.BUY and self.best_ask is not None:
                return self.best_ask
            elif side == Side.SELL and self.best_bid is not None:
                return self.best_bid
            return None 
        
    def _get_or_create_player(self, team: CurrTeamState, player_name: str) -> CurrPlayerState:
            """Get player from roster or create new one"""
            if player_name not in team.roster:
                team.roster[player_name] = CurrPlayerState(
                    player_id=player_name,
                    real_offense=team.team_offensive_rating or 50.0,
                    real_defense=50.0
                )
            return team.roster[player_name]

    def _update_player_stats(self, player: CurrPlayerState, event: dict) -> None:
            """Update player statistics from event"""
            event_type = event.get('event_type')

            if player.on_floor:
                player.minutes_on_floor += 1.0/60.0
            
            if event.get('player_name') == player.player_id:
                if event_type == 'SCORE':
                    shot_type = event.get('shot_type')
                    if shot_type == 'THREE_POINT':
                        player.three_pt_makes += 1
                        player.three_pt_attempts += 1
                    elif shot_type in ['TWO_POINT', 'LAYUP', 'DUNK']:
                        player.two_pt_makes += 1
                        player.two_pt_attempts += 1
                    elif shot_type == 'FREE_THROW':
                        player.free_throws_makes += 1
                        player.free_throw_attempts += 1

                elif event_type == 'MISSED':
                    shot_type = event.get('shot_type')
                    if shot_type == 'THREE_POINT':
                        player.three_pt_attempts += 1
                    elif shot_type in ['TWO_POINT', 'LAYUP', 'DUNK']:
                        player.two_pt_attempts += 1
                    elif shot_type == 'FREE_THROW':
                        player.free_throw_attempts += 1
                
                elif event_type == 'REBOUND':
                    if event.get('rebound_type') == 'OFFENSIVE':
                        player.offensive_rebounds += 1
                    else:
                        player.defensive_rebounds += 1
                
                elif event_type == 'STEAL':
                    player.steals += 1
                elif event_type == 'BLOCK':
                    player.blocks += 1
                elif event_type == 'TURNOVER':
                    player.turnovers += 1
                elif event_type == 'FOUL':
                    player.defensive_fouls += 1

            
def _blend_ratings(self, player: CurrPlayerState, team_avg_offense: float) -> None:
    """
    This takes the offensive and defensive ratings, along with
    time on the court, to determine the blended rating, until the
    player has played enough players
    """

def apply_event_to_player(player: CurrPlayerState, event: dict) -> None:
    """
    This takes ingested datapoint that applies to a specific player
    and applies a specific piece of info 
    """
    pass

def compute_offensive_rating(player: CurrPlayerState) -> None:
    """
    Use Updated event to recalculate the offensive rating of
    a player
    """

def compute_defensive_rating(player: CurrPlayerState) -> None:
    """
    Use Updated event to recalculate the defensive rating of
    a player
    """

def compute_overall_rating(player: CurrPlayerState) -> None:
    """
    Use Updated off/def to recalculate the overall rating of
    a player
    """
    pass

def update_lineup(lineup: List[CurrPlayerState], event: dict) -> None:
    """
    Set the new lineup after a substitution, change currOverall,
    update ratings
    """
    
def update_average_possession_length(team: CurrGameState, new_possession_length: int) -> None:
    """
    Update average possesion length variable in CurrGameState
    """
    pass

def calculate_expected_remaining_possessions(team: CurrGameState, time_remaining: int) -> None:
    """
    Use the average posession length and time remaining to calculate
    """
    pass

def calculate_average_points_per_possessions(team: CurrGameState) -> int: 
    """
    Use the team points and possessions to calculate
    """
    pass

def calculate_expected_total_points() -> None:
    """
    Use remaining possessions and average points scored per drive
    """
    pass

def calculate_probability(expectedA: int, expectedB: int, differential = 12.0) -> float:
    """
    Use calculate expected total points to get them and pass them as parameters
    Then, use standard deviation of 12 for nba game point differential
    get z-score: expected / differential
    get p = phi(z) = 0.5*(1+erf(z / sqrt(2)))
    then return p
    """
    pass