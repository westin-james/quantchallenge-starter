from enum import Enum
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import math

class Side(Enum):
    BUY = 0
    SELL = 1

@dataclass
class Position:
    curr_active: bool = False
    time_of_entry: Optional[int] = None
    side_of_entry: Side = 0
    quantity_of_position_purchased: int = 0
    cost_of_position: float = 0.0
    
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
    two_pt_attempts: int = 0.0
    two_pt_makes: int = 0.0

    real_offense: float = 0.0
    real_defense: float = 50.0

    minutes_on_floor: float = 0.0
    on_floor: bool = False

    three_pt_attempts: int = 0
    three_pt_makes: int = 0
    free_throw_attempts: int = 0
    free_throws_makes: int = 0

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
    blended_defense_rating: float = 50.0
    blended_overall_rating: float = 0.0

@dataclass
class CurrTeamState:
    roster: Dict[str, CurrPlayerState] = field(default_factory=dict)
    active_lineup: List[str] = field(default_factory=list)
    team_offensive_rating: float = 0.0
    team_defensive_rating: float = 50.0
    team_overall_rating: float = 0.0
    curr_total_points: int = 0
    expected_total_points: int = 0
    num_possessions: int = 0

@dataclass
class CurrGameState:
    home_score: int = 0
    away_score: int = 0
    time_remaining: float = 0.0
    num_possessions_completed: int = 0
    average_possession_length: float = 0.0
    last_possession_start: float = 0

@dataclass
class KellyCriterion:
    fraction: float = 0.40
    cap_percentage_total_capital: float = 0.30
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
    
    def target_units(self, wealth: float, trade_price: float, f: Optional[float] = None) -> float:
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

        self.curr_position = Position()

    def __init__(self) -> None:
        """Initialize strategy"""
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

        self.curr_position = Position()
        
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
        if player.minutes_on_floor >= 10.0:
            player.blended_offense_rating = player.real_offensive_rating
            player.blended_defense_rating = player.real_defensive_rating
        else:
            blend_factor = player.minutes_on_floor / 10.0
            player.blended_offense_rating = (
                team_avg_offense * (1 - blend_factor) +
                player.real_offensive_rating * blend_factor
            )
            player.blended_defense_rating = (
                50.0 * (1 - blend_factor) +
                player.real_defensive_rating * blend_factor
            )
        player.blended_overall_rating = (
            0.7 * player.blended_offense_rating +
            0.3 * player.blended_defense_rating
        )

    def _update_team_ratings(self, team: CurrTeamState) -> None:
        """Update team overall ratings from active lineup"""
        if not team.active_lineup:
            return
        
        active_players = [team.roster[pid] for pid in team.active_lineup if pid in team.roster]
        if not team.active_lineup:
            return
        
        team.team_offensive_rating = sum(p.blended_offense_rating for p in active_players) / len(active_players)
        team.team_defensive_rating = sum(p.blended_defense_rating for p in active_players) / len(active_players)
        team.team_overall_rating = sum(p.blended_overall_rating for p in active_players) / len(active_players)


    def _handle_substitution(self, team: CurrTeamState, event: dict) -> None:
        """Handle player substitution"""
        player_in = event.get('player_name')
        player_out = event.get('substituted_player_name')

        if player_out and player_out in team.active_lineup:
            team.active_lineup.append(player_in)
            team.active_lineup.remove(player_out)
            player_obj = self._get_or_create_player(team, player_in)
            player_obj.on_floor = True
            
    
    def is_possession_ending_event(self, event_type: str) -> bool:
        return event_type in ['SCORE', 'REBOUND', 'TURNOVER']

    def _update_possession_tracking(self, event: dict) -> None:
        """Update possession length tracking"""
        if self.is_possession_ending_event(event.get('event_type', '')):
            current_time = event.get('time_seconds', 0)
            if self.game_state.last_possession_start > current_time:
                possession_length = self.game_state.last_possession_start - current_time

                total_time = self.game_state.num_possessions_completed * self.game_state.average_possession_length
                self.game_state.num_possessions_completed += 1
                self.game_state.average_possession_length = (total_time + possession_length) / self.game_state.num_possessions_completed

                self.game_state.last_possession_start = current_time

    def _calculate_win_probability(self) -> float:
        """calculate home team win probability"""

        remaining_possessions = 0.0
        if self.game_state.average_possession_length > 0:
            remaining_possessions = self.game_state.time_remaining / self.game_state.average_possession_length

        home_ppp = 0.8 + (self.home_team.team_overall_rating / 100.0) * 0.6
        away_ppp = 0.8 + (self.away_team.team_overall_rating / 100) * 0.6

        expected_home = self.game_state.home_score + (remaining_possessions * home_ppp)
        expected_away = self.game_state.away_score + (remaining_possessions * away_ppp)

        point_diff = expected_home - expected_away
        z_score = point_diff / 12.0
        probability = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))

        return max(0.01, min(0.99, probability))

    def _enough_time_elapsed_to_make_new_trade(self, current_time: float) -> bool:
        """Check if enough time has passed since last trade"""
        return (current_time - self.last_trade_time) >= self.min_trade_interval
    
    def calculate_current_edge(self) -> float:
        win_prob = self._calculate_win_probability()
        market_price = self._mid()
        if market_price is None:
            market_price = self.last_trade_price
        if market_price is None:
            return 0.0
        
        market_prob = market_price / 100.00
        market_prob = min(max(market_prob, 1e-9), 1 - 1e-9)
        edge = win_prob - market_prob
        return edge
        
    def should_close_position(self) -> bool:
        if not self.curr_position.curr_active:
            return False
        
        if self.game_state.time_remaining < 250:
            return True
        
        current_edge = self.calculate_current_edge()

        if self.curr_position.side_of_entry == Side.BUY and current_edge < -0.05:
            return True
        
        if self.curr_position.side_of_entry == Side.SELL and current_edge > 0.05:
            return True

        if abs(current_edge) < 0.02:
            return True
        
        return False


    def reset_curr_position(self) -> None:
        self.curr_position.curr_active = False
        self.curr_position.time_of_entry = None
        self.curr_position.side_of_entry = 0
        self.curr_position.quantity_of_position_purchased = 0
        self.curr_position.cost_of_position = 0.0

    def update_curr_position_on_order(self, side: Side, quantity: float, curr_price: float):
        self.curr_position.curr_active = True
        self.curr_position.quantity_of_position_purchased = quantity
        self.curr_position.side_of_entry = side
        self.curr_position.time_of_entry = self.game_state.time_remaining
        self.curr_position.cost_of_position = quantity * curr_price

    def execute_close_position(self) -> None:
        if not self.curr_position.curr_active:
            return
        
        if self.curr_position.side_of_entry == Side.BUY:
            place_market_order(Side.SELL, Ticker.TEAM_A, self.curr_position.quantity_of_position_purchased)

        if self.curr_position.side_of_entry == Side.SELL:
            place_market_order(Side.BUY, Ticker.TEAM_A, self.curr_position.quantity_of_position_purchased)
        
        self.reset_curr_position()
        

    def _execute_trading_decision(self, win_prob: float, current_time: float) -> None:
        """Execute trading decision based on calculated probabilities"""
        if self.game_state.time_remaining > 1500 or self.game_state.time_remaining < 500:
            return
        
        if not self._enough_time_elapsed_to_make_new_trade(current_time):
            return
        
        mid_price = self.last_trade_price
        if mid_price is None or self.capital_remaining is None:
            return
        
        if self.should_close_position():
            self.execute_close_position()
            return
        
        if self.curr_position.curr_active == True:
            return
        
        #Rest of function only possible if there is no currently active position
        kelly_fraction = self.kelly.kelly_fraction(win_prob, mid_price)
        if abs(kelly_fraction) < 0.01:
            return
        
        target_units = self.kelly.target_units(self.capital_remaining, mid_price, kelly_fraction)
        position_change = target_units - self.position

        if abs(position_change) < 1.0:
            return
        
        if position_change > 0:
            trade_price = self._get_tradeable_price(Side.BUY)
            if trade_price is not None:
                quantity = min(abs(position_change), self.capital_remaining / trade_price)
                if quantity >= 1.0:
                    place_market_order(Side.BUY, Ticker.TEAM_A, quantity)
                    self.update_curr_position_on_order(Side.BUY, quantity, trade_price)
                    self.last_trade_time = current_time
        else:
            trade_price = self._get_tradeable_price(Side.SELL)
            if trade_price is not None:
                quantity = min(abs(position_change), abs (self.position) if self.position < 0 else float('inf'))
                if quantity >= 1.0:
                    place_market_order(Side.SELL, Ticker.TEAM_A, quantity)
                    self.update_curr_position_on_order(Side.SELL, quantity, trade_price)
                    self.last_trade_time = current_time

    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        """Called when any trade occurs"""
        self.last_trade_price = price

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        """Called when orderbook changes"""
        book = self.bids if side == Side.BUY else self.asks
        if quantity <= 0.0:
            if price in book:
                del book[price]
        else:
            book[price] = quantity
        self._recompute_bbo()

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        """Called with complete orderbook snapshot"""
        self.bids = {price: qty for price, qty in bids}
        self.asks = {price: qty for price, qty in asks}
        self._recompute_bbo()

    def on_account_update(self,
        ticker: Ticker,
        side: Side,
        price: float,
        quantity: float,
        capital_remaining: float) -> None:
        """Called when our order fills"""
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
                           time_seconds: Optional[float]) -> None:
        """Main game event procession"""

        if event_type == "END_GAME":
            self.reset_state()
            return
        
        if event_type and event_type != "NOTHING":
            self.game_active = True

        self.game_state.home_score = home_score
        self.game_state.away_score = away_score
        if time_seconds is not None:
            self.game_state.time_remaining = time_seconds

        event = {
            'event_type': event_type,
            'home_away': home_away,
            'home_score': home_score,
            'away_score': away_score,
            'player_name': player_name,
            'substituted_player_name': substituted_player_name,
            'shot_type': shot_type,
            'assist_player': assist_player,
            'rebound_type': rebound_type,
            'coordinate_x': coordinate_x,
            'coordinate_y': coordinate_y,
            'time_seconds': time_seconds

        }

        target_team = None
        if home_away == "home":
            target_team = self.home_team
            target_team.curr_total_points = home_score
        elif home_away == "away":
            target_team = self.away_team
            target_team.curr_total_points = away_score

        if event_type == "JUMP_BALL" and target_team and player_name:
            if player_name not in target_team.active_lineup:
                target_team.active_lineup.append(player_name)
                player_obj = self._get_or_create_player(target_team, player_name)
                player_obj.on_floor = True

        for team in [self.home_team, self.away_team]:
            for player_id in team.active_lineup:
                if player_id in team.roster:
                    player = team.roster[player_id]
                    self._update_player_stats(player, event)
                    self.compute_offensive_rating(player)
                    self.compute_defensive_rating(player)

        self._update_possession_tracking(event)

        if self.game_active and time_seconds is not None:
            win_prob = self._calculate_win_probability()
            self._execute_trading_decision(win_prob, time_seconds)
            


    def apply_event_to_player(self, player: CurrPlayerState, event: dict) -> None:

        """
        This takes ingested datapoint that applies to a specific player
        and applies a specific piece of info 
        """

    def compute_offensive_rating(self, player: CurrPlayerState) -> None:
        """
        Use Updated event to recalculate the offensive rating of
        a player
        """
        three_pt_pct = 0.0
        if player.three_pt_attempts > 0:
            three_pt_pct = player.three_pt_makes / player.three_pt_attempts

        total_2pt_attempts = player.two_pt_attempts + player.free_throw_attempts
        two_pt_pct = 0.0
        if total_2pt_attempts > 0:
            total_2pt_makes = player.two_pt_makes + player.free_throws_makes
            two_pt_pct = total_2pt_makes / total_2pt_attempts

        shooting_rating = (0.6 * three_pt_pct + 0.4 * two_pt_pct) * 100

        rebounding_rating = min(player.offensive_rebounds * 20, 100)

        turnover_penalty = min(player.turnovers * 20, 100)
        turnover_rating = max(0, 100 - turnover_penalty)

        player.real_offensive_rating = (
            0.6 * shooting_rating +
            0.3 * rebounding_rating +
            0.1 * turnover_rating
        )

    def compute_defensive_rating(self, player: CurrPlayerState) -> None:
        """
        Use Updated event to recalculate the defensive rating of
        a player
        """
        base = 50.0
        positive = (player.steals + player.blocks + player.defensive_rebounds) * 5.0
        negative = player.defensive_fouls * 10.0

        player.real_defensive_rating = max(0.0, min(100.0, base + positive- negative))

        



    # def compute_overall_rating(self, player: CurrPlayerState) -> None:
    #     """
    #     Use Updated off/def to recalculate the overall rating of
    #     a player
    #     """

    # def update_lineup(self, lineup: List[CurrPlayerState], event: dict) -> None:
    #     """
    #     Set the new lineup after a substitution, change currOverall,
    #     update ratings
    #     """

    # def update_average_possession_length(self, team: CurrGameState, new_possession_length: int) -> None:
    #     """
    #     Update average possesion length variable in CurrGameState
    #     """ 

    # def calculate_expected_remaining_possessions(self, team: CurrGameState, time_remaining: int) -> None:
    #     """
    #     Use the average posession length and time remaining to calculate
    #     """

    # def calculate_average_points_per_possessions(self, team: CurrGameState) -> int: 
    #     """
    #     Use the team points and possessions to calculate
    #     """

    # def calculate_expected_total_points() -> None:
    #     """
    #     Use remaining possessions and average points scored per drive
    #     """

    # def calculate_probability(expectedA: int, expectedB: int, differential = 12.0) -> float:
    #     """
    #     Use calculate expected total points to get them and pass them as parameters
    #     Then, use standard deviation of 12 for nba game point differential
    #     get z-score: expected / differential
    #     get p = phi(z) = 0.5*(1+erf(z / sqrt(2)))
    #     then return p
    #     """

