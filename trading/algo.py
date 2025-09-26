from enum import Enum
from typing import Dict, Optional, List, Tuple
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
    # player_id: str
    # two_pt_attempts: int = 0
    # two_pt_makes: int = 0

    # real_offense: float = 0.0
    # real_defense: float = 50.0

    # minutes_on_floor: float = 0.0
    # on_floor: bool = False

    # three_pt_attempts: int = 0
    # three_pt_makes: int = 0
    # free_throw_attempts: int = 0
    # free_throws_makes: int = 0

    # offensive_rebounds: int = 0
    # turnovers: int = 0
    # assists: int = 0
    # offensive_fouls: int = 0

    # defensive_rebounds: int = 0
    # steals: int = 0
    # blocks: int = 0
    # defensive_fouls: int = 0

    # real_offensive_rating: float = 0.0
    # real_defensive_rating: float = 0.0
    # real_overall_rating: float = 0.0

    # blended_offense_rating: float = 0.0
    # blended_defense_rating: float = 50.0
    # blended_overall_rating: float = 0.0
    pass

@dataclass
class CurrTeamState:
    # roster: Dict[str, CurrPlayerState] = field(default_factory=dict)
    # active_lineup: List[str] = field(default_factory=list)
    # team_offensive_rating: float = 0.0
    # team_defensive_rating: float = 50.0
    # team_overall_rating: float = 0.0
    curr_total_points: int = 0
    expected_total_points: int = 0
    num_possessions: int = 0

    fgm: int = 0
    tpm: int = 0
    fga: int = 0
    fta: int = 0
    tov: int = 0
    orb: int = 0
    drb: int = 0
    fouls: int = 0

    efg_live: float = 0.5
    tov_pct_live: float = 0.12
    orb_pct_live: float = 0.25
    ftr_live: float = 0.2

    efg_smooth: float = 0.5
    tov_pct_smooth: float = 0.25
    ftr_smooth: float = 0.2

    team_weight: float = 0.5

@dataclass
class CurrGameState:
    home_score: int = 0
    away_score: int = 0
    time_remaining: float = 0.0
    num_possessions_completed: int = 0
    average_possession_length: float = 0.0
    last_possession_start: float = 0
    possessions_seen: int = 0

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
        result = min(f, cap) if f >= 0.0 else max(f, -cap)
        return result
    
    def target_units(self, wealth: float, trade_price: float, f: Optional[float] = None) -> float:
        S = min(max(trade_price, 1e-6), 100.0 - 1e-6)
        if f >= 0.0:
            return (f * wealth) / S
        else:
            return -((abs(f) * wealth) / (100.0 - S))


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

        self.peak_edge_abs_since_entry = 0.0
        self.scale_out_stage = 0
        # endgame helpers (no-OT locks)
        self._initalize_orderbook()

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

        self._initalize_orderbook()

        self.peak_edge_abs_since_entry: float = 0.0
        self.scale_out_stage: int = 0

        self.micro_edge_active: bool = False
        self.micro_edge_team: str = ""
        self.micro_edge_event_type: str = ""
        self.micro_edge_position_size: float = 0.0
        self.micro_edge_entry_possession: int = 0
        self.MICRO_EDGE_SIZE_PCT = 0.02
        self.MICRO_EDGE_MIN_THRESHOLD = 0.06
        self.MICRO_EDGE_STEAL_BOOST = 0.07
        self.MICRO_EDGE_ORB_BOOST = 0.04

        # endgame helpers
        self.last_endgame_action_tick = None
        
    def _initialize_orderbook(self) -> None:
        for price in range(1, 100):
            self.bids[float(price)] = 10000.0
            self.asks[float(price)] = 10000.0

        self._recompute_bbo()
        self.last_trade_price = 50.0

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
        
    # def _get_or_create_player(self, team: CurrTeamState, player_name: str) -> CurrPlayerState:
    #         """Get player from roster or create new one"""
    #         if player_name not in team.roster:
    #             team.roster[player_name] = CurrPlayerState(
    #                 player_id=player_name,
    #                 real_offense=team.team_offensive_rating or 50.0,
    #                 real_defense=50.0
    #             )
    #         return team.roster[player_name]
    
    # def _ensure_player_on_floor(self, team: CurrTeamState, player_name: Optional[str]) -> None:
    #     if not team or not player_name:
    #         return
    #     p = self._get_or_create_player(team, player_name)
    #     p.on_floor = True
    #     if player_name not in team.active_lineup:
    #         team.active_lineup.append(player_name)

    def _update_team_box_scores(self, event: dict) -> None:
        """Update player statistics from event"""
        event_type = event.get('event_type')
        shot_type = event.get('shot_type')
        rebound_type = event.get('rebound_type')
        home_away = event.get('home_away')
        team = self.home_team if home_away == 'home' else self.away_team if home_away == 'away' else None
        opp = self.away_team if home_away == 'home' else self.home_team if home_away == 'away' else None
        if team is None:
            return
        
        if event_type == 'SCORE':
            if shot_type == 'THREE_POINT':
                team.tpm += 1
                team.fgm += 1
                team.fga += 1
            elif shot_type in ['TWO_POINT', 'LAYUP', 'DUNK']:
                team.fgm += 1
                team.fga += 1
            elif shot_type == 'FREE_THROW':
                team.fta += 1
        elif event_type == 'MISSED':
            if shot_type in ['THREE_POINT','TWO_POINT', 'LAYUP', 'DUNK']:
                team.fga += 1
            elif shot_type == 'FREE_THROW':
                team.fta += 1
        elif event_type == 'TURNOVER':
            team.tov += 1
        elif event_type == 'REBOUND':
            if rebound_type == 'OFFENSIVE':
                team.orb += 1
            else:
                team.drb += 1
        elif event_type == 'FOUL':
            team.fouls += 1

    def _recompute_four_factors_and_team_weight(self) -> None:
        def compute_four_factors(team: CurrTeamState, opp: CurrTeamState) -> None:
            efg_num = team.fgm + 0.5 * team.tpm
            efg_den = max(team.fga, 1)
            team.efg_live = efg_num / efg_den
            tov_den = team.fga + 0.44 * team.fta + team.tov
            team.tov_pct_live = (team.tov / tov_den) if tov_den > 0 else 0.12
            orb_den = team.orb + opp.drb
            team.orb_pct_live = (team.orb / orb_den) if orb_den > 0 else 0.25
            team.ftr_live = (team.fta / efg_den) if efg_den > 0 else 0.2
    
            team.efg_smooth = team.efg_live
            team.tov_pct_smooth = team.tov_pct_live
            team.orb_pct_smooth = team.orb_pct_live
            team.ftr_smooth = team.ftr_live

            tw = (
                0.40 * team.efg_smooth +
                0.25 * (1.0 - team.tov_pct_smooth) +
                0.20 * team.orb_pct_smooth +
                0.15 * team.ftr_smooth
            )
        
            team.team_weight = min(max(tw, 0.0), 1.0)

        compute_four_factors(self.home_team, self.away_team)
        compute_four_factors(self.away_team, self.home_team)


            
    # def _blend_ratings(self, player: CurrPlayerState, team_avg_offense: float) -> None:
    #     """
    #     This takes the offensive and defensive ratings, along with
    #     time on the court, to determine the blended rating, until the
    #     player has played enough players
    #     """
    #     if player.minutes_on_floor >= 10.0:
    #         player.blended_offense_rating = player.real_offensive_rating
    #         player.blended_defense_rating = player.real_defensive_rating
    #     else:
    #         blend_factor = player.minutes_on_floor / 10.0
    #         player.blended_offense_rating = (
    #             team_avg_offense * (1 - blend_factor) +
    #             player.real_offensive_rating * blend_factor
    #         )
    #         player.blended_defense_rating = (
    #             50.0 * (1 - blend_factor) +
    #             player.real_defensive_rating * blend_factor
    #         )
    #     player.blended_overall_rating = (
    #         0.7 * player.blended_offense_rating +
    #         0.3 * player.blended_defense_rating
    #     )

    # def _update_team_ratings(self, team: CurrTeamState) -> None:
    #     """Update team overall ratings from active lineup"""
    #     if not team.active_lineup:
    #         return
        
    #     active_players = [team.roster[pid] for pid in team.active_lineup if pid in team.roster]
    #     if not team.active_lineup:
    #         return
        
    #     team.team_offensive_rating = sum(p.blended_offense_rating for p in active_players) / len(active_players)
    #     team.team_defensive_rating = sum(p.blended_defense_rating for p in active_players) / len(active_players)
    #     team.team_overall_rating = sum(p.blended_overall_rating for p in active_players) / len(active_players)


    # def _handle_substitution(self, team: CurrTeamState, event: dict) -> None:
    #     """Handle player substitution"""
    #     player_in = event.get('player_name')
    #     player_out = event.get('substituted_player_name')

    #     if player_in:
    #         self._ensure_player_on_floor(team, player_in)

    #     if player_out:
    #         if player_out in team.active_lineup:
    #             team.active_lineup.remove(player_out)
    #         if player_out in team.roster:
    #             team.roster[player_out].on_floor = False    
    
    def is_possession_ending_event(self, event_type: str) -> bool:
        return event_type in ['SCORE', 'REBOUND', 'TURNOVER']

    def _update_possession_tracking(self, event: dict) -> None:
        """Update possession length tracking"""
        if self.is_possession_ending_event(event.get('event_type', '')):
            current_time = event.get('time_seconds', 0)
            if self.game_state.last_possession_start > 0 and self.game_state.last_possession_start > current_time:
                possession_length = self.game_state.last_possession_start - current_time

                total_time = self.game_state.num_possessions_completed * self.game_state.average_possession_length
                self.game_state.num_possessions_completed += 1
                self.game_state.average_possession_length = (total_time + possession_length) / self.game_state.num_possessions_completed

                self.game_state.possessions_seen += 1

                self.game_state.last_possession_start = current_time

    def _calculate_win_probability(self) -> float:
        """calculate home team win probability"""

        # remaining_possessions = 0.0
        # if self.game_state.average_possession_length > 0:
        #     remaining_possessions = self.game_state.time_remaining / self.game_state.average_possession_length

        # home_ppp = 0.8 + (self.home_team.team_overall_rating / 100.0) * 0.6
        # away_ppp = 0.8 + (self.away_team.team_overall_rating / 100) * 0.6

        # expected_home = self.game_state.home_score + (remaining_possessions * home_ppp)
        # expected_away = self.game_state.away_score + (remaining_possessions * away_ppp)

        # point_diff = expected_home - expected_away
        # z_score = point_diff / 10.0
        # probability = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))

        # return max(0.005, min(0.995, probability))

        c = 3.0
        t_rem = max(self.game_state.time_remaining, 0.0)
        g = 1.0 / (1.0 + c * math.sqrt(t_rem / 2400.0))

        kappa = 1.0
        n0 = 30.0
        possessions = max(self.game_state.possessions_seen, 0)
        lam = kappa * (possessions / (possessions + n0))

        tw_edge = max(min(tw_edge, 0.10), -0.10)

        score_diff = float(self.game_state.home_score - self.game_state.away_score)
        beta_s = 0.08
        beta_w = 2.0

        L = g * beta_s * score_diff + lam * beta_w * tw_edge
        p = 1.0 / (1.0 + math.exp(-L))

        p = min(max(p, 0.01), 0.99)
        return p

    def _enough_time_elapsed_to_make_new_trade(self, current_time: float) -> bool:
        """Check if enough time has passed since last trade"""
        return (current_time - self.last_trade_time) >= self.min_trade_interval
    
    #     trade_price = self._get_tradeable_price(desired_side)
    #     if trade_price is None:
    #         trade_price = market_price # graceful fallback if book side missing

    #     # flip if needed
    #     if self.curr_position.curr_active and self.curr_position.side_of_entry != desired_side:
    #         self.execute_close_position()

    #     if desired_side == Side.BUY:
    #         qty = (alloc * self.capital_remaining) / max(trade_price, 1e-6)
    #     else:
    #         qty = (alloc * self.capital_remaining) / max(100.0 - trade_price, 1e-6)
    #     if qty < 1.0:
    #         return False
        
    #     place_market_order(desired_side, Ticker.TEAM_A, qty)
    #     self.update_curr_position_on_order(desired_side, qty, trade_price)
    #     self.last_endgame_action_tick = tick

    #     print(f"[ENDGAME-LOCK] t={self.game_state.time_remaining:.2f}s "
    #           f"lead={self.game_state.home_score - self.game_state.away_score} "
    #           f"edge={(edge*100):.1f}% side={desired_side.name} qty={qty:.2f} px={trade_price:.2f} "
    #           f"alloc={int(alloc*100)}%")
    #     return True
    
    # def _maybe_apply_endgame_lock(self) -> bool:
    #     """Detect & act; True if an order was placed."""
    #     is_lock, p_true, reason = self._is_endgame_lock()
    #     if not is_lock or p_true is None:
    #         return False
    #     return self._execute_endgame_strategy(p_true, reason)

    def calculate_current_edge(self) -> float:
        win_prob = self._calculate_win_probability()
        market_price = self.last_trade_price if self.last_trade_price is not None else self._mid()
        if market_price is None:
            return 0.0
        
        market_prob = market_price / 100.00
        market_prob = min(max(market_prob, 1e-9), 1 - 1e-9)
        edge = win_prob - market_prob
        return edge
        
    def should_close_position(self) -> bool:
        if  self.position == 0.0:
            return False
        
        if self.game_state.time_remaining <= 200:
            return True
        
        current_edge = self.calculate_current_edge()

        return abs(current_edge) < 0.02

        # if self.curr_position.side_of_entry == Side.BUY and current_edge < -0.05:
        #     return True
        
        # if self.curr_position.side_of_entry == Side.SELL and current_edge > 0.05:
        #     return True

        # if abs(current_edge) < 0.02:
        #     return True
        
        # return False


    def reset_curr_position(self) -> None:
        self.curr_position.curr_active = False
        self.curr_position.time_of_entry = None
        self.curr_position.side_of_entry = 0
        self.curr_position.quantity_of_position_purchased = 0
        self.curr_position.cost_of_position = 0.0

        self.last_trade_time = 0.0

    def update_curr_position_on_order(self, side: Side, quantity: float, curr_price: float):
        self.curr_position.curr_active = True
        self.curr_position.quantity_of_position_purchased = quantity
        self.curr_position.side_of_entry = side
        self.curr_position.time_of_entry = self.game_state.time_remaining
        self.curr_position.cost_of_position = quantity * curr_price

        if abs(self.position) < 1e-6:
            self.peak_edge_abs_since_entry = abs(self.calculate_current_edge())
            self.scale_out_stage = 0

    def execute_close_position(self) -> None:
        if self.position == 0.0:
            return

        if self.position > 0:
            place_market_order(Side.SELL, Ticker.TEAM_A, abs(self.position))
        else:
            place_market_order(Side.BUY, Ticker.TEAM_A, abs(self.position))
        
        # if self.curr_position.side_of_entry == Side.BUY:
        #     place_market_order(Side.SELL, Ticker.TEAM_A, self.curr_position.quantity_of_position_purchased)

        # if self.curr_position.side_of_entry == Side.SELL:
        #     place_market_order(Side.BUY, Ticker.TEAM_A, self.curr_position.quantity_of_position_purchased)
        
        self.reset_curr_position()
        

    def _execute_trading_decision(self, win_prob: float, current_time: float) -> None:
        """Execute trading decision based on calculated probabilities"""

        #new addition: allow endgame lock to run even when < 500s
        if self._maybe_apply_endgame_lock():
            return

        # if self.game_state.time_remaining > 1500 or self.game_state.time_remaining < 500:
        #     return
        
        # if not self._enough_time_elapsed_to_make_new_trade(current_time):
        #     return
        
        # mid_price = self.last_trade_price
        # if mid_price is None or self.capital_remaining is None:
        #     return
        
        if self.should_close_position():
            self.execute_close_position()
            return
        
        if self.game_state.time_remaining > 1800 or self.game_state.time_remaining < 200:
            return
        
        price_ref = self.last_trade_price if self.last_trade_price is not None else self._mid()
        if price_ref is None or self.capital_remaining is None:
            return
        
        market_prob = price_ref / 100.0
        market_prob = min(max(market_prob, 1e-9), 1 - 1e-9)
        model_edge = win_prob - market_prob

        slippage_bps = 0.002
        effective_edge = model_edge - slippage_bps if model_edge > 0 else model_edge + slippage_bps

        entry_threshold = 0.05
        if abs(effective_edge) < entry_threshold:
            return
        
        kelly_fraction = self.kelly.kelly_fraction(win_prob, price_ref)
        if abs(kelly_fraction) < 0.01:
            return
        
        target_units = self.kelly.target_units(self.capital_remaining, price_ref, kelly_fraction)
        curr_edge_abs = abs(model_edge)
        if curr_edge_abs > self.peak_edge_abs_since_entry:
            self.peak_edge_abs_since_entry = curr_edge_abs

        drop = self.peak_edge_abs_since_entry - curr_edge_abs
        if self.position != 0.0 and drop >= 0.02:
            if curr_edge_abs < 0.02:
                self.execute_close_position()
                return
            
            elif curr_edge_abs < 0.03 and self.scale_out_stage < 2:
                desired = 0.5 * self.position
                reduce_qty = abs(self.position - desired)
                if reduce_qty >= 0.1:
                    side = Side.SELL if self.position > 0 else Side.BUY
                    trade_price = self._get_tradeable_price(side)
                    if trade_price is not None:
                        place_market_order(side, Ticker.TEAM_A, reduce_qty)
                        self.last_trade_time = current_time
                self.scale_out_stage = 2
            elif self.scale_out_stage < 1:
                desired = 0.75 * self.position
                reduce_qty = abs(self.position - desired)
                if reduce_qty >= 0.1:
                    side = Side.SELL if self.position > 0 else Side.BUY
                    trade_price = self._get_tradeable_price(side)
                    if trade_price is not None:
                        place_market_order(side, Ticker.TEAM_A, reduce_qty)
                        self.last_trade_time = current_time
                self.scale_out_stage = 1
        if target_units > 0:
            if self.position >= 0:
                add_qty = target_units - self.position
                if add_qty >= 0.1:
                    trade_price = self._get_tradeable_price(Side.BUY)
                    if trade_price is not None:
                        quantity = min(add_qty, self.capital_remaining / trade_price)
                        if quantity >= 0.1:
                            place_market_order(Side.BUY, Ticker.TEAM_A, quantity)
                            self.update_curr_position_on_order(Side.BUY, quantity, trade_price)
                            self.last_trade_time = current_time
        elif target_units < 0:
            if self.position <= 0:
                add_qty = abs(target_units) - abs(self.position)
                if add_qty >= 0.1:
                    trade_price = self._get_tradeable_price(Side.SELL)
                    if trade_price is not None:
                        quantity = add_qty
                        if quantity >= 0.1:
                            place_market_order(Side.SELL, Ticker.TEAM_A, quantity)
                            self.update_curr_position_on_order(Side.SELL, quantity, trade_price)
                            self.last_trade_time = current_time
    

        # #Rest of function only possible if there is no currently active position
        # kelly_fraction = self.kelly.kelly_fraction(win_prob, mid_price)
        # if abs(kelly_fraction) < 0.01:
        #     return
        
        # target_units = self.kelly.target_units(self.capital_remaining, mid_price, kelly_fraction)
        # position_change = target_units - self.position

        # if abs(position_change) < 1.0:
        #     return
        
        # if position_change > 0:
        #     trade_price = self._get_tradeable_price(Side.BUY)
        #     if trade_price is not None:
        #         quantity = min(abs(position_change), self.capital_remaining / trade_price)
        #         if quantity >= 1.0:
        #             place_market_order(Side.BUY, Ticker.TEAM_A, quantity)
        #             self.update_curr_position_on_order(Side.BUY, quantity, trade_price)
        #             self.last_trade_time = current_time
        # else:
        #     trade_price = self._get_tradeable_price(Side.SELL)
        #     if trade_price is not None:
        #         quantity = min(abs(position_change), abs (self.position) if self.position < 0 else float('inf'))
        #         if quantity >= 1.0:
        #             place_market_order(Side.SELL, Ticker.TEAM_A, quantity)
        #             self.update_curr_position_on_order(Side.SELL, quantity, trade_price)
        #             self.last_trade_time = current_time

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

    def _handle_event_sequence_micro_edge(self, event: dict) -> None:
        event_type = event.get('event_type')
        home_away = event.get('home_away')
        time_seconds = event.get('time_seconds')

        if self.micro_edge_active:
            should_exit = False

            if event_type == 'SCORE':
                should_exit = True
            elif event_type == 'TURNOVER':
                should_exit = True
            elif event_type == 'REBOUND' and event.get('rebound_type') == 'DEFENSIVE':
                should_exit = True
            elif self.game_state.num_possessions_completed > self.micro_edge_entry_possession + 1:
                should_exit = True

            if should_exit:
                self._exit_micro_edge_position()
                return
                
        if not self.micro_edge_active and time_seconds is not None:
            if 200 <= time_seconds <= 1800:
                if event_type == 'STEAL' or (event_type == 'REBOUND' and event.get('rebound_type') == 'OFFENSIVE'):
                    if event_type == 'STEAL':
                        benefiting_team = home_away
                        prob_boost = self.MICRO_EDGE_STEAL_BOOST
                    else:
                        benefiting_team = home_away
                        prob_boost = self.MICRO_EDGE_ORB_BOOST
                    
                    if benefiting_team not in ['home', 'away']:
                        return
                    
                    base_prob = self._calculate_win_probability()

                    if benefiting_team == 'home':
                        boosted_prob = min(base_prob + prob_boost, 0.99)
                    else:
                        boosted_prob = max(base_prob - prob_boost, 0.01)
                    
                    market_price = self.last_trade_price if self.last_trade_price is not None else self._mid()
                    if market_price is None:
                        return
                    
                    market_prob = market_price / 100.0
                    micro_edge = boosted_prob - market_prob

                    if abs(micro_edge) >= self.MICRO_EDGE_MIN_THRESHOLD:
                        self._enter_micro_edge_position(
                            benefiting_team=benefiting_team,
                            edge=micro_edge,
                            event_type=event_type,
                            time_seconds=time_seconds
                        )

    def _enter_micro_edge_position(self, benefiting_team: str, edge: float, event_type: str, time_seconds: float) -> None:
        position_capital = self.capital_remaining * self.MICRO_EDGE_SIZE_PCT

        if benefiting_team == 'home':
            side = Side.BUY if edge > 0 else None
        else:
            side = Side.SELL if edge < 0 else None
        
        if side is None:
            return
        
        trade_price = self._get_tradeable_price(side)
        if trade_price is None:
            trade_price = self.last_trade_price if self.last_trade_price is not None else 50.0

        quantity = position_capital / trade_price
        if quantity < 1.0:
            return
        
        place_market_order(side, Ticker.TEAM_A, quantity)

        self.micro_edge_active = True
        self.micro_edge_team = benefiting_team
        self.micro_edge_event_type = event_type
        self.micro_edge_position_size = quantity if side == Side.BUY else -quantity
        self.micro_edge_entry_time = time_seconds
        self.micro_edge_entry_possession = self.game_state.num_possessions_completed

    def _exit_micro_edge_position(self) -> None:
        if not self.micro_edge_active or self.micro_edge_position_size == 0:
            return

        if not self.micro_edge_active or self.micro_edge_position_size == 0:
            return
        
        if self.micro_edge_position_size > 0:
            exit_side = Side.SELL
            exit_quantity = abs(self.micro_edge_position_size)
        else:
            exit_side = Side.BUY
            exit_quantity = abs(self.micro_edge_position_size)

        place_market_order(exit_side, Ticker.TEAM_A, exit_quantity)

        self.micro_edge_active = False
        self.micro_edge_team = ""
        self.micro_edge_event_type = ""
        self.micro_edge_position_size = 0.0
        self.micro_edge_entry_time = None
        self.micro_edge_entry_possession = 0

    def _is_endgame_lock(self):
        t = self.game_state.time_remaining or 0.0
        if t > 35.0:
            return False, None, ""
        lead = self.game_state.home_score - self.game_state.away_score
        abs_lead = abs(lead)

            # ---------- Endgame Lock for no OT helpers -----
        def home_prob_from_leader(p_leader: float) -> float:
            return p_leader if lead >= 0 else (1.0 - p_leader)
        # under 5 seconds
        if t <= 5.0:
            if abs_lead >= 4:
                return True, home_prob_from_leader(1.0), "<=5s & lead >=4 (no 4-pt plays)"
            if abs_lead == 3:
                return True, home_prob_from_leader(0.995), "<=5s & lead=3"
            if abs_lead == 2:
                return True, home_prob_from_leader(0.98), "<=5s & lead=2"
            
        # 6-10s
        if 6.0 <= t <= 10.0:
            if abs_lead >= 6:
                return True, home_prob_from_leader(0.999), "6-10s & lead>=6"
            if abs_lead >= 4:
                return True, home_prob_from_leader(0.995), "6-10s & lead>=4 & <2 poss"
        
        # <=24s
        if t <= 24.0:
            if abs_lead >= 9:
                return True, home_prob_from_leader(0.9995), "<=24s & lead>=7"
            
        # <=35s
        if t <= 35.0:
            if abs_lead >= 10:
                return True, home_prob_from_leader(0.9995), "<=35s & lead>=9"
        return False, None, ""
        
    def _execute_endgame_strategy(self, true_home_prob: float, reason: str) -> bool:
        
        desired_side = Side.BUY if true_home_prob > 0.5 else Side.SELL

        market_price = self._mid()
        if market_price is None:
            market_price = self.last_trade_price
        if market_price is None:
            return False
        
        if true_home_prob >= 0.999 or true_home_prob <= 0.001:
            target_alloc = 0.80
        elif true_home_prob >= 0.995 or true_home_prob <= 0.005:
            target_alloc = 0.60
        elif true_home_prob >= 0.98 or true_home_prob <= 0.02:
            target_alloc = 0.40
        else:
            return False
        
        trade_price = self._get_tradeable_price(desired_side)
        if trade_price is None:
            trade_price = market_price

        current_alloc = 0.0
        if self.curr_position.curr_active:
            if self.curr_position.side_of_entry != desired_side:
                current_alloc = -1.0 * (self.curr_position.cost_of_position / (self.capital_remaining + self.curr_position.cost_of_position))
            else:
                current_alloc = self.curr_psoition.cost_of_position / (self.capital_remaining + self.curr_position.cost_of_position)


        alloc_change = target_alloc - current_alloc

        if abs(alloc_change) < 0.05:
            return False
        
        if current_alloc < 0:
            self.execute_close_position()
            current_alloc = 0.0
            alloc_change = target_alloc

        total_capital = self.capital_remaining
        if self.curr_position.curr_active and self.curr_position.side_of_entry == desired_side:
            total_capital = self.capital_remaining + self.curr_position.cost_of_position

        if alloc_change > 0:
            additional_capital_needed = alloc_change * total_capital
            if desired_side == Side.BUY:
                qty = additional_capital_needed / max(trade_price, 1e-6)
            else:
                qty = additional_capital_needed / max(100.0 - trade_price, 1e-6)

            if desired_side == Side.BUY:
                max_qty = self.capital_remaining / max(trade_price, 1e-6)
            else:
                max_qty = self.capital_remaining / max(100.0 - trade_price, 1e-6)

            qty = min(qty, max_qty)

            if qty < 1.0:
                return False
            
            if self.curr_position.curr_active:
                self.curr_position.quantity_of_position_purchased += qty
                self.curr_position.cost_of_position += qty * trade_price
            else:
                self.update_curr_posiiton_on_order(desired_side, qty, trade_price)
        else:
            reduction_capital = abs(alloc_change) * total_capital
            if desired_side == Side.BUY:
                qty_to_reduce = reduction_capital / max(trade_price, 1e-6)
            else:
                qty_to_reduce = reduction_capital / max(100.0 - trade_price, 1e-6)

            qty_to_reduce = min(qty_to_reduce, self.curr_position.quantity_of_position_purchased)

            if qty_to_reduce < 1.0:
                return False
            
            opposite_side = Side.SELL if desired_side == Side.BUY else Side.BUY
            place_market_order(opposite_side, Ticker.TEAM_A, qty_to_reduce)

            self.curr_position.cost_of_position -= qty_to_reduce * trade_price

            if self.curr_position.quantity_of_position_purchased < 1.0:
                self.reset_curr_position()
        
        return True

    def _maybe_apply_endgame_lock(self) -> bool:
        """Detect & act; True if an order was placed."""
        is_lock, p_true, reason = self._is_endgame_lock()
        if not is_lock or p_true is None:
            return False
        
        return self._execute_endgame_strategy(p_true, reason)

            # market_prob = min(max(market_price / 100.0, 1e-6), 1 - 1e-6)
            # edge = abs(true_home_prob - market_prob)
            # if edge < 0.10:
            #     return False
            
            # if edge >= 0.30:
            #     alloc = 0.80
            # elif edge >= 0.20:
            #     alloc = 0.60
            # else:
            #     alloc = 0.40

            # desired_side = Side.BUY if true_home_prob > market_prob else Side.SELL



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

        if home_away == "home":
            self.home_team.curr_total_points = home_score
        elif home_away == "away":
            self.away_team.curr_total_points = away_score

        # if event_type == "JUMP_BALL" and target_team and player_name:
        #     if player_name not in target_team.active_lineup:
        #         target_team.active_lineup.append(player_name)
        #         player_obj = self._get_or_create_player(target_team, player_name)
        #         player_obj.on_floor = True

        # if event_type == "SUBSTITUTION" and target_team:
        #     self._handle_substitution(target_team, event)

        # action_events = {"SCORE", "MISSED", "REBOUND", "STEAL", "BLOCK", "TURNOVER", "FOUL"}
        # if target_team and event_type in action_events:
        #     if player_name:
        #         self._ensure_player_on_floor(target_team, player_name)
        #     if assist_player:
        #         self._ensure_player_on_floor(target_team, assist_player)
        
                         

        # for team in [self.home_team, self.away_team]:
        #     for player_id in team.active_lineup:
        #         if player_id in team.roster:
        #             player = team.roster[player_id]
        #             self._update_player_stats(player, event)
        #             self.compute_offensive_rating(player)
        #             self.compute_defensive_rating(player)
        #             self._blend_ratings(player, team.team_offensive_rating)
        #     self._update_team_ratings(team)

        self._update_possession_tracking(event)

        self._update_team_box_scores(event)
        self._recompute_four_factors_and_team_weight()

        if event_type == "NOTHING":
            pass

        if time_seconds is not None:
            win_prob = self._calculate_win_probability()
            self._execute_trading_decision(win_prob, time_seconds)
            
        if event_type and event_type != "NOTHING":
            self.game_active = True


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

        
