from typing import Dict, Optional, List
from dataclasses import dataclass

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


def blend_ratings(player: CurrPlayerState) -> None:
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

def update_lineup(lineup: List[CurrPlayerState], event: dict) -> None:
    """
    Set the new lineup after a substitution, change currOverall,
    update ratings
    """
    
def update_average_possession_length(team: CurrGameState, new_possession_length: int) -> None:
    """
    Update average possesion length variable in CurrGameState
    """ 

def calculate_expected_remaining_possessions(team: CurrGameState, time_remaining: int) -> None:
    """
    Use the average posession length and time remaining to calculate
    """

def calculate_average_points_per_possessions(team: CurrGameState) -> int: 
    """
    Use the team points and possessions to calculate
    """

def calculate_expected_total_points() -> None:
    """
    Use remaining possessions and average points scored per drive
    """

def calculate_probability(expectedA: int, expectedB: int, differential = 12.0) -> float:
    """
    Use calculate expected total points to get them and pass them as parameters
    Then, use standard deviation of 12 for nba game point differential
    get z-score: expected / differential
    get p = phi(z) = 0.5*(1+erf(z / sqrt(2)))
    then return p
    """