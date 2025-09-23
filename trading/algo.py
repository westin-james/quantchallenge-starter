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

@dataclass
class CurrGameState:
    team_a_score: int = 0
    team_b_score: int = 0
    time_remeaning: int = 2000




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