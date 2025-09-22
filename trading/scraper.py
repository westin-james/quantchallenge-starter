from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any

_ALLOWED_KEYS = [
    "home_away",
    "home_score",
    "away_score",
    "event_type",
    "player_name",
    "substituted_player_name",
    "shot_type",
    "assist_player",
    "rebound_type",
    "coordinate_x",
    "coordinate_y",
    "time_seconds",
]

class GameScraper:

    def __init__(self, out_path: Optional[str] = None) -> None:
        self.events = []
        self.out_path = Path(out_path) if out_path else None
        self._last_home_score: Optional[int] = 0
        self._last_away_score: Optional[int] = 0
        self._last_time_seconds: Optional[float] = None
    
    def start_new_game(self, out_path: Optional[str] = None) -> None:
        self.events = []
        if out_path is not None:
            self.out_path = Path(out_path)
        self._last_home_score = 0
        self._last_away_score = 0
        self._last_time_seconds = None

    def _append(self, entry: Dict[str, Any]) -> None:
        normalized = {k: entry.get(k, None) for k in _ALLOWED_KEYS}
        self.events.append(normalized)

        if normalized["home_score"] is not None:
            self._last_home_score = normalized["home_score"]
        if normalized["away_score"] is not None:
            self._last_away_score = normalized["away_score"]
        if normalized["time_seconds"] is not None:
            self._last_time_seconds = normalized["time_seconds"]
            
    def record_game_event(
        self,
        *,
        home_away: Optional[str],
        home_score: Optional[int],
        away_score: Optional[int],
        event_type: str,
        player_name: Optional[str],
        substituted_player_name: Optional[str],
        shot_type: Optional[str],
        assist_player: Optional[str],
        rebound_type: Optional[str],
        coordinate_x: Optional[float],
        coordinate_y: Optional[float],
        time_seconds: Optional[float],
    ) -> None:
        self._append(
            {
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
                "time_seconds": float(time_seconds) if time_seconds is not None else None,
            }
        )

    def record_generic(self, *, event_type: str, home_away: Optional[str] = "unknown") -> None:
        self._append(
            {
                "home_away": home_away if home_away is not None else "unknown",
                "home_score": self._last_home_score,
                "away_score": self._last_away_score,
                "event_type": event_type,
                "player_name": None,
                "substituted_player_name": None,
                "shot_type": None,
                "assist_player": None,
                "rebound_type": None,
                "coordinate_x": None,
                "coordinate_y": None,
                "time_seconds": self._last_time_seconds,
            }
        )

    def finalize(self) -> None:
        if not self.out_path:
            self.out_path = Path("game_debug.json")
        self.out_path.parten.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w", encoding="utf_8") as f:
            json.dump(self.events, f, indent=2)