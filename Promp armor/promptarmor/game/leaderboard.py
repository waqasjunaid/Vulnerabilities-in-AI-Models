"""Leaderboard persistence for the Escape Room game."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class LeaderboardEntry:
    """A single leaderboard entry."""

    player_name: str
    score: int
    challenges_solved: int
    total_attempts: int
    timestamp: str
    successful_payloads: List[str]


class Leaderboard:
    """
    Persistent leaderboard for tracking top scores.

    Saves successful attacks for analysis and defense improvement.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = Path(persist_path) if persist_path else None
        self.entries: List[LeaderboardEntry] = []
        self._load()

    def _load(self) -> None:
        """Load leaderboard from disk."""
        if self.persist_path and self.persist_path.exists():
            try:
                with open(self.persist_path, 'r') as f:
                    data = json.load(f)
                    self.entries = [
                        LeaderboardEntry(**entry)
                        for entry in data.get("entries", [])
                    ]
            except Exception:
                self.entries = []

    def _save(self) -> None:
        """Save leaderboard to disk."""
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, 'w') as f:
                json.dump({
                    "entries": [asdict(e) for e in self.entries],
                    "updated": datetime.utcnow().isoformat(),
                }, f, indent=2)

    def add_entry(
        self,
        player_name: str,
        score: int,
        challenges_solved: int,
        total_attempts: int,
        successful_payloads: List[str],
    ) -> int:
        """
        Add a new leaderboard entry.

        Returns:
            The rank achieved (1-indexed)
        """
        entry = LeaderboardEntry(
            player_name=player_name,
            score=score,
            challenges_solved=challenges_solved,
            total_attempts=total_attempts,
            timestamp=datetime.utcnow().isoformat(),
            successful_payloads=successful_payloads[:10],  # Limit stored payloads
        )

        self.entries.append(entry)
        self.entries.sort(key=lambda e: e.score, reverse=True)

        # Keep top 100
        self.entries = self.entries[:100]

        self._save()

        # Return rank
        return next(
            i + 1 for i, e in enumerate(self.entries)
            if e.timestamp == entry.timestamp
        )

    def get_top(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N entries."""
        return [
            {
                "rank": i + 1,
                "player": e.player_name,
                "score": e.score,
                "challenges_solved": e.challenges_solved,
                "date": e.timestamp[:10],
            }
            for i, e in enumerate(self.entries[:n])
        ]

    def get_successful_payloads(self) -> List[str]:
        """
        Get all successful attack payloads for analysis.

        These can be used to improve defenses.
        """
        payloads = []
        for entry in self.entries:
            payloads.extend(entry.successful_payloads)
        return list(set(payloads))

    def get_stats(self) -> Dict[str, Any]:
        """Get leaderboard statistics."""
        if not self.entries:
            return {
                "total_players": 0,
                "total_challenges_solved": 0,
                "average_score": 0,
            }

        return {
            "total_players": len(self.entries),
            "total_challenges_solved": sum(e.challenges_solved for e in self.entries),
            "average_score": sum(e.score for e in self.entries) / len(self.entries),
            "highest_score": self.entries[0].score if self.entries else 0,
            "total_successful_payloads": len(self.get_successful_payloads()),
        }
