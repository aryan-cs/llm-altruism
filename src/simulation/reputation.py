"""Public reputation tracking for society simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ReputationConfig:
    """Configuration for the reputation system."""

    enabled: bool = True
    decay: float = 1.0
    anonymous_ratings: bool = False
    min_rating: int = 1
    max_rating: int = 5


@dataclass
class Rating:
    """One reputation rating event."""

    rater: str
    target: str
    score: int
    timestep: int
    note: str | None = None

    def to_dict(self, anonymous: bool = False) -> dict[str, Any]:
        """Serialize the rating for logs or prompts."""
        return {
            "rater": None if anonymous else self.rater,
            "target": self.target,
            "score": self.score,
            "timestep": self.timestep,
            "note": self.note,
        }


class ReputationSystem:
    """Maintain public reputation scores over time."""

    def __init__(self, config: ReputationConfig | None = None):
        self.config = config or ReputationConfig()
        self.ratings: list[Rating] = []

    def add_rating(
        self,
        *,
        rater: str,
        target: str,
        score: int,
        timestep: int,
        note: str | None = None,
    ) -> Rating | None:
        """Add a validated rating to the ledger."""
        if target == rater:
            return None
        if score < self.config.min_rating or score > self.config.max_rating:
            return None

        rating = Rating(rater=rater, target=target, score=score, timestep=timestep, note=note)
        self.ratings.append(rating)
        return rating

    def apply_decay(self) -> None:
        """Apply multiplicative decay to older ratings."""
        if self.config.decay >= 1.0:
            return
        decayed: list[Rating] = []
        for rating in self.ratings:
            adjusted = max(self.config.min_rating, round(rating.score * self.config.decay))
            decayed.append(
                Rating(
                    rater=rating.rater,
                    target=rating.target,
                    score=adjusted,
                    timestep=rating.timestep,
                    note=rating.note,
                )
            )
        self.ratings = decayed

    def score_for(self, agent_id: str) -> float:
        """Average visible reputation score for an agent."""
        scores = [rating.score for rating in self.ratings if rating.target == agent_id]
        if not scores:
            midpoint = (self.config.min_rating + self.config.max_rating) / 2
            return float(midpoint)
        return sum(scores) / len(scores)

    def summary(self) -> dict[str, dict[str, Any]]:
        """Return public reputation stats for all known targets."""
        targets = {rating.target for rating in self.ratings}
        return {
            target: {
                "score": self.score_for(target),
                "ratings": [
                    rating.to_dict(anonymous=self.config.anonymous_ratings)
                    for rating in self.ratings
                    if rating.target == target
                ],
            }
            for target in sorted(targets)
        }

