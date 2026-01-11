"""Feature extraction helpers for declarer hands."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .cards import card_value, is_trump, rank_of, suit_of, trump_strength


@dataclass(frozen=True)
class HandFeatures:
    trump_count: int
    strongest_trump_strength: Optional[int]
    weakest_trump_strength: Optional[int]
    max_suit_run: int
    trump_strength_sum: int
    color_aces: int
    total_points: int
    non_trump_points: int

    def signature(self) -> Tuple[int, int, int, int, int, int]:
        strong = self.strongest_trump_strength if self.strongest_trump_strength is not None else -1
        weak = self.weakest_trump_strength if self.weakest_trump_strength is not None else -1
        return (
            self.trump_count,
            strong,
            weak,
            self.max_suit_run,
            self.trump_strength_sum,
            self.color_aces,
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            'trump_count': self.trump_count,
            'strongest_trump_strength': self.strongest_trump_strength,
            'weakest_trump_strength': self.weakest_trump_strength,
            'max_suit_run': self.max_suit_run,
            'trump_strength_sum': self.trump_strength_sum,
            'color_aces': self.color_aces,
            'total_points': self.total_points,
            'non_trump_points': self.non_trump_points,
        }


def featurize_hand(hand: Sequence[str]) -> HandFeatures:
    trump_cards = [card for card in hand if is_trump(card)]
    trump_strengths = sorted((trump_strength(card) for card in trump_cards), reverse=True)
    strong = trump_strengths[0] if trump_strengths else None
    weak = trump_strengths[-1] if trump_strengths else None

    suit_counts = Counter(suit_of(card) for card in hand if not is_trump(card))
    max_suit_run = max(suit_counts.values()) if suit_counts else 0

    total_points = sum(card_value(card) for card in hand)
    non_trump_points = sum(card_value(card) for card in hand if not is_trump(card))
    trump_strength_sum = sum(trump_strength(card) for card in trump_cards)
    color_aces = sum(1 for card in hand if rank_of(card) == 'A' and suit_of(card) != 'Heart')

    return HandFeatures(
        trump_count=len(trump_cards),
        strongest_trump_strength=strong,
        weakest_trump_strength=weak,
        max_suit_run=max_suit_run,
        trump_strength_sum=trump_strength_sum,
        color_aces=color_aces,
        total_points=total_points,
        non_trump_points=non_trump_points,
    )


def signature_key(features: HandFeatures) -> str:
    a, b, c, d, e, f = features.signature()
    return f"{a}-{b}-{c}-{d}-{e}-{f}"
