"""Game engine utilities for Schafkopf Sauspiel simulations."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

from .cards import (
    FULL_DECK,
    POINT_VALUES,
    RANK_PRIORITY,
    SUITS,
    card_value,
    is_trump,
    lead_context,
    rank_of,
    suit_of,
    trump_rank,
)


@dataclass
class TrickLog:
    trick_index: int
    leader: int
    plays: List[Tuple[int, str]]
    winner: int
    points: int


@dataclass
class GameResult:
    declarer_index: int
    partner_index: Optional[int]
    called_ace: str
    declarer_points: int
    opponent_points: int
    win: bool
    schneider: bool
    schwarz: bool
    tricks: List[TrickLog]

    def to_record(self) -> dict:
        payload = asdict(self)
        payload['tricks'] = [
            {
                'trick_index': t.trick_index,
                'leader': t.leader,
                'winner': t.winner,
                'points': t.points,
                'plays': t.plays,
            }
            for t in self.tricks
        ]
        return payload


class SauspielSimulator:
    """Encapsulates stateful helpers for repeated simulations."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def deal_hands(self, declarer_hand: Optional[Sequence[str]] = None) -> List[List[str]]:
        deck = FULL_DECK[:]
        self.rng.shuffle(deck)

        if declarer_hand:
            declarer_cards = list(declarer_hand)
            for card in declarer_cards:
                if card not in deck:
                    raise ValueError(f"Invalid card in declarer hand: {card}")
                deck.remove(card)
            hands = [declarer_cards]
            for i in range(3):
                start = i * 8
                hands.append(deck[start:start + 8])
            return hands

        return [deck[i * 8:(i + 1) * 8] for i in range(4)]

    def choose_called_ace(self, hand: Sequence[str]) -> str:
        # Only non-trump aces (Acorn, Leaf, Bell) can be called in a Sauspiel.
        candidates = [f"A_{s}" for s in ('Acorn', 'Leaf', 'Bell') if f"A_{s}" not in hand]
        if not candidates:
            return 'A_Acorn'
        return self.rng.choice(candidates)

    def random_legal_card(self, hand: List[str], lead_ctx: Optional[str]) -> str:
        if not hand:
            raise ValueError("Cannot select a card from an empty hand")

        if lead_ctx is None:
            return self.rng.choice(hand)

        if lead_ctx == 'TRUMP':
            trumps = [c for c in hand if is_trump(c)]
            if trumps:
                return self.rng.choice(trumps)
            return self.rng.choice(hand)

        same_suit = [c for c in hand if suit_of(c) == lead_ctx and not is_trump(c)]
        if same_suit:
            return self.rng.choice(same_suit)
        return self.rng.choice(hand)

    def beats(self, challenger: str, incumbent: str, lead_ctx: str) -> bool:
        if is_trump(challenger):
            if not is_trump(incumbent):
                return True
            return trump_rank(challenger) < trump_rank(incumbent)

        if is_trump(incumbent):
            return False

        if lead_ctx == 'TRUMP':
            return False  # cannot beat a trump with a non-trump

        if suit_of(challenger) == suit_of(incumbent) == lead_ctx:
            return RANK_PRIORITY[rank_of(challenger)] < RANK_PRIORITY[rank_of(incumbent)]

        if suit_of(challenger) == lead_ctx and suit_of(incumbent) != lead_ctx:
            return True

        return False

    def play_trick(self, hands: List[List[str]], leader: int, trick_index: int) -> TrickLog:
        plays: List[Tuple[int, str]] = []
        ctx: Optional[str] = None
        for offset in range(4):
            player = (leader + offset) % 4
            card = self.random_legal_card(hands[player], ctx)
            hands[player].remove(card)
            plays.append((player, card))
            if ctx is None:
                ctx = lead_context(card)

        winner, winning_card = plays[0]
        for player, card in plays[1:]:
            if ctx is None:
                raise RuntimeError("Lead context missing during trick resolution")
            if self.beats(card, winning_card, ctx):
                winner, winning_card = player, card

        points = sum(card_value(card) for _, card in plays)
        return TrickLog(trick_index=trick_index, leader=leader, plays=plays, winner=winner, points=points)

    def simulate_game(self, declarer_index: int = 0, declarer_hand: Optional[Sequence[str]] = None) -> GameResult:
        hands = self.deal_hands(declarer_hand=declarer_hand)
        called_ace = self.choose_called_ace(hands[declarer_index])
        partner_index = next(
            (idx for idx, hand in enumerate(hands) if idx != declarer_index and called_ace in hand),
            None,
        )

        points = [0, 0, 0, 0]
        leader = declarer_index
        trick_logs: List[TrickLog] = []

        for trick_idx in range(8):
            trick = self.play_trick(hands, leader, trick_idx)
            points[trick.winner] += trick.points
            leader = trick.winner
            trick_logs.append(trick)

        declarer_points = points[declarer_index]
        if partner_index is not None:
            declarer_points += points[partner_index]

        opponent_points = sum(points) - declarer_points
        win = declarer_points >= 61
        schneider = declarer_points >= 91
        schwarz = declarer_points == 120

        return GameResult(
            declarer_index=declarer_index,
            partner_index=partner_index,
            called_ace=called_ace,
            declarer_points=declarer_points,
            opponent_points=opponent_points,
            win=win,
            schneider=schneider,
            schwarz=schwarz,
            tricks=trick_logs,
        )


def simulate_random_game(seed: Optional[int] = None) -> GameResult:
    return SauspielSimulator(seed=seed).simulate_game()
