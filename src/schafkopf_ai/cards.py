"""Card helper utilities for Schafkopf Sauspiel (Sucherb) simulations."""

from __future__ import annotations

from typing import Iterable, List

SUITS: List[str] = ['Acorn', 'Leaf', 'Heart', 'Bell']
RANKS: List[str] = ['A', '10', 'K', 'O', 'U', '9', '8', '7']
POINT_VALUES = {'A': 11, '10': 10, 'K': 4, 'O': 3, 'U': 2, '9': 0, '8': 0, '7': 0}

# Sauspiel trumps: Obers, Unters, then all Hearts.
TRUMP_ORDER: List[str] = [
    'O_Acorn', 'O_Leaf', 'O_Heart', 'O_Bell',
    'U_Acorn', 'U_Leaf', 'U_Heart', 'U_Bell',
    'A_Heart', '10_Heart', 'K_Heart', '9_Heart', '8_Heart', '7_Heart'
]
TRUMP_INDEX = {card: idx for idx, card in enumerate(TRUMP_ORDER)}
TRUMP_STRENGTH = {
    'O_Acorn': 14,
    'O_Leaf': 13,
    'O_Heart': 12,
    'O_Bell': 11,
    'U_Acorn': 10,
    'U_Leaf': 9,
    'U_Heart': 8,
    'U_Bell': 7,
    'A_Heart': 6,
    '10_Heart': 5,
    'K_Heart': 4,
    '9_Heart': 3,
    '8_Heart': 2,
    '7_Heart': 0,
}

FULL_DECK: List[str] = [f"{rank}_{suit}" for suit in SUITS for rank in RANKS]
RANK_PRIORITY = {rank: idx for idx, rank in enumerate(RANKS)}  # smaller idx == stronger card
TRUMP_SENTINEL = 100


def rank_of(card: str) -> str:
    return card.split('_')[0]


def suit_of(card: str) -> str:
    return card.split('_')[1]


def card_value(card: str) -> int:
    return POINT_VALUES.get(rank_of(card), 0)


def is_trump(card: str) -> bool:
    return card in TRUMP_INDEX


def trump_rank(card: str) -> int:
    return TRUMP_INDEX.get(card, TRUMP_SENTINEL)


def trump_strength(card: str) -> int:
    return TRUMP_STRENGTH.get(card, 0)


def lead_context(card: str) -> str:
    """Return 'TRUMP' if the card is trump, otherwise its suit."""
    return 'TRUMP' if is_trump(card) else suit_of(card)


def sort_by_strength(cards: Iterable[str]) -> List[str]:
    """Return cards ordered by trump dominance then suit strength."""
    def strength(card: str) -> tuple[int, int, int]:
        if is_trump(card):
            return (0, trump_rank(card), 0)
        return (1, SUITS.index(suit_of(card)), RANK_PRIORITY[rank_of(card)])

    return sorted(cards, key=strength)
