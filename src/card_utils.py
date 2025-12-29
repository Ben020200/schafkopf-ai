# -*- coding: utf-8 -*-
"""Card utilities and definitions for Schafkopf Rufer game."""

SUITS = ['Acorn', 'Leaf', 'Heart', 'Bell']
RANKS = ['A', '10', 'K', 'O', 'U', '9', '8', '7']
POINT_VALUES = {'A': 11, '10': 10, 'K': 4, 'O': 3, 'U': 2, '9': 0, '8': 0, '7': 0}

# Trump order: 0 = strongest, 13 = weakest
TRUMP_ORDER = [
    'O_Acorn', 'O_Leaf', 'O_Heart', 'O_Bell',
    'U_Acorn', 'U_Leaf', 'U_Heart', 'U_Bell',
    'A_Heart', '10_Heart', 'K_Heart', '9_Heart', '8_Heart', '7_Heart'
]
TRUMP_INDEX = {c: i for i, c in enumerate(TRUMP_ORDER)}

FULL_DECK = [f"{r}_{s}" for s in SUITS for r in RANKS]


def trump_rank(card):
    """Return 0..13 for trumps, 100 for non-trumps."""
    return TRUMP_INDEX.get(card, 100)


def card_value(card):
    """Get point value of a card."""
    return POINT_VALUES.get(card.split('_')[0], 0)