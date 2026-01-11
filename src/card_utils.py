# -*- coding: utf-8 -*-
"""Backward-compatible card helpers. Prefer :mod:`schafkopf_ai.cards`."""

from schafkopf_ai.cards import (  # noqa: F401
    FULL_DECK,
    POINT_VALUES,
    SUITS,
    TRUMP_INDEX,
    TRUMP_ORDER,
    card_value,
    rank_of,
    suit_of,
    trump_rank,
    trump_strength,
)

__all__ = [
    'SUITS',
    'TRUMP_ORDER',
    'TRUMP_INDEX',
    'POINT_VALUES',
    'FULL_DECK',
    'trump_rank',
    'card_value',
    'rank_of',
    'suit_of',
    'trump_strength',
]