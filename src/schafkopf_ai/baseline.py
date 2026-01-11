"""Baseline agents for Schafkopf comparison."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random

import numpy as np

from .cards import (
    FULL_DECK, TRUMP_STRENGTH, card_value, is_trump, lead_context,
    rank_of, suit_of, trump_rank, sort_by_strength, RANK_PRIORITY
)
from .features import featurize_hand, HandFeatures


# ============================================================================
# Hand Strength Thresholds (from 20k game analysis)
# ============================================================================

# Win rates by trump count (from our analysis)
TRUMP_COUNT_WIN_RATES = {
    0: 0.241,
    1: 0.299,
    2: 0.412,
    3: 0.476,
    4: 0.605,
    5: 0.730,
    6: 0.846,
    7: 0.936,
    8: 1.000,
}

# Win rates by Ober count
OBER_COUNT_WIN_RATES = {
    0: 0.408,
    1: 0.543,
    2: 0.716,
    3: 0.869,
    4: 0.973,
}

# Impact of specific cards (delta from baseline)
CARD_IMPACT = {
    'O_Acorn': 0.170,   # +17.0%
    'O_Leaf': 0.157,    # +15.7%
    'O_Heart': 0.10,    # estimated
    'O_Bell': 0.08,     # estimated
    'U_Acorn': 0.067,   # +6.7%
    'U_Leaf': 0.05,     # estimated
    'U_Heart': 0.04,    # estimated
    'U_Bell': 0.03,     # estimated
    'A_Heart': 0.026,   # +2.6%
}


@dataclass
class HandEvaluation:
    """Evaluation of a Schafkopf hand."""
    hand: List[str]
    features: HandFeatures
    estimated_win_rate: float
    should_play: bool
    reasoning: str


def evaluate_hand_strength(hand: List[str], threshold: float = 0.55) -> HandEvaluation:
    """
    Evaluate hand strength and decide if it should be played.
    
    Uses statistics from 20k game analysis to estimate win probability.
    """
    features = featurize_hand(hand)
    
    # Base win rate from trump count
    base_rate = TRUMP_COUNT_WIN_RATES.get(features.trump_count, 0.5)
    
    # Adjust for Ober count (weighted blend)
    ober_count = sum(1 for c in hand if c.startswith('O_'))
    ober_rate = OBER_COUNT_WIN_RATES.get(ober_count, 0.5)
    
    # Blend trump and ober effects (ober is more predictive for high counts)
    if ober_count >= 2:
        estimated = 0.4 * base_rate + 0.6 * ober_rate
    else:
        estimated = 0.7 * base_rate + 0.3 * ober_rate
    
    # Add bonuses for specific high cards
    for card in hand:
        if card in CARD_IMPACT:
            # Diminishing returns for stacking bonuses
            estimated += CARD_IMPACT[card] * 0.3
    
    # Penalty for very low trump strength
    if features.trump_count >= 3 and features.strongest_trump_strength is not None:
        if features.strongest_trump_strength < 8:  # No Obers
            estimated -= 0.05
    
    # Bonus for color aces (can capture tricks)
    estimated += features.color_aces * 0.02
    
    # Clamp to valid range
    estimated = max(0.1, min(0.95, estimated))
    
    # Decision
    should_play = estimated >= threshold
    
    # Build reasoning
    reasons = []
    reasons.append(f"{features.trump_count} trumps")
    reasons.append(f"{ober_count} Obers")
    if features.color_aces > 0:
        reasons.append(f"{features.color_aces} color aces")
    
    high_trumps = [c for c in hand if c in CARD_IMPACT and CARD_IMPACT[c] > 0.05]
    if high_trumps:
        reasons.append(f"has {', '.join(high_trumps)}")
    
    reasoning = f"Estimated {estimated:.1%} win rate ({', '.join(reasons)})"
    
    return HandEvaluation(
        hand=hand,
        features=features,
        estimated_win_rate=estimated,
        should_play=should_play,
        reasoning=reasoning,
    )


# ============================================================================
# Abstract Agent Interface
# ============================================================================

class BaseAgent(ABC):
    """Abstract base class for Schafkopf agents."""
    
    @abstractmethod
    def select_action(
        self,
        hand: List[str],
        legal_actions: List[str],
        current_trick: List[Tuple[int, str]],
        played_cards: List[str],
        player_idx: int,
        declarer_idx: int,
        partner_idx: Optional[int],
        points: List[int],
        trick_number: int,
    ) -> str:
        """Select a card to play."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name for display."""
        pass


# ============================================================================
# Random Agent
# ============================================================================

class RandomAgent(BaseAgent):
    """Plays random legal cards."""
    
    @property
    def name(self) -> str:
        return "Random"
    
    def select_action(
        self,
        hand: List[str],
        legal_actions: List[str],
        current_trick: List[Tuple[int, str]],
        played_cards: List[str],
        player_idx: int,
        declarer_idx: int,
        partner_idx: Optional[int],
        points: List[int],
        trick_number: int,
    ) -> str:
        return random.choice(legal_actions)


# ============================================================================
# Rule-Based Agent (Schafkopf Heuristics)
# ============================================================================

class RuleBasedAgent(BaseAgent):
    """
    Rule-based agent using classic Schafkopf heuristics.
    
    Key strategies:
    1. As declarer: Pull trumps, then run suits
    2. Play low when can't win, high when can win
    3. Protect high cards, sacrifice low ones
    4. Consider partner coordination
    """
    
    @property
    def name(self) -> str:
        return "RuleBased"
    
    def _is_teammate(
        self,
        player_idx: int,
        other_idx: int,
        declarer_idx: int,
        partner_idx: Optional[int],
    ) -> bool:
        """Check if two players are on the same team."""
        declarer_team = {declarer_idx}
        if partner_idx is not None:
            declarer_team.add(partner_idx)
        
        return (player_idx in declarer_team) == (other_idx in declarer_team)
    
    def _current_winner(self, trick: List[Tuple[int, str]]) -> Tuple[int, str]:
        """Determine current trick winner."""
        if not trick:
            return (-1, "")
        
        lead_player, lead_card = trick[0]
        lead_ctx = lead_context(lead_card)
        
        winner_idx, winner_card = lead_player, lead_card
        
        for player_idx, card in trick[1:]:
            card_ctx = lead_context(card)
            
            # Trump beats non-trump
            if card_ctx == "TRUMP" and lead_context(winner_card) != "TRUMP":
                winner_idx, winner_card = player_idx, card
            # Higher trump beats lower trump
            elif card_ctx == "TRUMP" and lead_context(winner_card) == "TRUMP":
                if trump_rank(card) < trump_rank(winner_card):
                    winner_idx, winner_card = player_idx, card
            # Same suit, higher rank wins
            elif card_ctx == suit_of(winner_card) and not is_trump(winner_card):
                if RANK_PRIORITY[rank_of(card)] < RANK_PRIORITY[rank_of(winner_card)]:
                    winner_idx, winner_card = player_idx, card
        
        return winner_idx, winner_card
    
    def _can_beat(self, card: str, current_best: str, lead_ctx: str) -> bool:
        """Check if card can beat the current best card."""
        card_ctx = lead_context(card)
        best_ctx = lead_context(current_best)
        
        # Trump beats non-trump
        if card_ctx == "TRUMP" and best_ctx != "TRUMP":
            return True
        
        # Compare trumps
        if card_ctx == "TRUMP" and best_ctx == "TRUMP":
            return trump_rank(card) < trump_rank(current_best)
        
        # Compare same suit
        if card_ctx == best_ctx and card_ctx != "TRUMP":
            return RANK_PRIORITY[rank_of(card)] < RANK_PRIORITY[rank_of(current_best)]
        
        return False
    
    def select_action(
        self,
        hand: List[str],
        legal_actions: List[str],
        current_trick: List[Tuple[int, str]],
        played_cards: List[str],
        player_idx: int,
        declarer_idx: int,
        partner_idx: Optional[int],
        points: List[int],
        trick_number: int,
    ) -> str:
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        # Categorize legal actions
        trumps = [c for c in legal_actions if is_trump(c)]
        non_trumps = [c for c in legal_actions if not is_trump(c)]
        
        # Sort by strength
        trumps_sorted = sorted(trumps, key=lambda c: trump_rank(c))  # Strongest first
        non_trumps_sorted = sorted(
            non_trumps,
            key=lambda c: (suit_of(c), RANK_PRIORITY[rank_of(c)])
        )
        
        # Am I on declarer's team?
        am_declarer_team = player_idx == declarer_idx or player_idx == partner_idx
        
        # === LEADING ===
        if not current_trick:
            return self._lead_strategy(
                legal_actions, trumps_sorted, non_trumps_sorted,
                am_declarer_team, trick_number, played_cards, hand
            )
        
        # === FOLLOWING ===
        lead_card = current_trick[0][1]
        lead_ctx = lead_context(lead_card)
        winner_idx, winner_card = self._current_winner(current_trick)
        
        # Is my partner currently winning?
        partner_winning = self._is_teammate(player_idx, winner_idx, declarer_idx, partner_idx)
        
        # Find cards that can beat current winner
        can_win = [c for c in legal_actions if self._can_beat(c, winner_card, lead_ctx)]
        cannot_win = [c for c in legal_actions if c not in can_win]
        
        # Trick points so far
        trick_points = sum(card_value(c) for _, c in current_trick)
        
        return self._follow_strategy(
            legal_actions, can_win, cannot_win,
            partner_winning, trick_points, lead_ctx,
            trumps_sorted, non_trumps_sorted, current_trick
        )
    
    def _lead_strategy(
        self,
        legal_actions: List[str],
        trumps: List[str],
        non_trumps: List[str],
        am_declarer_team: bool,
        trick_number: int,
        played_cards: List[str],
        hand: List[str],
    ) -> str:
        """Strategy when leading a trick."""
        
        # Early game: Pull trumps if we have strength
        if trick_number < 4 and trumps:
            # Count trumps remaining in play
            trumps_played = sum(1 for c in played_cards if is_trump(c))
            trumps_remaining = 14 - trumps_played - len([c for c in hand if is_trump(c)])
            
            # If we have good trumps and opponents have trumps, lead trump
            if trumps and len(trumps) >= 3:
                # Lead our strongest trump
                return trumps[0]
            elif trumps and trumps_remaining > 2:
                # Lead a mid-range trump to probe
                return trumps[len(trumps) // 2]
        
        # If we have aces in non-trump suits, cash them
        aces = [c for c in non_trumps if rank_of(c) == 'A']
        if aces:
            return aces[0]
        
        # Lead from longest suit
        suits = {}
        for c in non_trumps:
            s = suit_of(c)
            suits[s] = suits.get(s, []) + [c]
        
        if suits:
            longest_suit = max(suits.values(), key=len)
            # Lead low from long suit
            return sorted(longest_suit, key=lambda c: -RANK_PRIORITY[rank_of(c)])[0]
        
        # Default: lead lowest trump
        if trumps:
            return trumps[-1]
        
        return legal_actions[0]
    
    def _follow_strategy(
        self,
        legal_actions: List[str],
        can_win: List[str],
        cannot_win: List[str],
        partner_winning: bool,
        trick_points: int,
        lead_ctx: str,
        trumps: List[str],
        non_trumps: List[str],
        current_trick: List[Tuple[int, str]],
    ) -> str:
        """Strategy when following in a trick."""
        
        # If partner is winning with decent points, throw points
        if partner_winning and trick_points >= 10:
            # Throw highest value card we can't use to win
            point_cards = sorted(cannot_win, key=card_value, reverse=True)
            if point_cards and card_value(point_cards[0]) > 0:
                return point_cards[0]
        
        # If partner winning low-value trick, play low
        if partner_winning:
            # Play lowest card
            return sorted(legal_actions, key=lambda c: (card_value(c), RANK_PRIORITY.get(rank_of(c), 99)))[-1]
        
        # Partner NOT winning - try to win if worth it
        if can_win:
            # Worth winning if trick has points
            if trick_points >= 6:
                # Win with minimum necessary card
                return sorted(can_win, key=lambda c: (
                    -trump_rank(c) if is_trump(c) else RANK_PRIORITY[rank_of(c)]
                ))[-1]
            
            # Low points - win with cheap card if possible
            cheap_winners = [c for c in can_win if card_value(c) <= 2]
            if cheap_winners:
                return cheap_winners[0]
        
        # Can't win or not worth it - play low
        low_cards = sorted(legal_actions, key=lambda c: (card_value(c), -RANK_PRIORITY.get(rank_of(c), 0)))
        return low_cards[0]


# ============================================================================
# Threshold + Rules Combined Agent
# ============================================================================

class ThresholdRuleAgent(RuleBasedAgent):
    """
    Combines threshold-based hand evaluation with rule-based play.
    
    Tracks whether this hand "should" have been played based on
    statistical win rate expectations.
    """
    
    def __init__(self, play_threshold: float = 0.55):
        self.play_threshold = play_threshold
        self.last_evaluation: Optional[HandEvaluation] = None
    
    @property
    def name(self) -> str:
        return f"Threshold({self.play_threshold:.0%})+Rules"
    
    def evaluate_hand(self, hand: List[str]) -> HandEvaluation:
        """Evaluate whether this hand should be played."""
        self.last_evaluation = evaluate_hand_strength(hand, self.play_threshold)
        return self.last_evaluation
    
    def should_declare(self, hand: List[str]) -> bool:
        """Return True if this hand should be declared (played)."""
        return self.evaluate_hand(hand).should_play


# ============================================================================
# Agent Factory
# ============================================================================

def create_agent(agent_type: str, **kwargs) -> BaseAgent:
    """Create an agent by type name."""
    agents = {
        'random': RandomAgent,
        'rules': RuleBasedAgent,
        'threshold': ThresholdRuleAgent,
    }
    
    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}. Options: {list(agents.keys())}")
    
    return agents[agent_type](**kwargs)


# ============================================================================
# Testing / Demo
# ============================================================================

if __name__ == "__main__":
    # Test hand evaluation
    test_hands = [
        ['O_Acorn', 'O_Leaf', 'U_Acorn', 'A_Heart', '10_Heart', 'A_Acorn', 'K_Bell', '7_Leaf'],
        ['O_Acorn', 'U_Heart', '9_Heart', '7_Heart', 'A_Bell', 'K_Acorn', '10_Leaf', '8_Leaf'],
        ['U_Bell', '7_Heart', 'A_Bell', 'K_Bell', '10_Acorn', '9_Acorn', '8_Leaf', '7_Leaf'],
        ['9_Heart', '8_Heart', '7_Heart', 'A_Leaf', 'K_Leaf', '10_Leaf', '9_Leaf', '8_Leaf'],
    ]
    
    print("=" * 60)
    print("HAND EVALUATION TEST")
    print("=" * 60)
    
    for hand in test_hands:
        eval_result = evaluate_hand_strength(hand, threshold=0.55)
        status = "✓ PLAY" if eval_result.should_play else "✗ PASS"
        print(f"\n{status}")
        print(f"  Hand: {' '.join(sort_by_strength(hand))}")
        print(f"  {eval_result.reasoning}")
