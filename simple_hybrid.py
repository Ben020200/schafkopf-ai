"""
Simple Hybrid Agent: Uses RuleBased + MC Q-value corrections.

Key insight: Don't learn a network. Just use the MC data directly:
1. Hash the game state to find similar situations in MC data
2. If we find a match AND the MC-best is significantly better, use it
3. Otherwise trust RuleBased

This is much simpler and more reliable than neural network approaches.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

from src.schafkopf_ai.baseline import RuleBasedAgent
from src.schafkopf_ai.env import CARD_TO_IDX, IDX_TO_CARD


class SimpleHybridAgent:
    """
    Hybrid agent that uses RuleBased + MC corrections.
    
    Strategy:
    1. Build a lookup table from MC data keyed by (hand_signature, trick_number, position)
    2. At inference, find matching situations
    3. If MC says a different card is significantly better (>0.1 Q-value), use it
    4. Otherwise trust RuleBased
    """
    
    def __init__(self, mc_data_path: str = "data/mc_training_data.csv",
                 min_improvement: float = 0.08):
        self.rulebased = RuleBasedAgent()
        self.min_improvement = min_improvement
        self.corrections = {}  # (hand_sig, trick_num, pos_in_trick, is_declarer) -> {card: q_value}
        self.stats = {"total": 0, "corrected": 0, "matched": 0}
        
        self._load_mc_data(mc_data_path)
    
    def _hand_signature(self, hand: List[str]) -> str:
        """Create a signature for the hand (sorted cards)."""
        return "|".join(sorted(hand))
    
    def _load_mc_data(self, path: str):
        """Load MC data into lookup table."""
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"Warning: {path} not found")
            return
        
        print(f"Loading MC data from {path}...")
        
        # Group by hand signature + game context
        for _, row in df.iterrows():
            hand = str(row['hand']).split('|') if pd.notna(row['hand']) else []
            if not hand:
                continue
            
            hand_sig = self._hand_signature(hand)
            trick_num = int(row['trick_number'])
            
            # Position in trick (0-3)
            trick_str = row['current_trick']
            if pd.notna(trick_str) and trick_str:
                pos = len(str(trick_str).split('|'))
            else:
                pos = 0
            
            is_declarer = bool(row['is_declarer'])
            
            key = (hand_sig, trick_num, pos, is_declarer)
            
            # Extract Q-values for all legal actions
            legal_str = row['legal_actions']
            legal_cards = str(legal_str).split('|') if pd.notna(legal_str) else []
            
            q_values = {}
            for card in legal_cards:
                col = f"value_{card}"
                if col in df.columns and pd.notna(row[col]):
                    q_values[card] = float(row[col])
            
            if q_values:
                # Store best action and its value
                best_card = max(q_values, key=q_values.get)
                self.corrections[key] = {
                    'best': best_card,
                    'best_q': q_values[best_card],
                    'all_q': q_values,
                    'rb_choice': row['rulebased_choice'],
                    'rb_q': float(row['rulebased_value']) if pd.notna(row['rulebased_value']) else 0.5,
                }
        
        print(f"  Loaded {len(self.corrections)} unique situations")
    
    def select_action(self, hand: List[str], legal_actions: List[str],
                      current_trick: List[Tuple[int, str]],
                      played_cards: List[str],
                      player_idx: int, declarer_idx: int,
                      partner_idx: Optional[int],
                      points: List[int], trick_number: int) -> str:
        """Select action with potential MC correction."""
        
        self.stats["total"] += 1
        
        # Get RuleBased choice
        rb_choice = self.rulebased.select_action(
            hand=hand, legal_actions=legal_actions,
            current_trick=current_trick, played_cards=played_cards,
            player_idx=player_idx, declarer_idx=declarer_idx,
            partner_idx=partner_idx, points=points, trick_number=trick_number
        )
        
        # Only one option - no decision to make
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        # Build lookup key
        hand_strs = [str(c) for c in hand]
        hand_sig = self._hand_signature(hand_strs)
        pos = len(current_trick)
        
        declarer_team = {declarer_idx}
        if partner_idx is not None:
            declarer_team.add(partner_idx)
        is_declarer = player_idx in declarer_team
        
        key = (hand_sig, trick_number, pos, is_declarer)
        
        # Check if we have MC data for this exact situation
        if key in self.corrections:
            self.stats["matched"] += 1
            corr = self.corrections[key]
            
            # Get Q-value for RuleBased choice
            rb_q = corr['all_q'].get(str(rb_choice), corr['rb_q'])
            
            # Get best action from MC
            best_card = corr['best']
            best_q = corr['best_q']
            
            # If MC-best is significantly better AND is legal, use it
            improvement = best_q - rb_q
            if improvement > self.min_improvement and best_card in [str(c) for c in legal_actions]:
                self.stats["corrected"] += 1
                return best_card
        
        return rb_choice
    
    def get_stats(self) -> Dict:
        """Return statistics."""
        stats = self.stats.copy()
        if stats["total"] > 0:
            stats["match_rate"] = stats["matched"] / stats["total"]
            stats["correction_rate"] = stats["corrected"] / stats["total"]
        return stats
    
    def reset_stats(self):
        self.stats = {"total": 0, "corrected": 0, "matched": 0}


def evaluate(num_games: int = 200):
    """Evaluate SimpleHybrid vs RuleBased."""
    from src.schafkopf_ai.env import make_env
    
    print("=" * 60)
    print("SIMPLE HYBRID AGENT EVALUATION")
    print("=" * 60)
    
    env = make_env()
    hybrid = SimpleHybridAgent(min_improvement=0.08)
    rb = RuleBasedAgent()
    
    wins = 0
    
    for game in range(num_games):
        env.reset()
        hybrid_plays_declarer = game % 2 == 0
        
        while not all(env.terminations.values()):
            player_name = env.agent_selection
            player_idx = int(player_name.split('_')[1])
            
            hand = env._hands[player_idx]
            legal_indices = env._get_legal_actions(player_idx)
            legal_cards = [IDX_TO_CARD[idx] for idx in legal_indices]
            
            declarer_team = {env._declarer_idx}
            if env._partner_idx is not None:
                declarer_team.add(env._partner_idx)
            is_declarer_team = player_idx in declarer_team
            use_hybrid = (hybrid_plays_declarer == is_declarer_team)
            
            agent = hybrid if use_hybrid else rb
            
            card = agent.select_action(
                hand=hand, legal_actions=legal_cards,
                current_trick=env._current_trick, played_cards=env._played_cards,
                player_idx=player_idx, declarer_idx=env._declarer_idx,
                partner_idx=env._partner_idx, points=env._points,
                trick_number=env._trick_number
            )
            
            action = CARD_TO_IDX[card]
            env.step(action)
        
        result = env.get_game_result()
        if hybrid_plays_declarer == result["win"]:
            wins += 1
        
        if (game + 1) % 50 == 0:
            print(f"  Games {game+1}/{num_games}: Win rate = {wins/(game+1)*100:.1f}%")
    
    print(f"\nFinal: {wins}/{num_games} = {wins/num_games*100:.1f}% win rate")
    
    stats = hybrid.get_stats()
    print(f"\nHybrid stats:")
    print(f"  Match rate: {stats.get('match_rate', 0)*100:.1f}%")
    print(f"  Correction rate: {stats.get('correction_rate', 0)*100:.1f}%")
    
    return wins / num_games


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--min-improvement", type=float, default=0.08)
    args = parser.parse_args()
    
    evaluate(args.games)
