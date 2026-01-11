"""
Hybrid agent that uses RuleBased but corrects known mistakes.

This agent starts with RuleBased logic but uses a lookup table
of known mistake situations to override suboptimal decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import hashlib

from .baseline import RuleBasedAgent
from .env import CARD_TO_IDX, IDX_TO_CARD


class HybridAgent:
    """
    Agent that combines RuleBased with MC-based corrections.
    
    Stores a hash table of game situations where RuleBased
    makes mistakes, and overrides those decisions.
    """
    
    def __init__(self, mc_data_path: str = "data/mc_training_data.csv", 
                 mistake_threshold: float = 0.05):
        self.rulebased = RuleBasedAgent()
        self.corrections = {}  # hash -> best_action
        self._load_corrections(mc_data_path, mistake_threshold)
    
    def _situation_hash(self, hand: List[str], current_trick: List[Tuple], 
                        played_cards: List[str], trick_number: int,
                        is_declarer: bool) -> str:
        """Create a hash of the current game situation."""
        # Sort hand for consistent hashing
        hand_sorted = sorted([str(c) for c in hand])
        trick_str = '|'.join(f"{p}:{c}" for p, c in current_trick)
        played_sorted = sorted([str(c) for c in played_cards])
        
        key = f"{','.join(hand_sorted)}|{trick_str}|{','.join(played_sorted)}|{trick_number}|{is_declarer}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _load_corrections(self, mc_data_path: str, threshold: float):
        """Load MC data and build correction table."""
        try:
            df = pd.read_csv(mc_data_path)
        except FileNotFoundError:
            print(f"Warning: {mc_data_path} not found, using pure RuleBased")
            return
        
        # Filter to mistakes
        mistakes = df[df['mistake_severity'] > threshold]
        
        print(f"HybridAgent: Loaded {len(mistakes)} corrections from MC data")
        
        for _, row in mistakes.iterrows():
            # Parse situation
            hand = row['hand'].split('|') if pd.notna(row['hand']) else []
            
            trick_str = row['current_trick']
            current_trick = []
            if pd.notna(trick_str) and trick_str:
                for item in str(trick_str).split('|'):
                    if ':' in item:
                        pos, card = item.split(':')
                        current_trick.append((int(pos), card))
            
            played_str = row['played_cards']
            played = str(played_str).split('|') if pd.notna(played_str) and played_str else []
            
            trick_number = int(row['trick_number'])
            is_declarer = bool(row['is_declarer'])
            
            # Create hash
            h = self._situation_hash(hand, current_trick, played, trick_number, is_declarer)
            
            # Store correction
            self.corrections[h] = row['best_action']
    
    def select_action(self, hand, legal_actions, current_trick, played_cards,
                      player_idx, declarer_idx, partner_idx, points, trick_number) -> str:
        """Select action, correcting RuleBased mistakes when recognized."""
        
        # Check if this is a known mistake situation
        declarer_team = {declarer_idx}
        if partner_idx is not None:
            declarer_team.add(partner_idx)
        is_declarer = player_idx in declarer_team
        
        hand_strs = [str(c) for c in hand]
        current_trick_strs = [(p, str(c)) for p, c in current_trick]
        played_strs = [str(c) for c in played_cards]
        
        h = self._situation_hash(hand_strs, current_trick_strs, played_strs, 
                                  trick_number, is_declarer)
        
        # Check for correction
        if h in self.corrections:
            correction = self.corrections[h]
            if correction in [str(c) for c in legal_actions]:
                return correction
        
        # Fall back to RuleBased
        return self.rulebased.select_action(
            hand=hand,
            legal_actions=legal_actions,
            current_trick=current_trick,
            played_cards=played_cards,
            player_idx=player_idx,
            declarer_idx=declarer_idx,
            partner_idx=partner_idx,
            points=points,
            trick_number=trick_number,
        )


def evaluate_hybrid_agent(num_games: int = 500, seed: int = 42) -> float:
    """Evaluate hybrid agent against pure RuleBased."""
    from .env import make_env
    
    env = make_env(seed=seed)
    hybrid = HybridAgent()
    rulebased = RuleBasedAgent()
    rng = np.random.default_rng(seed)
    
    hybrid_wins = 0
    
    print(f"Evaluating Hybrid agent vs RuleBased over {num_games} games...")
    
    for game in range(num_games):
        # Random declarer
        declarer = int(rng.integers(0, 4))
        
        # Hybrid plays 50% as declarer team, 50% as opponent
        hybrid_plays_declarer = game % 2 == 0
        
        env.reset(options={"fixed_declarer": declarer})
        
        while not all(env.terminations.values()):
            player_name = env.agent_selection
            player_idx = int(player_name.split("_")[1])
            
            hand = env._hands[player_idx]
            legal_indices = env._get_legal_actions(player_idx)
            legal_cards = [IDX_TO_CARD[idx] for idx in legal_indices]
            current_trick = [(p, c) for p, c in env._current_trick]
            
            # Determine if this player uses hybrid agent
            declarer_team = {env._declarer_idx}
            if env._partner_idx is not None:
                declarer_team.add(env._partner_idx)
            is_declarer_team = player_idx in declarer_team
            
            use_hybrid = (hybrid_plays_declarer == is_declarer_team)
            
            if use_hybrid:
                card = hybrid.select_action(
                    hand=hand,
                    legal_actions=legal_cards,
                    current_trick=current_trick,
                    played_cards=env._played_cards,
                    player_idx=player_idx,
                    declarer_idx=env._declarer_idx,
                    partner_idx=env._partner_idx,
                    points=env._points,
                    trick_number=env._trick_number,
                )
            else:
                card = rulebased.select_action(
                    hand=hand,
                    legal_actions=legal_cards,
                    current_trick=current_trick,
                    played_cards=env._played_cards,
                    player_idx=player_idx,
                    declarer_idx=env._declarer_idx,
                    partner_idx=env._partner_idx,
                    points=env._points,
                    trick_number=env._trick_number,
                )
            
            action = CARD_TO_IDX[card]
            env.step(action)
        
        result = env.get_game_result()
        declarer_won = result["win"]
        
        if hybrid_plays_declarer == declarer_won:
            hybrid_wins += 1
        
        if (game + 1) % 100 == 0:
            print(f"  Games {game+1}/{num_games}: Hybrid win rate = {hybrid_wins/(game+1)*100:.1f}%")
    
    win_rate = hybrid_wins / num_games
    print(f"\nFinal: Hybrid vs RuleBased: {hybrid_wins}/{num_games} = {win_rate*100:.1f}%")
    
    return win_rate


if __name__ == "__main__":
    evaluate_hybrid_agent()
