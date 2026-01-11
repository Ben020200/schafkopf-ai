#!/usr/bin/env python3
"""Compare different agents: Random, Rule-Based, and PPO."""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
import torch

# Fix module path for checkpoint loading
sys.modules['schafkopf_ai'] = __import__('src.schafkopf_ai', fromlist=[''])
sys.modules['schafkopf_ai.ppo'] = __import__('src.schafkopf_ai.ppo', fromlist=[''])

from src.schafkopf_ai.env import SchafkopfEnv, IDX_TO_CARD, CARD_TO_IDX
from src.schafkopf_ai.ppo import SchafkopfNetwork
from src.schafkopf_ai.baseline import (
    RandomAgent, RuleBasedAgent, ThresholdRuleAgent,
    BaseAgent, evaluate_hand_strength
)
from src.schafkopf_ai.cards import sort_by_strength


class PPOAgent(BaseAgent):
    """Wrapper for trained PPO model as an agent."""
    
    def __init__(self, model_path: str):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        self.network = SchafkopfNetwork(hidden_size=config.hidden_size)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.network.eval()
        self._name = f"PPO ({model_path.split('/')[-1]})"
    
    @property
    def name(self) -> str:
        return self._name
    
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
        # This is a simplified version - in practice we need the full observation
        # For now, return random (we'll use the env-based approach below)
        return random.choice(legal_actions)
    
    def select_action_from_obs(self, obs: dict, mask: np.ndarray) -> int:
        """Select action from environment observation."""
        obs_dict = {
            'own_hand': torch.FloatTensor(obs['own_hand']).unsqueeze(0),
            'played_cards': torch.FloatTensor(obs['played_cards']).unsqueeze(0),
            'current_trick': torch.FloatTensor(obs['current_trick']).unsqueeze(0),
            'points': torch.FloatTensor(obs['points']).unsqueeze(0),
            'game_info': torch.FloatTensor(obs['game_info']).unsqueeze(0),
        }
        mask_tensor = torch.FloatTensor(mask).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = self.network(obs_dict, mask_tensor)
            action = torch.argmax(logits, dim=1).item()
        
        return action


@dataclass
class MatchResult:
    """Result of a single game."""
    declarer_wins: bool
    declarer_points: int
    schneider: bool
    schwarz: bool
    declarer_hand_eval: float  # Estimated win rate for declarer's hand


@dataclass 
class MatchupStats:
    """Aggregated stats for a matchup."""
    games: int = 0
    declarer_wins: int = 0
    total_points: int = 0
    schneider_count: int = 0
    schwarz_count: int = 0
    expected_wins: float = 0.0  # Sum of expected win rates
    
    @property
    def win_rate(self) -> float:
        return self.declarer_wins / self.games if self.games > 0 else 0.0
    
    @property
    def avg_points(self) -> float:
        return self.total_points / self.games if self.games > 0 else 0.0
    
    @property
    def expected_win_rate(self) -> float:
        return self.expected_wins / self.games if self.games > 0 else 0.0


def run_game_with_agents(
    declarer_agent: BaseAgent,
    opponent_agent: BaseAgent,
    seed: Optional[int] = None,
) -> MatchResult:
    """
    Run a single game with specified agents.
    
    Declarer team (players 0, 1) uses declarer_agent.
    Opponent team (players 2, 3) uses opponent_agent.
    """
    env = SchafkopfEnv(seed=seed)
    env.reset()
    
    # Get declarer's initial hand for evaluation
    declarer_hand = list(env._hands[0])
    hand_eval = evaluate_hand_strength(declarer_hand)
    
    # Play the game
    while env.agents:
        agent_name = env.agent_selection
        player_idx = env.agent_name_mapping[agent_name]
        
        obs, reward, term, trunc, info = env.last()
        
        if term or trunc:
            env.step(None)
            continue
        
        mask = obs['action_mask']
        legal_actions = [i for i in range(32) if mask[i]]
        legal_cards = [IDX_TO_CARD[i] for i in legal_actions]
        
        # Determine which agent to use
        # Declarer team = player 0 and partner (player who has called ace)
        is_declarer_team = player_idx in [0, env._partner_idx]
        
        if is_declarer_team:
            agent = declarer_agent
        else:
            agent = opponent_agent
        
        # Get action
        if isinstance(agent, PPOAgent):
            action = agent.select_action_from_obs(obs, mask)
        else:
            # Build context for rule-based agents
            hand = env._hands[player_idx]
            current_trick = env._current_trick
            played_cards = env._played_cards
            points = env._points
            
            card = agent.select_action(
                hand=hand,
                legal_actions=legal_cards,
                current_trick=current_trick,
                played_cards=played_cards,
                player_idx=player_idx,
                declarer_idx=env._declarer_idx,
                partner_idx=env._partner_idx,
                points=points,
                trick_number=env._trick_number,
            )
            action = CARD_TO_IDX[card]
        
        env.step(action)
    
    # Get result
    result = env.get_game_result()
    
    return MatchResult(
        declarer_wins=result['win'],
        declarer_points=result['declarer_points'],
        schneider=result['schneider'],
        schwarz=result['schwarz'],
        declarer_hand_eval=hand_eval.estimated_win_rate,
    )


def run_matchup(
    declarer_agent: BaseAgent,
    opponent_agent: BaseAgent,
    num_games: int,
    seed: Optional[int] = None,
) -> MatchupStats:
    """Run multiple games between two agents."""
    stats = MatchupStats()
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    for i in range(num_games):
        result = run_game_with_agents(declarer_agent, opponent_agent)
        
        stats.games += 1
        if result.declarer_wins:
            stats.declarer_wins += 1
        stats.total_points += result.declarer_points
        if result.schneider:
            stats.schneider_count += 1
        if result.schwarz:
            stats.schwarz_count += 1
        stats.expected_wins += result.declarer_hand_eval
        
        if (i + 1) % 500 == 0:
            print(f"  Games: {i+1}/{num_games} | Win Rate: {stats.win_rate:.1%}")
    
    return stats


def run_self_play(agent: BaseAgent, num_games: int, seed: Optional[int] = None) -> MatchupStats:
    """Run self-play games (all players use same agent)."""
    return run_matchup(agent, agent, num_games, seed)


def run_comparison(
    ppo_path: Optional[str],
    num_games: int = 1000,
    seed: Optional[int] = None,
):
    """Run full comparison between all agent types."""
    
    # Create agents
    agents = {
        'Random': RandomAgent(),
        'Rules': RuleBasedAgent(),
        'Threshold55': ThresholdRuleAgent(play_threshold=0.55),
    }
    
    if ppo_path:
        agents['PPO'] = PPOAgent(ppo_path)
    
    print("=" * 70)
    print("AGENT COMPARISON")
    print("=" * 70)
    print(f"Games per matchup: {num_games}")
    print(f"Agents: {', '.join(agents.keys())}")
    print()
    
    results = {}
    
    # Self-play for each agent
    print("\n--- SELF-PLAY RESULTS ---")
    print("(Declarer team win rate when all players use same agent)")
    print()
    
    for name, agent in agents.items():
        print(f"Testing {name} self-play...")
        stats = run_self_play(agent, num_games, seed)
        results[f"{name}_self"] = stats
        print(f"  {name}: {stats.win_rate:.1%} win rate, {stats.avg_points:.1f} avg points")
        print(f"         Expected: {stats.expected_win_rate:.1%} (based on hand strength)")
        print()
    
    # Head-to-head matchups
    if len(agents) > 1:
        print("\n--- HEAD-TO-HEAD MATCHUPS ---")
        print("(Declarer team uses Agent A, Opponent team uses Agent B)")
        print()
        
        agent_names = list(agents.keys())
        for i, name_a in enumerate(agent_names):
            for name_b in agent_names[i+1:]:
                # A as declarer vs B
                print(f"{name_a} (declarer) vs {name_b} (opponent)...")
                stats_ab = run_matchup(agents[name_a], agents[name_b], num_games, seed)
                results[f"{name_a}_vs_{name_b}"] = stats_ab
                
                # B as declarer vs A
                print(f"{name_b} (declarer) vs {name_a} (opponent)...")
                stats_ba = run_matchup(agents[name_b], agents[name_a], num_games, seed)
                results[f"{name_b}_vs_{name_a}"] = stats_ba
                
                print(f"  {name_a} as declarer: {stats_ab.win_rate:.1%}")
                print(f"  {name_b} as declarer: {stats_ba.win_rate:.1%}")
                print()
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nSelf-Play Win Rates:")
    print("-" * 40)
    for name in agents.keys():
        stats = results[f"{name}_self"]
        diff = stats.win_rate - stats.expected_win_rate
        print(f"  {name:15s}: {stats.win_rate:5.1%} (expected {stats.expected_win_rate:.1%}, Δ={diff:+.1%})")
    
    if len(agents) > 1:
        print("\nHead-to-Head (Declarer Win Rate):")
        print("-" * 50)
        
        # Build matrix
        agent_names = list(agents.keys())
        print(f"{'Declarer':<12} | " + " | ".join(f"{n:>10}" for n in agent_names))
        print("-" * (14 + 13 * len(agent_names)))
        
        for name_a in agent_names:
            row = []
            for name_b in agent_names:
                if name_a == name_b:
                    row.append(f"{'--':>10}")
                else:
                    key = f"{name_a}_vs_{name_b}"
                    if key in results:
                        row.append(f"{results[key].win_rate:>9.1%}")
                    else:
                        row.append(f"{'N/A':>10}")
            print(f"{name_a:<12} | " + " | ".join(row))
    
    print()
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare Schafkopf agents')
    parser.add_argument('--ppo', type=str, default=None, 
                       help='Path to PPO model checkpoint')
    parser.add_argument('--games', type=int, default=1000,
                       help='Number of games per matchup')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    run_comparison(
        ppo_path=args.ppo,
        num_games=args.games,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
