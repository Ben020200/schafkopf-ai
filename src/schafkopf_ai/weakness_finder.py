"""
Analysis tools to find weaknesses in RuleBased agent.

Uses Monte Carlo simulation to compare RuleBased's decisions
against alternatives, identifying suboptimal plays.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import copy
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from .env import SchafkopfEnv, make_env, CARD_TO_IDX, IDX_TO_CARD
from .baseline import RuleBasedAgent


@dataclass
class DecisionAnalysis:
    """Analysis of a single RuleBased decision."""
    game_id: int
    trick_number: int
    player_idx: int
    is_declarer: bool
    
    # Game state
    hand: List[str] = field(default_factory=list)
    current_trick: List[Tuple[int, str]] = field(default_factory=list)
    played_cards: List[str] = field(default_factory=list)
    points_so_far: Dict[str, int] = field(default_factory=dict)
    
    # Decision analysis
    legal_actions: List[str] = field(default_factory=list)
    rulebased_choice: str = ""
    action_values: Dict[str, float] = field(default_factory=dict)  # Expected win rate per action
    best_action: str = ""
    best_value: float = 0.0
    rulebased_value: float = 0.0
    
    # Is this a mistake?
    @property
    def is_mistake(self) -> bool:
        return self.best_value - self.rulebased_value > 0.05  # 5% threshold
    
    @property
    def mistake_severity(self) -> float:
        return self.best_value - self.rulebased_value


def simulate_game_to_end(env: SchafkopfEnv, rng: np.random.Generator) -> bool:
    """
    Simulate random play until game ends.
    Returns True if declarer team won.
    """
    while not all(env.terminations.values()):
        player_name = env.agent_selection
        player_idx = int(player_name.split("_")[1])
        
        legal_actions = env._get_legal_actions(player_idx)
        action = rng.choice(legal_actions)
        env.step(action)
    
    result = env.get_game_result()
    return result["win"]


def analyze_action_value(
    env: SchafkopfEnv,
    action: int,
    player_idx: int,
    is_declarer_team: bool,
    num_simulations: int,
    rng: np.random.Generator,
) -> float:
    """
    Estimate expected win rate if we take this action.
    Returns win rate from current player's team perspective.
    """
    team_wins = 0
    
    for _ in range(num_simulations):
        # Deep copy and take action
        sim_env = copy.deepcopy(env)
        sim_env.step(action)
        
        # If game ended
        if all(sim_env.terminations.values()):
            result = sim_env.get_game_result()
            declarer_won = result["win"]
            if is_declarer_team == declarer_won:
                team_wins += 1
            continue
        
        # Simulate to completion
        declarer_won = simulate_game_to_end(sim_env, rng)
        if is_declarer_team == declarer_won:
            team_wins += 1
    
    return team_wins / num_simulations


def analyze_single_decision(
    env: SchafkopfEnv,
    rulebased_action: int,
    game_id: int,
    num_simulations: int,
    rng: np.random.Generator,
) -> DecisionAnalysis:
    """
    Analyze one decision point using Monte Carlo.
    """
    player_name = env.agent_selection
    player_idx = int(player_name.split("_")[1])
    
    # Determine team membership
    declarer_team = {env._declarer_idx}
    if env._partner_idx is not None:
        declarer_team.add(env._partner_idx)
    is_declarer_team = player_idx in declarer_team
    
    # Capture state
    hand = [str(c) for c in env._hands[player_idx]]
    current_trick = [(p, str(c)) for p, c in env._current_trick]
    played_cards = [str(c) for c in env._played_cards]
    points_dict = {f"player_{i}": p for i, p in enumerate(env._points)}
    
    # Get legal actions
    legal_indices = env._get_legal_actions(player_idx)
    legal_cards = [IDX_TO_CARD[idx] for idx in legal_indices]
    
    # Evaluate each action
    action_values = {}
    for action_idx in legal_indices:
        card = IDX_TO_CARD[action_idx]
        value = analyze_action_value(
            env, action_idx, player_idx, is_declarer_team,
            num_simulations, rng
        )
        action_values[card] = value
    
    # Find best
    best_action = max(action_values.keys(), key=lambda x: action_values[x])
    best_value = action_values[best_action]
    rulebased_card = IDX_TO_CARD[rulebased_action]
    rulebased_value = action_values[rulebased_card]
    
    return DecisionAnalysis(
        game_id=game_id,
        trick_number=env._trick_number,
        player_idx=player_idx,
        is_declarer=player_idx in declarer_team,
        hand=hand,
        current_trick=current_trick,
        played_cards=played_cards,
        points_so_far=points_dict,
        legal_actions=legal_cards,
        rulebased_choice=rulebased_card,
        action_values=action_values,
        best_action=best_action,
        best_value=best_value,
        rulebased_value=rulebased_value,
    )


def find_rulebased_mistakes(
    num_games: int = 100,
    simulations_per_action: int = 50,
    mistake_threshold: float = 0.05,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[DecisionAnalysis], List[DecisionAnalysis]]:
    """
    Play games and find all RuleBased mistakes.
    
    Returns:
        Tuple of (all_decisions, mistakes)
    """
    env = make_env(seed=seed)
    agent = RuleBasedAgent()
    rng = np.random.default_rng(seed)
    
    all_decisions = []
    mistakes = []
    
    if verbose:
        print(f"Analyzing {num_games} games for RuleBased mistakes...")
        print(f"Simulations per action: {simulations_per_action}")
        print(f"Mistake threshold: {mistake_threshold*100:.0f}%")
        print("-" * 50)
    
    start_time = time.time()
    
    for game in range(num_games):
        random_declarer = int(rng.integers(0, 4))
        env.reset(options={"fixed_declarer": random_declarer})
        
        game_decisions = 0
        game_mistakes = 0
        
        while not all(env.terminations.values()):
            player_name = env.agent_selection
            player_idx = int(player_name.split("_")[1])
            
            # Get RuleBased action
            hand = env._hands[player_idx]
            legal_indices = env._get_legal_actions(player_idx)
            legal_cards = [IDX_TO_CARD[idx] for idx in legal_indices]
            current_trick = [(p, c) for p, c in env._current_trick]
            
            card = agent.select_action(
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
            
            # Only analyze if there's a choice (>1 legal action)
            if len(legal_indices) > 1:
                analysis = analyze_single_decision(
                    env, action, game, simulations_per_action, rng
                )
                all_decisions.append(analysis)
                game_decisions += 1
                
                if analysis.mistake_severity > mistake_threshold:
                    mistakes.append(analysis)
                    game_mistakes += 1
            
            env.step(action)
        
        if verbose and (game + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Game {game+1}/{num_games}: {game_decisions} decisions, "
                  f"{game_mistakes} mistakes | {elapsed:.1f}s")
    
    if verbose:
        total_time = time.time() - start_time
        print("-" * 50)
        print(f"Total decisions analyzed: {len(all_decisions)}")
        print(f"Total mistakes found: {len(mistakes)}")
        if len(all_decisions) > 0:
            print(f"Mistake rate: {len(mistakes)/len(all_decisions)*100:.1f}%")
        print(f"Total time: {total_time:.1f}s")
    
    return all_decisions, mistakes


def summarize_mistakes(mistakes: List[DecisionAnalysis]) -> None:
    """Print detailed summary of mistakes found."""
    if not mistakes:
        print("\nNo mistakes found!")
        return
    
    print("\n" + "=" * 60)
    print("RULEBASED WEAKNESS ANALYSIS")
    print("=" * 60)
    
    # Severity distribution
    severe = [m for m in mistakes if m.mistake_severity > 0.15]
    moderate = [m for m in mistakes if 0.10 < m.mistake_severity <= 0.15]
    minor = [m for m in mistakes if m.mistake_severity <= 0.10]
    
    print(f"\nMistake Severity Distribution:")
    print(f"  Severe (>15% loss):     {len(severe)}")
    print(f"  Moderate (10-15% loss): {len(moderate)}")
    print(f"  Minor (5-10% loss):     {len(minor)}")
    
    avg_severity = np.mean([m.mistake_severity for m in mistakes])
    print(f"\nAverage severity: {avg_severity*100:.1f}%")
    
    # By trick number
    by_trick = defaultdict(list)
    for m in mistakes:
        by_trick[m.trick_number].append(m)
    
    print(f"\nMistakes by Trick Number:")
    for trick in sorted(by_trick.keys()):
        avg = np.mean([m.mistake_severity for m in by_trick[trick]])
        print(f"  Trick {trick}: {len(by_trick[trick]):3d} mistakes (avg severity: {avg*100:.1f}%)")
    
    # By role
    declarer_mistakes = [m for m in mistakes if m.is_declarer]
    opponent_mistakes = [m for m in mistakes if not m.is_declarer]
    
    print(f"\nMistakes by Role:")
    print(f"  Declarer team: {len(declarer_mistakes)}")
    print(f"  Opponent team: {len(opponent_mistakes)}")
    
    # Pattern analysis: what cards are involved?
    print(f"\nCard Pattern Analysis:")
    card_mistakes = defaultdict(list)
    for m in mistakes:
        # What suit was the mistake in?
        suit = m.rulebased_choice[-1]  # Last char is suit
        card_mistakes[suit].append(m)
    
    for suit in ['E', 'G', 'H', 'S']:  # Eichel, Gras, Herz, Schellen
        if suit in card_mistakes:
            print(f"  {suit}: {len(card_mistakes[suit])} mistakes")
    
    # Show worst mistakes with explanation
    print(f"\n" + "=" * 60)
    print("TOP 10 WORST RULEBASED MISTAKES")
    print("=" * 60)
    
    sorted_mistakes = sorted(mistakes, key=lambda x: x.mistake_severity, reverse=True)
    
    for i, m in enumerate(sorted_mistakes[:10]):
        print(f"\n{'='*60}")
        print(f"MISTAKE #{i+1} - Severity: {m.mistake_severity*100:.1f}%")
        print(f"{'='*60}")
        print(f"Game {m.game_id}, Trick {m.trick_number}")
        print(f"Player {m.player_idx} ({'DECLARER' if m.is_declarer else 'OPPONENT'})")
        print(f"\nHand: {' '.join(m.hand)}")
        print(f"Current trick: {m.current_trick if m.current_trick else 'Empty (leading)'}")
        print(f"\nLegal options:")
        
        # Sort by value
        sorted_actions = sorted(m.action_values.items(), key=lambda x: x[1], reverse=True)
        for card, value in sorted_actions:
            marker = "✓ BEST" if card == m.best_action else ""
            marker = "✗ CHOSEN" if card == m.rulebased_choice else marker
            print(f"  {card}: {value*100:5.1f}% {marker}")
        
        print(f"\nRuleBased chose: {m.rulebased_choice} ({m.rulebased_value*100:.1f}%)")
        print(f"Better choice:   {m.best_action} ({m.best_value*100:.1f}%)")
        print(f"Lost: {m.mistake_severity*100:.1f}% win equity")


def analyze_mistake_patterns(mistakes: List[DecisionAnalysis]) -> Dict:
    """
    Analyze patterns in mistakes to find exploitable weaknesses.
    """
    patterns = {
        "leading_mistakes": [],
        "following_mistakes": [],
        "trump_mistakes": [],
        "non_trump_mistakes": [],
        "endgame_mistakes": [],  # Trick 5-7
        "declarer_leading": [],
        "declarer_following": [],
        "opponent_leading": [],
        "opponent_following": [],
    }
    
    trumps = {'O_Acorn', 'O_Leaf', 'O_Heart', 'O_Bell',
              'U_Acorn', 'U_Leaf', 'U_Heart', 'U_Bell',
              'A_Heart', '10_Heart', 'K_Heart', '9_Heart', '8_Heart', '7_Heart'}
    
    for m in mistakes:
        is_leading = len(m.current_trick) == 0
        
        # Leading vs following
        if is_leading:
            patterns["leading_mistakes"].append(m)
        else:
            patterns["following_mistakes"].append(m)
        
        # Trump vs non-trump
        if m.rulebased_choice in trumps:
            patterns["trump_mistakes"].append(m)
        else:
            patterns["non_trump_mistakes"].append(m)
        
        # Endgame
        if m.trick_number >= 5:
            patterns["endgame_mistakes"].append(m)
        
        # Combined role + position
        if m.is_declarer and is_leading:
            patterns["declarer_leading"].append(m)
        elif m.is_declarer and not is_leading:
            patterns["declarer_following"].append(m)
        elif not m.is_declarer and is_leading:
            patterns["opponent_leading"].append(m)
        else:
            patterns["opponent_following"].append(m)
    
    print("\n" + "=" * 60)
    print("PATTERN ANALYSIS")
    print("=" * 60)
    
    for pattern_name, pattern_mistakes in patterns.items():
        if pattern_mistakes:
            avg = np.mean([m.mistake_severity for m in pattern_mistakes])
            print(f"\n{pattern_name.replace('_', ' ').title()}:")
            print(f"  Count: {len(pattern_mistakes)}")
            print(f"  Avg severity: {avg*100:.1f}%")
    
    return patterns


def export_training_data(
    all_decisions: List[DecisionAnalysis],
    filepath: str = "data/mc_training_data.csv",
) -> None:
    """
    Export all decisions with MC-computed action values for training.
    This creates a dataset where each row is a decision point with
    the true (MC-estimated) value of each action.
    """
    import csv
    import os
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    rows = []
    for d in all_decisions:
        # Find best action and its value
        best_action = max(d.action_values.keys(), key=lambda x: d.action_values[x])
        best_value = d.action_values[best_action]
        
        row = {
            'game_id': d.game_id,
            'trick_number': d.trick_number,
            'player_idx': d.player_idx,
            'is_declarer': d.is_declarer,
            'hand': '|'.join(d.hand),
            'current_trick': '|'.join([f"{p}:{c}" for p, c in d.current_trick]),
            'played_cards': '|'.join(d.played_cards),
            'legal_actions': '|'.join(d.legal_actions),
            'rulebased_choice': d.rulebased_choice,
            'rulebased_value': d.rulebased_value,
            'best_action': best_action,
            'best_value': best_value,
            'is_mistake': d.is_mistake,
            'mistake_severity': d.mistake_severity,
            'num_options': len(d.legal_actions),
        }
        
        # Add individual action values
        for card, value in d.action_values.items():
            row[f'value_{card}'] = value
        
        rows.append(row)
    
    # Get all columns
    all_cols = set()
    for row in rows:
        all_cols.update(row.keys())
    
    # Sort columns for consistent ordering
    base_cols = ['game_id', 'trick_number', 'player_idx', 'is_declarer', 
                 'hand', 'current_trick', 'played_cards', 'legal_actions',
                 'rulebased_choice', 'rulebased_value', 'best_action', 'best_value',
                 'is_mistake', 'mistake_severity', 'num_options']
    value_cols = sorted([c for c in all_cols if c.startswith('value_')])
    all_cols = base_cols + value_cols
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nExported {len(rows)} decisions to {filepath}")
    
    # Statistics
    mistakes = [d for d in all_decisions if d.is_mistake]
    print(f"  Total decisions: {len(all_decisions)}")
    print(f"  Mistakes (>5% loss): {len(mistakes)}")
    print(f"  Mistake rate: {len(mistakes)/len(all_decisions)*100:.1f}%")


# ============================================================================
# FAST PARALLEL VERSION
# ============================================================================

def _analyze_game_worker(args):
    """Worker function to analyze a single game (for parallel processing)."""
    game_id, seed, simulations_per_action, mistake_threshold = args
    
    # Create fresh env and agent for this process
    env = make_env(seed=seed + game_id)
    agent = RuleBasedAgent()
    rng = np.random.default_rng(seed + game_id)
    
    decisions = []
    
    random_declarer = int(rng.integers(0, 4))
    env.reset(options={"fixed_declarer": random_declarer})
    
    while not all(env.terminations.values()):
        player_name = env.agent_selection
        player_idx = int(player_name.split("_")[1])
        
        hand = env._hands[player_idx]
        legal_indices = env._get_legal_actions(player_idx)
        legal_cards = [IDX_TO_CARD[idx] for idx in legal_indices]
        current_trick = [(p, c) for p, c in env._current_trick]
        
        card = agent.select_action(
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
        
        # Only analyze if there's a choice
        if len(legal_indices) > 1:
            analysis = analyze_single_decision(
                env, action, game_id, simulations_per_action, rng
            )
            decisions.append(analysis)
        
        env.step(action)
    
    return decisions


def find_rulebased_mistakes_fast(
    num_games: int = 100,
    simulations_per_action: int = 50,
    mistake_threshold: float = 0.05,
    seed: int = 42,
    num_workers: int = None,
    verbose: bool = True,
) -> Tuple[List[DecisionAnalysis], List[DecisionAnalysis]]:
    """
    Fast parallel version of find_rulebased_mistakes.
    
    Uses multiprocessing to analyze multiple games in parallel.
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)
    
    if verbose:
        print(f"Fast parallel analysis with {num_workers} workers")
        print(f"Analyzing {num_games} games for RuleBased mistakes...")
        print(f"Simulations per action: {simulations_per_action}")
        print(f"Mistake threshold: {mistake_threshold*100:.0f}%")
        print("-" * 50)
    
    start_time = time.time()
    
    # Prepare arguments for each game
    args_list = [
        (game_id, seed, simulations_per_action, mistake_threshold)
        for game_id in range(num_games)
    ]
    
    all_decisions = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_analyze_game_worker, args): args[0] 
                   for args in args_list}
        
        for future in as_completed(futures):
            game_id = futures[future]
            try:
                decisions = future.result()
                all_decisions.extend(decisions)
                completed += 1
                
                if verbose and completed % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (num_games - completed) / rate if rate > 0 else 0
                    print(f"Completed {completed}/{num_games} games | "
                          f"{rate:.1f} games/s | ETA: {eta:.0f}s")
            except Exception as e:
                print(f"Game {game_id} failed: {e}")
    
    # Filter mistakes
    mistakes = [d for d in all_decisions if d.mistake_severity > mistake_threshold]
    
    if verbose:
        total_time = time.time() - start_time
        print("-" * 50)
        print(f"Total time: {total_time:.1f}s ({num_games/total_time:.1f} games/s)")
        print(f"Total decisions analyzed: {len(all_decisions)}")
        print(f"Total mistakes found: {len(mistakes)}")
        if len(all_decisions) > 0:
            print(f"Mistake rate: {len(mistakes)/len(all_decisions)*100:.1f}%")
    
    return all_decisions, mistakes


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Find RuleBased agent weaknesses")
    parser.add_argument("--games", type=int, default=50, help="Number of games to analyze")
    parser.add_argument("--sims", type=int, default=50, help="MC simulations per action")
    parser.add_argument("--threshold", type=float, default=0.05, help="Mistake threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fast", action="store_true", help="Use parallel processing")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    args = parser.parse_args()
    
    if args.fast:
        all_decisions, mistakes = find_rulebased_mistakes_fast(
            num_games=args.games,
            simulations_per_action=args.sims,
            mistake_threshold=args.threshold,
            seed=args.seed,
            num_workers=args.workers,
        )
    else:
        all_decisions, mistakes = find_rulebased_mistakes(
            num_games=args.games,
            simulations_per_action=args.sims,
            mistake_threshold=args.threshold,
            seed=args.seed,
        )
    
    summarize_mistakes(mistakes)
    analyze_mistake_patterns(mistakes)
