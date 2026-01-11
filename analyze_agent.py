#!/usr/bin/env python3
"""Analyze trained Schafkopf agent performance by hand characteristics."""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random

import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fix for checkpoint module path
sys.modules['schafkopf_ai'] = __import__('src.schafkopf_ai', fromlist=[''])
sys.modules['schafkopf_ai.ppo'] = __import__('src.schafkopf_ai.ppo', fromlist=[''])

from src.schafkopf_ai.env import SchafkopfEnv
from src.schafkopf_ai.ppo import SchafkopfNetwork, PPOConfig
from src.schafkopf_ai.features import featurize_hand, HandFeatures
from src.schafkopf_ai.cards import (
    TRUMP_STRENGTH, TRUMP_ORDER, is_trump, trump_strength, 
    sort_by_strength, card_value
)


@dataclass
class HandResult:
    """Record of a single game's outcome with hand features."""
    declarer_hand: List[str]
    features: HandFeatures
    win: bool
    declarer_points: int
    schneider: bool
    schwarz: bool
    
    # Specific trump cards held
    has_ober_acorn: bool = False
    has_ober_leaf: bool = False
    has_ober_heart: bool = False
    has_ober_bell: bool = False
    has_unter_acorn: bool = False
    has_unter_leaf: bool = False
    has_unter_heart: bool = False
    has_unter_bell: bool = False
    has_ace_heart: bool = False


def get_specific_trumps(hand: List[str]) -> Dict[str, bool]:
    """Check which specific high trumps are in the hand."""
    return {
        'has_ober_acorn': 'O_Acorn' in hand,
        'has_ober_leaf': 'O_Leaf' in hand,
        'has_ober_heart': 'O_Heart' in hand,
        'has_ober_bell': 'O_Bell' in hand,
        'has_unter_acorn': 'U_Acorn' in hand,
        'has_unter_leaf': 'U_Leaf' in hand,
        'has_unter_heart': 'U_Heart' in hand,
        'has_unter_bell': 'U_Bell' in hand,
        'has_ace_heart': 'A_Heart' in hand,
    }


def run_analysis_games(
    model_path: str,
    num_games: int = 1000,
    seed: Optional[int] = None,
) -> List[HandResult]:
    """Run games and collect detailed hand statistics."""
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # Network uses hidden_size from config
    network = SchafkopfNetwork(hidden_size=config.hidden_size)
    network.load_state_dict(checkpoint['network_state_dict'])
    network.eval()
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    results: List[HandResult] = []
    
    for game_idx in range(num_games):
        env = SchafkopfEnv()
        env.reset()
        
        # Record declarer's hand (player 0) - use _hands (private attribute)
        declarer_hand = list(env._hands[0])
        features = featurize_hand(declarer_hand)
        trump_flags = get_specific_trumps(declarer_hand)
        
        # Play full game with AI
        while env.agents:
            agent = env.agent_selection
            obs, reward, term, trunc, info = env.last()
            
            if term or trunc:
                env.step(None)
                continue
            
            mask = obs['action_mask']
            
            # Create observation dict for network
            obs_dict = {
                'own_hand': torch.FloatTensor(obs['own_hand']).unsqueeze(0),
                'played_cards': torch.FloatTensor(obs['played_cards']).unsqueeze(0),
                'current_trick': torch.FloatTensor(obs['current_trick']).unsqueeze(0),
                'points': torch.FloatTensor(obs['points']).unsqueeze(0),
                'game_info': torch.FloatTensor(obs['game_info']).unsqueeze(0),
            }
            mask_tensor = torch.FloatTensor(mask).unsqueeze(0)
            
            with torch.no_grad():
                logits, _ = network(obs_dict, mask_tensor)
                action = torch.argmax(logits, dim=1).item()
            
            env.step(action)
        
        # Get result using the environment's method
        game_result = env.get_game_result()
        declarer_pts = game_result['declarer_points']
        opponent_pts = game_result['opponent_points']
        win = game_result['win']
        schneider = game_result['schneider']
        schwarz = game_result['schwarz']
        
        results.append(HandResult(
            declarer_hand=declarer_hand,
            features=features,
            win=win,
            declarer_points=declarer_pts,
            schneider=schneider,
            schwarz=schwarz,
            **trump_flags
        ))
        
        if (game_idx + 1) % 200 == 0:
            print(f"Games: {game_idx + 1}/{num_games}")
    
    return results


def results_to_dataframe(results: List[HandResult]) -> pd.DataFrame:
    """Convert results to a DataFrame for analysis."""
    rows = []
    for r in results:
        row = {
            'hand': ' '.join(sort_by_strength(r.declarer_hand)),
            'win': r.win,
            'declarer_points': r.declarer_points,
            'schneider': r.schneider,
            'schwarz': r.schwarz,
            # Features
            'trump_count': r.features.trump_count,
            'strongest_trump': r.features.strongest_trump_strength,
            'weakest_trump': r.features.weakest_trump_strength,
            'trump_strength_sum': r.features.trump_strength_sum,
            'color_aces': r.features.color_aces,
            'max_suit_run': r.features.max_suit_run,
            'total_points': r.features.total_points,
            # Specific trumps
            'has_ober_acorn': r.has_ober_acorn,
            'has_ober_leaf': r.has_ober_leaf,
            'has_ober_heart': r.has_ober_heart,
            'has_ober_bell': r.has_ober_bell,
            'has_unter_acorn': r.has_unter_acorn,
            'has_unter_leaf': r.has_unter_leaf,
            'has_unter_heart': r.has_unter_heart,
            'has_unter_bell': r.has_unter_bell,
            'has_ace_heart': r.has_ace_heart,
        }
        # Count high trumps (Ober + Unter)
        row['ober_count'] = sum([r.has_ober_acorn, r.has_ober_leaf, r.has_ober_heart, r.has_ober_bell])
        row['unter_count'] = sum([r.has_unter_acorn, r.has_unter_leaf, r.has_unter_heart, r.has_unter_bell])
        row['high_trump_count'] = row['ober_count'] + row['unter_count']
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_win_rate_by_trump_count(df: pd.DataFrame, save_path: str = 'plots/win_by_trump_count.png'):
    """Bar chart: win rate by number of trumps in hand."""
    grouped = df.groupby('trump_count').agg(
        games=('win', 'count'),
        wins=('win', 'sum'),
        avg_points=('declarer_points', 'mean')
    ).reset_index()
    grouped['win_rate'] = grouped['wins'] / grouped['games'] * 100
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = grouped['trump_count']
    bars = ax1.bar(x, grouped['win_rate'], color='steelblue', alpha=0.8, label='Win Rate')
    ax1.set_xlabel('Number of Trumps in Hand', fontsize=12)
    ax1.set_ylabel('Win Rate (%)', color='steelblue', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # Add game counts on bars
    for bar, games in zip(bars, grouped['games']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'n={games}', ha='center', va='bottom', fontsize=9)
    
    # Secondary axis for avg points
    ax2 = ax1.twinx()
    ax2.plot(x, grouped['avg_points'], 'ro-', linewidth=2, markersize=8, label='Avg Points')
    ax2.set_ylabel('Average Declarer Points', color='red', fontsize=12)
    ax2.set_ylim(30, 90)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.axhline(60, color='gray', linestyle='--', alpha=0.5, label='Win threshold (60)')
    
    plt.title('Agent Performance by Trump Count', fontsize=14, fontweight='bold')
    fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.88))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    return grouped


def plot_win_rate_by_specific_trump(df: pd.DataFrame, save_path: str = 'plots/win_by_specific_trump.png'):
    """Bar chart: win rate when holding specific high trumps."""
    trump_cards = [
        ('O_Acorn', 'has_ober_acorn', 'Ober\nAcorn'),
        ('O_Leaf', 'has_ober_leaf', 'Ober\nLeaf'),
        ('O_Heart', 'has_ober_heart', 'Ober\nHeart'),
        ('O_Bell', 'has_ober_bell', 'Ober\nBell'),
        ('U_Acorn', 'has_unter_acorn', 'Unter\nAcorn'),
        ('U_Leaf', 'has_unter_leaf', 'Unter\nLeaf'),
        ('U_Heart', 'has_unter_heart', 'Unter\nHeart'),
        ('U_Bell', 'has_unter_bell', 'Unter\nBell'),
        ('A_Heart', 'has_ace_heart', 'Ace\nHeart'),
    ]
    
    results = []
    for card, col, label in trump_cards:
        has_card = df[df[col] == True]
        no_card = df[df[col] == False]
        results.append({
            'card': label,
            'strength': TRUMP_STRENGTH.get(card, 0),
            'win_rate_with': has_card['win'].mean() * 100 if len(has_card) > 0 else 0,
            'win_rate_without': no_card['win'].mean() * 100 if len(no_card) > 0 else 0,
            'count_with': len(has_card),
            'count_without': len(no_card),
        })
    
    results_df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(results_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, results_df['win_rate_with'], width, 
                   label='With Card', color='forestgreen', alpha=0.8)
    bars2 = ax.bar(x + width/2, results_df['win_rate_without'], width,
                   label='Without Card', color='indianred', alpha=0.8)
    
    ax.set_xlabel('Trump Card', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Win Rate by Specific Trump Card Ownership', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['card'])
    ax.set_ylim(0, 100)
    ax.legend()
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
    
    # Add strength labels
    for i, (bar, strength) in enumerate(zip(bars1, results_df['strength'])):
        ax.text(bar.get_x() + bar.get_width()/2, 3, f'str={strength}', 
               ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    return results_df


def plot_win_rate_by_ober_count(df: pd.DataFrame, save_path: str = 'plots/win_by_ober_count.png'):
    """Win rate by number of Obers (highest trumps)."""
    grouped = df.groupby('ober_count').agg(
        games=('win', 'count'),
        wins=('win', 'sum'),
        avg_points=('declarer_points', 'mean')
    ).reset_index()
    grouped['win_rate'] = grouped['wins'] / grouped['games'] * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']
    bars = ax.bar(grouped['ober_count'], grouped['win_rate'], 
                  color=[colors[min(i, 4)] for i in grouped['ober_count']], 
                  edgecolor='black', linewidth=1)
    
    for bar, games, wr in zip(bars, grouped['games'], grouped['win_rate']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{wr:.1f}%\n(n={games})', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Number of Obers (Highest Trumps)', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Win Rate by Ober Count', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_xticks(range(5))
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    return grouped


def plot_win_rate_by_color_aces(df: pd.DataFrame, save_path: str = 'plots/win_by_color_aces.png'):
    """Win rate by number of color aces (non-heart aces)."""
    grouped = df.groupby('color_aces').agg(
        games=('win', 'count'),
        wins=('win', 'sum'),
        avg_points=('declarer_points', 'mean')
    ).reset_index()
    grouped['win_rate'] = grouped['wins'] / grouped['games'] * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(grouped['color_aces'], grouped['win_rate'], 
                  color='goldenrod', edgecolor='black', linewidth=1, alpha=0.8)
    
    for bar, games, wr in zip(bars, grouped['games'], grouped['win_rate']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{wr:.1f}%\n(n={games})', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Number of Color Aces (Acorn, Leaf, Bell)', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Win Rate by Color Ace Count', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_xticks(range(4))
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    return grouped


def plot_trump_strength_heatmap(df: pd.DataFrame, save_path: str = 'plots/trump_strength_heatmap.png'):
    """Heatmap: win rate by trump count vs trump strength sum."""
    # Create bins for trump strength
    df['strength_bin'] = pd.cut(df['trump_strength_sum'], 
                                 bins=[0, 10, 20, 30, 40, 50, 100],
                                 labels=['0-10', '11-20', '21-30', '31-40', '41-50', '50+'])
    
    pivot = df.pivot_table(values='win', index='strength_bin', columns='trump_count',
                           aggfunc=['mean', 'count'])
    
    win_rates = pivot['mean'] * 100
    counts = pivot['count']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    im = ax.imshow(win_rates.values, cmap='RdYlGn', aspect='auto', vmin=30, vmax=80)
    
    # Labels
    ax.set_xticks(range(len(win_rates.columns)))
    ax.set_xticklabels(win_rates.columns)
    ax.set_yticks(range(len(win_rates.index)))
    ax.set_yticklabels(win_rates.index)
    
    ax.set_xlabel('Trump Count', fontsize=12)
    ax.set_ylabel('Trump Strength Sum', fontsize=12)
    ax.set_title('Win Rate Heatmap: Trump Count vs Strength', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(win_rates.index)):
        for j in range(len(win_rates.columns)):
            wr = win_rates.values[i, j]
            cnt = counts.values[i, j]
            if not np.isnan(wr):
                text = f'{wr:.0f}%\n({int(cnt)})'
                color = 'white' if wr < 45 or wr > 65 else 'black'
                ax.text(j, i, text, ha='center', va='center', fontsize=9, color=color)
    
    plt.colorbar(im, ax=ax, label='Win Rate (%)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_points_distribution(df: pd.DataFrame, save_path: str = 'plots/points_distribution.png'):
    """Histogram of declarer points by trump count."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, trump_count in enumerate(range(6)):
        ax = axes[i]
        subset = df[df['trump_count'] == trump_count]['declarer_points']
        
        if len(subset) > 0:
            ax.hist(subset, bins=20, range=(0, 120), color='steelblue', 
                   edgecolor='black', alpha=0.7)
            ax.axvline(60, color='red', linestyle='--', linewidth=2, label='Win threshold')
            ax.axvline(subset.mean(), color='orange', linestyle='-', linewidth=2, label=f'Mean: {subset.mean():.1f}')
            
            win_rate = (subset > 60).mean() * 100
            ax.set_title(f'{trump_count} Trumps (n={len(subset)}, WR={win_rate:.1f}%)', fontsize=11)
        else:
            ax.set_title(f'{trump_count} Trumps (n=0)', fontsize=11)
        
        ax.set_xlabel('Declarer Points')
        ax.set_ylabel('Frequency')
        ax.set_xlim(0, 120)
    
    axes[0].legend(loc='upper left', fontsize=8)
    plt.suptitle('Points Distribution by Trump Count', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_best_worst_hands(df: pd.DataFrame, save_path: str = 'plots/best_worst_hands.png'):
    """Show example hands with best and worst performance."""
    # Group by hand and get stats
    hand_stats = df.groupby('hand').agg(
        games=('win', 'count'),
        wins=('win', 'sum'),
        avg_points=('declarer_points', 'mean'),
        trump_count=('trump_count', 'first'),
        ober_count=('ober_count', 'first'),
    ).reset_index()
    hand_stats['win_rate'] = hand_stats['wins'] / hand_stats['games'] * 100
    
    # Filter to hands with enough samples
    hand_stats = hand_stats[hand_stats['games'] >= 3].copy()
    
    # Best and worst
    best = hand_stats.nlargest(5, 'win_rate')
    worst = hand_stats.nsmallest(5, 'win_rate')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Best hands
    y_pos = range(len(best))
    ax1.barh(y_pos, best['win_rate'], color='forestgreen', alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([h[:40] + '...' if len(h) > 40 else h for h in best['hand']], fontsize=8)
    ax1.set_xlabel('Win Rate (%)')
    ax1.set_title('Best Performing Hands', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 100)
    for i, (wr, n) in enumerate(zip(best['win_rate'], best['games'])):
        ax1.text(wr + 1, i, f'{wr:.0f}% (n={n})', va='center', fontsize=9)
    
    # Worst hands
    y_pos = range(len(worst))
    ax2.barh(y_pos, worst['win_rate'], color='indianred', alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([h[:40] + '...' if len(h) > 40 else h for h in worst['hand']], fontsize=8)
    ax2.set_xlabel('Win Rate (%)')
    ax2.set_title('Worst Performing Hands', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 100)
    for i, (wr, n) in enumerate(zip(worst['win_rate'], worst['games'])):
        ax2.text(wr + 1, i, f'{wr:.0f}% (n={n})', va='center', fontsize=9)
    
    plt.suptitle('Hand Performance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics to console."""
    print("\n" + "="*70)
    print("AGENT HAND ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nTotal Games Analyzed: {len(df)}")
    print(f"Overall Win Rate: {df['win'].mean()*100:.1f}%")
    print(f"Average Declarer Points: {df['declarer_points'].mean():.1f}")
    print(f"Schneider Rate: {df['schneider'].mean()*100:.1f}%")
    
    print("\n--- Win Rate by Trump Count ---")
    for tc in sorted(df['trump_count'].unique()):
        subset = df[df['trump_count'] == tc]
        print(f"  {tc} trumps: {subset['win'].mean()*100:5.1f}% (n={len(subset):4d})")
    
    print("\n--- Win Rate by Ober Count ---")
    for oc in sorted(df['ober_count'].unique()):
        subset = df[df['ober_count'] == oc]
        print(f"  {oc} Obers: {subset['win'].mean()*100:5.1f}% (n={len(subset):4d})")
    
    print("\n--- Impact of Specific Cards ---")
    trump_cols = [
        ('has_ober_acorn', 'Ober Acorn'),
        ('has_ober_leaf', 'Ober Leaf'),
        ('has_unter_acorn', 'Unter Acorn'),
        ('has_ace_heart', 'Ace Heart'),
    ]
    for col, name in trump_cols:
        with_card = df[df[col] == True]['win'].mean() * 100
        without_card = df[df[col] == False]['win'].mean() * 100
        diff = with_card - without_card
        print(f"  {name:12s}: {with_card:5.1f}% with, {without_card:5.1f}% without (Δ={diff:+5.1f}%)")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Analyze Schafkopf agent by hand type')
    parser.add_argument('model', type=str, help='Path to model checkpoint')
    parser.add_argument('--games', type=int, default=2000, help='Number of games to analyze')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='plots', help='Output directory for plots')
    parser.add_argument('--save-csv', type=str, default=None, help='Save raw data to CSV')
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Running {args.games} games with model: {args.model}")
    print("This may take a few minutes...\n")
    
    # Run analysis games
    results = run_analysis_games(args.model, num_games=args.games, seed=args.seed)
    df = results_to_dataframe(results)
    
    # Save raw data if requested
    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"\nSaved raw data to: {args.save_csv}")
    
    # Print summary
    print_summary_stats(df)
    
    # Generate all plots
    print("\nGenerating plots...")
    plot_win_rate_by_trump_count(df, f'{args.output_dir}/win_by_trump_count.png')
    plot_win_rate_by_specific_trump(df, f'{args.output_dir}/win_by_specific_trump.png')
    plot_win_rate_by_ober_count(df, f'{args.output_dir}/win_by_ober_count.png')
    plot_win_rate_by_color_aces(df, f'{args.output_dir}/win_by_color_aces.png')
    plot_trump_strength_heatmap(df, f'{args.output_dir}/trump_strength_heatmap.png')
    plot_points_distribution(df, f'{args.output_dir}/points_distribution.png')
    plot_best_worst_hands(df, f'{args.output_dir}/best_worst_hands.png')
    
    print(f"\n✅ All plots saved to: {args.output_dir}/")
    print("Open the plots folder to view visualizations!")


if __name__ == '__main__':
    main()
