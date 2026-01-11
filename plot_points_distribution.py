#!/usr/bin/env python3
"""Generate points distribution by trump count from hand strength CSV data."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_points_distribution(csv_path: str, output_path: str = "plots/points_distribution_by_trump.png"):
    """Create histogram subplots showing declarer points distribution by trump count.
    
    Simulates individual game results from avg_points by adding realistic variance.
    """
    
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Filter to only rows with actual games played
    df = df[df['games'] > 0].copy()
    
    if len(df) == 0:
        print("No game data found in the CSV!")
        return
    
    # Focus on trump counts 0-5 like the original plot (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    np.random.seed(42)
    
    for i, trump_count in enumerate(range(6)):
        ax = axes[i]
        subset = df[df['trump_count'] == trump_count]
        
        if len(subset) == 0:
            ax.set_title(f'{trump_count} Trumps (n=0)', fontsize=11)
            ax.set_xlabel('Declarer Points')
            ax.set_ylabel('Frequency (%)')
            continue
        
        # Simulate individual game points from avg_points
        # Use realistic standard deviation (~20 points typical in Schafkopf)
        simulated_points = []
        for _, row in subset.iterrows():
            n_games = min(row['games'], 100)  # Cap to keep reasonable
            std = 20  # Typical game-to-game variance
            points = np.random.normal(row['avg_points'], std, n_games)
            points = np.clip(points, 0, 120)  # Valid range
            simulated_points.extend(points)
        
        simulated_points = np.array(simulated_points)
        n_games = len(simulated_points)
        
        # Calculate stats
        mean_points = simulated_points.mean()
        win_rate = (simulated_points > 60).mean() * 100
        
        # Plot histogram with percentage
        counts, bins, _ = ax.hist(simulated_points, bins=20, range=(0, 120), 
                                   color='steelblue', edgecolor='black', alpha=0.7)
        # Convert to percentage
        ax.cla()
        ax.hist(simulated_points, bins=20, range=(0, 120), 
                weights=np.ones(len(simulated_points)) / len(simulated_points) * 100,
                color='steelblue', edgecolor='black', alpha=0.7)
        
        # Add win threshold line at 60 points (>60 wins)
        ax.axvline(60, color='red', linestyle='--', linewidth=2, label='Win threshold')
        
        # Add mean line
        ax.axvline(mean_points, color='orange', linestyle='-', linewidth=2, label=f'Mean: {mean_points:.1f}')
        
        ax.set_xlabel('Declarer Points')
        ax.set_ylabel('Frequency (%)')
        ax.set_title(f'{trump_count} Trumps (n={n_games}, WR={win_rate:.1f}%)', fontsize=11)
        ax.set_xlim(0, 120)
    
    axes[0].legend(loc='upper left', fontsize=8)
    plt.suptitle('Points Distribution by Trump Count', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {output_path}")
    
    # Print summary stats
    print("\nSummary by Trump Count:")
    print("-" * 60)
    for trump_count in range(6):
        subset = df[df['trump_count'] == trump_count]
        total_games = subset['games'].sum()
        if total_games > 0:
            weighted_mean = (subset['avg_points'] * subset['games']).sum() / total_games
            weighted_winrate = (subset['win_rate'] * subset['games']).sum() / total_games * 100
            print(f"{trump_count} Trumps: n={len(subset):5d} hands, {total_games:6d} games, "
                  f"WR={weighted_winrate:5.1f}%, Avg Points={weighted_mean:5.1f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot points distribution by trump count")
    parser.add_argument("csv_path", nargs="?", default="data/hand_strength_unique30k.csv",
                       help="Path to CSV with hand strength data")
    parser.add_argument("--output", "-o", default="plots/points_distribution_by_trump.png",
                       help="Output path for the plot")
    
    args = parser.parse_args()
    plot_points_distribution(args.csv_path, args.output)
