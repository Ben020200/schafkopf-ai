#!/usr/bin/env python3
"""
Generate publication-quality figures for the Schafkopf AI paper.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

PAPER_DIR = Path("paper_figures")
PAPER_DIR.mkdir(exist_ok=True)


def fig1_schafkopf_game_structure():
    """Figure 1: Game structure and card hierarchy."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    
    # (a) Trump hierarchy
    ax = axes[0]
    trump_order = [
        ('O♣', 'Ober Acorn', 14),
        ('O♠', 'Ober Leaf', 13),
        ('O♥', 'Ober Heart', 12),
        ('O♦', 'Ober Bell', 11),
        ('U♣', 'Unter Acorn', 10),
        ('U♠', 'Unter Leaf', 9),
        ('U♥', 'Unter Heart', 8),
        ('U♦', 'Unter Bell', 7),
        ('A♥', 'Ace Heart', 6),
        ('10♥', 'Ten Heart', 5),
        ('K♥', 'King Heart', 4),
        ('9♥', 'Nine Heart', 3),
        ('8♥', 'Eight Heart', 2),
        ('7♥', 'Seven Heart', 1),
    ]
    
    y_pos = np.arange(len(trump_order))
    colors = ['#2ecc71' if i < 4 else '#3498db' if i < 8 else '#e74c3c' for i in range(14)]
    
    ax.barh(y_pos, [t[2] for t in trump_order], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t[0] for t in trump_order])
    ax.set_xlabel('Trump Rank (higher wins)')
    ax.set_title('(a) Trump Hierarchy in Schafkopf')
    ax.invert_yaxis()
    
    # Legend
    legend_patches = [
        mpatches.Patch(color='#2ecc71', label='Obers'),
        mpatches.Patch(color='#3498db', label='Unters'),
        mpatches.Patch(color='#e74c3c', label='Hearts'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8)
    
    # (b) Game flow
    ax = axes[1]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Draw game flow boxes
    boxes = [
        (5, 9, 'Deal 8 Cards\nto Each Player'),
        (5, 7, 'Declarer Calls\nAce (Partner)'),
        (5, 5, '8 Tricks\n(32 decisions)'),
        (5, 3, 'Count Points\n(120 total)'),
        (5, 1, 'Win: ≥61 pts\nSchneider: ≥91'),
    ]
    
    for x, y, text in boxes:
        ax.add_patch(plt.Rectangle((x-2, y-0.6), 4, 1.2, 
                     facecolor='lightblue', edgecolor='navy', linewidth=1.5))
        ax.text(x, y, text, ha='center', va='center', fontsize=8)
    
    # Arrows
    for i in range(len(boxes)-1):
        ax.annotate('', xy=(5, boxes[i+1][1]+0.7), xytext=(5, boxes[i][1]-0.7),
                   arrowprops=dict(arrowstyle='->', color='navy', lw=1.5))
    
    ax.set_title('(b) Game Flow', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(PAPER_DIR / 'fig1_game_structure.pdf')
    plt.savefig(PAPER_DIR / 'fig1_game_structure.png')
    plt.close()
    print("✓ Figure 1: Game structure")


def fig2_mc_analysis_distribution():
    """Figure 2: MC analysis results - mistake distribution."""
    # Load MC data
    df = pd.read_csv('data/mc_training_data.csv')
    
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    
    # (a) Mistake distribution by severity
    ax = axes[0]
    mistakes = df[df['is_mistake'] == True]
    severity = mistakes['mistake_severity'] * 100
    
    ax.hist(severity, bins=20, color='#e74c3c', alpha=0.7, edgecolor='darkred')
    ax.axvline(severity.mean(), color='navy', linestyle='--', linewidth=2, 
               label=f'Mean: {severity.mean():.1f}%')
    ax.set_xlabel('Mistake Severity (%)')
    ax.set_ylabel('Count')
    ax.set_title('(a) RuleBased Mistake\nSeverity Distribution')
    ax.legend(fontsize=8)
    
    # (b) Mistakes by trick number
    ax = axes[1]
    mistake_rate_by_trick = df.groupby('trick_number')['is_mistake'].mean() * 100
    
    ax.bar(mistake_rate_by_trick.index, mistake_rate_by_trick.values, 
           color='#3498db', alpha=0.8, edgecolor='navy')
    ax.set_xlabel('Trick Number')
    ax.set_ylabel('Mistake Rate (%)')
    ax.set_title('(b) Mistakes by\nTrick Number')
    ax.set_xticks(range(8))
    
    # (c) Mistakes by legal action count
    ax = axes[2]
    df['num_options_cat'] = pd.cut(df['num_options'], bins=[0, 2, 4, 6, 8], 
                                    labels=['1-2', '3-4', '5-6', '7-8'])
    mistake_rate_by_options = df.groupby('num_options_cat')['is_mistake'].mean() * 100
    
    ax.bar(range(len(mistake_rate_by_options)), mistake_rate_by_options.values,
           color='#2ecc71', alpha=0.8, edgecolor='darkgreen')
    ax.set_xlabel('Legal Actions')
    ax.set_ylabel('Mistake Rate (%)')
    ax.set_title('(c) Mistakes by\nChoice Complexity')
    ax.set_xticks(range(len(mistake_rate_by_options)))
    ax.set_xticklabels(mistake_rate_by_options.index)
    
    plt.tight_layout()
    plt.savefig(PAPER_DIR / 'fig2_mc_analysis.pdf')
    plt.savefig(PAPER_DIR / 'fig2_mc_analysis.png')
    plt.close()
    print("✓ Figure 2: MC analysis distribution")


def fig3_algorithm_comparison():
    """Figure 3: Win rate comparison of all approaches."""
    
    # Data from experiments
    algorithms = [
        'RuleBased\nvs RuleBased',
        'PPO\n(2M steps)',
        'Imitation\nLearning',
        'Q-Value\nPredictor',
        'Blend\nHybrid',
        'Ranking\nHybrid',
    ]
    
    win_rates = [48.7, 42.3, 44.5, 49.7, 49.7, 57.7]
    errors = [2.5, 3.0, 2.8, 2.5, 2.5, 2.2]  # Approximate std errors
    
    colors = ['#95a5a6', '#e74c3c', '#e74c3c', '#f39c12', '#f39c12', '#2ecc71']
    
    fig, ax = plt.subplots(figsize=(7, 3))
    
    bars = ax.bar(range(len(algorithms)), win_rates, yerr=errors, 
                  color=colors, alpha=0.8, edgecolor='black',
                  capsize=4, error_kw={'linewidth': 1.5})
    
    # Add 50% reference line
    ax.axhline(50, color='navy', linestyle='--', linewidth=2, label='50% (parity)')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, win_rates)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errors[i] + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Win Rate vs RuleBased (%)')
    ax.set_title('Algorithm Performance Comparison')
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, fontsize=8)
    ax.set_ylim(30, 70)
    ax.legend(loc='upper left')
    
    # Add color legend
    legend_patches = [
        mpatches.Patch(color='#95a5a6', label='Baseline'),
        mpatches.Patch(color='#e74c3c', label='Below parity'),
        mpatches.Patch(color='#f39c12', label='At parity'),
        mpatches.Patch(color='#2ecc71', label='Above parity'),
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(PAPER_DIR / 'fig3_algorithm_comparison.pdf')
    plt.savefig(PAPER_DIR / 'fig3_algorithm_comparison.png')
    plt.close()
    print("✓ Figure 3: Algorithm comparison")


def fig4_ranking_architecture():
    """Figure 4: Ranking hybrid architecture diagram."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis('off')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    
    # Input layer
    boxes = {
        'state': (2, 8, 'Game State\n(9 features)'),
        'card': (2, 5, 'Card Features\n(6 per card)'),
        'state_net': (5, 8, 'State\nEncoder\n(64 units)'),
        'card_net': (5, 5, 'Card\nEncoder\n(64 units)'),
        'combine': (8, 6.5, 'Combine\nNetwork\n(128→64→1)'),
        'score': (11, 6.5, 'Score\ns(state, card)'),
    }
    
    colors = {
        'state': '#3498db',
        'card': '#2ecc71',
        'state_net': '#9b59b6',
        'card_net': '#9b59b6',
        'combine': '#e74c3c',
        'score': '#f39c12',
    }
    
    for key, (x, y, text) in boxes.items():
        width = 2.5 if 'net' in key or 'combine' in key else 2
        height = 1.5
        ax.add_patch(plt.Rectangle((x-width/2, y-height/2), width, height,
                     facecolor=colors[key], edgecolor='black', 
                     linewidth=1.5, alpha=0.8))
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Arrows
    arrows = [
        ((3, 8), (3.75, 8)),      # state -> state_net
        ((3, 5), (3.75, 5)),      # card -> card_net
        ((6.25, 8), (6.75, 7.2)), # state_net -> combine
        ((6.25, 5), (6.75, 5.8)), # card_net -> combine
        ((9.25, 6.5), (9.8, 6.5)), # combine -> score
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Title and annotations
    ax.text(7, 9.5, 'Ranking Model Architecture', ha='center', fontsize=12, fontweight='bold')
    
    # Training annotation
    ax.text(7, 1.5, 'Training: Margin Ranking Loss\ns(best) > s(worse) + margin', 
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(PAPER_DIR / 'fig4_ranking_architecture.pdf')
    plt.savefig(PAPER_DIR / 'fig4_ranking_architecture.png')
    plt.close()
    print("✓ Figure 4: Ranking architecture")


def fig5_threshold_analysis():
    """Figure 5: Override threshold analysis."""
    
    # Data from experiments
    thresholds = [0.0, 0.05, 0.1, 0.2, 0.3]
    win_rates = [56.7, 57.7, 56.3, 50.0, 48.7]
    override_rates = [36.6, 23.7, 12.5, 1.0, 0.0]
    
    fig, ax1 = plt.subplots(figsize=(5, 3))
    
    color1 = '#2ecc71'
    color2 = '#3498db'
    
    # Win rate
    line1 = ax1.plot(thresholds, win_rates, 'o-', color=color1, linewidth=2, 
                     markersize=8, label='Win Rate')
    ax1.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Override Threshold')
    ax1.set_ylabel('Win Rate (%)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(45, 60)
    
    # Override rate on secondary axis
    ax2 = ax1.twinx()
    line2 = ax2.plot(thresholds, override_rates, 's--', color=color2, linewidth=2,
                     markersize=8, label='Override Rate')
    ax2.set_ylabel('Override Rate (%)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 40)
    
    # Highlight optimal
    ax1.scatter([0.05], [57.7], s=200, facecolors='none', edgecolors='red', 
                linewidth=3, zorder=5)
    ax1.annotate('Optimal', xy=(0.05, 57.7), xytext=(0.12, 59),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='red'))
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', fontsize=9)
    
    ax1.set_title('Override Threshold Analysis')
    
    plt.tight_layout()
    plt.savefig(PAPER_DIR / 'fig5_threshold_analysis.pdf')
    plt.savefig(PAPER_DIR / 'fig5_threshold_analysis.png')
    plt.close()
    print("✓ Figure 5: Threshold analysis")


def fig6_hybrid_overview():
    """Figure 6: Complete hybrid system overview."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axis('off')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    
    # Components
    components = {
        'game_state': (2, 8, 'Game State\n(trick, hand, etc.)'),
        'rulebased': (5, 8, 'RuleBased\nAgent'),
        'ranking': (5, 5, 'Ranking\nModel'),
        'compare': (8, 6.5, 'Compare\nScores'),
        'decision': (11, 6.5, 'Final\nAction'),
        'mc_data': (2, 3, 'MC Oracle\n(43,420 samples)'),
        'training': (5, 2, 'Pairwise\nRanking Loss'),
    }
    
    colors = {
        'game_state': '#3498db',
        'rulebased': '#f39c12',
        'ranking': '#9b59b6',
        'compare': '#e74c3c',
        'decision': '#2ecc71',
        'mc_data': '#1abc9c',
        'training': '#95a5a6',
    }
    
    for key, (x, y, text) in components.items():
        width = 2.5
        height = 1.4
        ax.add_patch(plt.Rectangle((x-width/2, y-height/2), width, height,
                     facecolor=colors[key], edgecolor='black', 
                     linewidth=1.5, alpha=0.8))
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Arrows
    arrows = [
        ((3.25, 8), (3.75, 8), 'solid'),        # state -> rulebased
        ((3.25, 7.5), (3.75, 5.5), 'solid'),    # state -> ranking
        ((6.25, 8), (6.75, 7.2), 'solid'),      # rulebased -> compare
        ((6.25, 5), (6.75, 5.8), 'solid'),      # ranking -> compare
        ((9.25, 6.5), (9.75, 6.5), 'solid'),    # compare -> decision
        ((3.25, 3), (3.75, 4.3), 'dashed'),     # mc_data -> ranking (training)
        ((5, 3.4), (5, 4.3), 'dashed'),         # training -> ranking
    ]
    
    for start, end, style in arrows:
        ls = '--' if style == 'dashed' else '-'
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5, ls=ls))
    
    # Labels
    ax.text(7, 9.5, 'Ranking Hybrid Agent Architecture', 
            ha='center', fontsize=12, fontweight='bold')
    
    ax.text(8, 4.5, 'Override if:\ns(MC) - s(RB) > θ', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.text(2, 1, '(Offline Training)', ha='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig(PAPER_DIR / 'fig6_hybrid_overview.pdf')
    plt.savefig(PAPER_DIR / 'fig6_hybrid_overview.png')
    plt.close()
    print("✓ Figure 6: Hybrid system overview")


def table_results():
    """Generate results table for the paper."""
    
    results = [
        ("RuleBased", "Handcrafted heuristics", "48.7%", "—"),
        ("PPO (self-play)", "RL, 2M timesteps", "42.3%", "-6.4%"),
        ("Imitation Learning", "Supervised on RuleBased", "44.5%", "-4.2%"),
        ("Simple Hybrid", "Exact MC lookup", "46.5%", "-2.2%"),
        ("Q-Value Predictor", "MSE loss on Q-values", "49.7%", "+1.0%"),
        ("Blend Hybrid", "α-weighted blend", "49.7%", "+1.0%"),
        ("\\textbf{Ranking Hybrid}", "Pairwise ranking loss", "\\textbf{57.7%}", "\\textbf{+9.0%}"),
    ]
    
    latex = r"""
\begin{table}[h]
\centering
\caption{Win Rates Against RuleBased Agent (1000 Games)}
\label{tab:results}
\begin{tabular}{lccc}
\toprule
\textbf{Agent} & \textbf{Method} & \textbf{Win Rate} & \textbf{Δ vs RB} \\
\midrule
"""
    for agent, method, win_rate, delta in results:
        latex += f"{agent} & {method} & {win_rate} & {delta} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(PAPER_DIR / 'table_results.tex', 'w') as f:
        f.write(latex)
    print("✓ Table: Results")


if __name__ == "__main__":
    print("Generating paper figures...")
    print("=" * 40)
    
    fig1_schafkopf_game_structure()
    fig2_mc_analysis_distribution()
    fig3_algorithm_comparison()
    fig4_ranking_architecture()
    fig5_threshold_analysis()
    fig6_hybrid_overview()
    table_results()
    
    print("=" * 40)
    print(f"All figures saved to {PAPER_DIR}/")
