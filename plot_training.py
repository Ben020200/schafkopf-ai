#!/usr/bin/env python3
"""Plot PPO training progress from CSV log."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load training history
df = pd.read_csv('checkpoints/training_history_vs_rulebased.csv')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('PPO Training Progress (Dense Rewards vs RuleBased)', fontsize=14, fontweight='bold')

# Convert timesteps to thousands for readability
timesteps_k = df['timestep'] / 1000

# 1. Win Rate (with smoothing)
ax1 = axes[0, 0]
ax1.plot(timesteps_k, df['win_rate'] * 100, alpha=0.3, color='blue', label='Raw')
# Rolling average for smoothing
window = 10
win_rate_smooth = df['win_rate'].rolling(window=window, center=True).mean() * 100
ax1.plot(timesteps_k, win_rate_smooth, color='blue', linewidth=2, label=f'Smoothed (window={window})')
# Plot eval win rate
eval_mask = df['eval_win_rate'].notna()
ax1.scatter(timesteps_k[eval_mask], df.loc[eval_mask, 'eval_win_rate'] * 100, 
            color='red', s=30, zorder=5, label='Eval Win Rate')
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
ax1.set_xlabel('Timesteps (thousands)')
ax1.set_ylabel('Win Rate (%)')
ax1.set_title('Win Rate During Training')
ax1.legend(loc='lower right')
ax1.set_ylim(30, 70)
ax1.grid(True, alpha=0.3)

# 2. Average Points
ax2 = axes[0, 1]
ax2.plot(timesteps_k, df['avg_points'], alpha=0.3, color='green', label='Raw')
points_smooth = df['avg_points'].rolling(window=window, center=True).mean()
ax2.plot(timesteps_k, points_smooth, color='green', linewidth=2, label=f'Smoothed')
ax2.axhline(y=61, color='gray', linestyle='--', alpha=0.5, label='61 pts (win threshold)')
ax2.set_xlabel('Timesteps (thousands)')
ax2.set_ylabel('Average Points')
ax2.set_title('Average Points per Game')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

# 3. Loss and Entropy
ax3 = axes[1, 0]
ax3.plot(timesteps_k, df['total_loss'], color='red', alpha=0.5, label='Total Loss')
loss_smooth = df['total_loss'].rolling(window=window, center=True).mean()
ax3.plot(timesteps_k, loss_smooth, color='red', linewidth=2, label='Loss (smoothed)')
ax3.set_xlabel('Timesteps (thousands)')
ax3.set_ylabel('Loss', color='red')
ax3.tick_params(axis='y', labelcolor='red')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

# Entropy on secondary axis
ax3b = ax3.twinx()
ax3b.plot(timesteps_k, df['entropy'], color='purple', alpha=0.5, label='Entropy')
entropy_smooth = df['entropy'].rolling(window=window, center=True).mean()
ax3b.plot(timesteps_k, entropy_smooth, color='purple', linewidth=2, label='Entropy (smoothed)')
ax3b.set_ylabel('Entropy', color='purple')
ax3b.tick_params(axis='y', labelcolor='purple')
ax3b.legend(loc='upper right')
ax3.set_title('Loss and Entropy')

# 4. Training FPS
ax4 = axes[1, 1]
ax4.plot(timesteps_k, df['fps'], color='orange', linewidth=1.5)
ax4.set_xlabel('Timesteps (thousands)')
ax4.set_ylabel('Frames Per Second')
ax4.set_title('Training Speed (FPS)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/training_progress.png', dpi=150, bbox_inches='tight')
plt.savefig('plots/training_progress.pdf', bbox_inches='tight')
print("Saved plots/training_progress.png and plots/training_progress.pdf")

# Show summary stats
print("\n=== Training Summary ===")
print(f"Total timesteps: {df['timestep'].max():,}")
print(f"Final win rate: {df['win_rate'].iloc[-1]*100:.1f}%")
print(f"Best eval win rate: {df['eval_win_rate'].max()*100:.1f}%")
print(f"Final entropy: {df['entropy'].iloc[-1]:.3f} (started at {df['entropy'].iloc[0]:.3f})")
print(f"Avg points (last 10 updates): {df['avg_points'].tail(10).mean():.1f}")

plt.show()
