# schafkopf-ai

Toolkit for simulating Schafkopf Sucherb (Sauspiel) games, mining declarer hand win-rates, and exporting structured data for downstream reinforcement learning experiments.

## Requirements

- Python 3.10+
- pandas
- numpy
- matplotlib (optional, for custom plotting)
- seaborn (optional, for plotting)

Install dependencies and point `PYTHONPATH` at `src` so the package can be imported without installation:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

## Usage

All commands live under the module `schafkopf_ai.cli`.

### 1. Simulate random games

```bash
python -m schafkopf_ai.cli simulate-games --games 100000 --out data/random_games.csv
```

Outputs one row per simulated Sucherb game with declarer/opponent points, Schneider/Schwarz flags, and the called ace.

### 2. Mine hand statistics

```bash
python -m schafkopf_ai.cli mine-hands --hands 2000 --sims-per-hand 250 --progress --out data/hand_strength_stats.csv
```

This samples declarer starting hands, runs repeated simulations per hand, and groups the results by the new signature definition:

- number of trumps
- strongest trump strength (Obers score 14, weakest Heart 7 scores 0)
- weakest trump strength
- longest non-trump suit run
- sum of trump strength scores in the hand
- number of color aces (Acorn/Leaf/Bell)

The exported CSV is suitable for exploratory data analysis or as a feature table for RL policies that must decide whether to declare based solely on their 8 cards.

Additional knobs:

- `--unique-target 5000` keeps sampling until it has produced 5k distinct signature buckets, even if that requires dealing more than `--hands` hands.
- `--skip-duplicates` avoids re-simulating hands once their signature was already evaluated, which is useful when hunting for new, rare signatures.
- `--resume-from data/hand_strength_50k_100sims.csv` seeds the run with an existing CSV so new simulations focus on unseen signatures; combine with `--skip-duplicates` to extend the table efficiently.
- `--signatures-only` records unique feature combinations first (no simulations) and stores one representative `hand_example` per signature. Once satisfied with coverage, rerun without this flag (and with `--resume-from` pointing to the signature CSV) to gather win-rate estimates.
- `--seed-only` consumes the `hand_example` data from `--resume-from` and simulates those exact hands without sampling new ones—perfect for “simulate N games per catalogued signature”.

### 3. Estimate a win rate for a specific hand

```bash
python -m schafkopf_ai.cli estimate --stats data/hand_strength_stats.csv --hand "O_Acorn,O_Leaf,U_Acorn,A_Acorn,10_Acorn,A_Bell,10_Bell,K_Bell"
```

The command locates the matching signature (or the nearest one) and prints the predicted win rate and average declarer points.

### 4. Visualize mined hand statistics

```bash
python -m schafkopf_ai.cli visualize --stats data/hand_strength_seed300.csv --variant overview --out figures/hand_stats.png
```

Available `--variant` layouts (combine with `--min-trumps` / `--max-trumps` to focus on specific trump-count bands):

- `overview` (default): 2×2 dashboard with trump-count trend, total points scatter, strongest-trump scatter, and a 3D multi-metric view.
- `comparisons`: regression and box plots comparing win-rate distributions across trump count, strongest trump strength, trump-strength sum (or points), and color aces.
- `heatmaps`: two heat maps that highlight how win rate changes across (trump count × strongest strength) and (total points × color aces) grids.
- `clusters`: splits signatures into winning (≥50% win rate) and losing cohorts, runs a lightweight k-means clustering on trump-centric features, and renders cluster-colored scatter plots.
- `cluster-trump-win`: focuses on the winning cohort, plotting trump count vs win rate with the same cluster colors (bubble size = sample weight) for a quick read of how clusters perform.
- `cluster-trump-win-all`: reuses the winning-cluster colors but shows side-by-side panels for winning and losing hands so you can compare how each cluster behaves across the outcome boundary.

Use these visualizations to inspect how early-hand composition drives win probability before feeding the data into downstream RL tasks.

## Data schema

`hand_strength_stats.csv` columns:

| Column | Description |
| --- | --- |
| `signature` | Aggregated key `trumpCount-strongest-weakest-maxSuit-sumStrength-colorAces` |
| `trump_count` | Number of trumps in the hand |
| `strongest_trump_strength` | 14 (O♣) down to 0 (7♥). NaN if no trump |
| `weakest_trump_strength` | Lowest strength among trumps; NaN if none |
| `max_suit_run` | Longest count of a non-trump suit |
| `trump_strength_sum` | Sum of inverted trump strengths in the hand |
| `color_aces` | Count of Acorn/Leaf/Bell aces |
| `total_points` | Total Bavarian points in the hand (still useful for analysis) |
| `non_trump_points` | Points carried by non-trumps |
| `hand_example` | First observed 8-card hand for that signature (space-separated) |
| `hand_instances` | How many raw hands contributed to the signature |
| `games` | Total simulated games for that signature |
| `win_rate` | Declarer win probability |
| `win_rate_std` | Standard error for `win_rate` |
| `avg_points` | Average declarer points |

`random_games.csv` columns capture per-game outcomes and can power broader statistical summaries.

## Reinforcement learning scope

The mined hand table can serve as:

1. A prior for policy initialization (e.g., probability of declaring given trump composition).
2. A lookup table for reward shaping when training agents that only observe the declarer hand.
3. A dataset for supervised warm-start—train a value estimator that predicts win probability from hand features, then embed it into RL algorithms (DQN, PPO, etc.).

For RL, feed `hand_strength_stats.csv` into your pipeline (e.g., convert to tensors, augment with card-level encodings, and integrate as state-value targets).

## Project layout

```
src/
  schafkopf_ai/
    cards.py        # card constants + helpers
    game.py         # simulator for random legal Sucherb play
    features.py     # declarer-hand feature extraction
    analysis.py     # batching + aggregation helpers
    cli.py          # argparse-based command surface
card_utils.py       # compatibility layer pointing to schafkopf_ai.cards
untitled8.py        # original Colab script (legacy reference)
```
