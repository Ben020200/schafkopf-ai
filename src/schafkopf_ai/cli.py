"""Command line interface for Schafkopf Sucherb data mining."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from .analysis import collect_hand_statistics, estimate_win_rate, run_game_batch
from .plots import (
    create_cluster_figure,
    create_cluster_trump_win_figure,
    create_cluster_trump_win_all_figure,
    create_hand_stats_figure,
    create_heatmap_figure,
    create_metric_comparison_figure,
)


def _comma_separated_cards(value: str) -> List[str]:
    cards = [part.strip() for part in value.split(',') if part.strip()]
    if len(cards) != 8:
        raise argparse.ArgumentTypeError("Provide exactly 8 cards separated by commas.")
    return cards


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Schafkopf Sucherb data mining toolkit")
    sub = parser.add_subparsers(dest='command', required=True)

    sim_parser = sub.add_parser('simulate-games', help="Run plain random Sucherb games")
    sim_parser.add_argument('--games', type=int, default=10000, help='Number of games to simulate')
    sim_parser.add_argument('--seed', type=int, default=None, help='Optional RNG seed')
    sim_parser.add_argument('--out', type=Path, default=None, help='Optional CSV output path')

    hands_parser = sub.add_parser('mine-hands', help="Estimate win rates grouped by trump stats")
    hands_parser.add_argument('--hands', type=int, default=500, help='Number of declarer hands to sample')
    hands_parser.add_argument('--sims-per-hand', type=int, default=200, help='Simulations per sampled hand')
    hands_parser.add_argument('--seed', type=int, default=None, help='Optional RNG seed')
    hands_parser.add_argument('--progress', action='store_true', help='Print coarse progress updates')
    hands_parser.add_argument('--out', type=Path, default=Path('hand_strength_stats.csv'), help='CSV target path')
    hands_parser.add_argument('--unique-target', type=int, default=None, help='Stop after collecting this many unique signatures')
    hands_parser.add_argument('--skip-duplicates', action='store_true', help='Skip simulation runs when a signature was already simulated')
    hands_parser.add_argument('--resume-from', type=Path, default=None, help='Existing stats CSV to extend')
    hands_parser.add_argument('--signatures-only', action='store_true', help='Collect unique signatures without playing simulations')
    hands_parser.add_argument('--seed-only', action='store_true', help='Simulate only the hands provided via --resume-from (requires stored hand examples)')

    estimate_parser = sub.add_parser('estimate', help="Estimate win rate for a concrete hand")
    estimate_parser.add_argument('--stats', type=Path, required=True, help='CSV produced via mine-hands command')
    estimate_parser.add_argument('--hand', type=_comma_separated_cards, required=True, help='Eight cards, comma separated')

    viz_parser = sub.add_parser('visualize', help="Create summary plots from mined hand stats")
    viz_parser.add_argument('--stats', type=Path, required=True, help='CSV produced via mine-hands command')
    viz_parser.add_argument('--out', type=Path, default=Path('hand_stats_overview.png'), help='Target image path (PNG)')
    viz_parser.add_argument(
        '--variant',
        choices=('overview', 'comparisons', 'heatmaps', 'clusters', 'cluster-trump-win', 'cluster-trump-win-all'),
        default='overview',
        help='Visualization layout to render',
    )
    viz_parser.add_argument('--min-trumps', type=int, default=None, help='Optional lower bound for trump count filtering')
    viz_parser.add_argument('--max-trumps', type=int, default=None, help='Optional upper bound for trump count filtering')

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == 'simulate-games':
        df = run_game_batch(args.games, seed=args.seed)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.out, index=False)
            print(f"Saved {len(df)} rows to {args.out}")
        else:
            print(df.describe(include='all'))
        return

    if args.command == 'mine-hands':
        seed_df = None
        if args.resume_from:
            if not args.resume_from.exists():
                parser.error(f"Resume source {args.resume_from} not found")
            seed_df = pd.read_csv(args.resume_from)

        if args.seed_only and not args.resume_from:
            parser.error('--seed-only requires --resume-from to supply stored hands')

        df = collect_hand_statistics(
            num_hands=args.hands,
            sims_per_hand=args.sims_per_hand,
            seed=args.seed,
            progress=args.progress,
            unique_target=args.unique_target,
            skip_duplicates=args.skip_duplicates,
            seed_stats=seed_df,
            signatures_only=args.signatures_only,
            seed_only=args.seed_only,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Saved {len(df)} signature rows to {args.out}")
        if not args.signatures_only and not df.empty:
            cols = [c for c in ['signature', 'trump_count', 'win_rate', 'avg_points'] if c in df.columns]
            top = df.head(5)[cols]
            print("Top 5 signature buckets:\n", top.to_string(index=False))
        return

    if args.command == 'estimate':
        stats = pd.read_csv(args.stats)
        info = estimate_win_rate(args.hand, stats)
        print(
            "Hand", args.hand,
            "=> win_rate", f"{info['win_rate']:.3f}",
            "avg_points", f"{info['avg_points']:.1f}",
            "source", info['source'],
        )
        return

    if args.command == 'visualize':
        if args.variant == 'overview':
            target = create_hand_stats_figure(args.stats, args.out, args.min_trumps, args.max_trumps)
        elif args.variant == 'comparisons':
            target = create_metric_comparison_figure(args.stats, args.out, args.min_trumps, args.max_trumps)
        elif args.variant == 'heatmaps':
            target = create_heatmap_figure(args.stats, args.out, args.min_trumps, args.max_trumps)
        elif args.variant == 'clusters':
            target = create_cluster_figure(args.stats, args.out, args.min_trumps, args.max_trumps)
        elif args.variant == 'cluster-trump-win':
            target = create_cluster_trump_win_figure(args.stats, args.out, args.min_trumps, args.max_trumps)
        else:
            target = create_cluster_trump_win_all_figure(args.stats, args.out, args.min_trumps, args.max_trumps)
        print(f"Saved visualization to {target}")
        return

    parser.error("Unknown command")


if __name__ == '__main__':
    main()
