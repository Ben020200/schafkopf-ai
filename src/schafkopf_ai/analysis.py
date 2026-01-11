"""Data mining helpers for Schafkopf Sauspiel hands and games."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .features import HandFeatures, featurize_hand, signature_key
from .game import GameResult, SauspielSimulator


def _result_to_row(result: GameResult) -> Dict[str, object]:
    return {
        'declarer_pts': result.declarer_points,
        'opponent_pts': result.opponent_points,
        'win': result.win,
        'schneider': result.schneider,
        'schwarz': result.schwarz,
        'called_ace': result.called_ace,
        'partner_index': result.partner_index,
    }


def run_game_batch(num_games: int, seed: Optional[int] = None) -> pd.DataFrame:
    simulator = SauspielSimulator(seed=seed)
    rows: List[Dict[str, object]] = []
    for _ in range(num_games):
        rows.append(_result_to_row(simulator.simulate_game()))
    return pd.DataFrame(rows)


def _simulate_hand(simulator: SauspielSimulator, hand: Sequence[str], sims_per_hand: int) -> Dict[str, float]:
    wins = 0
    total_points = 0
    for _ in range(sims_per_hand):
        result = simulator.simulate_game(declarer_hand=hand)
        wins += int(result.win)
        total_points += result.declarer_points
    return {
        'games': sims_per_hand,
        'wins': wins,
        'points': total_points,
    }


def _aggregates_to_frame(aggregates: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for key, bucket in aggregates.items():
        features: HandFeatures = bucket['features']
        games = bucket['games']
        wins = bucket['wins']
        points = bucket['points']
        if games > 0:
            win_rate = wins / games
            avg_points = points / games
            variance = win_rate * (1 - win_rate)
            std = math.sqrt(variance / games)
        else:
            win_rate = 0.0
            avg_points = 0.0
            std = 0.0

        example = bucket.get('example_hand')
        row = {
            'signature': key,
            **features.to_dict(),
            'hand_instances': bucket['hand_instances'],
            'games': games,
            'win_rate': win_rate,
            'win_rate_std': std,
            'avg_points': avg_points,
            'hand_example': ' '.join(example) if example else '',
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values('win_rate', ascending=False).reset_index(drop=True)


def collect_hand_statistics(
    num_hands: int,
    sims_per_hand: int,
    seed: Optional[int] = None,
    progress: bool = False,
    unique_target: Optional[int] = None,
    skip_duplicates: bool = False,
    seed_stats: Optional[pd.DataFrame] = None,
    signatures_only: bool = False,
    seed_only: bool = False,
) -> pd.DataFrame:
    if signatures_only and seed_only:
        raise ValueError("signatures_only and seed_only cannot be used together")
    if seed_only and (seed_stats is None or seed_stats.empty):
        raise ValueError("seed_only requires --resume-from data with stored hand examples")

    simulator = SauspielSimulator(seed=seed)
    aggregates: Dict[str, Dict[str, object]] = {}

    def _coerce_optional_int(value: object) -> Optional[int]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(float(value))

    def _coerce_int(value: object, default: int = 0) -> int:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(float(value))

    if seed_stats is not None and not seed_stats.empty:
        for _, row in seed_stats.iterrows():
            features = HandFeatures(
                trump_count=_coerce_int(row.get('trump_count', 0)),
                strongest_trump_strength=_coerce_optional_int(
                    row.get('strongest_trump_strength') or row.get('strongest_trump_rank')
                ),
                weakest_trump_strength=_coerce_optional_int(
                    row.get('weakest_trump_strength') or row.get('weakest_trump_rank')
                ),
                max_suit_run=_coerce_int(row.get('max_suit_run', 0)),
                trump_strength_sum=_coerce_int(row.get('trump_strength_sum', row.get('total_points', 0))),
                color_aces=_coerce_int(row.get('color_aces', 0)),
                total_points=_coerce_int(row.get('total_points', 0)),
                non_trump_points=_coerce_int(row.get('non_trump_points', row.get('total_points', 0))),
            )
            games = _coerce_int(row.get('games', 0))
            win_rate = float(row.get('win_rate', 0))
            avg_points = float(row.get('avg_points', 0))
            wins = float(row.get('wins', win_rate * games if games else 0))
            points = float(row.get('points', avg_points * games if games else 0))
            example_raw = row.get('hand_example', '')
            if isinstance(example_raw, str) and example_raw.strip():
                example_hand = tuple(example_raw.split())
            else:
                example_hand = None

            aggregates[row['signature']] = {
                'features': features,
                'games': games,
                'wins': wins,
                'points': points,
                'hand_instances': int(row.get('hand_instances', 0)),
                'example_hand': example_hand,
            }

    if seed_only:
        missing_examples = [key for key, bucket in aggregates.items() if not bucket.get('example_hand')]
        if missing_examples:
            raise ValueError("seed_only requires hand_example data for every seeded signature")
        total = len(aggregates)
        if total == 0:
            raise ValueError("seed_only mode needs at least one seeded signature")
        progress_interval = max(1, total // 10)
        if progress:
            print(f"Simulating {total} stored signatures")
        for idx, bucket in enumerate(aggregates.values(), start=1):
            stats = _simulate_hand(simulator, list(bucket['example_hand']), sims_per_hand)
            bucket['games'] += stats['games']
            bucket['wins'] += stats['wins']
            bucket['points'] += stats['points']
            if progress and idx % progress_interval == 0:
                print(f"Simulated {idx}/{total} signatures")
        return _aggregates_to_frame(aggregates)

    processed = 0
    skipped_duplicates = 0
    progress_interval = max(1, num_hands // 10)

    if progress and aggregates:
        print(f"Seeded {len(aggregates)} signatures from existing stats")

    def should_continue() -> bool:
        if processed >= num_hands:
            return False
        if unique_target is not None and len(aggregates) >= unique_target:
            return False
        return True

    while should_continue():
        hands = simulator.deal_hands()
        declarer_hand = hands[0]
        features = featurize_hand(declarer_hand)
        key = signature_key(features)

        bucket = aggregates.setdefault(
            key,
            {
                'features': features,
                'games': 0,
                'wins': 0,
                'points': 0,
                'hand_instances': 0,
                'example_hand': None,
            },
        )
        bucket['hand_instances'] += 1
        if bucket['example_hand'] is None:
            bucket['example_hand'] = tuple(declarer_hand)

        if signatures_only:
            processed += 1
            if progress and processed % progress_interval == 0:
                status = f"Processed {processed} hands | {len(aggregates)} unique signatures"
                if unique_target is not None:
                    status += f" / target {unique_target}"
                print(status)
            continue

        if skip_duplicates and bucket['games'] > 0:
            skipped_duplicates += 1
            processed += 1
            if progress and processed % progress_interval == 0:
                status = f"Processed {processed} hands | {len(aggregates)} unique signatures"
                if unique_target is not None:
                    status += f" / target {unique_target}"
                if skipped_duplicates:
                    status += f" (skipped {skipped_duplicates} duplicates)"
                print(status)
            continue

        stats = _simulate_hand(simulator, declarer_hand, sims_per_hand)
        bucket['games'] += stats['games']
        bucket['wins'] += stats['wins']
        bucket['points'] += stats['points']

        processed += 1
        if progress and processed % progress_interval == 0:
            status = f"Processed {processed} hands | {len(aggregates)} unique signatures"
            if unique_target is not None:
                status += f" / target {unique_target}"
            if skipped_duplicates:
                status += f" (skipped {skipped_duplicates})"
            print(status)

    return _aggregates_to_frame(aggregates)


def estimate_win_rate(hand: Sequence[str], stats: pd.DataFrame) -> Dict[str, float]:
    if stats.empty:
        raise ValueError("Statistics DataFrame is empty; run collect_hand_statistics first.")

    features = featurize_hand(hand)
    key = signature_key(features)
    match = stats.loc[stats['signature'] == key]
    if not match.empty:
        row = match.iloc[0]
        return {
            'win_rate': float(row['win_rate']),
            'avg_points': float(row['avg_points']),
            'source': 'exact_signature',
        }

    # Fallback: nearest neighbor on trump count + strength + points.
    stats = stats.copy()
    for col in ['trump_strength_sum', 'color_aces']:
        if col not in stats:
            stats[col] = 0
    stats['distance'] = (
        (stats['trump_count'] - features.trump_count).abs()
        + (stats['max_suit_run'] - features.max_suit_run).abs()
        + (stats['trump_strength_sum'] - features.trump_strength_sum).abs() / 10.0
        + (stats['color_aces'] - features.color_aces).abs()
    )
    candidate = stats.sort_values('distance').iloc[0]
    return {
        'win_rate': float(candidate['win_rate']),
        'avg_points': float(candidate['avg_points']),
        'source': f"nearest_signature:{candidate['signature']}",
    }
