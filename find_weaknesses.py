#!/usr/bin/env python3
"""
Find weaknesses in the RuleBased Schafkopf agent.

Uses Monte Carlo simulation to identify situations where
RuleBased makes suboptimal decisions.
"""

import argparse
from src.schafkopf_ai.weakness_finder import (
    find_rulebased_mistakes,
    find_rulebased_mistakes_fast,
    summarize_mistakes,
    analyze_mistake_patterns,
    export_training_data,
)


def main():
    parser = argparse.ArgumentParser(description="Find RuleBased weaknesses")
    parser.add_argument("--games", type=int, default=100, 
                        help="Number of games to analyze")
    parser.add_argument("--sims", type=int, default=100,
                        help="MC simulations per action (higher = more accurate)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Mistake threshold (0.05 = 5%)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export", type=str, default="data/mc_training_data.csv",
                        help="Export training data to this file")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 30 games, 50 sims")
    parser.add_argument("--fast", action="store_true", default=True,
                        help="Use parallel processing (default: True)")
    parser.add_argument("--no-fast", action="store_false", dest="fast",
                        help="Disable parallel processing")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    args = parser.parse_args()
    
    if args.quick:
        args.games = 30
        args.sims = 50
    
    print("=" * 60)
    print("RULEBASED WEAKNESS FINDER")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Games to analyze: {args.games}")
    print(f"  MC simulations per action: {args.sims}")
    print(f"  Mistake threshold: {args.threshold*100:.0f}%")
    print(f"  Parallel processing: {args.fast}")
    print()
    
    # Run analysis
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
    
    # Show results
    summarize_mistakes(mistakes)
    patterns = analyze_mistake_patterns(mistakes)
    
    # Export training data
    if args.export:
        export_training_data(all_decisions, args.export)
    
    # Final summary
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if len(all_decisions) > 0 and len(mistakes) > 0:
        mistake_rate = len(mistakes) / len(all_decisions) * 100
        avg_severity = sum(m.mistake_severity for m in mistakes) / len(mistakes) * 100
        total_lost = sum(m.mistake_severity for m in mistakes) / len(all_decisions) * 100
        
        print(f"\n✓ RuleBased makes mistakes in {mistake_rate:.1f}% of decisions")
        print(f"✓ Average mistake severity: {avg_severity:.1f}% win equity")
        print(f"✓ Expected loss per decision: {total_lost:.2f}%")
        print(f"\n→ This proves RuleBased is NOT optimal!")
        print(f"→ Training data exported to: {args.export}")
    else:
        print("\nNo significant mistakes found.")


if __name__ == "__main__":
    main()
