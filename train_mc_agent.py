#!/usr/bin/env python3
"""
Train an agent using MC (Monte Carlo) oracle data.

This trains a neural network on the "perfect play" data generated
by Monte Carlo analysis of RuleBased decisions.
"""

import argparse
from src.schafkopf_ai.mc_training import (
    train_mc_agent,
    evaluate_mc_agent,
    MCTrainingConfig,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train agent from MC analysis data"
    )
    parser.add_argument("--data", type=str, default="data/mc_training_data.csv",
                        help="Path to MC training data CSV")
    parser.add_argument("--output", type=str, default="checkpoints/mc_agent.pt",
                        help="Output path for trained model")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs")
    parser.add_argument("--hidden", type=int, default=256,
                        help="Hidden layer size")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--eval-games", type=int, default=500,
                        help="Number of games for evaluation")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="Only evaluate this model (skip training)")
    args = parser.parse_args()
    
    if args.eval_only:
        print("=" * 60)
        print("EVALUATION ONLY MODE")
        print("=" * 60)
        evaluate_mc_agent(args.eval_only, num_games=args.eval_games)
        return
    
    print("=" * 60)
    print("MC-BASED AGENT TRAINING")
    print("=" * 60)
    print(f"\nThis trains a neural network to predict MC-optimal actions.")
    print(f"The network learns from 'perfect play' data, not from RuleBased.\n")
    
    print(f"Configuration:")
    print(f"  Data: {args.data}")
    print(f"  Output: {args.output}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Hidden size: {args.hidden}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print()
    
    # Train
    trainer = train_mc_agent(
        data_path=args.data,
        output_path=args.output,
        epochs=args.epochs,
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION VS RULEBASED")
    print("=" * 60)
    win_rate = evaluate_mc_agent(args.output, num_games=args.eval_games)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if win_rate > 0.50:
        print(f"\n✓ SUCCESS! MC Agent beats RuleBased with {win_rate*100:.1f}% win rate!")
    else:
        print(f"\n→ MC Agent achieves {win_rate*100:.1f}% win rate")
        print("  Need more training data or larger network")


if __name__ == "__main__":
    main()
