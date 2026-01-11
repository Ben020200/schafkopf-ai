#!/usr/bin/env python3
"""Training script for Schafkopf PPO agent."""

import argparse
import sys
sys.path.insert(0, 'src')

from schafkopf_ai.ppo import train_ppo, PPOConfig, PPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Schafkopf AI with PPO")
    parser.add_argument(
        "--timesteps", "-t", 
        type=int, 
        default=500_000,
        help="Total training timesteps (default: 500,000)"
    )
    parser.add_argument(
        "--envs", "-e",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=20,
        help="Evaluate every N updates (default: 20)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default=None,
        choices=["rulebased", "bidding", "mixed", "focused"],
        help="Training mode: 'focused' = 100%% vs RuleBased (RECOMMENDED), 'mixed' = 50%% self-play + 50%% vs RuleBased"
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="sparse",
        choices=["sparse", "shaped", "dense"],
        help="Reward shaping mode: 'sparse' (default), 'shaped', or 'dense'"
    )
    parser.add_argument(
        "--no-lstm",
        action="store_true",
        help="Disable LSTM memory (use feedforward network instead)"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=512,
        help="Hidden layer size (default: 512)"
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.05,
        help="Entropy coefficient for exploration (default: 0.05)"
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Initialize network from imitation model (e.g., checkpoints/imitation_model.pt)"
    )
    
    args = parser.parse_args()
    
    use_lstm = not args.no_lstm
    if args.opponent is None:
        training_mode = "Self-play"
    elif args.opponent == "mixed":
        training_mode = "Mixed (50% self-play + 50% vs RuleBased)"
    elif args.opponent == "focused":
        training_mode = "FOCUSED (100% vs RuleBased)"
    else:
        training_mode = f"vs {args.opponent.title()}"
    network_type = "LSTM" if use_lstm else "Feedforward"
    
    print("=" * 60)
    print("Schafkopf AI - PPO Training")
    print("=" * 60)
    print(f"Timesteps:     {args.timesteps:,}")
    print(f"Environments:  {args.envs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Entropy Coef:  {args.entropy_coef}")
    print(f"Training Mode: {training_mode}")
    print(f"Reward Mode:   {args.reward}")
    print(f"Network:       {network_type} (hidden={args.hidden_size})")
    print(f"Save Dir:      {args.save_dir}")
    if args.init_from:
        print(f"Init From:     {args.init_from}")
    print("=" * 60)
    
    if args.resume:
        print(f"Resuming from: {args.resume}")
        config = PPOConfig(
            total_timesteps=args.timesteps,
            num_envs=args.envs,
            learning_rate=args.lr,
            save_dir=args.save_dir,
            eval_interval=args.eval_interval,
            opponent_type=args.opponent,
            reward_mode=args.reward,
            use_lstm=use_lstm,
            hidden_size=args.hidden_size,
            entropy_coef=args.entropy_coef,
        )
        trainer = PPOTrainer(config)
        trainer.load(args.resume)
        trainer.train()
    else:
        train_ppo(
            total_timesteps=args.timesteps,
            num_envs=args.envs,
            learning_rate=args.lr,
            save_dir=args.save_dir,
            eval_interval=args.eval_interval,
            opponent_type=args.opponent,
            reward_mode=args.reward,
            use_lstm=use_lstm,
            hidden_size=args.hidden_size,
            entropy_coef=args.entropy_coef,
            init_from=args.init_from,
        )


if __name__ == "__main__":
    main()
