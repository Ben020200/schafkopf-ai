#!/usr/bin/env python3
"""Play games with a trained Schafkopf agent."""

import argparse
import sys
sys.path.insert(0, 'src')

import numpy as np
import torch

from schafkopf_ai.env import make_env, IDX_TO_CARD
from schafkopf_ai.ppo import SchafkopfNetwork


def load_agent(checkpoint_path: str, device: str = "cpu") -> SchafkopfNetwork:
    """Load a trained agent from checkpoint."""
    network = SchafkopfNetwork()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    network.load_state_dict(checkpoint["network_state_dict"])
    network.eval()
    return network


def play_game(network: SchafkopfNetwork, render: bool = True, device: str = "cpu"):
    """Play a single game with the trained agent."""
    env = make_env(render_mode="human" if render else None)
    env.reset()
    
    if render:
        print("\n" + "=" * 60)
        print("NEW GAME - Schafkopf Sauspiel")
        print("=" * 60)
        print(f"Declarer: Player 0")
        print(f"Called Ace: {env._called_ace}")
        print(f"Partner: Player {env._partner_idx} (hidden until ace is played)")
        print("=" * 60)
    
    trick_num = 0
    while not all(env.terminations.values()):
        agent = env.agent_selection
        player_idx = env.agent_name_mapping[agent]
        obs = env.observe(agent)
        
        # Get action from network
        with torch.no_grad():
            obs_tensor = {
                k: torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
                for k, v in obs.items() if k != "action_mask"
            }
            mask_tensor = torch.tensor(
                obs["action_mask"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            
            logits, value = network.forward(obs_tensor, mask_tensor)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(logits, dim=-1).item()
        
        card = IDX_TO_CARD[action]
        
        if render:
            # Show player's hand and decision
            hand = env._hands[player_idx]
            legal = [IDX_TO_CARD[i] for i, v in enumerate(obs["action_mask"]) if v]
            
            if len(env._current_trick) == 0:
                trick_num += 1
                print(f"\n--- Trick {trick_num} ---")
            
            # Get top 3 choices
            top_k = torch.topk(probs[0], min(3, len(legal)))
            choices = [
                f"{IDX_TO_CARD[idx.item()]} ({prob.item():.1%})"
                for idx, prob in zip(top_k.indices, top_k.values)
                if obs["action_mask"][idx.item()]
            ]
            
            marker = "→" if player_idx == 0 else " "
            print(f"{marker} {agent}: plays {card:<12} (value: {value.item():.2f}) | options: {', '.join(choices[:3])}")
        
        env.step(action)
    
    result = env.get_game_result()
    
    if render:
        print("\n" + "=" * 60)
        print("GAME RESULT")
        print("=" * 60)
        print(f"Declarer Team (Players {result['declarer_idx']}, {result['partner_idx']}): {result['declarer_points']} points")
        print(f"Opponent Team: {result['opponent_points']} points")
        print(f"Winner: {'DECLARER TEAM' if result['win'] else 'OPPONENTS'}")
        if result['schneider']:
            print("🏆 SCHNEIDER!")
        if result['schwarz']:
            print("🏆🏆 SCHWARZ!")
        print("=" * 60)
    
    return result


def evaluate_agent(network: SchafkopfNetwork, num_games: int = 1000, device: str = "cpu"):
    """Evaluate agent over many games."""
    wins = 0
    total_points = 0
    schneiders = 0
    schwarzs = 0
    
    for i in range(num_games):
        result = play_game(network, render=False, device=device)
        if result['win']:
            wins += 1
        if result['schneider']:
            schneiders += 1
        if result['schwarz']:
            schwarzs += 1
        total_points += result['declarer_points']
        
        if (i + 1) % 100 == 0:
            print(f"Games: {i+1}/{num_games} | Win Rate: {wins/(i+1):.1%}")
    
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ({num_games} games)")
    print("=" * 60)
    print(f"Win Rate:      {wins/num_games:.1%}")
    print(f"Avg Points:    {total_points/num_games:.1f}")
    print(f"Schneider:     {schneiders} ({schneiders/num_games:.1%})")
    print(f"Schwarz:       {schwarzs} ({schwarzs/num_games:.1%})")
    print("=" * 60)
    
    return {
        "win_rate": wins / num_games,
        "avg_points": total_points / num_games,
        "schneider_rate": schneiders / num_games,
        "schwarz_rate": schwarzs / num_games,
    }


def compare_agents(checkpoint1: str, checkpoint2: str, num_games: int = 500, device: str = "cpu"):
    """Compare two trained agents."""
    print(f"Loading agent 1: {checkpoint1}")
    network1 = load_agent(checkpoint1, device)
    
    print(f"Loading agent 2: {checkpoint2}")
    network2 = load_agent(checkpoint2, device)
    
    print(f"\nComparing over {num_games} games each...\n")
    
    print("Agent 1 as all players:")
    stats1 = evaluate_agent(network1, num_games, device)
    
    print("\nAgent 2 as all players:")
    stats2 = evaluate_agent(network2, num_games, device)
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Agent 1 Win Rate: {stats1['win_rate']:.1%}")
    print(f"Agent 2 Win Rate: {stats2['win_rate']:.1%}")
    print(f"Difference: {(stats1['win_rate'] - stats2['win_rate'])*100:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Play/evaluate Schafkopf with trained agent")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["play", "eval", "compare"],
        default="play",
        help="Mode: play (single game), eval (many games), compare (two models)"
    )
    parser.add_argument(
        "--games", "-g",
        type=int,
        default=1000,
        help="Number of games for evaluation (default: 1000)"
    )
    parser.add_argument(
        "--compare-to",
        type=str,
        default=None,
        help="Second checkpoint for comparison mode"
    )
    parser.add_argument(
        "--num-plays", "-n",
        type=int,
        default=1,
        help="Number of games to play in play mode (default: 1)"
    )
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    network = load_agent(args.checkpoint, device)
    print(f"Loaded model from {args.checkpoint}")
    
    if args.mode == "play":
        for i in range(args.num_plays):
            if args.num_plays > 1:
                print(f"\n{'='*60}")
                print(f"GAME {i+1}/{args.num_plays}")
            play_game(network, render=True, device=device)
    
    elif args.mode == "eval":
        evaluate_agent(network, args.games, device)
    
    elif args.mode == "compare":
        if not args.compare_to:
            print("Error: --compare-to required for compare mode")
            sys.exit(1)
        compare_agents(args.checkpoint, args.compare_to, args.games, device)


if __name__ == "__main__":
    main()
