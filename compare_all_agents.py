#!/usr/bin/env python3
"""
Compare all agents against each other in a comprehensive table.
Tests each agent as attacker (declarer team) and defender (opponent team).
"""

import numpy as np
import torch
from tabulate import tabulate
from src.schafkopf_ai.env import make_env, CARD_TO_IDX, IDX_TO_CARD
from src.schafkopf_ai.baseline import RuleBasedAgent, RandomAgent
from src.schafkopf_ai.hybrid_agent import HybridAgent
from src.schafkopf_ai.ppo import SchafkopfNetwork, SchafkopfLSTMNetwork


class LegacySchafkopfNetwork(torch.nn.Module):
    """Legacy network with 2 encoder layers (for old checkpoints)."""
    
    def __init__(self, hidden_size: int = 256, num_encoder_layers: int = 2):
        super().__init__()
        input_size = 32 + 32 + 128 + 4 + 8  # 204
        
        layers = []
        layers.append(torch.nn.Linear(input_size, hidden_size))
        layers.append(torch.nn.ReLU())
        for _ in range(num_encoder_layers - 1):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(torch.nn.ReLU())
        
        self.encoder = torch.nn.Sequential(*layers)
        
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, 32),
        )
        
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, 1),
        )
    
    def forward(self, obs, action_mask=None):
        x = torch.cat([
            obs["own_hand"].float(),
            obs["played_cards"].float(),
            obs["current_trick"].float().flatten(start_dim=-2),
            obs["points"],
            obs["game_info"],
        ], dim=-1)
        
        features = self.encoder(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e8)
        
        return logits, value


class PPOAgentWrapper:
    """Wrapper to use trained PPO network as an agent."""
    
    def __init__(self, checkpoint_path: str):
        import sys
        import src.schafkopf_ai as schafkopf_ai
        sys.modules['schafkopf_ai'] = schafkopf_ai  # Legacy module alias
        
        self.device = torch.device("cpu")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("network_state_dict", checkpoint.get("model_state_dict", {}))
        
        # Auto-detect architecture from state dict
        has_lstm = any("lstm" in k for k in state_dict.keys())
        hidden_size = state_dict["encoder.0.weight"].shape[0]
        # Count encoder layers by looking at layer indices (0, 2, 4, etc for Linear layers)
        encoder_layer_indices = set()
        for k in state_dict.keys():
            if k.startswith("encoder.") and "weight" in k:
                idx = int(k.split(".")[1])
                encoder_layer_indices.add(idx)
        num_encoder_layers = len(encoder_layer_indices)
        
        if has_lstm:
            self.network = SchafkopfLSTMNetwork(hidden_size=hidden_size)
            self.use_lstm = True
        else:
            self.network = LegacySchafkopfNetwork(hidden_size=hidden_size, 
                                                   num_encoder_layers=num_encoder_layers)
            self.use_lstm = False
        
        self.network.load_state_dict(state_dict)
        self.network.eval()
        self.lstm_state = None
    
    def reset_lstm(self):
        """Reset LSTM state for new game."""
        self.lstm_state = None
    
    def get_action(self, obs_dict, action_mask):
        """Get action from the network."""
        with torch.no_grad():
            # Convert to tensors
            obs_tensors = {
                k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) 
                for k, v in obs_dict.items()
            }
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)
            
            if self.use_lstm:
                logits, _, self.lstm_state = self.network.forward(
                    obs_tensors, mask_tensor, self.lstm_state
                )
            else:
                logits, _ = self.network.forward(obs_tensors, mask_tensor)
            
            # Get best valid action
            action = logits.argmax(dim=-1).item()
            return action


def load_agents():
    """Load all available agents."""
    from src.schafkopf_ai.smart_hybrid import SmartHybridAgent
    
    agents = {
        "Random": RandomAgent(),
        "RuleBased": RuleBasedAgent(),
        "Hybrid": HybridAgent("data/mc_training_data.csv", mistake_threshold=0.05),
    }
    
    # Try to load SmartHybrid agent
    try:
        agents["SmartHybrid"] = SmartHybridAgent(correction_threshold=0.6)
    except Exception as e:
        print(f"Could not load SmartHybrid: {e}")
    
    # Try to load PPO agent
    for checkpoint_name in ["schafkopf_ppo_final.pt", "schafkopf_ppo_vs_rulebased_final.pt", 
                            "schafkopf_ppo_focused_final.pt"]:
        checkpoint_path = f"checkpoints/{checkpoint_name}"
        try:
            ppo = PPOAgentWrapper(checkpoint_path)
            short_name = checkpoint_name.replace("schafkopf_ppo_", "PPO_").replace("_final.pt", "")
            agents[short_name] = ppo
            print(f"Loaded PPO agent: {short_name}")
        except Exception as e:
            print(f"Could not load {checkpoint_name}: {e}")
    
    return agents


def evaluate_matchup(agent_a, agent_b, agent_a_name: str, agent_b_name: str,
                     a_plays_declarer: bool, num_games: int = 200, seed: int = 42) -> float:
    """
    Evaluate agent A vs agent B.
    
    Args:
        agent_a: First agent
        agent_b: Second agent  
        a_plays_declarer: If True, agent A plays declarer team
        num_games: Number of games to play
        seed: Random seed
    
    Returns:
        Win rate for agent A
    """
    env = make_env(seed=seed)
    rng = np.random.default_rng(seed)
    
    a_wins = 0
    
    for game in range(num_games):
        declarer = int(rng.integers(0, 4))
        env.reset(options={"fixed_declarer": declarer})
        
        # Reset LSTM states for PPO agents at start of each game
        if isinstance(agent_a, PPOAgentWrapper):
            agent_a.reset_lstm()
        if isinstance(agent_b, PPOAgentWrapper):
            agent_b.reset_lstm()
        
        while not all(env.terminations.values()):
            player_name = env.agent_selection
            player_idx = int(player_name.split("_")[1])
            
            hand = env._hands[player_idx]
            legal_indices = env._get_legal_actions(player_idx)
            legal_cards = [IDX_TO_CARD[idx] for idx in legal_indices]
            current_trick = [(p, c) for p, c in env._current_trick]
            
            # Determine if this player is on declarer team
            declarer_team = {env._declarer_idx}
            if env._partner_idx is not None:
                declarer_team.add(env._partner_idx)
            is_declarer_team = player_idx in declarer_team
            
            # Choose which agent to use
            use_agent_a = (a_plays_declarer == is_declarer_team)
            agent = agent_a if use_agent_a else agent_b
            
            # Get action from agent
            card = get_agent_action(
                agent, hand, legal_cards, current_trick, env, player_idx
            )
            
            action = CARD_TO_IDX[card]
            env.step(action)
        
        result = env.get_game_result()
        declarer_won = result["win"]
        
        # Agent A wins if they were declarer team and won, or opponent team and declarer lost
        if a_plays_declarer == declarer_won:
            a_wins += 1
    
    return a_wins / num_games


def get_agent_action(agent, hand, legal_cards, current_trick, env, player_idx) -> str:
    """Get action from any agent type."""
    
    if isinstance(agent, PPOAgentWrapper):
        # PPO agent needs dict observation
        obs_dict = env._encode_observation(player_idx)
        action_mask = obs_dict["action_mask"]
        action_idx = agent.get_action(obs_dict, action_mask)
        return IDX_TO_CARD[action_idx]
    elif hasattr(agent, 'select_action'):
        # RuleBased, Hybrid
        return agent.select_action(
            hand=hand,
            legal_actions=legal_cards,
            current_trick=current_trick,
            played_cards=env._played_cards,
            player_idx=player_idx,
            declarer_idx=env._declarer_idx,
            partner_idx=env._partner_idx,
            points=env._points,
            trick_number=env._trick_number,
        )
    elif hasattr(agent, 'choose_action'):
        # RandomAgent fallback
        import random
        return str(random.choice(legal_cards))
    else:
        raise ValueError(f"Unknown agent type: {type(agent)}")


def run_full_comparison(num_games: int = 200, seed: int = 42):
    """Run full comparison between all agents."""
    
    print("=" * 70)
    print("COMPREHENSIVE AGENT COMPARISON")
    print("=" * 70)
    print(f"\nGames per matchup: {num_games}")
    print()
    
    agents = load_agents()
    agent_names = list(agents.keys())
    
    # Results tables
    results_as_attacker = {}  # results[row_agent][col_agent] = win rate when row attacks
    results_as_defender = {}
    
    for name in agent_names:
        results_as_attacker[name] = {}
        results_as_defender[name] = {}
    
    # Run all matchups
    total_matchups = len(agent_names) * (len(agent_names) - 1)
    current = 0
    
    for a_name in agent_names:
        for b_name in agent_names:
            if a_name == b_name:
                continue
            
            current += 1
            print(f"[{current}/{total_matchups}] {a_name} vs {b_name}...", end=" ", flush=True)
            
            agent_a = agents[a_name]
            agent_b = agents[b_name]
            
            # A as attacker (declarer)
            win_rate_attack = evaluate_matchup(
                agent_a, agent_b, a_name, b_name,
                a_plays_declarer=True, num_games=num_games, seed=seed
            )
            results_as_attacker[a_name][b_name] = win_rate_attack
            
            # A as defender (opponent)
            win_rate_defend = evaluate_matchup(
                agent_a, agent_b, a_name, b_name,
                a_plays_declarer=False, num_games=num_games, seed=seed
            )
            results_as_defender[a_name][b_name] = win_rate_defend
            
            print(f"Attack: {win_rate_attack*100:.1f}%, Defend: {win_rate_defend*100:.1f}%")
    
    # Print results tables
    print("\n" + "=" * 70)
    print("RESULTS: WIN RATE AS ATTACKER (Declarer Team)")
    print("=" * 70)
    print("Row agent attacks, Column agent defends\n")
    
    # Build table for attacker results
    headers = ["Attacker \\ Defender"] + [n for n in agent_names]
    table_attack = []
    for a_name in agent_names:
        row = [a_name]
        for b_name in agent_names:
            if a_name == b_name:
                row.append("-")
            else:
                rate = results_as_attacker[a_name].get(b_name, 0)
                row.append(f"{rate*100:.1f}%")
        table_attack.append(row)
    
    print(tabulate(table_attack, headers=headers, tablefmt="grid"))
    
    print("\n" + "=" * 70)
    print("RESULTS: WIN RATE AS DEFENDER (Opponent Team)")
    print("=" * 70)
    print("Row agent defends, Column agent attacks\n")
    
    # Build table for defender results
    headers = ["Defender \\ Attacker"] + [n for n in agent_names]
    table_defend = []
    for a_name in agent_names:
        row = [a_name]
        for b_name in agent_names:
            if a_name == b_name:
                row.append("-")
            else:
                rate = results_as_defender[a_name].get(b_name, 0)
                row.append(f"{rate*100:.1f}%")
        table_defend.append(row)
    
    print(tabulate(table_defend, headers=headers, tablefmt="grid"))
    
    # Summary stats
    print("\n" + "=" * 70)
    print("OVERALL PERFORMANCE SUMMARY")
    print("=" * 70)
    
    summary = []
    for name in agent_names:
        attack_rates = [v for k, v in results_as_attacker[name].items()]
        defend_rates = [v for k, v in results_as_defender[name].items()]
        
        avg_attack = np.mean(attack_rates) if attack_rates else 0
        avg_defend = np.mean(defend_rates) if defend_rates else 0
        overall = (avg_attack + avg_defend) / 2
        
        summary.append([
            name,
            f"{avg_attack*100:.1f}%",
            f"{avg_defend*100:.1f}%",
            f"{overall*100:.1f}%"
        ])
    
    # Sort by overall performance
    summary.sort(key=lambda x: float(x[3].replace('%', '')), reverse=True)
    
    print()
    print(tabulate(summary, 
                   headers=["Agent", "Avg Attack", "Avg Defend", "Overall"],
                   tablefmt="grid"))
    
    print("\n✓ Higher is better for all metrics")
    print("✓ Attack = playing as declarer team")
    print("✓ Defend = playing as opponent team")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200, help="Games per matchup")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_full_comparison(num_games=args.games, seed=args.seed)
