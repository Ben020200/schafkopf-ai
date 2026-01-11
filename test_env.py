"""Test script for the Schafkopf PettingZoo environment."""

import sys
sys.path.insert(0, 'src')

from schafkopf_ai.env import SchafkopfEnv, make_env


def test_basic_game():
    """Run a complete game with random actions."""
    print("=== Testing Basic Game ===\n")
    
    env = make_env(render_mode="human", seed=42)
    env.reset(seed=42)
    
    step_count = 0
    while not all(env.terminations.values()):
        agent = env.agent_selection
        obs = env.observe(agent)
        
        # Get legal actions from mask
        action_mask = obs["action_mask"]
        legal_actions = [i for i, v in enumerate(action_mask) if v == 1]
        
        # Random legal action
        import numpy as np
        action = np.random.choice(legal_actions)
        
        print(f"\n{agent} plays card index {action}")
        env.step(action)
        step_count += 1
        
        if step_count % 4 == 0:
            env.render()
    
    print("\n=== Game Over ===")
    result = env.get_game_result()
    print(f"Declarer (Player {result['declarer_idx']}) + Partner (Player {result['partner_idx']})")
    print(f"Called ace: {result['called_ace']}")
    print(f"Declarer team points: {result['declarer_points']}")
    print(f"Opponent points: {result['opponent_points']}")
    print(f"Winner: {'Declarer team' if result['win'] else 'Opponents'}")
    print(f"Schneider: {result['schneider']}, Schwarz: {result['schwarz']}")
    print(f"Final rewards: {env._cumulative_rewards}")
    
    env.close()
    print("\n✓ Basic game test passed!")


def test_observation_space():
    """Test observation encoding."""
    print("\n=== Testing Observation Space ===\n")
    
    env = make_env(seed=123)
    env.reset()
    
    obs = env.observe("player_0")
    
    print("Observation keys:", list(obs.keys()))
    print(f"  own_hand shape: {obs['own_hand'].shape}, sum: {obs['own_hand'].sum()}")
    print(f"  played_cards shape: {obs['played_cards'].shape}")
    print(f"  current_trick shape: {obs['current_trick'].shape}")
    print(f"  points shape: {obs['points'].shape}")
    print(f"  game_info shape: {obs['game_info'].shape}")
    print(f"  action_mask shape: {obs['action_mask'].shape}, legal: {obs['action_mask'].sum()}")
    
    assert obs['own_hand'].sum() == 8, "Should have 8 cards in hand"
    assert obs['action_mask'].sum() == 8, "All cards should be legal when leading"
    
    print("\n✓ Observation space test passed!")


def test_legal_actions():
    """Test legal action masking."""
    print("\n=== Testing Legal Actions ===\n")
    
    env = make_env(seed=456)
    env.reset()
    
    # Play through a few steps and check masking
    for i in range(8):  # Two tricks
        agent = env.agent_selection
        player_idx = env.agent_name_mapping[agent]
        obs = env.observe(agent)
        
        legal_actions = [i for i, v in enumerate(obs['action_mask']) if v == 1]
        hand = env._hands[player_idx]
        
        print(f"Turn {i}: {agent}")
        print(f"  Hand: {hand}")
        print(f"  Legal actions: {len(legal_actions)} cards")
        print(f"  Current trick: {env._current_trick}")
        
        # Pick first legal action
        action = legal_actions[0]
        env.step(action)
        
        if all(env.terminations.values()):
            break
    
    print("\n✓ Legal actions test passed!")


def test_multiple_games():
    """Run multiple games to check stability."""
    print("\n=== Testing Multiple Games ===\n")
    
    import numpy as np
    
    env = make_env(reward_mode="sparse")
    wins = 0
    total_games = 100
    
    for game_num in range(total_games):
        env.reset(seed=game_num)
        
        while not all(env.terminations.values()):
            agent = env.agent_selection
            obs = env.observe(agent)
            legal = [i for i, v in enumerate(obs['action_mask']) if v == 1]
            action = np.random.choice(legal)
            env.step(action)
        
        result = env.get_game_result()
        if result['win']:
            wins += 1
    
    win_rate = wins / total_games
    print(f"Declarer win rate over {total_games} games: {win_rate:.1%}")
    print(f"(Random play should be ~50%)")
    
    env.close()
    print("\n✓ Multiple games test passed!")


if __name__ == "__main__":
    test_observation_space()
    test_legal_actions()
    test_basic_game()
    test_multiple_games()
    
    print("\n" + "="*50)
    print("All tests passed! Environment is ready for RL training.")
    print("="*50)
