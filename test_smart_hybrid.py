"""Quick test of SmartHybrid agent."""
from src.schafkopf_ai.smart_hybrid import SmartHybridAgent
from src.schafkopf_ai.baseline import RuleBasedAgent
from src.schafkopf_ai.env import SchafkopfEnv, IDX_TO_CARD, CARD_TO_IDX

print('Testing SmartHybrid vs RuleBased...')

smart_hybrid = SmartHybridAgent('checkpoints/smart_hybrid.pt')
rulebased = RuleBasedAgent()

agents = {
    'player_0': smart_hybrid,
    'player_1': rulebased,
    'player_2': rulebased,
    'player_3': rulebased,
}

# Run 100 games
wins = {0: 0, 1: 0}  # 0=SmartHybrid, 1=RuleBased team
env = SchafkopfEnv()

for game in range(100):
    env.reset()
    done = False
    
    while not done:
        for agent_id in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            
            if term or trunc:
                action = None
            else:
                agent = agents[agent_id]
                
                # Extract player index from agent_id (e.g. 'player_0' -> 0)
                player_idx = int(agent_id.split('_')[1])
                
                # Get legal actions from action_mask
                mask = obs['action_mask']
                legal_actions = [IDX_TO_CARD[i] for i, v in enumerate(mask) if v]
                
                # Get hand and other game state from env's internal state
                hand = env._hands[player_idx]
                
                action = agent.select_action(
                    hand=hand,
                    legal_actions=legal_actions,
                    current_trick=env._current_trick,
                    played_cards=env._played_cards,
                    player_idx=player_idx,
                    declarer_idx=env._declarer_idx,
                    partner_idx=env._partner_idx,
                    points=env._points,
                    trick_number=env._trick_number
                )
                # Convert to action index
                if action:
                    action = CARD_TO_IDX[action]
            
            env.step(action)
            
            if term or trunc:
                break
        
        # Check if game is done
        if all(env.terminations.values()):
            done = True
            # Check who won - SmartHybrid is player_0
            rewards = env.rewards
            if rewards['player_0'] > 0:
                wins[0] += 1
            else:
                wins[1] += 1

print(f'\nResults after 100 games:')
print(f'SmartHybrid (P0): {wins[0]} wins')
print(f'RuleBased (P1-3): {wins[1]} wins')

# Correction stats
print(f'\nSmartHybrid correction rate: {smart_hybrid.get_correction_rate():.1%}')
                
                action = agent.select_action(
                    hand=obs['hand'],
                    legal_actions=legal_actions,
                    current_trick=obs['current_trick'],
                    played_cards=obs['played_cards'],
                    player_idx=player_idx,
                    declarer_idx=obs['declarer'],
                    partner_idx=obs.get('partner'),
                    points=obs.get('points', [0,0,0,0]),
                    trick_number=obs.get('trick_number', 0)
                )
                # Convert to action index
                if action:
                    action = CARD_TO_IDX[action]
            
            env.step(action)
            
            if term or trunc:
                break
        
        # Check if game is done
        if all(env.terminations.values()):
            done = True
            # Check who won - SmartHybrid is player_0
            rewards = env.rewards
            if rewards['player_0'] > 0:
                wins[0] += 1
            else:
                wins[1] += 1

print(f'\nResults after 100 games:')
print(f'SmartHybrid (P0): {wins[0]} wins')
print(f'RuleBased (P1-3): {wins[1]} wins')

# Correction stats
print(f'\nSmartHybrid correction rate: {smart_hybrid.get_correction_rate():.1%}')
