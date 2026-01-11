"""
Evaluate the legal-action-only MC agent vs RuleBased for 200 games.
"""
import torch
from src.schafkopf_ai.env import make_env, CARD_TO_IDX, IDX_TO_CARD
from src.schafkopf_ai.baseline import RuleBasedAgent
import numpy as np

class LegalMCNet(torch.nn.Module):
    def __init__(self, context_dim=3, card_embed_dim=16, hidden=64):
        super().__init__()
        self.card_embed = torch.nn.Embedding(32, card_embed_dim)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(8 * card_embed_dim + context_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 8)
        )
    def forward(self, legal, context):
        emb = self.card_embed(legal.clamp(min=0))
        emb[legal == -1] = 0
        flat = emb.view(emb.size(0), -1)
        x = torch.cat([flat, context], dim=1)
        logits = self.fc(x)
        return logits

def select_action(model, hand, legal_cards, context, device):
    # Prepare input
    legal_idx = [CARD_TO_IDX[c] for c in legal_cards]
    padded = np.full(8, -1, dtype=np.int64)
    padded[:len(legal_idx)] = legal_idx
    context = torch.tensor([context], dtype=torch.float32).to(device)
    legal = torch.tensor([padded], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(legal, context)
        mask = torch.zeros(1,8).to(device)
        mask[0,:len(legal_cards)] = 1.0
        logits = logits + (mask - 1) * 1e9
        best = logits.argmax(dim=1).item()
    return legal_cards[best]

def evaluate(model_path, num_games=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LegalMCNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    rb_agent = RuleBasedAgent()
    env = make_env()
    rng = np.random.default_rng(42)
    wins = 0
    for game in range(num_games):
        env.reset()
        mc_plays_declarer = game % 2 == 0
        while not all(env.terminations.values()):
            player_name = env.agent_selection
            player_idx = int(player_name.split('_')[1])
            hand = env._hands[player_idx]
            legal_indices = env._get_legal_actions(player_idx)
            legal_cards = [IDX_TO_CARD[idx] for idx in legal_indices]
            trick_number = env._trick_number
            is_declarer = float(player_idx == env._declarer_idx or player_idx == env._partner_idx)
            context = [trick_number/7.0, is_declarer, player_idx/3.0]
            declarer_team = {env._declarer_idx}
            if env._partner_idx is not None:
                declarer_team.add(env._partner_idx)
            is_declarer_team = player_idx in declarer_team
            use_mc = (mc_plays_declarer == is_declarer_team)
            if use_mc:
                card = select_action(model, hand, legal_cards, context, device)
            else:
                card = rb_agent.select_action(
                    hand=hand, legal_actions=legal_cards,
                    current_trick=env._current_trick, played_cards=env._played_cards,
                    player_idx=player_idx, declarer_idx=env._declarer_idx,
                    partner_idx=env._partner_idx, points=env._points,
                    trick_number=env._trick_number
                )
            action = CARD_TO_IDX[card]
            env.step(action)
        result = env.get_game_result()
        if mc_plays_declarer == result["win"]:
            wins += 1
        if (game+1) % 50 == 0:
            print(f"  Games {game+1}/{num_games}: Win rate = {wins/(game+1)*100:.1f}%")
    print(f"\nFinal: {wins}/{num_games} = {wins/num_games*100:.1f}% win rate")

if __name__ == '__main__':
    evaluate('checkpoints/legal_mc_agent.pt', 200)
