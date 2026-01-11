"""
Final Hybrid: Blend RuleBased heuristics with MC Q-values.

Simple approach:
1. Get RuleBased choice and score it with heuristics
2. Get MC model's Q-value predictions
3. Blend: pick the card with highest (RB_score * alpha + MC_Q * (1-alpha))

This way we leverage both the heuristics and the learned values.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from collections import defaultdict

from src.schafkopf_ai.baseline import RuleBasedAgent
from src.schafkopf_ai.env import CARD_TO_IDX, IDX_TO_CARD


TRUMP_CARDS = {
    "O_Acorn", "O_Leaf", "O_Heart", "O_Bell",
    "U_Acorn", "U_Leaf", "U_Heart", "U_Bell",
    "A_Heart", "10_Heart", "K_Heart", "9_Heart", "8_Heart", "7_Heart"
}

TRUMP_ORDER = [
    "O_Acorn", "O_Leaf", "O_Heart", "O_Bell",
    "U_Acorn", "U_Leaf", "U_Heart", "U_Bell",
    "A_Heart", "10_Heart", "K_Heart", "9_Heart", "8_Heart", "7_Heart"
]

def is_trump(card: str) -> bool:
    return card in TRUMP_CARDS

def card_points(card: str) -> int:
    rank = card.split("_")[0]
    return {"A": 11, "10": 10, "K": 4, "O": 3, "U": 2}.get(rank, 0)

def card_strength(card: str) -> float:
    if card in TRUMP_ORDER:
        return 1.0 - (TRUMP_ORDER.index(card) / 14.0)
    rank = card.split("_")[0]
    return {"A": 0.4, "10": 0.35, "K": 0.3, "9": 0.15, "8": 0.1, "7": 0.05}.get(rank, 0.0)


class BlendDataset(Dataset):
    """Load MC data with Q-values for training."""
    
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.samples = []
        
        for _, row in df.iterrows():
            sample = self._process(row, df.columns)
            if sample is not None:
                self.samples.append(sample)
        
        print(f"  Loaded {len(self.samples)} samples")
    
    def _process(self, row, columns):
        hand = str(row['hand']).split('|') if pd.notna(row['hand']) else []
        legal = str(row['legal_actions']).split('|') if pd.notna(row['legal_actions']) else []
        
        if len(legal) < 2:
            return None
        
        # Get Q-values
        q_values = []
        for card in legal:
            col = f"value_{card}"
            if col in columns and pd.notna(row[col]):
                q_values.append(float(row[col]))
            else:
                q_values.append(0.5)
        
        best_idx = int(np.argmax(q_values))
        
        # Features
        trick_str = row['current_trick']
        trick_cards = []
        if pd.notna(trick_str) and trick_str:
            for item in str(trick_str).split('|'):
                if ':' in item:
                    pos, card = item.split(':')
                    trick_cards.append((int(pos), card))
        
        features = self._features(hand, legal, trick_cards, row)
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'q_values': torch.tensor(q_values, dtype=torch.float32),
            'best_idx': best_idx,
            'num_legal': len(legal),
        }
    
    def _features(self, hand, legal, trick_cards, row):
        f = []
        
        # Game state
        f.append(float(row['trick_number']) / 7.0)
        f.append(float(row['is_declarer']))
        f.append(len(trick_cards) / 3.0)
        f.append(len(legal) / 8.0)
        
        # Hand
        f.append(sum(1 for c in hand if is_trump(c)) / 8.0)
        f.append(sum(card_points(c) for c in hand) / 120.0)
        
        # Trick
        f.append(sum(card_points(c) for _, c in trick_cards) / 40.0)
        f.append(float(any(is_trump(c) for _, c in trick_cards)))
        
        # Per-card features (8 cards max)
        for card in legal[:8]:
            f.extend([
                float(is_trump(card)),
                card_strength(card),
                card_points(card) / 11.0,
            ])
        
        # Pad
        for _ in range(8 - len(legal)):
            f.extend([0.0, 0.0, 0.0])
        
        return f  # 8 + 8*3 = 32 features
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    features = torch.stack([s['features'] for s in batch])
    
    q_padded = torch.zeros(len(batch), 8)
    masks = torch.zeros(len(batch), 8)
    best_indices = []
    
    for i, s in enumerate(batch):
        n = s['num_legal']
        q_padded[i, :n] = s['q_values']
        masks[i, :n] = 1.0
        best_indices.append(s['best_idx'])
    
    return {
        'features': features,
        'q_values': q_padded,
        'mask': masks,
        'best_idx': torch.tensor(best_indices, dtype=torch.long),
    }


class QModel(nn.Module):
    def __init__(self, input_size=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 8),
        )
    
    def forward(self, x):
        return self.net(x)


def train_qmodel(epochs=100):
    print("=" * 60)
    print("TRAINING BLEND HYBRID Q-MODEL")
    print("=" * 60)
    
    ds = BlendDataset("data/mc_training_data.csv")
    n = len(ds)
    train_ds = torch.utils.data.Subset(ds, range(int(0.8*n)))
    val_ds = torch.utils.data.Subset(ds, range(int(0.8*n), n))
    
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=128, collate_fn=collate)
    
    device = torch.device('cpu')
    model = QModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            f = batch['features'].to(device)
            q = batch['q_values'].to(device)
            m = batch['mask'].to(device)
            
            opt.zero_grad()
            pred = model(f)
            loss = F.mse_loss(pred * m, q * m)
            loss.backward()
            opt.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_dl:
                f = batch['features'].to(device)
                m = batch['mask'].to(device)
                best = batch['best_idx'].to(device)
                
                pred = model(f)
                pred = pred + (m - 1) * 1e9
                pred_best = pred.argmax(dim=1)
                correct += (pred_best == best).sum().item()
                total += len(best)
        
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch+1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: Val Acc = {acc*100:.1f}%")
    
    torch.save({'model': best_state, 'val_acc': best_acc}, "checkpoints/blend_q.pt")
    print(f"\nBest: {best_acc*100:.1f}%")
    return model


class BlendHybridAgent:
    """Blends RuleBased ranking with learned Q-values."""
    
    def __init__(self, alpha: float = 0.3):
        """alpha: weight for RuleBased (0=pure MC, 1=pure RuleBased)"""
        self.rulebased = RuleBasedAgent()
        self.alpha = alpha
        self.device = torch.device('cpu')
        
        try:
            ckpt = torch.load("checkpoints/blend_q.pt", map_location=self.device, weights_only=False)
            self.model = QModel().to(self.device)
            self.model.load_state_dict(ckpt['model'])
            self.model.eval()
            print(f"Loaded Q-model (val_acc={ckpt['val_acc']*100:.1f}%), alpha={alpha}")
        except:
            self.model = None
            print("No model, using pure RuleBased")
    
    def select_action(self, hand, legal_actions, current_trick, played_cards,
                      player_idx, declarer_idx, partner_idx, points, trick_number):
        
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        # Get RuleBased choice
        rb_choice = self.rulebased.select_action(
            hand=hand, legal_actions=legal_actions,
            current_trick=current_trick, played_cards=played_cards,
            player_idx=player_idx, declarer_idx=declarer_idx,
            partner_idx=partner_idx, points=points, trick_number=trick_number
        )
        
        if self.model is None:
            return rb_choice
        
        # Build features
        features = self._build_features(hand, legal_actions, current_trick,
                                         trick_number, player_idx, declarer_idx, partner_idx)
        
        # Get Q-values from model
        with torch.no_grad():
            f = torch.tensor([features], dtype=torch.float32).to(self.device)
            q_pred = self.model(f)[0].numpy()
        
        # Score each legal action
        scores = []
        for i, card in enumerate(legal_actions):
            # RuleBased score: 1.0 for chosen, 0.5 for others
            rb_score = 1.0 if card == rb_choice else 0.5
            
            # MC Q-value (normalized to 0-1)
            mc_score = q_pred[i] if i < len(q_pred) else 0.5
            
            # Blend
            blend = self.alpha * rb_score + (1 - self.alpha) * mc_score
            scores.append(blend)
        
        # Pick best
        best_idx = int(np.argmax(scores))
        return legal_actions[best_idx]
    
    def _build_features(self, hand, legal, trick_cards, trick_num, player_idx, declarer_idx, partner_idx):
        f = []
        
        declarer_team = {declarer_idx}
        if partner_idx is not None:
            declarer_team.add(partner_idx)
        is_declarer = float(player_idx in declarer_team)
        
        f.extend([trick_num / 7.0, is_declarer, len(trick_cards) / 3.0, len(legal) / 8.0])
        f.extend([sum(1 for c in hand if is_trump(c)) / 8.0, sum(card_points(c) for c in hand) / 120.0])
        f.extend([sum(card_points(c) for _, c in trick_cards) / 40.0, float(any(is_trump(c) for _, c in trick_cards))])
        
        for card in legal[:8]:
            f.extend([float(is_trump(card)), card_strength(card), card_points(card) / 11.0])
        for _ in range(8 - len(legal)):
            f.extend([0.0, 0.0, 0.0])
        
        return f


def evaluate(num_games=200, alpha=0.3):
    from src.schafkopf_ai.env import make_env
    
    print(f"\n{'='*60}")
    print(f"EVALUATING BLEND HYBRID (alpha={alpha})")
    print("=" * 60)
    
    env = make_env()
    hybrid = BlendHybridAgent(alpha=alpha)
    rb = RuleBasedAgent()
    
    wins = 0
    
    for game in range(num_games):
        env.reset()
        hybrid_plays_declarer = game % 2 == 0
        
        while not all(env.terminations.values()):
            player_name = env.agent_selection
            player_idx = int(player_name.split('_')[1])
            
            hand = env._hands[player_idx]
            legal_indices = env._get_legal_actions(player_idx)
            legal_cards = [IDX_TO_CARD[idx] for idx in legal_indices]
            
            declarer_team = {env._declarer_idx}
            if env._partner_idx is not None:
                declarer_team.add(env._partner_idx)
            is_declarer_team = player_idx in declarer_team
            use_hybrid = (hybrid_plays_declarer == is_declarer_team)
            
            agent = hybrid if use_hybrid else rb
            
            card = agent.select_action(
                hand=hand, legal_actions=legal_cards,
                current_trick=env._current_trick, played_cards=env._played_cards,
                player_idx=player_idx, declarer_idx=env._declarer_idx,
                partner_idx=env._partner_idx, points=env._points,
                trick_number=env._trick_number
            )
            
            action = CARD_TO_IDX[card]
            env.step(action)
        
        result = env.get_game_result()
        if hybrid_plays_declarer == result["win"]:
            wins += 1
        
        if (game + 1) % 50 == 0:
            print(f"  Games {game+1}/{num_games}: Win rate = {wins/(game+1)*100:.1f}%")
    
    print(f"\nFinal: {wins}/{num_games} = {wins/num_games*100:.1f}% win rate")
    return wins / num_games


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.3)
    args = parser.parse_args()
    
    if args.train:
        train_qmodel(args.epochs)
    
    if args.eval or not args.train:
        evaluate(args.games, args.alpha)
