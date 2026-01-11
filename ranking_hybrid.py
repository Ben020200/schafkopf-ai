"""
Ranking Hybrid: Learn to rank legal actions, not predict Q-values.

Key insight: We don't need accurate Q-values, just correct rankings.
Train with pairwise ranking loss: best card should score higher than others.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import random

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

def get_suit(card: str) -> str:
    if is_trump(card):
        return "Trump"
    return card.split("_")[1]

def card_points(card: str) -> int:
    rank = card.split("_")[0]
    return {"A": 11, "10": 10, "K": 4, "O": 3, "U": 2}.get(rank, 0)

def card_strength(card: str) -> float:
    if card in TRUMP_ORDER:
        return 1.0 - (TRUMP_ORDER.index(card) / 14.0)
    rank = card.split("_")[0]
    return {"A": 0.4, "10": 0.35, "K": 0.3, "9": 0.15, "8": 0.1, "7": 0.05}.get(rank, 0.0)


class RankingDataset(Dataset):
    """Pairs: (state, best_card, worse_card)"""
    
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.samples = []
        
        for _, row in df.iterrows():
            pairs = self._extract_pairs(row, df.columns)
            self.samples.extend(pairs)
        
        random.shuffle(self.samples)
        print(f"  Loaded {len(self.samples)} ranking pairs")
    
    def _extract_pairs(self, row, columns):
        legal = str(row['legal_actions']).split('|') if pd.notna(row['legal_actions']) else []
        
        if len(legal) < 2:
            return []
        
        # Get Q-values
        q_values = {}
        for card in legal:
            col = f"value_{card}"
            if col in columns and pd.notna(row[col]):
                q_values[card] = float(row[col])
            else:
                q_values[card] = 0.5
        
        # Sort by Q-value
        sorted_cards = sorted(legal, key=lambda c: q_values[c], reverse=True)
        best_card = sorted_cards[0]
        best_q = q_values[best_card]
        
        # Create pairs: (best vs each worse)
        pairs = []
        for worse_card in sorted_cards[1:]:
            worse_q = q_values[worse_card]
            
            # Only create pair if difference is meaningful
            if best_q - worse_q > 0.05:
                features = self._features(row, legal)
                best_vec = self._card_features(best_card, row, legal)
                worse_vec = self._card_features(worse_card, row, legal)
                
                pairs.append({
                    'state': torch.tensor(features, dtype=torch.float32),
                    'best': torch.tensor(best_vec, dtype=torch.float32),
                    'worse': torch.tensor(worse_vec, dtype=torch.float32),
                    'margin': best_q - worse_q,
                })
        
        return pairs
    
    def _features(self, row, legal):
        hand = str(row['hand']).split('|') if pd.notna(row['hand']) else []
        trick_str = row['current_trick']
        trick_cards = []
        if pd.notna(trick_str) and trick_str:
            for item in str(trick_str).split('|'):
                if ':' in item:
                    pos, card = item.split(':')
                    trick_cards.append((int(pos), card))
        
        f = []
        f.append(float(row['trick_number']) / 7.0)
        f.append(float(row['is_declarer']))
        f.append(len(trick_cards) / 3.0)
        f.append(len(legal) / 8.0)
        f.append(sum(1 for c in hand if is_trump(c)) / 8.0)
        f.append(sum(card_points(c) for c in hand) / 120.0)
        f.append(sum(card_points(c) for _, c in trick_cards) / 40.0)
        f.append(float(any(is_trump(c) for _, c in trick_cards)))
        
        # Led suit if trick started
        if trick_cards:
            led = trick_cards[0][1]
            f.append(float(is_trump(led)))
        else:
            f.append(0.0)
        
        return f  # 9 features
    
    def _card_features(self, card, row, legal):
        hand = str(row['hand']).split('|') if pd.notna(row['hand']) else []
        trick_str = row['current_trick']
        trick_cards = []
        if pd.notna(trick_str) and trick_str:
            for item in str(trick_str).split('|'):
                if ':' in item:
                    pos, card_ = item.split(':')
                    trick_cards.append((int(pos), card_))
        
        f = []
        f.append(float(is_trump(card)))
        f.append(card_strength(card))
        f.append(card_points(card) / 11.0)
        
        # Relative to trick
        if trick_cards:
            led_suit = get_suit(trick_cards[0][1])
            f.append(float(get_suit(card) == led_suit))
            
            # Would win?
            can_win = True
            for _, tc in trick_cards:
                if is_trump(tc):
                    if not is_trump(card):
                        can_win = False
                    elif card_strength(tc) > card_strength(card):
                        can_win = False
                elif is_trump(card):
                    pass
                elif get_suit(tc) == get_suit(card):
                    if card_strength(tc) > card_strength(card):
                        can_win = False
            f.append(float(can_win))
        else:
            f.append(0.0)  # Leading
            f.append(1.0)  # Can always "win" when leading
        
        # Position in hand strength
        my_suit_cards = [c for c in legal if get_suit(c) == get_suit(card)]
        if len(my_suit_cards) > 1:
            ranks = sorted([card_strength(c) for c in my_suit_cards], reverse=True)
            f.append(ranks.index(card_strength(card)) / len(ranks))
        else:
            f.append(0.0)
        
        return f  # 6 features
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class RankingModel(nn.Module):
    """Scores a single (state, card) pair."""
    
    def __init__(self, state_size=9, card_size=6, hidden=64):
        super().__init__()
        self.state_net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
        )
        self.card_net = nn.Sequential(
            nn.Linear(card_size, hidden),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    
    def forward(self, state, card):
        s = self.state_net(state)
        c = self.card_net(card)
        x = torch.cat([s, c], dim=-1)
        return self.combine(x).squeeze(-1)


def train_ranking(epochs=100):
    print("=" * 60)
    print("TRAINING RANKING MODEL")
    print("=" * 60)
    
    ds = RankingDataset("data/mc_training_data.csv")
    n = len(ds)
    train_ds = torch.utils.data.Subset(ds, range(int(0.8*n)))
    val_ds = torch.utils.data.Subset(ds, range(int(0.8*n), n))
    
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256)
    
    device = torch.device('cpu')
    model = RankingModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            state = batch['state'].to(device)
            best = batch['best'].to(device)
            worse = batch['worse'].to(device)
            margin = batch['margin'].to(device)
            
            opt.zero_grad()
            
            best_score = model(state, best)
            worse_score = model(state, worse)
            
            # Margin ranking loss: best should be higher than worse by margin
            loss = F.margin_ranking_loss(
                best_score, worse_score,
                torch.ones_like(best_score),
                margin=0.1
            )
            
            loss.backward()
            opt.step()
        
        # Validation: what % pairs ranked correctly?
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_dl:
                state = batch['state'].to(device)
                best = batch['best'].to(device)
                worse = batch['worse'].to(device)
                
                best_score = model(state, best)
                worse_score = model(state, worse)
                
                correct += (best_score > worse_score).sum().item()
                total += len(best_score)
        
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch+1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: Ranking Acc = {acc*100:.1f}%")
    
    torch.save({'model': best_state, 'val_acc': best_acc}, "checkpoints/ranking_model.pt")
    print(f"\nBest: {best_acc*100:.1f}%")
    return model


class RankingHybridAgent:
    """Uses ranking model to override RuleBased when confident."""
    
    def __init__(self, confidence_threshold: float = 0.2):
        self.rulebased = RuleBasedAgent()
        self.threshold = confidence_threshold
        self.device = torch.device('cpu')
        self.stats = {'total': 0, 'overrides': 0}
        
        try:
            ckpt = torch.load("checkpoints/ranking_model.pt", map_location=self.device, weights_only=False)
            self.model = RankingModel().to(self.device)
            self.model.load_state_dict(ckpt['model'])
            self.model.eval()
            print(f"Loaded ranking model (val_acc={ckpt['val_acc']*100:.1f}%), threshold={confidence_threshold}")
        except Exception as e:
            self.model = None
            print(f"No model, using pure RuleBased: {e}")
    
    def select_action(self, hand, legal_actions, current_trick, played_cards,
                      player_idx, declarer_idx, partner_idx, points, trick_number):
        
        self.stats['total'] += 1
        
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        rb_choice = self.rulebased.select_action(
            hand=hand, legal_actions=legal_actions,
            current_trick=current_trick, played_cards=played_cards,
            player_idx=player_idx, declarer_idx=declarer_idx,
            partner_idx=partner_idx, points=points, trick_number=trick_number
        )
        
        if self.model is None:
            return rb_choice
        
        # Score all legal actions
        state_feat = self._state_features(hand, legal_actions, current_trick, 
                                          trick_number, player_idx, declarer_idx, partner_idx)
        
        scores = []
        with torch.no_grad():
            state = torch.tensor([state_feat], dtype=torch.float32).to(self.device)
            for card in legal_actions:
                card_feat = self._card_features(card, legal_actions, current_trick)
                card_t = torch.tensor([card_feat], dtype=torch.float32).to(self.device)
                score = self.model(state, card_t).item()
                scores.append(score)
        
        # Find best and RuleBased scores
        best_idx = int(np.argmax(scores))
        best_card = legal_actions[best_idx]
        best_score = scores[best_idx]
        
        rb_idx = legal_actions.index(rb_choice)
        rb_score = scores[rb_idx]
        
        # Override if model is confident best is better than RB choice
        if best_card != rb_choice and (best_score - rb_score) > self.threshold:
            self.stats['overrides'] += 1
            return best_card
        
        return rb_choice
    
    def _state_features(self, hand, legal, trick_cards, trick_num, player_idx, declarer_idx, partner_idx):
        declarer_team = {declarer_idx}
        if partner_idx is not None:
            declarer_team.add(partner_idx)
        is_declarer = float(player_idx in declarer_team)
        
        return [
            trick_num / 7.0,
            is_declarer,
            len(trick_cards) / 3.0,
            len(legal) / 8.0,
            sum(1 for c in hand if is_trump(c)) / 8.0,
            sum(card_points(c) for c in hand) / 120.0,
            sum(card_points(c) for _, c in trick_cards) / 40.0,
            float(any(is_trump(c) for _, c in trick_cards)),
            float(is_trump(trick_cards[0][1])) if trick_cards else 0.0,
        ]
    
    def _card_features(self, card, legal, trick_cards):
        f = []
        f.append(float(is_trump(card)))
        f.append(card_strength(card))
        f.append(card_points(card) / 11.0)
        
        if trick_cards:
            led_suit = get_suit(trick_cards[0][1])
            f.append(float(get_suit(card) == led_suit))
            
            can_win = True
            for _, tc in trick_cards:
                if is_trump(tc):
                    if not is_trump(card):
                        can_win = False
                    elif card_strength(tc) > card_strength(card):
                        can_win = False
                elif is_trump(card):
                    pass
                elif get_suit(tc) == get_suit(card):
                    if card_strength(tc) > card_strength(card):
                        can_win = False
            f.append(float(can_win))
        else:
            f.append(0.0)
            f.append(1.0)
        
        my_suit_cards = [c for c in legal if get_suit(c) == get_suit(card)]
        if len(my_suit_cards) > 1:
            ranks = sorted([card_strength(c) for c in my_suit_cards], reverse=True)
            f.append(ranks.index(card_strength(card)) / len(ranks))
        else:
            f.append(0.0)
        
        return f


def evaluate(num_games=200, threshold=0.2):
    from src.schafkopf_ai.env import make_env
    
    print(f"\n{'='*60}")
    print(f"EVALUATING RANKING HYBRID (threshold={threshold})")
    print("=" * 60)
    
    env = make_env()
    hybrid = RankingHybridAgent(threshold)
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
            override_rate = hybrid.stats['overrides'] / max(1, hybrid.stats['total']) * 100
            print(f"  Games {game+1}/{num_games}: Win={wins/(game+1)*100:.1f}%, Override={override_rate:.1f}%")
    
    override_rate = hybrid.stats['overrides'] / max(1, hybrid.stats['total']) * 100
    print(f"\nFinal: {wins}/{num_games} = {wins/num_games*100:.1f}% win rate")
    print(f"Overrides: {hybrid.stats['overrides']}/{hybrid.stats['total']} ({override_rate:.1f}%)")
    return wins / num_games


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    
    if args.train:
        train_ranking(args.epochs)
    
    if args.eval or not args.train:
        evaluate(args.games, args.threshold)
