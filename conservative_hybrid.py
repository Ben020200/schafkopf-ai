"""
Conservative Hybrid: RuleBased + High-Confidence MC Corrections Only.

Key insight: RuleBased is already ~75% correct. We only want to correct
the ~25% of mistakes, and only when we're VERY confident.

Approach:
1. Use RuleBased by default
2. Only override when:
   - MC data shows improvement > 0.15 (significant)
   - The situation is common (seen multiple times in training)
3. Use a simple learned model to estimate "should we override RuleBased?"

This is a GATING model, not a playing model.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

from src.schafkopf_ai.baseline import RuleBasedAgent
from src.schafkopf_ai.env import CARD_TO_IDX, IDX_TO_CARD


TRUMP_CARDS = {
    "O_Acorn", "O_Leaf", "O_Heart", "O_Bell",
    "U_Acorn", "U_Leaf", "U_Heart", "U_Bell",
    "A_Heart", "10_Heart", "K_Heart", "9_Heart", "8_Heart", "7_Heart"
}

def is_trump(card: str) -> bool:
    return card in TRUMP_CARDS

def card_points(card: str) -> int:
    rank = card.split("_")[0]
    return {"A": 11, "10": 10, "K": 4, "O": 3, "U": 2}.get(rank, 0)


class GatingDataset(Dataset):
    """
    Dataset for training a gating model:
    - Input: game features + RuleBased choice features
    - Output: 0 if RuleBased is correct, 1 if RuleBased should be overridden
    - Also stores the best alternative when RuleBased is wrong
    """
    
    def __init__(self, csv_path: str, min_severity: float = 0.10):
        df = pd.read_csv(csv_path)
        self.samples = []
        
        mistakes = 0
        correct = 0
        
        for _, row in df.iterrows():
            is_mistake = row['is_mistake'] if pd.notna(row['is_mistake']) else False
            severity = float(row['mistake_severity']) if pd.notna(row['mistake_severity']) else 0.0
            
            # Only significant mistakes count
            significant_mistake = is_mistake and severity >= min_severity
            
            sample = self._process_row(row, significant_mistake, df.columns)
            if sample is not None:
                self.samples.append(sample)
                if significant_mistake:
                    mistakes += 1
                else:
                    correct += 1
        
        print(f"  Dataset: {len(self.samples)} samples ({mistakes} mistakes, {correct} correct)")
        
        # Balance the dataset
        self._balance()
    
    def _balance(self):
        """Downsample majority class for balance."""
        mistakes = [s for s in self.samples if s['should_override']]
        correct = [s for s in self.samples if not s['should_override']]
        
        min_count = min(len(mistakes), len(correct))
        if min_count > 0:
            np.random.seed(42)
            mistakes = list(np.random.choice(mistakes, min_count, replace=False)) if len(mistakes) > min_count else mistakes
            correct = list(np.random.choice(correct, min_count, replace=False)) if len(correct) > min_count else correct
            self.samples = mistakes + correct
            np.random.shuffle(self.samples)
            print(f"  Balanced to {len(self.samples)} samples")
    
    def _process_row(self, row, should_override: bool, columns) -> Optional[Dict]:
        hand = str(row['hand']).split('|') if pd.notna(row['hand']) else []
        legal = str(row['legal_actions']).split('|') if pd.notna(row['legal_actions']) else []
        
        if len(legal) <= 1:  # No decision to make
            return None
        
        rb_choice = row['rulebased_choice']
        best_action = row['best_action']
        
        if pd.isna(rb_choice) or pd.isna(best_action):
            return None
        
        # Parse trick
        trick_str = row['current_trick']
        trick_cards = []
        if pd.notna(trick_str) and trick_str:
            for item in str(trick_str).split('|'):
                if ':' in item:
                    pos, card = item.split(':')
                    trick_cards.append((int(pos), card))
        
        # Build features
        features = self._build_features(hand, legal, trick_cards, row, rb_choice)
        
        # Find index of best action among legal
        best_idx = legal.index(best_action) if best_action in legal else 0
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'should_override': should_override,
            'best_idx': best_idx,
            'num_legal': len(legal),
        }
    
    def _build_features(self, hand, legal, trick_cards, row, rb_choice) -> List[float]:
        """Features describing the situation and RuleBased's choice."""
        features = []
        
        # Game context
        trick_num = float(row['trick_number']) / 7.0
        is_declarer = float(row['is_declarer'])
        pos = len(trick_cards) / 3.0
        num_options = len(legal) / 8.0
        
        features.extend([trick_num, is_declarer, pos, num_options])
        
        # Hand stats
        trump_count = sum(1 for c in hand if is_trump(c)) / 8.0
        points_in_hand = sum(card_points(c) for c in hand) / 120.0
        
        features.extend([trump_count, points_in_hand])
        
        # RuleBased choice properties
        rb_is_trump = float(is_trump(rb_choice))
        rb_points = card_points(rb_choice) / 11.0
        
        features.extend([rb_is_trump, rb_points])
        
        # Trick context
        trick_points = sum(card_points(c) for _, c in trick_cards) / 40.0
        trick_has_trump = float(any(is_trump(c) for _, c in trick_cards))
        
        features.extend([trick_points, trick_has_trump])
        
        return features  # 10 features
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'features': s['features'],
            'target': torch.tensor(1.0 if s['should_override'] else 0.0),
            'best_idx': s['best_idx'],
            'num_legal': s['num_legal'],
        }


class GatingModel(nn.Module):
    """Simple binary classifier: should we override RuleBased?"""
    
    def __init__(self, input_size: int = 10, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class ActionPredictor(nn.Module):
    """When overriding, which action to take?"""
    
    def __init__(self, input_size: int = 10, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 8),  # Max 8 legal actions
        )
    
    def forward(self, x):
        return self.net(x)


def train_gating_model(epochs: int = 50):
    """Train the gating model."""
    print("=" * 60)
    print("TRAINING CONSERVATIVE HYBRID (GATING MODEL)")
    print("=" * 60)
    
    ds = GatingDataset("data/mc_training_data.csv", min_severity=0.10)
    
    # Split
    n = len(ds)
    train_size = int(0.8 * n)
    train_ds = torch.utils.data.Subset(ds, range(train_size))
    val_ds = torch.utils.data.Subset(ds, range(train_size, n))
    
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GatingModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_acc = 0.0
    best_state = None
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0
        
        for batch in train_dl:
            features = batch['features'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            pred = model(features)
            loss = F.binary_cross_entropy(pred, target)
            loss.backward()
            optimizer.step()
            
            pred_class = (pred > 0.5).float()
            train_correct += (pred_class == target).sum().item()
            train_total += len(target)
        
        model.eval()
        val_correct = 0
        val_total = 0
        val_tp, val_fp, val_fn = 0, 0, 0
        
        with torch.no_grad():
            for batch in val_dl:
                features = batch['features'].to(device)
                target = batch['target'].to(device)
                
                pred = model(features)
                pred_class = (pred > 0.5).float()
                val_correct += (pred_class == target).sum().item()
                val_total += len(target)
                
                val_tp += ((pred_class == 1) & (target == 1)).sum().item()
                val_fp += ((pred_class == 1) & (target == 0)).sum().item()
                val_fn += ((pred_class == 0) & (target == 1)).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train={train_acc*100:.1f}%, Val={val_acc*100:.1f}%, P={precision*100:.1f}%, R={recall*100:.1f}%")
    
    torch.save({'model': best_state, 'val_acc': best_val_acc}, "checkpoints/gating_model.pt")
    print(f"\nBest validation accuracy: {best_val_acc*100:.1f}%")
    
    return model


class ConservativeHybridAgent:
    """
    Uses RuleBased by default, only overrides when gating model is confident.
    """
    
    def __init__(self, gating_threshold: float = 0.7):
        self.rulebased = RuleBasedAgent()
        self.threshold = gating_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load gating model
        try:
            checkpoint = torch.load("checkpoints/gating_model.pt", map_location=self.device, weights_only=False)
            self.gating = GatingModel().to(self.device)
            self.gating.load_state_dict(checkpoint['model'])
            self.gating.eval()
            print(f"Loaded gating model (val_acc={checkpoint['val_acc']*100:.1f}%)")
        except:
            self.gating = None
            print("No gating model found, using pure RuleBased")
        
        # Load MC data for corrections
        self._load_corrections()
        
        self.stats = {"total": 0, "overrides": 0}
    
    def _load_corrections(self):
        """Load MC data to know what to correct to."""
        self.corrections = {}
        try:
            df = pd.read_csv("data/mc_training_data.csv")
            for _, row in df.iterrows():
                if row['is_mistake'] and row['mistake_severity'] >= 0.10:
                    hand = str(row['hand'])
                    best = row['best_action']
                    self.corrections[hand] = best
            print(f"  Loaded {len(self.corrections)} correction mappings")
        except:
            pass
    
    def select_action(self, hand: List[str], legal_actions: List[str],
                      current_trick: List[Tuple[int, str]],
                      played_cards: List[str],
                      player_idx: int, declarer_idx: int,
                      partner_idx: Optional[int],
                      points: List[int], trick_number: int) -> str:
        
        self.stats["total"] += 1
        
        # Get RuleBased choice
        rb_choice = self.rulebased.select_action(
            hand=hand, legal_actions=legal_actions,
            current_trick=current_trick, played_cards=played_cards,
            player_idx=player_idx, declarer_idx=declarer_idx,
            partner_idx=partner_idx, points=points, trick_number=trick_number
        )
        
        if len(legal_actions) <= 1 or self.gating is None:
            return rb_choice
        
        # Build features
        features = self._build_features(hand, legal_actions, current_trick,
                                         trick_number, player_idx, declarer_idx,
                                         partner_idx, rb_choice)
        
        # Ask gating model
        with torch.no_grad():
            feat_t = torch.tensor([features], dtype=torch.float32).to(self.device)
            override_prob = self.gating(feat_t).item()
        
        # Only override if very confident
        if override_prob > self.threshold:
            # Look up correction
            hand_key = "|".join(sorted(hand))
            if hand_key in self.corrections:
                correction = self.corrections[hand_key]
                if correction in legal_actions:
                    self.stats["overrides"] += 1
                    return correction
        
        return rb_choice
    
    def _build_features(self, hand, legal, trick_cards, trick_num, 
                        player_idx, declarer_idx, partner_idx, rb_choice) -> List[float]:
        declarer_team = {declarer_idx}
        if partner_idx is not None:
            declarer_team.add(partner_idx)
        is_declarer = float(player_idx in declarer_team)
        
        features = [
            trick_num / 7.0,
            is_declarer,
            len(trick_cards) / 3.0,
            len(legal) / 8.0,
            sum(1 for c in hand if is_trump(c)) / 8.0,
            sum(card_points(c) for c in hand) / 120.0,
            float(is_trump(rb_choice)),
            card_points(rb_choice) / 11.0,
            sum(card_points(c) for _, c in trick_cards) / 40.0,
            float(any(is_trump(c) for _, c in trick_cards)),
        ]
        return features
    
    def get_stats(self):
        return self.stats


def evaluate(num_games: int = 200, threshold: float = 0.7):
    """Evaluate conservative hybrid."""
    from src.schafkopf_ai.env import make_env
    
    print("\n" + "=" * 60)
    print(f"EVALUATING CONSERVATIVE HYBRID (threshold={threshold})")
    print("=" * 60)
    
    env = make_env()
    hybrid = ConservativeHybridAgent(gating_threshold=threshold)
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
    print(f"Override rate: {hybrid.stats['overrides']}/{hybrid.stats['total']} = {hybrid.stats['overrides']/max(1,hybrid.stats['total'])*100:.1f}%")
    
    return wins / num_games


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()
    
    if args.train:
        train_gating_model(args.epochs)
    
    if args.eval or not args.train:
        evaluate(args.games, args.threshold)
