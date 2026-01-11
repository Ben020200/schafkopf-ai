"""
Feature-Based Hybrid Agent: Uses RuleBased + learned Q-value adjustments.

Instead of exact matching, we:
1. Extract features from the game state
2. Train a simple model to predict Q-value adjustments for each legal card
3. Pick the card with highest adjusted Q-value

This combines the strengths of both approaches:
- RuleBased provides strong baseline heuristics
- MC data provides Q-value corrections
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
from src.schafkopf_ai.env import CARD_TO_IDX, IDX_TO_CARD, FULL_DECK


# Trump cards
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

def card_strength(card: str) -> float:
    """Relative strength of a card (0-1)."""
    TRUMP_ORDER = [
        "O_Acorn", "O_Leaf", "O_Heart", "O_Bell",
        "U_Acorn", "U_Leaf", "U_Heart", "U_Bell",
        "A_Heart", "10_Heart", "K_Heart", "9_Heart", "8_Heart", "7_Heart"
    ]
    if card in TRUMP_ORDER:
        return 1.0 - (TRUMP_ORDER.index(card) / 14.0)
    rank = card.split("_")[0]
    return {"A": 0.4, "10": 0.35, "K": 0.3, "9": 0.15, "8": 0.1, "7": 0.05}.get(rank, 0.0)


class FeatureDataset(Dataset):
    """Dataset with game features + Q-values for each legal card."""
    
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.samples = []
        
        for _, row in df.iterrows():
            sample = self._process_row(row, df.columns)
            if sample is not None:
                self.samples.append(sample)
        
        print(f"  Loaded {len(self.samples)} samples")
    
    def _process_row(self, row, columns) -> Optional[Dict]:
        hand = str(row['hand']).split('|') if pd.notna(row['hand']) else []
        legal = str(row['legal_actions']).split('|') if pd.notna(row['legal_actions']) else []
        
        if not hand or not legal:
            return None
        
        # Get Q-values for each legal action
        q_values = []
        for card in legal:
            col = f"value_{card}"
            if col in columns and pd.notna(row[col]):
                q_values.append(float(row[col]))
            else:
                q_values.append(0.5)
        
        if len(q_values) == 0:
            return None
        
        best_idx = int(np.argmax(q_values))
        
        # Parse trick
        trick_str = row['current_trick']
        trick_cards = []
        if pd.notna(trick_str) and trick_str:
            for item in str(trick_str).split('|'):
                if ':' in item:
                    pos, card = item.split(':')
                    trick_cards.append((int(pos), card))
        
        # Build features
        features = self._build_features(hand, legal, trick_cards, row)
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'q_values': torch.tensor(q_values, dtype=torch.float32),
            'best_idx': best_idx,
            'num_legal': len(legal),
        }
    
    def _build_features(self, hand, legal, trick_cards, row) -> List[float]:
        """Build feature vector for the state + each legal action."""
        features = []
        
        # Global features
        trick_num = float(row['trick_number']) / 7.0
        is_declarer = float(row['is_declarer'])
        pos_in_trick = len(trick_cards) / 3.0
        
        features.extend([trick_num, is_declarer, pos_in_trick])
        
        # Hand stats
        trump_count = sum(1 for c in hand if is_trump(c)) / 8.0
        high_cards = sum(1 for c in hand if card_points(c) >= 10) / 8.0
        total_points = sum(card_points(c) for c in hand) / 120.0
        
        features.extend([trump_count, high_cards, total_points])
        
        # Trick stats
        trick_points = sum(card_points(c) for _, c in trick_cards) / 40.0
        trick_has_trump = float(any(is_trump(c) for _, c in trick_cards))
        
        features.extend([trick_points, trick_has_trump])
        
        # For each legal action (up to 8)
        for i, card in enumerate(legal[:8]):
            card_feat = self._card_features(card, hand, trick_cards)
            features.extend(card_feat)
        
        # Pad to 8 cards
        for _ in range(8 - len(legal)):
            features.extend([0.0] * 5)  # 5 features per card
        
        return features  # 8 global + 8*5 card = 48 features
    
    def _card_features(self, card: str, hand: List[str], trick_cards) -> List[float]:
        """Features for a single card."""
        is_trump_card = float(is_trump(card))
        strength = card_strength(card)
        points = card_points(card) / 11.0
        
        # Would this card likely win the trick?
        if trick_cards:
            trick_trumps = [c for _, c in trick_cards if is_trump(c)]
            if is_trump(card):
                if trick_trumps:
                    would_win = float(card_strength(card) > max(card_strength(c) for c in trick_trumps))
                else:
                    would_win = 1.0
            else:
                would_win = 0.0 if trick_trumps else 0.5
        else:
            would_win = 0.5  # Leading
        
        # Is this highest trump in hand?
        hand_trumps = [c for c in hand if is_trump(c)]
        is_highest = 0.0
        if is_trump(card) and hand_trumps:
            is_highest = float(card_strength(card) >= max(card_strength(c) for c in hand_trumps))
        
        return [is_trump_card, strength, points, would_win, is_highest]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Custom collate for variable number of legal actions."""
    features = torch.stack([s['features'] for s in batch])
    
    # Pad Q-values to 8
    max_legal = 8
    q_padded = torch.zeros(len(batch), max_legal)
    masks = torch.zeros(len(batch), max_legal)
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


class QPredictor(nn.Module):
    """Predicts Q-value for each of 8 possible actions."""
    
    def __init__(self, input_size: int = 48, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 8),  # Q-value for each of 8 possible actions
        )
    
    def forward(self, x):
        return self.net(x)


def train_model(epochs: int = 100, save_path: str = "checkpoints/feature_hybrid.pt"):
    """Train the Q-predictor model."""
    print("=" * 60)
    print("TRAINING FEATURE-BASED HYBRID")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    ds = FeatureDataset("data/mc_training_data.csv")
    
    # Split
    n = len(ds)
    train_size = int(0.8 * n)
    train_ds = torch.utils.data.Subset(ds, range(train_size))
    val_ds = torch.utils.data.Subset(ds, range(train_size, n))
    
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_acc = 0.0
    best_state = None
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_dl:
            features = batch['features'].to(device)
            target_q = batch['q_values'].to(device)
            mask = batch['mask'].to(device)
            best_idx = batch['best_idx'].to(device)
            
            optimizer.zero_grad()
            pred_q = model(features)
            
            # MSE loss on Q-values
            loss = F.mse_loss(pred_q * mask, target_q * mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy: pick best
            pred_q_masked = pred_q + (mask - 1) * 1e9
            pred_best = pred_q_masked.argmax(dim=1)
            train_correct += (pred_best == best_idx).sum().item()
            train_total += len(best_idx)
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_dl:
                features = batch['features'].to(device)
                mask = batch['mask'].to(device)
                best_idx = batch['best_idx'].to(device)
                
                pred_q = model(features)
                pred_q_masked = pred_q + (mask - 1) * 1e9
                pred_best = pred_q_masked.argmax(dim=1)
                val_correct += (pred_best == best_idx).sum().item()
                val_total += len(best_idx)
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train={train_acc*100:.1f}%, Val={val_acc*100:.1f}%")
    
    # Save best
    model.load_state_dict(best_state)
    torch.save({'model': best_state, 'val_acc': best_val_acc}, save_path)
    print(f"\nBest validation accuracy: {best_val_acc*100:.1f}%")
    print(f"Saved to {save_path}")
    
    return model


class FeatureHybridAgent:
    """Agent that uses learned Q-predictions + RuleBased fallback."""
    
    def __init__(self, model_path: str = "checkpoints/feature_hybrid.pt"):
        self.rulebased = RuleBasedAgent()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = QPredictor().to(self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        
        print(f"Loaded FeatureHybrid (val_acc={checkpoint['val_acc']*100:.1f}%)")
    
    def select_action(self, hand: List[str], legal_actions: List[str],
                      current_trick: List[Tuple[int, str]],
                      played_cards: List[str],
                      player_idx: int, declarer_idx: int,
                      partner_idx: Optional[int],
                      points: List[int], trick_number: int) -> str:
        """Select action using learned Q-values."""
        
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        # Build features
        features = self._build_features(hand, legal_actions, current_trick, trick_number,
                                        player_idx, declarer_idx, partner_idx)
        
        # Predict Q-values
        with torch.no_grad():
            feat_t = torch.tensor([features], dtype=torch.float32).to(self.device)
            q_pred = self.model(feat_t)[0]
        
        # Mask and pick best
        n = len(legal_actions)
        q_pred[n:] = float('-inf')
        best_idx = q_pred.argmax().item()
        
        if best_idx < n:
            return legal_actions[best_idx]
        else:
            # Fallback to RuleBased
            return self.rulebased.select_action(
                hand=hand, legal_actions=legal_actions,
                current_trick=current_trick, played_cards=played_cards,
                player_idx=player_idx, declarer_idx=declarer_idx,
                partner_idx=partner_idx, points=points, trick_number=trick_number
            )
    
    def _build_features(self, hand, legal, trick_cards, trick_num, 
                        player_idx, declarer_idx, partner_idx) -> List[float]:
        """Build feature vector."""
        features = []
        
        # Global features
        declarer_team = {declarer_idx}
        if partner_idx is not None:
            declarer_team.add(partner_idx)
        is_declarer = float(player_idx in declarer_team)
        
        features.extend([trick_num / 7.0, is_declarer, len(trick_cards) / 3.0])
        
        # Hand stats
        trump_count = sum(1 for c in hand if is_trump(c)) / 8.0
        high_cards = sum(1 for c in hand if card_points(c) >= 10) / 8.0
        total_points = sum(card_points(c) for c in hand) / 120.0
        features.extend([trump_count, high_cards, total_points])
        
        # Trick stats
        trick_points = sum(card_points(c) for _, c in trick_cards) / 40.0
        trick_has_trump = float(any(is_trump(c) for _, c in trick_cards))
        features.extend([trick_points, trick_has_trump])
        
        # Card features
        for i, card in enumerate(legal[:8]):
            is_trump_card = float(is_trump(card))
            strength = card_strength(card)
            pts = card_points(card) / 11.0
            
            # Would win?
            if trick_cards:
                trick_trumps = [c for _, c in trick_cards if is_trump(c)]
                if is_trump(card):
                    would_win = 1.0 if not trick_trumps else float(
                        card_strength(card) > max(card_strength(c) for c in trick_trumps))
                else:
                    would_win = 0.0 if trick_trumps else 0.5
            else:
                would_win = 0.5
            
            # Highest trump?
            hand_trumps = [c for c in hand if is_trump(c)]
            is_highest = 0.0
            if is_trump(card) and hand_trumps:
                is_highest = float(card_strength(card) >= max(card_strength(c) for c in hand_trumps))
            
            features.extend([is_trump_card, strength, pts, would_win, is_highest])
        
        # Pad
        for _ in range(8 - len(legal)):
            features.extend([0.0] * 5)
        
        return features


def evaluate(num_games: int = 200):
    """Evaluate FeatureHybrid vs RuleBased."""
    from src.schafkopf_ai.env import make_env
    
    print("\n" + "=" * 60)
    print("EVALUATING FEATURE HYBRID VS RULEBASED")
    print("=" * 60)
    
    env = make_env()
    hybrid = FeatureHybridAgent()
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
    args = parser.parse_args()
    
    if args.train:
        train_model(epochs=args.epochs)
    
    if args.eval or not args.train:
        evaluate(args.games)
