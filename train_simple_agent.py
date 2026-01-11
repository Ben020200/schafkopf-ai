#!/usr/bin/env python3
"""
Simple MC Agent Training - Only considers legal actions (max 8 cards).

Key insight: Don't predict among 32 cards. Only predict among legal actions.
This is a RANKING problem: which of the 1-8 legal cards is best?
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import os

# Card mappings
FULL_DECK = [
    f"{rank}_{suit}" 
    for suit in ["Acorn", "Leaf", "Heart", "Bell"]
    for rank in ["A", "10", "K", "O", "U", "9", "8", "7"]
]
CARD_TO_IDX = {card: idx for idx, card in enumerate(FULL_DECK)}
IDX_TO_CARD = {idx: card for card, idx in CARD_TO_IDX.items()}

# Trump cards in Schafkopf
TRUMP_CARDS = {
    "O_Acorn", "O_Leaf", "O_Heart", "O_Bell",  # All Obers
    "U_Acorn", "U_Leaf", "U_Heart", "U_Bell",  # All Unters
    "A_Heart", "10_Heart", "K_Heart", "9_Heart", "8_Heart", "7_Heart"  # All Hearts
}


def is_trump(card: str) -> bool:
    return card in TRUMP_CARDS


def card_strength(card: str) -> float:
    """Return normalized strength of a card (0-1)."""
    # Trump strength order
    TRUMP_ORDER = [
        "O_Acorn", "O_Leaf", "O_Heart", "O_Bell",
        "U_Acorn", "U_Leaf", "U_Heart", "U_Bell",
        "A_Heart", "10_Heart", "K_Heart", "9_Heart", "8_Heart", "7_Heart"
    ]
    if card in TRUMP_ORDER:
        return 1.0 - (TRUMP_ORDER.index(card) / len(TRUMP_ORDER))
    
    # Non-trump: A > 10 > K > 9 > 8 > 7
    rank = card.split("_")[0]
    rank_values = {"A": 0.5, "10": 0.4, "K": 0.3, "9": 0.2, "8": 0.1, "7": 0.0}
    return rank_values.get(rank, 0.0)


def card_points(card: str) -> int:
    """Point value of a card."""
    rank = card.split("_")[0]
    points = {"A": 11, "10": 10, "K": 4, "O": 3, "U": 2}
    return points.get(rank, 0)


class SimpleDataset(Dataset):
    """
    Dataset that represents each decision as:
    - State features (fixed size)
    - List of legal action features (variable, max 8)
    - Q-values for each legal action
    - Index of best action
    """
    
    def __init__(self, df: pd.DataFrame):
        self.samples = []
        self._preprocess(df)
        print(f"  Loaded {len(self.samples)} samples")
    
    def _preprocess(self, df: pd.DataFrame):
        for idx in range(len(df)):
            row = df.iloc[idx]
            sample = self._process_row(row)
            if sample is not None:
                self.samples.append(sample)
    
    def _process_row(self, row) -> dict:
        # Parse legal actions
        legal_str = row['legal_actions']
        if pd.isna(legal_str) or not legal_str:
            return None
        legal_cards = str(legal_str).split('|')
        if len(legal_cards) == 0:
            return None
        
        # Get Q-values for legal actions
        q_values = []
        for card in legal_cards:
            col = f'value_{card}'
            if col in row.index and pd.notna(row[col]):
                q_values.append(float(row[col]))
            else:
                q_values.append(0.5)  # Default
        
        if len(q_values) == 0:
            return None
        
        # Find best action index (among legal actions only!)
        best_idx = int(np.argmax(q_values))
        
        # Parse hand
        hand_str = row['hand']
        hand_cards = str(hand_str).split('|') if pd.notna(hand_str) else []
        
        # Parse current trick
        trick_str = row['current_trick']
        trick_cards = []
        if pd.notna(trick_str) and trick_str:
            for item in str(trick_str).split('|'):
                if ':' in item:
                    pos, card = item.split(':')
                    trick_cards.append((int(pos), card))
        
        # Parse played cards
        played_str = row['played_cards']
        played_cards = str(played_str).split('|') if pd.notna(played_str) and played_str else []
        
        # Build state features
        state = self._build_state_features(
            hand_cards, trick_cards, played_cards,
            row['trick_number'], row['is_declarer']
        )
        
        # Build action features for each legal action
        action_features = []
        for card in legal_cards:
            feat = self._build_action_features(card, hand_cards, trick_cards)
            action_features.append(feat)
        
        return {
            'state': torch.tensor(state, dtype=torch.float32),
            'action_features': torch.tensor(action_features, dtype=torch.float32),
            'q_values': torch.tensor(q_values, dtype=torch.float32),
            'best_idx': best_idx,
            'num_actions': len(legal_cards),
        }
    
    def _build_state_features(self, hand, trick_cards, played_cards, trick_num, is_declarer) -> List[float]:
        """Build fixed-size state representation."""
        features = []
        
        # Trick number (0-7) normalized
        features.append(trick_num / 7.0)
        
        # Is declarer
        features.append(1.0 if is_declarer else 0.0)
        
        # Position in trick (0-3)
        pos_in_trick = len(trick_cards)
        for i in range(4):
            features.append(1.0 if i == pos_in_trick else 0.0)
        
        # Hand stats
        features.append(len(hand) / 8.0)  # Cards remaining
        trump_count = sum(1 for c in hand if is_trump(c))
        features.append(trump_count / 8.0)
        
        # Points in hand
        hand_points = sum(card_points(c) for c in hand)
        features.append(hand_points / 120.0)
        
        # Current trick stats
        trick_points = sum(card_points(c) for _, c in trick_cards)
        features.append(trick_points / 40.0)  # Max ~40 points in one trick
        
        # Is trick led by trump?
        trick_has_trump = any(is_trump(c) for _, c in trick_cards)
        features.append(1.0 if trick_has_trump else 0.0)
        
        # Highest card in trick (strength)
        if trick_cards:
            max_strength = max(card_strength(c) for _, c in trick_cards)
            features.append(max_strength)
        else:
            features.append(0.0)
        
        # Game progress
        features.append(len(played_cards) / 32.0)
        
        return features  # 13 features
    
    def _build_action_features(self, card: str, hand: List[str], trick_cards: List[Tuple[int, str]]) -> List[float]:
        """Build features for a single action/card."""
        features = []
        
        # Card properties
        features.append(1.0 if is_trump(card) else 0.0)
        features.append(card_strength(card))
        features.append(card_points(card) / 11.0)  # Normalize by max (Ace = 11)
        
        # Is this the highest trump in hand?
        hand_trumps = [c for c in hand if is_trump(c)]
        if hand_trumps and is_trump(card):
            best_trump = max(hand_trumps, key=card_strength)
            features.append(1.0 if card == best_trump else 0.0)
        else:
            features.append(0.0)
        
        # Would this card win the current trick?
        if trick_cards:
            # Simplified: trump beats non-trump, higher strength wins
            trick_trumps = [c for _, c in trick_cards if is_trump(c)]
            if is_trump(card):
                if trick_trumps:
                    max_trick_trump = max(trick_trumps, key=card_strength)
                    would_win = card_strength(card) > card_strength(max_trick_trump)
                else:
                    would_win = True  # Trump beats non-trump
            else:
                would_win = not trick_trumps and all(
                    card_strength(card) > card_strength(c) for _, c in trick_cards
                )
            features.append(1.0 if would_win else 0.0)
        else:
            features.append(0.5)  # Leading, unclear
        
        return features  # 5 features
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Custom collate to handle variable number of actions."""
    # Pad action features and q_values to max 8
    max_actions = 8
    
    states = torch.stack([s['state'] for s in batch])
    
    # Pad action features
    action_feat_size = batch[0]['action_features'].shape[1]
    padded_actions = torch.zeros(len(batch), max_actions, action_feat_size)
    padded_q = torch.zeros(len(batch), max_actions)
    masks = torch.zeros(len(batch), max_actions)
    best_indices = []
    
    for i, s in enumerate(batch):
        n = s['num_actions']
        padded_actions[i, :n] = s['action_features']
        padded_q[i, :n] = s['q_values']
        masks[i, :n] = 1.0
        best_indices.append(s['best_idx'])
    
    return {
        'state': states,
        'action_features': padded_actions,
        'q_values': padded_q,
        'mask': masks,
        'best_idx': torch.tensor(best_indices, dtype=torch.long),
    }


class SimpleQNetwork(nn.Module):
    """
    Network that predicts Q-value for each legal action.
    
    Architecture:
    1. Encode state -> state embedding
    2. Encode each action -> action embedding  
    3. Combine: Q(s,a) = f(state_embed, action_embed)
    """
    
    def __init__(self, state_size: int = 13, action_size: int = 5, hidden: int = 64):
        super().__init__()
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_size, hidden),
            nn.ReLU(),
        )
        
        # Combined Q-value predictor
        self.q_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    
    def forward(self, state: torch.Tensor, action_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch, state_size]
            action_features: [batch, max_actions, action_size]
            mask: [batch, max_actions] - 1 for valid actions, 0 for padding
        
        Returns:
            q_values: [batch, max_actions]
        """
        batch_size, max_actions, action_size = action_features.shape
        
        # Encode state
        state_embed = self.state_encoder(state)  # [batch, hidden]
        state_embed = state_embed.unsqueeze(1).expand(-1, max_actions, -1)  # [batch, max_actions, hidden]
        
        # Encode actions
        action_embed = self.action_encoder(action_features.view(-1, action_size))  # [batch*max_actions, hidden]
        action_embed = action_embed.view(batch_size, max_actions, -1)  # [batch, max_actions, hidden]
        
        # Combine and predict Q
        combined = torch.cat([state_embed, action_embed], dim=-1)  # [batch, max_actions, hidden*2]
        q_values = self.q_head(combined.view(-1, combined.shape[-1]))  # [batch*max_actions, 1]
        q_values = q_values.view(batch_size, max_actions)  # [batch, max_actions]
        
        # Mask invalid actions
        q_values = q_values.masked_fill(mask == 0, float('-inf'))
        
        return q_values


def train(data_path: str = "data/mc_training_data.csv", epochs: int = 50, 
          output_path: str = "checkpoints/simple_agent.pt"):
    """Train the simple Q-network."""
    
    print("=" * 60)
    print("SIMPLE MC AGENT TRAINING")
    print("=" * 60)
    print("\nThis approach only considers legal actions (max 8 cards).")
    print("Much simpler than predicting among all 32 cards!\n")
    
    # Load data
    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Total rows: {len(df)}")
    
    # Split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(len(df) * 0.8)
    train_df, val_df = df.iloc[:split], df.iloc[split:]
    
    print(f"\nCreating datasets...")
    train_dataset = SimpleDataset(train_df)
    val_dataset = SimpleDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleQNetwork(state_size=13, action_size=5, hidden=64).to(device)
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Device: {device}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    best_val_acc = 0.0
    best_state = None
    
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0
        
        for batch in train_loader:
            state = batch['state'].to(device)
            actions = batch['action_features'].to(device)
            mask = batch['mask'].to(device)
            target_q = batch['q_values'].to(device)
            best_idx = batch['best_idx'].to(device)
            
            optimizer.zero_grad()
            
            pred_q = model(state, actions, mask)
            
            # Loss: MSE on Q-values for valid actions only
            valid_mask = mask.bool()
            loss = F.mse_loss(pred_q[valid_mask], target_q[valid_mask])
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy: did we pick the best action?
            pred_best = pred_q.argmax(dim=-1)
            train_correct += (pred_best == best_idx).sum().item()
            train_total += len(best_idx)
        
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_correct = 0
        val_near_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                state = batch['state'].to(device)
                actions = batch['action_features'].to(device)
                mask = batch['mask'].to(device)
                target_q = batch['q_values'].to(device)
                best_idx = batch['best_idx'].to(device)
                
                pred_q = model(state, actions, mask)
                pred_best = pred_q.argmax(dim=-1)
                
                val_correct += (pred_best == best_idx).sum().item()
                
                # Near-correct: within 0.05 of best Q-value
                pred_q_val = pred_q.gather(1, pred_best.unsqueeze(1)).squeeze()
                best_q_val = target_q.gather(1, best_idx.unsqueeze(1)).squeeze()
                val_near_correct += ((best_q_val - pred_q_val).abs() < 0.05).sum().item()
                
                val_total += len(best_idx)
        
        val_acc = val_correct / val_total
        val_near = val_near_correct / val_total
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train={train_acc*100:.1f}%, Val={val_acc*100:.1f}% (near={val_near*100:.1f}%)")
    
    # Save best model
    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'model_state': best_state,
        'val_acc': best_val_acc,
    }, output_path)
    
    print("-" * 60)
    print(f"\nBest Validation Accuracy: {best_val_acc*100:.1f}%")
    print(f"Model saved to: {output_path}")
    
    return model, best_val_acc


def evaluate(model_path: str = "checkpoints/simple_agent.pt", num_games: int = 200):
    """Evaluate trained agent vs RuleBased."""
    from src.schafkopf_ai.env import make_env, CARD_TO_IDX as ENV_CARD_IDX, IDX_TO_CARD as ENV_IDX_CARD
    from src.schafkopf_ai.baseline import RuleBasedAgent
    
    print(f"\nEvaluating {model_path} vs RuleBased...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = SimpleQNetwork(state_size=13, action_size=5, hidden=64).to(device)
    model.load_state_dict(checkpoint['model_state'])
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
            player_idx = int(player_name.split("_")[1])
            
            hand = env._hands[player_idx]
            legal_indices = env._get_legal_actions(player_idx)
            legal_cards = [ENV_IDX_CARD[idx] for idx in legal_indices]
            trick_cards = [(p, c) for p, c in env._current_trick]
            
            # Which team?
            declarer_team = {env._declarer_idx}
            if env._partner_idx is not None:
                declarer_team.add(env._partner_idx)
            is_declarer_team = player_idx in declarer_team
            use_mc = (mc_plays_declarer == is_declarer_team)
            
            if use_mc:
                # Use our model
                state_feat = build_state_features(
                    hand, trick_cards, env._played_cards,
                    env._trick_number, player_idx in declarer_team
                )
                action_feats = [build_action_features(c, hand, trick_cards) for c in legal_cards]
                
                state_t = torch.tensor([state_feat], dtype=torch.float32).to(device)
                action_t = torch.zeros(1, 8, 5)
                mask_t = torch.zeros(1, 8)
                for i, af in enumerate(action_feats):
                    action_t[0, i] = torch.tensor(af)
                    mask_t[0, i] = 1.0
                action_t = action_t.to(device)
                mask_t = mask_t.to(device)
                
                with torch.no_grad():
                    q_vals = model(state_t, action_t, mask_t)
                    best_idx = q_vals[0].argmax().item()
                
                card = legal_cards[best_idx]
            else:
                card = rb_agent.select_action(
                    hand=hand, legal_actions=legal_cards,
                    current_trick=trick_cards, played_cards=env._played_cards,
                    player_idx=player_idx, declarer_idx=env._declarer_idx,
                    partner_idx=env._partner_idx, points=env._points,
                    trick_number=env._trick_number
                )
            
            action = ENV_CARD_IDX[card]
            env.step(action)
        
        result = env.get_game_result()
        if mc_plays_declarer == result["win"]:
            wins += 1
        
        if (game + 1) % 50 == 0:
            print(f"  Games {game+1}/{num_games}: Win rate = {wins/(game+1)*100:.1f}%")
    
    win_rate = wins / num_games
    print(f"\nFinal: {wins}/{num_games} = {win_rate*100:.1f}% win rate")
    return win_rate


# Helper functions for evaluation (copy of dataset methods)
def build_state_features(hand, trick_cards, played_cards, trick_num, is_declarer):
    features = []
    features.append(trick_num / 7.0)
    features.append(1.0 if is_declarer else 0.0)
    pos_in_trick = len(trick_cards)
    for i in range(4):
        features.append(1.0 if i == pos_in_trick else 0.0)
    features.append(len(hand) / 8.0)
    trump_count = sum(1 for c in hand if is_trump(c))
    features.append(trump_count / 8.0)
    hand_points = sum(card_points(c) for c in hand)
    features.append(hand_points / 120.0)
    trick_points = sum(card_points(c) for _, c in trick_cards)
    features.append(trick_points / 40.0)
    trick_has_trump = any(is_trump(c) for _, c in trick_cards)
    features.append(1.0 if trick_has_trump else 0.0)
    if trick_cards:
        max_strength = max(card_strength(c) for _, c in trick_cards)
        features.append(max_strength)
    else:
        features.append(0.0)
    features.append(len(played_cards) / 32.0)
    return features


def build_action_features(card, hand, trick_cards):
    features = []
    features.append(1.0 if is_trump(card) else 0.0)
    features.append(card_strength(card))
    features.append(card_points(card) / 11.0)
    hand_trumps = [c for c in hand if is_trump(c)]
    if hand_trumps and is_trump(card):
        best_trump = max(hand_trumps, key=card_strength)
        features.append(1.0 if card == best_trump else 0.0)
    else:
        features.append(0.0)
    if trick_cards:
        trick_trumps = [c for _, c in trick_cards if is_trump(c)]
        if is_trump(card):
            if trick_trumps:
                max_trick_trump = max(trick_trumps, key=card_strength)
                would_win = card_strength(card) > card_strength(max_trick_trump)
            else:
                would_win = True
        else:
            would_win = not trick_trumps and all(
                card_strength(card) > card_strength(c) for _, c in trick_cards
            )
        features.append(1.0 if would_win else 0.0)
    else:
        features.append(0.5)
    return features


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data", type=str, default="data/mc_training_data.csv")
    parser.add_argument("--output", type=str, default="checkpoints/simple_agent.pt")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-games", type=int, default=200)
    args = parser.parse_args()
    
    if args.eval_only:
        evaluate(args.output, args.eval_games)
    else:
        train(args.data, args.epochs, args.output)
        evaluate(args.output, args.eval_games)
