"""
Smart Hybrid Agent: Combines RuleBased with learned corrections from MC analysis.

The key insight is that RuleBased makes mistakes in specific PATTERNS:
1. Early tricks (0-2): 35% mistake rate but low severity (5%)
2. More options = more chance of mistake
3. Declarer makes slightly more mistakes than defender

Instead of trying to learn exact states (which fails), we:
1. Learn a CORRECTION NETWORK that predicts when RuleBased is wrong
2. When confidence is high, use MC-learned action instead
3. Otherwise, trust RuleBased

This is essentially "learning to correct" rather than "learning to play".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import os

from .baseline import RuleBasedAgent
from .cards import FULL_DECK, is_trump, rank_of


class CorrectionNetwork(nn.Module):
    """
    Simplified binary classifier: Should we trust RuleBased or pick the MC-best action?
    
    This is much simpler than Q-value prediction:
    - Input: game state + which card RuleBased picked + which card MC says is best
    - Output: probability that MC-best is actually better
    
    At inference time, if confidence > threshold, use MC-best, else use RuleBased.
    """
    
    def __init__(self, input_size: int = 195 + 64, hidden_size: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        
        # Card embeddings (32 cards -> 32 dim embedding)
        self.card_embed = nn.Embedding(32, 32)
        
        # Main network
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state: torch.Tensor, rb_card: torch.Tensor, 
                best_card: torch.Tensor) -> torch.Tensor:
        """
        Returns probability that best_card is better than rb_card.
        
        Args:
            state: [batch, 195] game state features
            rb_card: [batch] index of RuleBased choice
            best_card: [batch] index of MC-best card
        """
        rb_emb = self.card_embed(rb_card)  # [batch, 32]
        best_emb = self.card_embed(best_card)  # [batch, 32]
        
        # Concatenate: state + rb_embed + best_embed
        x = torch.cat([state, rb_emb, best_emb], dim=1)  # [batch, 195+64]
        
        return self.network(x)


# Also keep original Q-value network for comparison
class QValueNetwork(nn.Module):
    """
    Predicts Q-values for each action given the game state.
    Uses the MC analysis to learn better action values than RuleBased.
    """
    
    def __init__(self, input_size: int = 195, hidden_size: int = 256, num_actions: int = 32,
                 dropout: float = 0.5):
        super().__init__()
        
        # Simpler network with heavy dropout to prevent overfitting
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size)
        
        # Simpler architecture - just 2 blocks with heavy regularization
        self.block1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Removed block3 - simpler is better to avoid overfitting
        
        # Output: Q-value for each action
        self.q_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        # Confidence head: how sure are we that RuleBased is wrong?
        self.confidence_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (q_values, confidence)"""
        h = self.input_bn(F.relu(self.input_proj(x)))
        
        # Simple feedforward with residual-like structure
        h = h + self.block1(h)
        h = h + self.block2(h)
        
        q_values = self.q_head(h)
        confidence = self.confidence_head(h)
        
        return q_values, confidence


def encode_game_state(hand: List[str], current_trick: List[Tuple[int, str]], 
                      played_cards: List[str], is_declarer: bool,
                      trick_number: int) -> np.ndarray:
    """
    Encode game state into feature vector matching the training data format.
    Cards are strings like 'A_Heart', 'O_Acorn', etc.
    """
    features = []
    
    # 32 features: cards in hand (one-hot)
    hand_set = set(hand)
    for card in FULL_DECK:
        features.append(1.0 if card in hand_set else 0.0)
    
    # 32 features: cards played in current trick
    trick_cards = set(c for _, c in current_trick)
    for card in FULL_DECK:
        features.append(1.0 if card in trick_cards else 0.0)
    
    # 32 features: all previously played cards
    played_set = set(played_cards)
    for card in FULL_DECK:
        features.append(1.0 if card in played_set else 0.0)
    
    # Position in trick (0-3)
    position_in_trick = len(current_trick)
    for i in range(4):
        features.append(1.0 if i == position_in_trick else 0.0)
    
    # Trick number (0-7) - one-hot
    for i in range(8):
        features.append(1.0 if i == trick_number else 0.0)
    
    # Is declarer
    features.append(1.0 if is_declarer else 0.0)
    
    # Additional context features
    features.append(len(hand) / 8.0)  # Fraction of cards remaining
    features.append(len(played_cards) / 32.0)  # Fraction of game completed
    
    # Trump count in hand (using is_trump function)
    trump_count = sum(1 for c in hand if is_trump(c))
    features.append(trump_count / 8.0)
    
    # High card count (Aces and 10s)
    high_count = sum(1 for c in hand if rank_of(c) in ['A', '10'])
    features.append(high_count / 8.0)
    
    # Points in current trick
    POINT_VALUES = {'A': 11, '10': 10, 'K': 4, 'O': 3, 'U': 2, '9': 0, '8': 0, '7': 0}
    trick_points = sum(POINT_VALUES.get(rank_of(c), 0) for _, c in current_trick)
    features.append(trick_points / 30.0)  # Normalize by max possible
    
    # Number of legal actions will be determined at action time
    # Pad to 195 features
    while len(features) < 195:
        features.append(0.0)
    
    return np.array(features[:195], dtype=np.float32)


class MCQDataset(torch.utils.data.Dataset):
    """
    Dataset for binary classification: Is RuleBased correct or wrong?
    
    Returns:
    - features: game state encoding
    - rb_card_idx: index of RuleBased choice
    - best_card_idx: index of MC-best choice  
    - is_mistake: 1 if RuleBased was wrong (label to predict)
    - severity: how wrong (for weighting)
    """
    
    def __init__(self, csv_path: str, min_severity: float = 0.05):
        self.df = pd.read_csv(csv_path)
        
        # Build card index mapping
        self.card_to_idx = {card: idx for idx, card in enumerate(FULL_DECK)}
        
        # Get Q-value columns for finding best action
        self.q_columns = [col for col in self.df.columns if col.startswith('value_')]
        
        # We want samples with CLEAR signal:
        # 1. Severe mistakes (severity >= threshold) - label=1 (RuleBased wrong)
        # 2. Correct choices (severity == 0) - label=0 (RuleBased correct)
        severe = self.df[self.df['mistake_severity'] >= min_severity]
        correct = self.df[self.df['mistake_severity'] == 0]
        
        print(f"Found {len(severe)} severe mistakes (>= {min_severity:.0%} severity)")
        print(f"Found {len(correct)} correct samples")
        
        # Balance the dataset
        n_each = min(len(severe), len(correct))
        severe = severe.sample(n=n_each, random_state=42)
        correct = correct.sample(n=n_each, random_state=42)
        
        filtered_df = pd.concat([severe, correct]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Using {len(filtered_df)} balanced samples ({n_each} of each class)")
        
        # Precompute
        self.features = []
        self.rb_card_idx = []
        self.best_card_idx = []
        self.is_mistake = []
        self.severities = []
        self.legal_masks = []
        
        for _, row in filtered_df.iterrows():
            # Parse game state
            hand = self._parse_cards(row['hand'])
            trick = self._parse_trick(row['current_trick'])
            played = self._parse_cards(row['played_cards'])
            is_declarer = row['is_declarer']
            trick_num = row['trick_number']
            
            # Encode features
            feat = encode_game_state(hand, trick, played, is_declarer, trick_num)
            self.features.append(feat)
            
            # Legal mask
            legal_mask = np.zeros(32, dtype=np.float32)
            for col in self.q_columns:
                if pd.notna(row[col]):
                    card_name = col.replace('value_', '')
                    if card_name in self.card_to_idx:
                        legal_mask[self.card_to_idx[card_name]] = 1.0
            self.legal_masks.append(legal_mask)
            
            # Best action from MC
            best_card = row['best_action'].replace(' of ', '_')
            best_idx = self.card_to_idx.get(best_card, 0)
            self.best_card_idx.append(best_idx)
            
            # RuleBased choice
            rb_card = row['rulebased_choice'].replace(' of ', '_')
            rb_idx = self.card_to_idx.get(rb_card, 0)
            self.rb_card_idx.append(rb_idx)
            
            # Labels
            self.is_mistake.append(float(row['is_mistake']))
            self.severities.append(row['mistake_severity'])
        
        self.features = np.array(self.features)
        self.rb_card_idx = np.array(self.rb_card_idx)
        self.best_card_idx = np.array(self.best_card_idx)
        self.is_mistake = np.array(self.is_mistake)
        self.severities = np.array(self.severities)
        self.legal_masks = np.array(self.legal_masks)
        
        print(f"Dataset ready: {len(self)} samples, {self.is_mistake.mean():.1%} mistakes")
    
    def _parse_cards(self, s: str) -> List[str]:
        """Parse card string 'A_Heart|O_Acorn' into list of card strings."""
        if pd.isna(s) or s == '':
            return []
        cards = []
        for part in str(s).split('|'):
            if part and '_' in part:
                # Cards are stored as 'Rank_Suit' e.g. 'A_Heart'
                cards.append(part.strip())
        return cards
    
    def _parse_trick(self, s: str) -> List[Tuple[int, str]]:
        """Parse trick string '0:A_Heart|1:O_Acorn' into list of (player, card) tuples."""
        if pd.isna(s) or s == '':
            return []
        trick = []
        for part in str(s).split('|'):
            if ':' in part:
                player, card_str = part.split(':')
                if '_' in card_str:
                    trick.append((int(player), card_str.strip()))
        return trick
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]),
            torch.tensor(self.rb_card_idx[idx], dtype=torch.long),
            torch.tensor(self.best_card_idx[idx], dtype=torch.long),
            torch.tensor(self.legal_masks[idx]),
            torch.tensor(self.is_mistake[idx], dtype=torch.float32),
            torch.tensor(self.severities[idx], dtype=torch.float32)
        )


def train_correction_network(csv_path: str, epochs: int = 200, 
                             batch_size: int = 128, lr: float = 0.001,
                             save_path: str = 'checkpoints/smart_hybrid.pt',
                             min_severity: float = 0.05) -> CorrectionNetwork:
    """
    Train binary classifier: Is RuleBased wrong?
    
    Simpler task than Q-value prediction - just classify mistake vs correct.
    """
    
    # Load data
    dataset = MCQDataset(csv_path, min_severity=min_severity)
    
    # Split 80/20
    n_val = len(dataset) // 5
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CorrectionNetwork(hidden_size=128, dropout=0.3).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_acc = 0
    patience = 30
    no_improve = 0
    
    print(f"\nTraining on {n_train} samples, validating on {n_val}")
    print(f"Device: {device}")
    print(f"Task: Binary classification (is RuleBased wrong?)")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, rb_idx, best_idx, legal_masks, is_mistake, severities in train_loader:
            features = features.to(device)
            rb_idx = rb_idx.to(device)
            best_idx = best_idx.to(device)
            is_mistake = is_mistake.to(device)
            severities = severities.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(features, rb_idx, best_idx).squeeze()
            
            # Binary cross-entropy, weighted by severity
            weight = 1.0 + severities * 5  # Higher weight for severe mistakes
            loss = F.binary_cross_entropy(pred, is_mistake, weight=weight)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy
            pred_labels = (pred > 0.5).float()
            train_correct += (pred_labels == is_mistake).sum().item()
            train_total += len(is_mistake)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_tp, val_fp, val_tn, val_fn = 0, 0, 0, 0
        
        with torch.no_grad():
            for features, rb_idx, best_idx, legal_masks, is_mistake, severities in val_loader:
                features = features.to(device)
                rb_idx = rb_idx.to(device)
                best_idx = best_idx.to(device)
                is_mistake = is_mistake.to(device)
                
                pred = model(features, rb_idx, best_idx).squeeze()
                pred_labels = (pred > 0.5).float()
                
                val_correct += (pred_labels == is_mistake).sum().item()
                val_total += len(is_mistake)
                
                # Confusion matrix
                val_tp += ((pred_labels == 1) & (is_mistake == 1)).sum().item()
                val_fp += ((pred_labels == 1) & (is_mistake == 0)).sum().item()
                val_tn += ((pred_labels == 0) & (is_mistake == 0)).sum().item()
                val_fn += ((pred_labels == 0) & (is_mistake == 1)).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # Precision/Recall for detecting mistakes
        precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}: Train={train_acc:.1%}, Val={val_acc:.1%}, P={precision:.1%}, R={recall:.1%}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save({
                'model_state': model.state_dict(),
                'val_acc': val_acc,
                'precision': precision,
                'recall': recall,
                'epoch': epoch,
                'hidden_size': 128
            }, save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nBest validation accuracy: {best_val_acc:.1%}")
    
    # Load best model
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    
    return model


class SmartHybridAgent:
    """
    Hybrid agent that uses RuleBased as base but corrects using learned classifier.
    
    Strategy:
    1. Get RuleBased recommendation
    2. For each legal action, ask "is this better than RuleBased?"
    3. If any action is predicted as better with high confidence, use it
    4. Otherwise, trust RuleBased
    """
    
    def __init__(self, model_path: str = 'checkpoints/smart_hybrid.pt',
                 correction_threshold: float = 0.6):
        self.rulebased = RuleBasedAgent()
        self.correction_threshold = correction_threshold
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        hidden_size = 128
        self.model = CorrectionNetwork(hidden_size=hidden_size, dropout=0.3).to(self.device)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            val_acc = checkpoint.get('val_acc', 0)
            print(f"Loaded SmartHybrid from {model_path} (val_acc={val_acc:.1%})")
        else:
            print(f"Warning: No model at {model_path}, using RuleBased only")
        
        self.model.eval()
        
        # Card index mapping
        self.card_to_idx = {card: idx for idx, card in enumerate(FULL_DECK)}
        
        # Stats
        self.corrections_made = 0
        self.total_decisions = 0
    
    def select_action(self, hand: List[str], legal_actions: List[str],
                      current_trick: List[Tuple[int, str]] = None,
                      played_cards: List[str] = None,
                      player_idx: int = 0,
                      declarer_idx: int = 0,
                      partner_idx: Optional[int] = None,
                      points: List[int] = None,
                      trick_number: int = 0,
                      **kwargs) -> str:
        """
        Select action, potentially correcting RuleBased.
        
        Compatible with the same signature as RuleBasedAgent.
        """
        # Handle defaults
        if current_trick is None:
            current_trick = []
        if played_cards is None:
            played_cards = []
        if points is None:
            points = [0, 0, 0, 0]
        
        # Determine if declarer
        declarer_team = {declarer_idx}
        if partner_idx is not None:
            declarer_team.add(partner_idx)
        is_declarer = player_idx in declarer_team
        
        self.total_decisions += 1
        
        # Get RuleBased choice
        rb_choice = self.rulebased.select_action(
            hand=hand, legal_actions=legal_actions,
            current_trick=current_trick, played_cards=played_cards,
            player_idx=player_idx, declarer_idx=declarer_idx,
            partner_idx=partner_idx, points=points, trick_number=trick_number
        )
        
        # If only one option, just use it
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        # Encode game state
        features = encode_game_state(hand, current_trick, played_cards, is_declarer, trick_number)
        feat_tensor = torch.tensor(features).unsqueeze(0).to(self.device)
        
        rb_idx = self.card_to_idx.get(rb_choice, 0)
        rb_idx_tensor = torch.tensor([rb_idx], dtype=torch.long).to(self.device)
        
        # Check each legal action: is it better than RuleBased?
        best_alt_card = None
        best_alt_prob = 0.0
        
        with torch.no_grad():
            for card in legal_actions:
                if card == rb_choice:
                    continue  # Skip comparing RuleBased to itself
                
                card_idx = self.card_to_idx.get(card, 0)
                card_idx_tensor = torch.tensor([card_idx], dtype=torch.long).to(self.device)
                
                # Probability that this card is better than RuleBased
                prob = self.model(feat_tensor, rb_idx_tensor, card_idx_tensor).item()
                
                if prob > best_alt_prob:
                    best_alt_prob = prob
                    best_alt_card = card
        
        # Decide whether to correct
        if best_alt_prob > self.correction_threshold:
            self.corrections_made += 1
            return best_alt_card
        else:
            return rb_choice
    
    def get_correction_rate(self) -> float:
        if self.total_decisions == 0:
            return 0.0
        return self.corrections_made / self.total_decisions
    
    def reset_stats(self):
        self.corrections_made = 0
        self.total_decisions = 0


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--data', type=str, default='data/mc_training_data.csv')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    args = parser.parse_args()
    
    if args.train:
        model = train_correction_network(args.data, epochs=args.epochs)
        print("\nTraining complete!")
    
    if args.eval:
        agent = SmartHybridAgent()
        print(f"Loaded agent, ready for evaluation")
