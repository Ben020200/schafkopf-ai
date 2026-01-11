"""
Train an agent using Monte Carlo "oracle" data.

This module trains a neural network to predict the MC-estimated
best action, effectively learning from perfect play rather than
from RuleBased decisions.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os

from .env import CARD_TO_IDX, IDX_TO_CARD, FULL_DECK, make_env
from .cards import is_trump, suit_of
from .baseline import RuleBasedAgent


@dataclass
class MCTrainingConfig:
    """Configuration for MC-based training."""
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    hidden_size: int = 256
    dropout: float = 0.2
    weight_decay: float = 1e-4
    train_split: float = 0.8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MCDataset(Dataset):
    """Dataset of MC-analyzed decisions."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self._preprocess()
    
    def _preprocess(self):
        """Preprocess dataframe into tensors."""
        self.states = []
        self.best_actions = []
        self.action_masks = []
        self.action_values = []
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            state, mask, best_action, values = self._process_row(row)
            self.states.append(state)
            self.action_masks.append(mask)
            self.best_actions.append(best_action)
            self.action_values.append(values)
        
        self.states = torch.stack(self.states).float()
        self.action_masks = torch.stack(self.action_masks).float()
        self.best_actions = torch.tensor(self.best_actions, dtype=torch.long)
        self.action_values = torch.stack(self.action_values).float()
    
    def _process_row(self, row) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Convert a row to tensors."""
        # Parse hand
        hand_str = row['hand']
        hand_cards = str(hand_str).split('|') if pd.notna(hand_str) and hand_str else []
        hand_vec = torch.zeros(32)
        for card in hand_cards:
            if card in CARD_TO_IDX:
                hand_vec[CARD_TO_IDX[card]] = 1.0
        
        # Parse played cards
        played_str = row['played_cards']
        played_cards = str(played_str).split('|') if pd.notna(played_str) and played_str else []
        played_vec = torch.zeros(32)
        for card in played_cards:
            if card and card in CARD_TO_IDX:
                played_vec[CARD_TO_IDX[card]] = 1.0
        
        # Parse current trick
        trick_str = row['current_trick']
        trick_vec = torch.zeros(4, 32)  # 4 positions, 32 cards
        if pd.notna(trick_str) and trick_str:
            for item in str(trick_str).split('|'):
                if ':' in item:
                    pos, card = item.split(':')
                    pos = int(pos)
                    if card in CARD_TO_IDX and pos < 4:
                        trick_vec[pos, CARD_TO_IDX[card]] = 1.0
        trick_vec = trick_vec.flatten()  # [128]
        
        # Game info
        trick_number = row['trick_number'] / 7.0  # Normalize
        is_declarer = float(row['is_declarer'])
        player_idx = row['player_idx'] / 3.0
        
        game_info = torch.tensor([trick_number, is_declarer, player_idx])
        
        # Combine state
        state = torch.cat([hand_vec, played_vec, trick_vec, game_info])  # 32+32+128+3 = 195
        
        # Legal action mask
        legal_str = row['legal_actions']
        legal_cards = str(legal_str).split('|') if pd.notna(legal_str) and legal_str else []
        mask = torch.zeros(32)
        for card in legal_cards:
            if card in CARD_TO_IDX:
                mask[CARD_TO_IDX[card]] = 1.0
        
        # Best action
        best_card = row['best_action']
        best_action = CARD_TO_IDX.get(best_card, 0)
        
        # Action values (for potential value-based training)
        values = torch.zeros(32)
        for col in self.df.columns:
            if col.startswith('value_'):
                card = col[6:]  # Remove 'value_' prefix
                if card in CARD_TO_IDX and pd.notna(row[col]):
                    values[CARD_TO_IDX[card]] = row[col]
        
        return state, mask, best_action, values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'mask': self.action_masks[idx],
            'best_action': self.best_actions[idx],
            'action_values': self.action_values[idx],
        }


class MCPolicyNetwork(nn.Module):
    """Network trained to predict MC-optimal actions."""
    
    def __init__(self, state_size: int = 195, hidden_size: int = 256, dropout: float = 0.2):
        super().__init__()
        
        # Stronger regularization with BatchNorm and higher dropout
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Less dropout at end
            nn.Linear(hidden_size // 2, 32),  # 32 possible cards
        )
    
    def forward(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with action masking.
        Returns log probabilities over actions.
        """
        logits = self.net(state)
        # Mask illegal actions
        logits = logits.masked_fill(mask == 0, float('-inf'))
        return F.log_softmax(logits, dim=-1)
    
    def get_action(self, state: torch.Tensor, mask: torch.Tensor) -> int:
        """Get best action (greedy)."""
        with torch.no_grad():
            log_probs = self.forward(state.unsqueeze(0), mask.unsqueeze(0))
            return log_probs.argmax(dim=-1).item()


class MCValueNetwork(nn.Module):
    """Network trained to predict action values (Q-values)."""
    
    def __init__(self, state_size: int = 195, hidden_size: int = 256, dropout: float = 0.2):
        super().__init__()
        
        # Stronger regularization with BatchNorm
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 32),  # Q-value per card
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for all actions."""
        return self.net(state)
    
    def get_action(self, state: torch.Tensor, mask: torch.Tensor) -> int:
        """Get action with highest Q-value among legal actions."""
        with torch.no_grad():
            q_values = self.forward(state.unsqueeze(0))[0]
            # Mask illegal actions
            q_values = q_values.masked_fill(mask == 0, float('-inf'))
            return q_values.argmax().item()


class MCTrainer:
    """Trainer for MC-based policy/value networks."""
    
    def __init__(self, config: MCTrainingConfig = None):
        self.config = config or MCTrainingConfig()
        self.device = torch.device(self.config.device)
        
        self.policy_net = None
        self.value_net = None
    
    def load_data(self, filepath: str) -> Tuple[DataLoader, DataLoader]:
        """Load and split data into train/val loaders."""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"  Total samples: {len(df)}")
        
        # Shuffle and split
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * self.config.train_split)
        
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        print(f"  Train samples: {len(train_df)}")
        print(f"  Val samples: {len(val_df)}")
        
        train_dataset = MCDataset(train_df)
        val_dataset = MCDataset(val_df)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        return train_loader, val_loader
    
    def train_policy(self, train_loader: DataLoader, val_loader: DataLoader) -> MCPolicyNetwork:
        """Train policy network to predict best action."""
        print("\nTraining Policy Network...")
        print(f"  Device: {self.device}")
        
        # Get state size from data
        sample = next(iter(train_loader))
        state_size = sample['state'].shape[1]
        
        self.policy_net = MCPolicyNetwork(
            state_size=state_size,
            hidden_size=self.config.hidden_size,
            dropout=self.config.dropout,
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(
            self.policy_net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        best_val_acc = 0.0
        best_state = None
        
        for epoch in range(self.config.epochs):
            # Training
            self.policy_net.train()
            train_loss = 0.0
            train_correct = 0
            train_near_correct = 0  # Within 0.05 of best value
            train_total = 0
            
            for batch in train_loader:
                state = batch['state'].to(self.device)
                mask = batch['mask'].to(self.device)
                target = batch['best_action'].to(self.device)
                values = batch['action_values'].to(self.device)
                
                optimizer.zero_grad()
                log_probs = self.policy_net(state, mask)
                loss = F.nll_loss(log_probs, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                preds = log_probs.argmax(dim=-1)
                train_correct += (preds == target).sum().item()
                
                # Check if prediction is "near optimal" (within 0.05 of best value)
                pred_values = values.gather(1, preds.unsqueeze(1)).squeeze()
                best_values = values.gather(1, target.unsqueeze(1)).squeeze()
                train_near_correct += ((best_values - pred_values) < 0.05).sum().item()
                
                train_total += len(target)
            
            train_acc = train_correct / train_total
            train_near_acc = train_near_correct / train_total
            
            # Validation
            self.policy_net.eval()
            val_correct = 0
            val_near_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    state = batch['state'].to(self.device)
                    mask = batch['mask'].to(self.device)
                    target = batch['best_action'].to(self.device)
                    values = batch['action_values'].to(self.device)
                    
                    log_probs = self.policy_net(state, mask)
                    preds = log_probs.argmax(dim=-1)
                    val_correct += (preds == target).sum().item()
                    
                    # Near optimal
                    pred_values = values.gather(1, preds.unsqueeze(1)).squeeze()
                    best_values = values.gather(1, target.unsqueeze(1)).squeeze()
                    val_near_correct += ((best_values - pred_values) < 0.05).sum().item()
                    
                    val_total += len(target)
            
            val_acc = val_correct / val_total
            val_near_acc = val_near_correct / val_total
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.policy_net.state_dict().items()}
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{self.config.epochs}: "
                      f"Train Acc={train_acc*100:.1f}% (near={train_near_acc*100:.1f}%), "
                      f"Val Acc={val_acc*100:.1f}% (near={val_near_acc*100:.1f}%)")
        
        # Load best model
        self.policy_net.load_state_dict(best_state)
        print(f"\nBest Policy Val Accuracy: {best_val_acc*100:.1f}%")
        
        return self.policy_net
    
    def train_value(self, train_loader: DataLoader, val_loader: DataLoader) -> MCValueNetwork:
        """Train value network to predict action values."""
        print("\nTraining Value Network...")
        
        sample = next(iter(train_loader))
        state_size = sample['state'].shape[1]
        
        self.value_net = MCValueNetwork(
            state_size=state_size,
            hidden_size=self.config.hidden_size,
            dropout=self.config.dropout,
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(
            self.value_net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        best_val_loss = float('inf')
        best_state = None
        
        for epoch in range(self.config.epochs):
            # Training
            self.value_net.train()
            train_loss = 0.0
            
            for batch in train_loader:
                state = batch['state'].to(self.device)
                mask = batch['mask'].to(self.device)
                target_values = batch['action_values'].to(self.device)
                
                optimizer.zero_grad()
                pred_values = self.value_net(state)
                
                # Only compute loss on legal actions
                loss = F.mse_loss(pred_values * mask, target_values * mask)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.value_net.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    state = batch['state'].to(self.device)
                    mask = batch['mask'].to(self.device)
                    target_values = batch['action_values'].to(self.device)
                    
                    pred_values = self.value_net(state)
                    loss = F.mse_loss(pred_values * mask, target_values * mask)
                    val_loss += loss.item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.value_net.state_dict().items()}
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{self.config.epochs}: "
                      f"Train Loss={train_loss/len(train_loader):.4f}, "
                      f"Val Loss={val_loss/len(val_loader):.4f}")
        
        self.value_net.load_state_dict(best_state)
        print(f"\nBest Value Val Loss: {best_val_loss/len(val_loader):.4f}")
        
        return self.value_net
    
    def save(self, filepath: str):
        """Save trained models."""
        state = {
            'policy_net': self.policy_net.state_dict() if self.policy_net else None,
            'value_net': self.value_net.state_dict() if self.value_net else None,
            'config': self.config,
        }
        torch.save(state, filepath)
        print(f"Saved models to {filepath}")
    
    def load(self, filepath: str):
        """Load trained models."""
        state = torch.load(filepath, map_location=self.device)
        self.config = state['config']
        
        if state['policy_net']:
            self.policy_net = MCPolicyNetwork(
                hidden_size=self.config.hidden_size,
                dropout=self.config.dropout,
            ).to(self.device)
            self.policy_net.load_state_dict(state['policy_net'])
        
        if state['value_net']:
            self.value_net = MCValueNetwork(
                hidden_size=self.config.hidden_size,
                dropout=self.config.dropout,
            ).to(self.device)
            self.value_net.load_state_dict(state['value_net'])
        
        print(f"Loaded models from {filepath}")


class MCAgent:
    """Agent that uses MC-trained networks for decision making."""
    
    def __init__(self, model_path: str, use_value: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_value = use_value
        
        # Load model
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        config = state['config']
        
        if use_value and state['value_net']:
            self.net = MCValueNetwork(
                hidden_size=config.hidden_size,
                dropout=0.0,  # No dropout during inference
            ).to(self.device)
            self.net.load_state_dict(state['value_net'])
        else:
            self.net = MCPolicyNetwork(
                hidden_size=config.hidden_size,
                dropout=0.0,
            ).to(self.device)
            self.net.load_state_dict(state['policy_net'])
        
        self.net.eval()
    
    def _build_state(self, hand, legal_actions, current_trick, played_cards,
                     player_idx, declarer_idx, partner_idx, trick_number) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build state tensor from game info."""
        # Hand
        hand_vec = torch.zeros(32)
        for card in hand:
            card_str = str(card)
            if card_str in CARD_TO_IDX:
                hand_vec[CARD_TO_IDX[card_str]] = 1.0
        
        # Played cards
        played_vec = torch.zeros(32)
        for card in played_cards:
            card_str = str(card)
            if card_str in CARD_TO_IDX:
                played_vec[CARD_TO_IDX[card_str]] = 1.0
        
        # Current trick
        trick_vec = torch.zeros(4, 32)
        for pos, card in current_trick:
            card_str = str(card)
            if card_str in CARD_TO_IDX and pos < 4:
                trick_vec[pos, CARD_TO_IDX[card_str]] = 1.0
        trick_vec = trick_vec.flatten()
        
        # Game info
        declarer_team = {declarer_idx}
        if partner_idx is not None:
            declarer_team.add(partner_idx)
        is_declarer = float(player_idx in declarer_team)
        
        game_info = torch.tensor([trick_number / 7.0, is_declarer, player_idx / 3.0])
        
        state = torch.cat([hand_vec, played_vec, trick_vec, game_info])
        
        # Mask
        mask = torch.zeros(32)
        for card in legal_actions:
            card_str = str(card)
            if card_str in CARD_TO_IDX:
                mask[CARD_TO_IDX[card_str]] = 1.0
        
        return state, mask
    
    def select_action(self, hand, legal_actions, current_trick, played_cards,
                      player_idx, declarer_idx, partner_idx, points, trick_number) -> str:
        """Select action using trained network."""
        state, mask = self._build_state(
            hand, legal_actions, current_trick, played_cards,
            player_idx, declarer_idx, partner_idx, trick_number
        )
        
        state = state.to(self.device)
        mask = mask.to(self.device)
        
        action_idx = self.net.get_action(state, mask)
        return IDX_TO_CARD[action_idx]


def evaluate_mc_agent(model_path: str, num_games: int = 500, seed: int = 42) -> float:
    """Evaluate MC agent against RuleBased."""
    from .baseline import RuleBasedAgent
    
    env = make_env(seed=seed)
    mc_agent = MCAgent(model_path)
    rb_agent = RuleBasedAgent()
    rng = np.random.default_rng(seed)
    
    mc_wins = 0
    
    print(f"Evaluating MC agent vs RuleBased over {num_games} games...")
    
    for game in range(num_games):
        # Random declarer
        declarer = int(rng.integers(0, 4))
        
        # MC plays 50% as declarer team, 50% as opponent
        mc_plays_declarer = game % 2 == 0
        
        env.reset(options={"fixed_declarer": declarer})
        
        while not all(env.terminations.values()):
            player_name = env.agent_selection
            player_idx = int(player_name.split("_")[1])
            
            hand = env._hands[player_idx]
            legal_indices = env._get_legal_actions(player_idx)
            legal_cards = [IDX_TO_CARD[idx] for idx in legal_indices]
            current_trick = [(p, c) for p, c in env._current_trick]
            
            # Determine if this player uses MC agent
            declarer_team = {env._declarer_idx}
            if env._partner_idx is not None:
                declarer_team.add(env._partner_idx)
            is_declarer_team = player_idx in declarer_team
            
            use_mc = (mc_plays_declarer == is_declarer_team)
            
            if use_mc:
                card = mc_agent.select_action(
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
            else:
                card = rb_agent.select_action(
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
            
            action = CARD_TO_IDX[card]
            env.step(action)
        
        result = env.get_game_result()
        declarer_won = result["win"]
        
        # MC wins if it was declarer team and won, or opponent team and declarer lost
        if mc_plays_declarer == declarer_won:
            mc_wins += 1
        
        if (game + 1) % 100 == 0:
            print(f"  Games {game+1}/{num_games}: MC win rate = {mc_wins/(game+1)*100:.1f}%")
    
    win_rate = mc_wins / num_games
    print(f"\nFinal: MC Agent vs RuleBased: {mc_wins}/{num_games} = {win_rate*100:.1f}%")
    
    return win_rate


def train_mc_agent(
    data_path: str = "data/mc_training_data.csv",
    output_path: str = "checkpoints/mc_agent.pt",
    epochs: int = 50,
) -> MCTrainer:
    """Main training function."""
    config = MCTrainingConfig(epochs=epochs)
    trainer = MCTrainer(config)
    
    train_loader, val_loader = trainer.load_data(data_path)
    
    # Train both networks
    trainer.train_policy(train_loader, val_loader)
    trainer.train_value(train_loader, val_loader)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trainer.save(output_path)
    
    return trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/mc_training_data.csv")
    parser.add_argument("--output", type=str, default="checkpoints/mc_agent.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-only", type=str, default=None, help="Only evaluate this model")
    args = parser.parse_args()
    
    if args.eval_only:
        evaluate_mc_agent(args.eval_only)
    else:
        trainer = train_mc_agent(args.data, args.output, args.epochs)
        print("\nEvaluating trained agent...")
        evaluate_mc_agent(args.output)
