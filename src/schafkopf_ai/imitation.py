"""
Imitation Learning for Schafkopf AI.

Pre-trains the PPO network to mimic the RuleBased agent's decisions.
This gives the agent a good starting point before RL fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import os

from .env import SchafkopfEnv, make_env, CARD_TO_IDX, IDX_TO_CARD
from .baseline import RuleBasedAgent
from .ppo import SchafkopfLSTMNetwork, SchafkopfNetwork


class SchafkopfImitationDataset(Dataset):
    """Dataset of (observation, action) pairs from RuleBased agent."""
    
    def __init__(self, num_games: int = 10000, seed: int = 42):
        """
        Generate dataset by playing games with RuleBased agent.
        
        Args:
            num_games: Number of games to collect data from
            seed: Random seed for reproducibility
        """
        self.observations = []
        self.actions = []
        self.action_masks = []
        
        env = make_env(seed=seed)
        agent = RuleBasedAgent()
        rng = np.random.default_rng(seed)
        
        print(f"Collecting {num_games} games of expert data...")
        
        for game in tqdm(range(num_games)):
            # Random declarer for variety
            random_declarer = int(rng.integers(0, 4))
            env.reset(options={"fixed_declarer": random_declarer})
            
            while not all(env.terminations.values()):
                player_name = env.agent_selection
                player_idx = int(player_name.split("_")[1])
                
                obs = env.observe(player_name)
                
                # Get RuleBased agent's action
                hand = env._hands[player_idx]
                legal_indices = env._get_legal_actions(player_idx)
                legal_actions = [IDX_TO_CARD[idx] for idx in legal_indices]
                current_trick = [(p, c) for p, c in env._current_trick]
                
                card = agent.select_action(
                    hand=hand,
                    legal_actions=legal_actions,
                    current_trick=current_trick,
                    played_cards=env._played_cards,
                    player_idx=player_idx,
                    declarer_idx=env._declarer_idx,
                    partner_idx=env._partner_idx,
                    points=env._points,
                    trick_number=env._trick_number,
                )
                action = CARD_TO_IDX[card]
                
                # Store observation and action
                self.observations.append({
                    "own_hand": obs["own_hand"].copy(),
                    "played_cards": obs["played_cards"].copy(),
                    "current_trick": obs["current_trick"].copy(),
                    "points": obs["points"].copy(),
                    "game_info": obs["game_info"].copy(),
                })
                self.actions.append(action)
                self.action_masks.append(obs["action_mask"].copy())
                
                env.step(action)
        
        print(f"Collected {len(self.actions)} state-action pairs")
        
        # Convert to numpy arrays
        self.obs_own_hand = np.array([o["own_hand"] for o in self.observations])
        self.obs_played_cards = np.array([o["played_cards"] for o in self.observations])
        self.obs_current_trick = np.array([o["current_trick"] for o in self.observations])
        self.obs_points = np.array([o["points"] for o in self.observations])
        self.obs_game_info = np.array([o["game_info"] for o in self.observations])
        self.actions = np.array(self.actions)
        self.action_masks = np.array(self.action_masks)
        
        # Clear original list to save memory
        self.observations = None
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        obs = {
            "own_hand": torch.tensor(self.obs_own_hand[idx], dtype=torch.float32),
            "played_cards": torch.tensor(self.obs_played_cards[idx], dtype=torch.float32),
            "current_trick": torch.tensor(self.obs_current_trick[idx], dtype=torch.float32),
            "points": torch.tensor(self.obs_points[idx], dtype=torch.float32),
            "game_info": torch.tensor(self.obs_game_info[idx], dtype=torch.float32),
        }
        action = torch.tensor(self.actions[idx], dtype=torch.long)
        mask = torch.tensor(self.action_masks[idx], dtype=torch.float32)
        
        return obs, action, mask


def collate_fn(batch):
    """Custom collate function for batching observations."""
    obs_batch = {
        "own_hand": torch.stack([b[0]["own_hand"] for b in batch]),
        "played_cards": torch.stack([b[0]["played_cards"] for b in batch]),
        "current_trick": torch.stack([b[0]["current_trick"] for b in batch]),
        "points": torch.stack([b[0]["points"] for b in batch]),
        "game_info": torch.stack([b[0]["game_info"] for b in batch]),
    }
    actions = torch.stack([b[1] for b in batch])
    masks = torch.stack([b[2] for b in batch])
    
    return obs_batch, actions, masks


class ImitationTrainer:
    """Train network to imitate RuleBased agent."""
    
    def __init__(
        self,
        hidden_size: int = 512,
        use_lstm: bool = False,  # Simpler network for imitation
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.use_lstm = use_lstm
        
        if use_lstm:
            self.network = SchafkopfLSTMNetwork(hidden_size, lstm_hidden_size=256).to(self.device)
        else:
            self.network = SchafkopfNetwork(hidden_size).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        total_params = sum(p.numel() for p in self.network.parameters())
        print(f"Network: {'LSTM' if use_lstm else 'Feedforward'} ({total_params:,} params)")
    
    def train(
        self,
        dataset: SchafkopfImitationDataset,
        epochs: int = 10,
        batch_size: int = 256,
        val_split: float = 0.1,
    ) -> Dict[str, List[float]]:
        """
        Train the network on the imitation dataset.
        
        Returns:
            Dictionary with training history
        """
        # Split into train/val
        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val]
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=0,
        )
        
        print(f"\nTraining on {n_train} samples, validating on {n_val} samples")
        print(f"Batch size: {batch_size}, Epochs: {epochs}")
        print("-" * 50)
        
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        
        for epoch in range(epochs):
            # Training
            self.network.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for obs, actions, masks in train_loader:
                # Move to device
                obs = {k: v.to(self.device) for k, v in obs.items()}
                actions = actions.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass - get logits directly
                self.optimizer.zero_grad()
                
                if self.use_lstm:
                    hidden = self.network.get_initial_hidden(obs["own_hand"].shape[0], self.device)
                    logits, _ = self.network.forward(obs, masks, hidden)
                else:
                    logits, _ = self.network.forward(obs, masks)
                
                # Cross-entropy loss
                loss = F.cross_entropy(logits, actions)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * actions.shape[0]
                
                # Accuracy
                preds = logits.argmax(dim=1)
                train_correct += (preds == actions).sum().item()
                train_total += actions.shape[0]
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # Validation
            self.network.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for obs, actions, masks in val_loader:
                    obs = {k: v.to(self.device) for k, v in obs.items()}
                    actions = actions.to(self.device)
                    masks = masks.to(self.device)
                    
                    if self.use_lstm:
                        hidden = self.network.get_initial_hidden(obs["own_hand"].shape[0], self.device)
                        logits, _ = self.network.forward(obs, masks, hidden)
                    else:
                        logits, _ = self.network.forward(obs, masks)
                    
                    loss = F.cross_entropy(logits, actions)
                    
                    val_loss += loss.item() * actions.shape[0]
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == actions).sum().item()
                    val_total += actions.shape[0]
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")
        
        return history
    
    def save(self, path: str):
        """Save the trained network."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "use_lstm": self.use_lstm,
        }, path)
        print(f"Saved imitation model to {path}")
    
    def evaluate_vs_rulebased(self, num_games: int = 100) -> float:
        """Evaluate the imitation agent against RuleBased."""
        from .baseline import RuleBasedAgent
        
        env = make_env()
        opponent = RuleBasedAgent()
        rng = np.random.default_rng(42)
        
        self.network.eval()
        wins = 0
        
        for game in range(num_games):
            random_declarer = int(rng.integers(0, 4))
            env.reset(options={"fixed_declarer": random_declarer})
            
            declarer_team = {env._declarer_idx}
            if env._partner_idx is not None:
                declarer_team.add(env._partner_idx)
            
            # Alternate: imitation plays declarer team in even games
            imitation_is_declarer = (game % 2 == 0)
            imitation_players = declarer_team if imitation_is_declarer else {0, 1, 2, 3} - declarer_team
            
            if self.use_lstm:
                hidden = self.network.get_initial_hidden(1, self.device)
            
            while not all(env.terminations.values()):
                player_name = env.agent_selection
                player_idx = int(player_name.split("_")[1])
                obs = env.observe(player_name)
                
                if player_idx in imitation_players:
                    # Imitation agent's turn
                    with torch.no_grad():
                        obs_tensor = {
                            k: torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)
                            for k, v in obs.items() if k != "action_mask"
                        }
                        mask_tensor = torch.tensor(
                            obs["action_mask"], dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        
                        if self.use_lstm:
                            action, _, _, _, hidden = self.network.get_action_and_value(
                                obs_tensor, mask_tensor, hidden
                            )
                        else:
                            action, _, _, _ = self.network.get_action_and_value(
                                obs_tensor, mask_tensor
                            )
                    action = action.item()
                else:
                    # RuleBased opponent
                    hand = env._hands[player_idx]
                    legal_indices = env._get_legal_actions(player_idx)
                    legal_actions = [IDX_TO_CARD[idx] for idx in legal_indices]
                    current_trick = [(p, c) for p, c in env._current_trick]
                    
                    card = opponent.select_action(
                        hand=hand,
                        legal_actions=legal_actions,
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
            
            if imitation_is_declarer:
                if declarer_won:
                    wins += 1
            else:
                if not declarer_won:
                    wins += 1
        
        return wins / num_games


def train_imitation(
    num_games: int = 20000,
    epochs: int = 20,
    batch_size: int = 256,
    hidden_size: int = 512,
    use_lstm: bool = False,
    learning_rate: float = 1e-3,
    save_path: str = "checkpoints/imitation_model.pt",
) -> ImitationTrainer:
    """
    Train an imitation learning model.
    
    Args:
        num_games: Number of games to collect for training data
        epochs: Training epochs
        batch_size: Batch size for training
        hidden_size: Network hidden size
        use_lstm: Whether to use LSTM network
        learning_rate: Learning rate
        save_path: Path to save the trained model
    
    Returns:
        Trained ImitationTrainer
    """
    print("=" * 60)
    print("IMITATION LEARNING - Pre-training on RuleBased Agent")
    print("=" * 60)
    
    # Collect dataset
    dataset = SchafkopfImitationDataset(num_games=num_games)
    
    # Create trainer
    trainer = ImitationTrainer(
        hidden_size=hidden_size,
        use_lstm=use_lstm,
        learning_rate=learning_rate,
    )
    
    # Train
    history = trainer.train(dataset, epochs=epochs, batch_size=batch_size)
    
    # Evaluate
    print("\nEvaluating against RuleBased...")
    win_rate = trainer.evaluate_vs_rulebased(num_games=200)
    print(f"Win rate vs RuleBased: {win_rate*100:.1f}%")
    
    # Save
    trainer.save(save_path)
    
    return trainer


if __name__ == "__main__":
    train_imitation()
