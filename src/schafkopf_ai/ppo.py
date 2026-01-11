"""PPO training for Schafkopf with self-play or opponent training."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import SchafkopfEnv, make_env, CARD_TO_IDX, IDX_TO_CARD

if TYPE_CHECKING:
    from .baseline import BaseAgent


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    # Training
    total_timesteps: int = 1_000_000
    num_envs: int = 8  # Parallel environments
    num_steps: int = 128  # Steps per rollout per env
    batch_size: int = 256
    num_epochs: int = 4
    
    # PPO specific
    learning_rate: float = 1e-4  # Reduced for stability
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.05  # Increased for more exploration
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Network
    hidden_size: int = 512  # Increased from 256
    lstm_hidden_size: int = 256  # LSTM memory size
    use_lstm: bool = True  # Enable LSTM memory
    
    # Logging
    log_interval: int = 10
    save_interval: int = 50
    eval_interval: int = 20
    eval_episodes: int = 100
    
    # Paths
    save_dir: str = "checkpoints"
    
    # Opponent training
    opponent_type: Optional[str] = None  # None = self-play, "rulebased" = vs RuleBasedAgent, "bidding" = full self-play, "mixed" = 50/50, "focused" = vs RuleBased only
    
    # Reward shaping
    reward_mode: str = "sparse"  # "sparse", "shaped", or "dense"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SchafkopfNetwork(nn.Module):
    """Actor-Critic network for Schafkopf (feedforward version)."""
    
    def __init__(self, hidden_size: int = 512):
        super().__init__()
        
        # Input: own_hand(32) + played_cards(32) + current_trick(4*32=128) + 
        #        points(4) + game_info(8) = 204
        input_size = 32 + 32 + 128 + 4 + 8
        
        # Larger shared encoder (3 layers instead of 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Policy head (actor) - larger
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 32),
        )
        
        # Value head (critic) - larger
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        obs: Dict[str, torch.Tensor],
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value."""
        # Flatten observation
        x = torch.cat([
            obs["own_hand"].float(),
            obs["played_cards"].float(),
            obs["current_trick"].float().flatten(start_dim=-2),
            obs["points"],
            obs["game_info"],
        ], dim=-1)
        
        # Encode
        features = self.encoder(x)
        
        # Get outputs
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        
        # Apply action mask
        if action_mask is not None:
            # Set invalid actions to very negative logits
            logits = logits.masked_fill(action_mask == 0, -1e8)
        
        return logits, value
    
    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log prob, entropy, and value."""
        logits, value = self.forward(obs, action_mask)
        
        # Create distribution
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value


class SchafkopfLSTMNetwork(nn.Module):
    """
    Actor-Critic network with LSTM memory for Schafkopf.
    
    The LSTM allows the agent to remember:
    - Which cards have been played (card counting)
    - Patterns in opponent play
    - Game flow and trick sequences
    """
    
    def __init__(self, hidden_size: int = 512, lstm_hidden_size: int = 256):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        
        # Input: own_hand(32) + played_cards(32) + current_trick(4*32=128) + 
        #        points(4) + game_info(8) = 204
        input_size = 32 + 32 + 128 + 4 + 8
        
        # Initial encoder (process raw observation)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # LSTM for temporal memory
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        
        # Post-LSTM processing
        self.post_lstm = nn.Sequential(
            nn.Linear(lstm_hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 32),
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        # LSTM specific initialization
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def get_initial_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial LSTM hidden state."""
        h0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        return (h0, c0)
    
    def _prepare_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten observation dict to tensor."""
        return torch.cat([
            obs["own_hand"].float(),
            obs["played_cards"].float(),
            obs["current_trick"].float().flatten(start_dim=-2),
            obs["points"],
            obs["game_info"],
        ], dim=-1)
    
    def forward(
        self, 
        obs: Dict[str, torch.Tensor],
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with LSTM memory.
        
        Returns: (logits, value, new_hidden_state)
        """
        x = self._prepare_obs(obs)
        batch_size = x.shape[0]
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, x.device)
        
        # Encode observation
        encoded = self.encoder(x)
        
        # Add sequence dimension for LSTM (batch, seq=1, features)
        encoded = encoded.unsqueeze(1)
        
        # LSTM forward
        lstm_out, new_hidden = self.lstm(encoded, hidden)
        
        # Remove sequence dimension
        lstm_out = lstm_out.squeeze(1)
        
        # Post-LSTM processing
        features = self.post_lstm(lstm_out)
        
        # Get outputs
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        
        # Apply action mask
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e8)
        
        return logits, value, new_hidden
    
    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get action, log prob, entropy, value, and new hidden state."""
        logits, value, new_hidden = self.forward(obs, hidden, action_mask)
        
        # Create distribution
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value, new_hidden
    
    def forward_sequence(
        self,
        obs_sequence: Dict[str, torch.Tensor],
        action_mask_sequence: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass on a sequence of observations (for training).
        
        obs_sequence: dict with tensors of shape (batch, seq_len, ...)
        Returns: (logits, values, final_hidden) with shapes (batch, seq_len, ...)
        """
        # Get sequence dimensions
        batch_size = obs_sequence["own_hand"].shape[0]
        seq_len = obs_sequence["own_hand"].shape[1]
        
        # Initialize hidden state
        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, obs_sequence["own_hand"].device)
        
        # Flatten batch and sequence for encoder
        obs_flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in obs_sequence.items()}
        x = self._prepare_obs(obs_flat)
        
        # Encode all observations
        encoded = self.encoder(x)
        
        # Reshape back to (batch, seq_len, hidden)
        encoded = encoded.reshape(batch_size, seq_len, -1)
        
        # LSTM forward on full sequence
        lstm_out, new_hidden = self.lstm(encoded, hidden)
        
        # Post-LSTM processing
        features = self.post_lstm(lstm_out.reshape(-1, self.lstm_hidden_size))
        features = features.reshape(batch_size, seq_len, -1)
        
        # Get outputs for all timesteps
        logits = self.policy_head(features.reshape(-1, self.hidden_size))
        logits = logits.reshape(batch_size, seq_len, 32)
        
        values = self.value_head(features.reshape(-1, self.hidden_size))
        values = values.reshape(batch_size, seq_len)
        
        # Apply action mask
        if action_mask_sequence is not None:
            mask_flat = action_mask_sequence.reshape(batch_size, seq_len, 32)
            logits = logits.masked_fill(mask_flat == 0, -1e8)
        
        return logits, values, new_hidden


class RolloutBuffer:
    """Buffer for storing rollout data with optional LSTM support."""
    
    def __init__(
        self, 
        num_steps: int, 
        num_envs: int, 
        device: str,
        use_lstm: bool = False,
        lstm_hidden_size: int = 256,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.use_lstm = use_lstm
        self.lstm_hidden_size = lstm_hidden_size
        self.reset()
    
    def reset(self):
        self.obs_own_hand = []
        self.obs_played_cards = []
        self.obs_current_trick = []
        self.obs_points = []
        self.obs_game_info = []
        self.action_masks = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.ptr = 0
        
        # LSTM hidden states (stored at each step for training)
        if self.use_lstm:
            self.hidden_h = []  # LSTM h states
            self.hidden_c = []  # LSTM c states
    
    def add(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        self.obs_own_hand.append(obs["own_hand"])
        self.obs_played_cards.append(obs["played_cards"])
        self.obs_current_trick.append(obs["current_trick"])
        self.obs_points.append(obs["points"])
        self.obs_game_info.append(obs["game_info"])
        self.action_masks.append(action_mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        
        # Store LSTM hidden states
        if self.use_lstm and hidden is not None:
            self.hidden_h.append(hidden[0].detach().cpu().numpy())
            self.hidden_c.append(hidden[1].detach().cpu().numpy())
        
        self.ptr += 1
    
    def compute_returns_and_advantages(
        self,
        last_value: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ):
        """Compute GAE advantages and returns."""
        advantages = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        last_gae = 0
        
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
                next_done = np.zeros(self.num_envs)
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]
            
            delta = (
                self.rewards[t] 
                + gamma * next_value * (1 - next_done) 
                - self.values[t]
            )
            advantages[t] = last_gae = (
                delta + gamma * gae_lambda * (1 - next_done) * last_gae
            )
        
        returns = advantages + np.array(self.values)
        self.advantages = advantages.flatten()
        self.returns = returns.flatten()
    
    def get_batches(self, batch_size: int):
        """Yield minibatches for training."""
        total_size = self.num_steps * self.num_envs
        indices = np.random.permutation(total_size)
        
        # Stack all data
        obs = {
            "own_hand": torch.tensor(
                np.array(self.obs_own_hand).reshape(total_size, -1),
                dtype=torch.float32, device=self.device
            ),
            "played_cards": torch.tensor(
                np.array(self.obs_played_cards).reshape(total_size, -1),
                dtype=torch.float32, device=self.device
            ),
            "current_trick": torch.tensor(
                np.array(self.obs_current_trick).reshape(total_size, 4, 32),
                dtype=torch.float32, device=self.device
            ),
            "points": torch.tensor(
                np.array(self.obs_points).reshape(total_size, -1),
                dtype=torch.float32, device=self.device
            ),
            "game_info": torch.tensor(
                np.array(self.obs_game_info).reshape(total_size, -1),
                dtype=torch.float32, device=self.device
            ),
        }
        action_masks = torch.tensor(
            np.array(self.action_masks).reshape(total_size, -1),
            dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            np.array(self.actions).flatten(),
            dtype=torch.long, device=self.device
        )
        log_probs = torch.tensor(
            np.array(self.log_probs).flatten(),
            dtype=torch.float32, device=self.device
        )
        advantages = torch.tensor(
            self.advantages, dtype=torch.float32, device=self.device
        )
        returns = torch.tensor(
            self.returns, dtype=torch.float32, device=self.device
        )
        
        for start in range(0, total_size, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            
            yield (
                {k: v[batch_idx] for k, v in obs.items()},
                action_masks[batch_idx],
                actions[batch_idx],
                log_probs[batch_idx],
                advantages[batch_idx],
                returns[batch_idx],
            )


class VectorizedSchafkopfEnv:
    """Vectorized environment wrapper for parallel self-play."""
    
    def __init__(self, num_envs: int, seed: int = 0, reward_mode: str = "sparse"):
        self.num_envs = num_envs
        self.reward_mode = reward_mode
        self.envs = [make_env(seed=seed + i, reward_mode=reward_mode) for i in range(num_envs)]
        
        # Track current agent per env
        self.current_agents = ["player_0"] * num_envs
        
    def reset(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Reset all environments."""
        obs_batch = {
            "own_hand": [],
            "played_cards": [],
            "current_trick": [],
            "points": [],
            "game_info": [],
        }
        action_masks = []
        
        for i, env in enumerate(self.envs):
            env.reset()
            self.current_agents[i] = env.agent_selection
            obs = env.observe(env.agent_selection)
            
            for key in obs_batch:
                obs_batch[key].append(obs[key])
            action_masks.append(obs["action_mask"])
        
        return (
            {k: np.array(v) for k, v in obs_batch.items()},
            np.array(action_masks),
        )
    
    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step all environments."""
        obs_batch = {
            "own_hand": [],
            "played_cards": [],
            "current_trick": [],
            "points": [],
            "game_info": [],
        }
        action_masks = []
        rewards = []
        dones = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            # Step the environment
            env.step(int(action))
            
            # Get reward for the agent that just acted
            agent = self.current_agents[i]
            reward = env._cumulative_rewards[agent]
            done = env.terminations[agent]
            
            # Reset cumulative reward after reading
            env._cumulative_rewards[agent] = 0
            
            # Check if game is over
            if all(env.terminations.values()):
                info = {"game_result": env.get_game_result()}
                env.reset()
                done = True
            else:
                info = {}
            
            # Move to next agent
            self.current_agents[i] = env.agent_selection
            obs = env.observe(env.agent_selection)
            
            for key in obs_batch:
                obs_batch[key].append(obs[key])
            action_masks.append(obs["action_mask"])
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return (
            {k: np.array(v) for k, v in obs_batch.items()},
            np.array(action_masks),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            infos,
        )


class VectorizedSchafkopfEnvWithOpponent:
    """
    Vectorized environment where PPO trains as declarer team vs fixed opponent.
    
    Player 0 (declarer) and Player 2 (partner) are controlled by PPO.
    Player 1 and Player 3 (opponents) use a fixed baseline agent.
    
    This trains PPO to win as declarer against competent opponents,
    avoiding the self-play trap where agents learn quirks rather than strategy.
    """
    
    def __init__(self, num_envs: int, opponent_agent: 'BaseAgent', seed: int = 0, reward_mode: str = "sparse"):
        self.num_envs = num_envs
        self.reward_mode = reward_mode
        self.envs = [make_env(seed=seed + i, reward_mode=reward_mode) for i in range(num_envs)]
        self.opponent = opponent_agent
        
        # Track current agent per env
        self.current_agents = ["player_0"] * num_envs
        
        # Players controlled by PPO (declarer team): 0 and 2
        # Players controlled by opponent: 1 and 3
        self.ppo_players = {0, 2}
        self.opponent_players = {1, 3}
        
    def _get_opponent_action(self, env: SchafkopfEnv, player_idx: int) -> int:
        """Get action from the opponent agent."""
        hand = env._hands[player_idx]
        legal_indices = env._get_legal_actions(player_idx)
        legal_actions = [IDX_TO_CARD[idx] for idx in legal_indices]
        
        # Build current trick as list of (player_idx, card)
        current_trick = [(p, c) for p, c in env._current_trick]
        
        # Get opponent's card choice
        card = self.opponent.select_action(
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
        
        return CARD_TO_IDX[card]
    
    def _advance_to_ppo_turn(self, env_idx: int) -> Tuple[bool, Optional[dict]]:
        """
        Advance the environment until it's a PPO player's turn.
        Returns (game_ended, info_if_ended).
        """
        env = self.envs[env_idx]
        
        while True:
            if all(env.terminations.values()):
                return True, {"game_result": env.get_game_result()}
            
            current_agent = env.agent_selection
            player_idx = int(current_agent.split("_")[1])
            
            if player_idx in self.ppo_players:
                # PPO's turn - stop advancing
                self.current_agents[env_idx] = current_agent
                return False, None
            
            # Opponent's turn - let baseline play
            action = self._get_opponent_action(env, player_idx)
            env.step(action)
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Reset all environments and advance to PPO's turn."""
        obs_batch = {
            "own_hand": [],
            "played_cards": [],
            "current_trick": [],
            "points": [],
            "game_info": [],
        }
        action_masks = []
        
        for i, env in enumerate(self.envs):
            env.reset()
            
            # Advance to PPO's turn (player 0 should be first anyway)
            self._advance_to_ppo_turn(i)
            
            obs = env.observe(self.current_agents[i])
            
            for key in obs_batch:
                obs_batch[key].append(obs[key])
            action_masks.append(obs["action_mask"])
        
        return (
            {k: np.array(v) for k, v in obs_batch.items()},
            np.array(action_masks),
        )
    
    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step all environments with PPO actions, then advance through opponent turns."""
        obs_batch = {
            "own_hand": [],
            "played_cards": [],
            "current_trick": [],
            "points": [],
            "game_info": [],
        }
        action_masks = []
        rewards = []
        dones = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            # Step with PPO's action
            env.step(int(action))
            
            # Get reward for the PPO agent that just acted
            agent = self.current_agents[i]
            reward = env._cumulative_rewards[agent]
            env._cumulative_rewards[agent] = 0
            
            # Advance through opponent turns
            game_ended, info = self._advance_to_ppo_turn(i)
            
            if game_ended:
                # Game finished - reset and collect final rewards
                result = info["game_result"]
                
                # Give the declarer team's result as reward
                # (PPO is always declarer team in this mode)
                final_reward = 1.0 if result["win"] else -1.0
                reward = final_reward  # Override with game outcome
                
                env.reset()
                self._advance_to_ppo_turn(i)
                done = True
            else:
                info = {}
                done = False
            
            obs = env.observe(self.current_agents[i])
            
            for key in obs_batch:
                obs_batch[key].append(obs[key])
            action_masks.append(obs["action_mask"])
            rewards.append(reward)
            dones.append(done)
            infos.append(info if info else {})
        
        return (
            {k: np.array(v) for k, v in obs_batch.items()},
            np.array(action_masks),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            infos,
        )


class VectorizedSchafkopfEnvSelfPlayBidding:
    """
    Vectorized environment for full self-play.
    
    PPO plays ALL 4 positions - learning both as declarer and opponent.
    
    Modes:
    - use_bidding=True: Best hand declares (inflated win rate ~77%)
    - use_bidding=False: Random player declares (realistic ~50% baseline)
    
    This teaches the agent:
    - Offensive play (as declarer team)
    - Defensive play (as opponent team)  
    """
    
    def __init__(self, num_envs: int, seed: int = 0, reward_mode: str = "sparse", use_bidding: bool = False):
        self.num_envs = num_envs
        self.reward_mode = reward_mode
        self.use_bidding = use_bidding
        self.rng = np.random.default_rng(seed)
        self.envs = [make_env(seed=seed + i, reward_mode=reward_mode) for i in range(num_envs)]
        
        # Track current agent and game state per env
        self.current_agents = ["player_0"] * num_envs
        
        # Track team assignment per env for reward calculation
        # These get set after reset
        self.declarer_teams = [set() for _ in range(num_envs)]
        
    def _get_team(self, env: SchafkopfEnv) -> set:
        """Get the declarer team player indices."""
        team = {env._declarer_idx}
        if env._partner_idx is not None:
            team.add(env._partner_idx)
        return team
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Reset all environments."""
        obs_batch = {
            "own_hand": [],
            "played_cards": [],
            "current_trick": [],
            "points": [],
            "game_info": [],
        }
        action_masks = []
        
        for i, env in enumerate(self.envs):
            if self.use_bidding:
                # Use bidding (best hand declares)
                env.reset()
            else:
                # Random declarer (more realistic, ~50% baseline)
                random_declarer = int(self.rng.integers(0, 4))
                env.reset(options={"fixed_declarer": random_declarer})
            
            # Track team assignments for this game
            self.declarer_teams[i] = self._get_team(env)
            
            self.current_agents[i] = env.agent_selection
            obs = env.observe(self.current_agents[i])
            
            for key in obs_batch:
                obs_batch[key].append(obs[key])
            action_masks.append(obs["action_mask"])
        
        return (
            {k: np.array(v) for k, v in obs_batch.items()},
            np.array(action_masks),
        )
    
    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step all environments - PPO plays all positions."""
        obs_batch = {
            "own_hand": [],
            "played_cards": [],
            "current_trick": [],
            "points": [],
            "game_info": [],
        }
        action_masks = []
        rewards = []
        dones = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            # Step with PPO's action
            env.step(int(action))
            
            # Get reward for the agent that just acted
            agent = self.current_agents[i]
            reward = env._cumulative_rewards[agent]
            env._cumulative_rewards[agent] = 0
            
            # Check if game ended
            if all(env.terminations.values()):
                result = env.get_game_result()
                
                # Get which team the last acting player was on
                player_idx = int(agent.split("_")[1])
                was_declarer_team = player_idx in self.declarer_teams[i]
                
                # Final reward: +1 for win, -1 for loss (from player's perspective)
                if was_declarer_team:
                    final_reward = 1.0 if result["win"] else -1.0
                else:
                    final_reward = -1.0 if result["win"] else 1.0
                
                reward = final_reward
                
                # Reset environment for next game
                if self.use_bidding:
                    env.reset()
                else:
                    random_declarer = int(self.rng.integers(0, 4))
                    env.reset(options={"fixed_declarer": random_declarer})
                
                self.declarer_teams[i] = self._get_team(env)
                done = True
                info = {"game_result": result, "was_declarer_team": was_declarer_team}
            else:
                done = False
                info = {}
            
            # Move to next agent
            self.current_agents[i] = env.agent_selection
            obs = env.observe(self.current_agents[i])
            
            for key in obs_batch:
                obs_batch[key].append(obs[key])
            action_masks.append(obs["action_mask"])
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return (
            {k: np.array(v) for k, v in obs_batch.items()},
            np.array(action_masks),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            infos,
        )


class VectorizedSchafkopfEnvFocused:
    """
    FOCUSED training environment: 100% PPO vs RuleBased.
    
    - No self-play confusion - clear objective
    - Random declarer (fair, no bidding advantage)
    - PPO plays on EITHER team (50% declarer, 50% opponent)
    - Learns both offense and defense against competent opponent
    """
    
    def __init__(self, num_envs: int, seed: int = 0, reward_mode: str = "sparse"):
        from .baseline import RuleBasedAgent
        
        self.num_envs = num_envs
        self.reward_mode = reward_mode
        self.rng = np.random.default_rng(seed)
        self.opponent = RuleBasedAgent()
        
        self.envs = [make_env(seed=seed + i, reward_mode=reward_mode) for i in range(num_envs)]
        
        # Track per-environment state
        self.current_agents = ["player_0"] * num_envs
        self.declarer_teams = [set() for _ in range(num_envs)]
        self.ppo_players = [set() for _ in range(num_envs)]
        self.ppo_is_declarer = [True] * num_envs
    
    def _get_team(self, env: SchafkopfEnv) -> set:
        """Get the declarer team player indices."""
        team = {env._declarer_idx}
        if env._partner_idx is not None:
            team.add(env._partner_idx)
        return team
    
    def _setup_game(self, env_idx: int):
        """Set up a new game with random declarer and team assignment."""
        env = self.envs[env_idx]
        
        # Random declarer for fair training
        random_declarer = int(self.rng.integers(0, 4))
        env.reset(options={"fixed_declarer": random_declarer})
        
        self.declarer_teams[env_idx] = self._get_team(env)
        
        # 50/50 chance: PPO plays declarer team or opponent team
        self.ppo_is_declarer[env_idx] = (self.rng.random() < 0.5)
        if self.ppo_is_declarer[env_idx]:
            self.ppo_players[env_idx] = self.declarer_teams[env_idx].copy()
        else:
            self.ppo_players[env_idx] = {0, 1, 2, 3} - self.declarer_teams[env_idx]
    
    def _get_opponent_action(self, env: SchafkopfEnv, player_idx: int) -> int:
        """Get action from the RuleBased opponent."""
        hand = env._hands[player_idx]
        legal_indices = env._get_legal_actions(player_idx)
        legal_actions = [IDX_TO_CARD[idx] for idx in legal_indices]
        current_trick = [(p, c) for p, c in env._current_trick]
        
        card = self.opponent.select_action(
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
        return CARD_TO_IDX[card]
    
    def _advance_to_ppo_turn(self, env_idx: int) -> Tuple[bool, Optional[dict]]:
        """Advance env until it's a PPO player's turn."""
        env = self.envs[env_idx]
        
        while True:
            if all(env.terminations.values()):
                return True, {"game_result": env.get_game_result()}
            
            current_agent = env.agent_selection
            player_idx = int(current_agent.split("_")[1])
            
            if player_idx in self.ppo_players[env_idx]:
                self.current_agents[env_idx] = current_agent
                return False, None
            
            # RuleBased opponent's turn
            action = self._get_opponent_action(env, player_idx)
            env.step(action)
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Reset all environments."""
        obs_batch = {
            "own_hand": [],
            "played_cards": [],
            "current_trick": [],
            "points": [],
            "game_info": [],
        }
        action_masks = []
        
        for i in range(self.num_envs):
            self._setup_game(i)
            self._advance_to_ppo_turn(i)
            
            obs = self.envs[i].observe(self.current_agents[i])
            
            for key in obs_batch:
                obs_batch[key].append(obs[key])
            action_masks.append(obs["action_mask"])
        
        return (
            {k: np.array(v) for k, v in obs_batch.items()},
            np.array(action_masks),
        )
    
    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step all environments."""
        obs_batch = {
            "own_hand": [],
            "played_cards": [],
            "current_trick": [],
            "points": [],
            "game_info": [],
        }
        action_masks = []
        rewards = []
        dones = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            # Execute PPO's action
            env.step(int(action))
            
            # Get intermediate reward (from dense shaping)
            agent = self.current_agents[i]
            reward = env._cumulative_rewards[agent]
            env._cumulative_rewards[agent] = 0
            
            # Advance through opponent turns
            game_ended, end_info = self._advance_to_ppo_turn(i)
            
            if game_ended:
                result = end_info["game_result"]
                declarer_won = result["win"]
                
                # Reward from PPO's team perspective
                if self.ppo_is_declarer[i]:
                    ppo_won = declarer_won
                else:
                    ppo_won = not declarer_won
                
                # Terminal reward
                reward = 1.0 if ppo_won else -1.0
                
                # Reset for next game
                self._setup_game(i)
                self._advance_to_ppo_turn(i)
                
                done = True
                info = {
                    "game_result": result,
                    "ppo_won": ppo_won,
                    "ppo_is_declarer": self.ppo_is_declarer[i],
                }
            else:
                done = False
                info = {}
            
            obs = self.envs[i].observe(self.current_agents[i])
            
            for key in obs_batch:
                obs_batch[key].append(obs[key])
            action_masks.append(obs["action_mask"])
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return (
            {k: np.array(v) for k, v in obs_batch.items()},
            np.array(action_masks),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            infos,
        )


class VectorizedSchafkopfEnvMixed:
    """
    Vectorized environment with MIXED training:
    - 50% pure self-play (PPO plays all 4 positions)
    - 50% vs RuleBased (PPO plays 2 positions, RuleBased plays 2)
    
    When playing vs RuleBased, PPO can be on EITHER team (declarer or opponent).
    This teaches the agent both offensive and defensive play against competent opponents.
    """
    
    def __init__(self, num_envs: int, seed: int = 0, reward_mode: str = "sparse", selfplay_ratio: float = 0.5):
        from .baseline import RuleBasedAgent
        
        self.num_envs = num_envs
        self.reward_mode = reward_mode
        self.selfplay_ratio = selfplay_ratio
        self.rng = np.random.default_rng(seed)
        self.opponent = RuleBasedAgent()
        
        self.envs = [make_env(seed=seed + i, reward_mode=reward_mode) for i in range(num_envs)]
        
        # Track per-environment state
        self.current_agents = ["player_0"] * num_envs
        self.declarer_teams = [set() for _ in range(num_envs)]
        
        # Track game mode per env: "selfplay" or "vs_opponent"
        self.game_modes = ["selfplay"] * num_envs
        
        # When vs_opponent: which players does PPO control? (can be declarer or opponent team)
        self.ppo_players = [set() for _ in range(num_envs)]
    
    def _get_team(self, env: SchafkopfEnv) -> set:
        """Get the declarer team player indices."""
        team = {env._declarer_idx}
        if env._partner_idx is not None:
            team.add(env._partner_idx)
        return team
    
    def _setup_game(self, env_idx: int):
        """Set up a new game - decide mode and team assignments."""
        env = self.envs[env_idx]
        
        # Random declarer
        random_declarer = int(self.rng.integers(0, 4))
        env.reset(options={"fixed_declarer": random_declarer})
        
        self.declarer_teams[env_idx] = self._get_team(env)
        
        # Decide game mode
        if self.rng.random() < self.selfplay_ratio:
            # Pure self-play: PPO plays all positions
            self.game_modes[env_idx] = "selfplay"
            self.ppo_players[env_idx] = {0, 1, 2, 3}
        else:
            # vs RuleBased: randomly assign PPO to declarer OR opponent team
            self.game_modes[env_idx] = "vs_opponent"
            if self.rng.random() < 0.5:
                # PPO is declarer team
                self.ppo_players[env_idx] = self.declarer_teams[env_idx].copy()
            else:
                # PPO is opponent team
                self.ppo_players[env_idx] = {0, 1, 2, 3} - self.declarer_teams[env_idx]
    
    def _get_opponent_action(self, env: SchafkopfEnv, player_idx: int) -> int:
        """Get action from the RuleBased opponent."""
        hand = env._hands[player_idx]
        legal_indices = env._get_legal_actions(player_idx)
        legal_actions = [IDX_TO_CARD[idx] for idx in legal_indices]
        current_trick = [(p, c) for p, c in env._current_trick]
        
        card = self.opponent.select_action(
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
        return CARD_TO_IDX[card]
    
    def _advance_to_ppo_turn(self, env_idx: int) -> Tuple[bool, Optional[dict]]:
        """Advance env until it's a PPO player's turn (in vs_opponent mode)."""
        env = self.envs[env_idx]
        
        while True:
            if all(env.terminations.values()):
                return True, {"game_result": env.get_game_result()}
            
            current_agent = env.agent_selection
            player_idx = int(current_agent.split("_")[1])
            
            if player_idx in self.ppo_players[env_idx]:
                self.current_agents[env_idx] = current_agent
                return False, None
            
            # Opponent's turn
            action = self._get_opponent_action(env, player_idx)
            env.step(action)
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Reset all environments."""
        obs_batch = {
            "own_hand": [],
            "played_cards": [],
            "current_trick": [],
            "points": [],
            "game_info": [],
        }
        action_masks = []
        
        for i in range(self.num_envs):
            self._setup_game(i)
            
            # In vs_opponent mode, advance to PPO's turn
            if self.game_modes[i] == "vs_opponent":
                self._advance_to_ppo_turn(i)
            else:
                self.current_agents[i] = self.envs[i].agent_selection
            
            obs = self.envs[i].observe(self.current_agents[i])
            
            for key in obs_batch:
                obs_batch[key].append(obs[key])
            action_masks.append(obs["action_mask"])
        
        return (
            {k: np.array(v) for k, v in obs_batch.items()},
            np.array(action_masks),
        )
    
    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step all environments."""
        obs_batch = {
            "own_hand": [],
            "played_cards": [],
            "current_trick": [],
            "points": [],
            "game_info": [],
        }
        action_masks = []
        rewards = []
        dones = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            # Execute PPO's action
            env.step(int(action))
            
            # Get intermediate reward
            agent = self.current_agents[i]
            reward = env._cumulative_rewards[agent]
            env._cumulative_rewards[agent] = 0
            
            # Handle based on game mode
            if self.game_modes[i] == "selfplay":
                # Pure self-play: just check if game ended
                if all(env.terminations.values()):
                    result = env.get_game_result()
                    player_idx = int(agent.split("_")[1])
                    was_declarer = player_idx in self.declarer_teams[i]
                    
                    # Reward from this player's perspective
                    if was_declarer:
                        reward = 1.0 if result["win"] else -1.0
                    else:
                        reward = -1.0 if result["win"] else 1.0
                    
                    # Reset for next game
                    self._setup_game(i)
                    done = True
                    info = {"game_result": result, "mode": "selfplay", "was_declarer": was_declarer}
                else:
                    # Move to next agent
                    self.current_agents[i] = env.agent_selection
                    done = False
                    info = {}
            else:
                # vs_opponent mode: advance through opponent turns
                game_ended, end_info = self._advance_to_ppo_turn(i)
                
                if game_ended:
                    result = end_info["game_result"]
                    player_idx = int(agent.split("_")[1])
                    was_declarer = player_idx in self.declarer_teams[i]
                    
                    # Reward from this player's perspective
                    if was_declarer:
                        reward = 1.0 if result["win"] else -1.0
                    else:
                        reward = -1.0 if result["win"] else 1.0
                    
                    # Reset for next game
                    self._setup_game(i)
                    if self.game_modes[i] == "vs_opponent":
                        self._advance_to_ppo_turn(i)
                    else:
                        self.current_agents[i] = self.envs[i].agent_selection
                    
                    done = True
                    info = {"game_result": result, "mode": "vs_opponent", "was_declarer": was_declarer}
                else:
                    done = False
                    info = {}
            
            obs = self.envs[i].observe(self.current_agents[i])
            
            for key in obs_batch:
                obs_batch[key].append(obs[key])
            action_masks.append(obs["action_mask"])
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return (
            {k: np.array(v) for k, v in obs_batch.items()},
            np.array(action_masks),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            infos,
        )


class PPOTrainer:
    """PPO trainer for Schafkopf with self-play or opponent training."""
    
    def __init__(self, config: Optional[PPOConfig] = None):
        self.config = config or PPOConfig()
        self.device = torch.device(self.config.device)
        
        # Create network and optimizer - choose LSTM or feedforward
        if self.config.use_lstm:
            self.network = SchafkopfLSTMNetwork(
                self.config.hidden_size,
                self.config.lstm_hidden_size,
            ).to(self.device)
            self.use_lstm = True
            print(f"Network: LSTM (hidden={self.config.hidden_size}, lstm={self.config.lstm_hidden_size})")
        else:
            self.network = SchafkopfNetwork(self.config.hidden_size).to(self.device)
            self.use_lstm = False
            print(f"Network: Feedforward (hidden={self.config.hidden_size})")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.network.parameters())
        print(f"Total parameters: {total_params:,}")
        
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=self.config.learning_rate,
            eps=1e-5,
        )
        
        # Create vectorized environments based on opponent config
        reward_mode = self.config.reward_mode
        print(f"Reward mode: {reward_mode}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Entropy coefficient: {self.config.entropy_coef}")
        
        if self.config.opponent_type == "focused":
            # FOCUSED training: 100% vs RuleBased, no self-play
            self.envs = VectorizedSchafkopfEnvFocused(
                self.config.num_envs,
                reward_mode=reward_mode,
            )
            self.training_mode = "focused"
            print("Training mode: FOCUSED (100% vs RuleBased)")
            print("  - Random declarer (fair, no bidding advantage)")
            print("  - PPO on either team (50% declarer, 50% opponent)")
        elif self.config.opponent_type == "mixed":
            # Mixed training: 50% self-play, 50% vs RuleBased
            self.envs = VectorizedSchafkopfEnvMixed(
                self.config.num_envs,
                reward_mode=reward_mode,
                selfplay_ratio=0.5,
            )
            self.training_mode = "mixed"
            print("Training mode: MIXED (50% self-play + 50% vs RuleBased)")
            print("  - Self-play: PPO plays all 4 positions")
            print("  - Vs RuleBased: PPO plays 2 positions (either team)")
        elif self.config.opponent_type == "bidding":
            # Full self-play - PPO plays all positions with RANDOM declarer
            # use_bidding=False ensures ~50% baseline (not inflated 77%)
            self.envs = VectorizedSchafkopfEnvSelfPlayBidding(
                self.config.num_envs,
                reward_mode=reward_mode,
                use_bidding=False,  # Random declarer for fair training
            )
            self.training_mode = "selfplay_all_positions"
            print("Training mode: Self-play ALL positions (random declarer, ~50% baseline)")
        elif self.config.opponent_type == "rulebased":
            # Training against fixed opponent
            from .baseline import RuleBasedAgent
            opponent = RuleBasedAgent()
            self.envs = VectorizedSchafkopfEnvWithOpponent(
                self.config.num_envs, 
                opponent_agent=opponent,
                reward_mode=reward_mode,
            )
            self.training_mode = f"vs_{self.config.opponent_type}"
            print(f"Training mode: PPO vs {opponent.name}")
        else:
            # Standard self-play training (no bidding)
            self.envs = VectorizedSchafkopfEnv(
                self.config.num_envs,
                reward_mode=reward_mode,
            )
            self.training_mode = "self_play"
            print("Training mode: Self-play (legacy, no bidding)")
        
        # Create rollout buffer (with LSTM support)
        self.buffer = RolloutBuffer(
            self.config.num_steps,
            self.config.num_envs,
            self.device,
            use_lstm=self.use_lstm,
            lstm_hidden_size=self.config.lstm_hidden_size if self.use_lstm else 0,
        )
        
        # LSTM hidden states for each environment
        if self.use_lstm:
            self.hidden_states = self.network.get_initial_hidden(
                self.config.num_envs, self.device
            )
        else:
            self.hidden_states = None
        
        # Tracking
        self.global_step = 0
        self.update_count = 0
        self.episode_rewards = []
        self.episode_wins = []
        self.training_history = []  # For logging metrics over time
        
        # Create save directory
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
    
    def _obs_to_tensor(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert observation dict to tensors."""
        return {
            k: torch.tensor(v, dtype=torch.float32, device=self.device)
            for k, v in obs.items()
        }
    
    def collect_rollout(self):
        """Collect a rollout of experience."""
        self.buffer.reset()
        obs, action_mask = self.envs.reset()
        
        # Reset LSTM hidden states at start of rollout
        if self.use_lstm:
            self.hidden_states = self.network.get_initial_hidden(
                self.config.num_envs, self.device
            )
        
        for step in range(self.config.num_steps):
            with torch.no_grad():
                obs_tensor = self._obs_to_tensor(obs)
                mask_tensor = torch.tensor(
                    action_mask, dtype=torch.float32, device=self.device
                )
                
                if self.use_lstm:
                    action, log_prob, _, value, new_hidden = self.network.get_action_and_value(
                        obs_tensor, mask_tensor, self.hidden_states
                    )
                else:
                    action, log_prob, _, value = self.network.get_action_and_value(
                        obs_tensor, mask_tensor
                    )
                    new_hidden = None
            
            action_np = action.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            value_np = value.cpu().numpy()
            
            # Step environments
            next_obs, next_mask, reward, done, infos = self.envs.step(action_np)
            
            # Track episode stats
            for i, info in enumerate(infos):
                if "game_result" in info:
                    result = info["game_result"]
                    self.episode_rewards.append(result["declarer_points"])
                    self.episode_wins.append(1.0 if result["win"] else 0.0)
                    
                    # Reset LSTM hidden state for this env when game ends
                    if self.use_lstm and done[i]:
                        h, c = self.hidden_states
                        h[:, i, :] = 0
                        c[:, i, :] = 0
            
            # Store transition (with hidden state for LSTM)
            self.buffer.add(
                obs, action_mask, action_np, log_prob_np, reward, done, value_np,
                hidden=self.hidden_states if self.use_lstm else None
            )
            
            # Update hidden states for next step
            if self.use_lstm:
                self.hidden_states = new_hidden
            
            obs = next_obs
            action_mask = next_mask
            self.global_step += self.config.num_envs
        
        # Compute last value for GAE
        with torch.no_grad():
            obs_tensor = self._obs_to_tensor(obs)
            mask_tensor = torch.tensor(
                action_mask, dtype=torch.float32, device=self.device
            )
            if self.use_lstm:
                _, _, _, last_value, _ = self.network.get_action_and_value(
                    obs_tensor, mask_tensor, self.hidden_states
                )
            else:
                _, _, _, last_value = self.network.get_action_and_value(
                    obs_tensor, mask_tensor
                )
        
        self.buffer.compute_returns_and_advantages(
            last_value.cpu().numpy(),
            self.config.gamma,
            self.config.gae_lambda,
        )
        
        return obs, action_mask
    
    def update(self) -> Dict[str, float]:
        """Perform PPO update."""
        total_pg_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_loss = 0
        num_batches = 0
        
        for epoch in range(self.config.num_epochs):
            for (
                obs_batch,
                mask_batch,
                action_batch,
                old_log_prob_batch,
                advantage_batch,
                return_batch,
            ) in self.buffer.get_batches(self.config.batch_size):
                
                # Normalize advantages
                advantage_batch = (advantage_batch - advantage_batch.mean()) / (
                    advantage_batch.std() + 1e-8
                )
                
                # Get current policy outputs
                # Note: For LSTM, we use fresh hidden states in training
                # (this is a simplification - full LSTM-PPO would use sequence-based training)
                if self.use_lstm:
                    batch_size = obs_batch["own_hand"].shape[0]
                    hidden = self.network.get_initial_hidden(batch_size, self.device)
                    _, new_log_prob, entropy, new_value, _ = self.network.get_action_and_value(
                        obs_batch, mask_batch, hidden, action_batch
                    )
                else:
                    _, new_log_prob, entropy, new_value = self.network.get_action_and_value(
                        obs_batch, mask_batch, action_batch
                    )
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_prob - old_log_prob_batch)
                pg_loss1 = -advantage_batch * ratio
                pg_loss2 = -advantage_batch * torch.clamp(
                    ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_value, return_batch)
                
                # Entropy bonus
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = (
                    pg_loss 
                    + self.config.value_loss_coef * value_loss 
                    - self.config.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), 
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                total_pg_loss += pg_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_loss += loss.item()
                num_batches += 1
        
        self.update_count += 1
        
        return {
            "pg_loss": total_pg_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "entropy": total_entropy / num_batches,
            "total_loss": total_loss / num_batches,
        }
    
    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate the current policy against appropriate opponents."""
        if self.config.opponent_type:
            # Evaluate against the RuleBasedAgent
            return self._evaluate_vs_opponent(num_episodes)
        else:
            # Evaluate in self-play
            return self._evaluate_self_play(num_episodes)
    
    def _evaluate_self_play(self, num_episodes: int) -> Dict[str, float]:
        """Evaluate in self-play mode (all 4 players use PPO)."""
        env = make_env()
        wins = 0
        total_points = 0
        
        for _ in range(num_episodes):
            env.reset()
            
            # Initialize LSTM hidden state for evaluation
            if self.use_lstm:
                hidden = self.network.get_initial_hidden(1, self.device)
            
            while not all(env.terminations.values()):
                agent = env.agent_selection
                obs = env.observe(agent)
                
                # Use policy
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
                
                env.step(action.item())
            
            result = env.get_game_result()
            if result["win"]:
                wins += 1
            total_points += result["declarer_points"]
        
        return {
            "eval_win_rate": wins / num_episodes,
            "eval_avg_points": total_points / num_episodes,
        }
    
    def _evaluate_vs_opponent(self, num_episodes: int) -> Dict[str, float]:
        """
        Evaluate PPO vs RuleBasedAgent with FAIR measurement.
        
        - Random declarer (no bidding advantage)
        - PPO plays BOTH as declarer team AND as opponent team (50/50)
        - Measures: how often PPO's team wins when PPO plays that team
        """
        from .baseline import RuleBasedAgent
        
        env = make_env()
        opponent = RuleBasedAgent()
        rng = np.random.default_rng(42)  # Fixed seed for reproducible eval
        
        wins = 0
        total_points = 0
        
        for ep in range(num_episodes):
            # Random declarer for fair evaluation
            random_declarer = int(rng.integers(0, 4))
            env.reset(options={"fixed_declarer": random_declarer})
            
            # Get actual declarer team
            declarer_team = {env._declarer_idx}
            if env._partner_idx is not None:
                declarer_team.add(env._partner_idx)
            opponent_team = {0, 1, 2, 3} - declarer_team
            
            # Alternate: PPO plays declarer team in even episodes, opponent team in odd
            ppo_is_declarer = (ep % 2 == 0)
            ppo_players = declarer_team if ppo_is_declarer else opponent_team
            
            # Initialize LSTM hidden state for evaluation
            if self.use_lstm:
                hidden = self.network.get_initial_hidden(1, self.device)
            
            while not all(env.terminations.values()):
                agent = env.agent_selection
                player_idx = int(agent.split("_")[1])
                obs = env.observe(agent)
                
                if player_idx in ppo_players:
                    # PPO's turn
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
                    # Opponent's turn
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
            # "win" is from declarer's perspective
            declarer_won = result["win"]
            
            # Did PPO's team win?
            if ppo_is_declarer:
                ppo_won = declarer_won
            else:
                ppo_won = not declarer_won
            
            if ppo_won:
                wins += 1
            total_points += result["declarer_points"]
        
        return {
            "eval_win_rate": wins / num_episodes,
            "eval_avg_points": total_points / num_episodes,
        }
    
    def save(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(
                self.config.save_dir, 
                f"schafkopf_ppo_{self.global_step}.pt"
            )
        
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "update_count": self.update_count,
            "config": self.config,
        }, path)
        
        return path
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.update_count = checkpoint["update_count"]
    
    def train(self):
        """Main training loop."""
        print(f"Starting PPO training on {self.config.device}")
        print(f"Total timesteps: {self.config.total_timesteps:,}")
        print(f"Environments: {self.config.num_envs}")
        print(f"Steps per rollout: {self.config.num_steps}")
        print("-" * 50)
        
        start_time = time.time()
        obs, action_mask = self.envs.reset()
        
        num_updates = self.config.total_timesteps // (
            self.config.num_envs * self.config.num_steps
        )
        
        for update in range(1, num_updates + 1):
            # Collect rollout
            obs, action_mask = self.collect_rollout()
            
            # Update policy
            losses = self.update()
            
            # Logging
            if update % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                fps = self.global_step / elapsed
                
                avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                avg_win_rate = np.mean(self.episode_wins[-100:]) if self.episode_wins else 0
                
                # Record metrics for plotting
                self.training_history.append({
                    'update': update,
                    'timestep': self.global_step,
                    'win_rate': avg_win_rate,
                    'avg_points': avg_reward,
                    'pg_loss': losses['pg_loss'],
                    'value_loss': losses['value_loss'],
                    'entropy': losses['entropy'],
                    'total_loss': losses['total_loss'],
                    'fps': fps,
                })
                
                print(
                    f"Update {update}/{num_updates} | "
                    f"Step {self.global_step:,} | "
                    f"FPS {fps:.0f} | "
                    f"Win Rate {avg_win_rate:.1%} | "
                    f"Avg Points {avg_reward:.1f} | "
                    f"Loss {losses['total_loss']:.4f} | "
                    f"Entropy {losses['entropy']:.4f}"
                )
            
            # Evaluation
            if update % self.config.eval_interval == 0:
                eval_stats = self.evaluate(self.config.eval_episodes)
                print(
                    f"  [Eval] Win Rate: {eval_stats['eval_win_rate']:.1%} | "
                    f"Avg Points: {eval_stats['eval_avg_points']:.1f}"
                )
                # Add eval metrics to last history entry
                if self.training_history:
                    self.training_history[-1]['eval_win_rate'] = eval_stats['eval_win_rate']
                    self.training_history[-1]['eval_avg_points'] = eval_stats['eval_avg_points']
            
            # Save checkpoint
            if update % self.config.save_interval == 0:
                path = self.save()
                print(f"  [Save] Checkpoint saved to {path}")
        
        # Final save - include training mode in filename
        suffix = f"_{self.training_mode}" if self.training_mode != "self_play" else ""
        final_path = self.save(
            os.path.join(self.config.save_dir, f"schafkopf_ppo{suffix}_final.pt")
        )
        print(f"\nTraining complete! Final model saved to {final_path}")
        
        # Save training history to CSV
        if self.training_history:
            import pandas as pd
            history_path = os.path.join(self.config.save_dir, f"training_history{suffix}.csv")
            df = pd.DataFrame(self.training_history)
            df.to_csv(history_path, index=False)
            print(f"Training history saved to {history_path}")
        
        return self.network


def train_ppo(
    total_timesteps: int = 500_000,
    num_envs: int = 8,
    learning_rate: float = 1e-4,
    save_dir: str = "checkpoints",
    opponent_type: Optional[str] = None,
    reward_mode: str = "sparse",
    use_lstm: bool = True,
    hidden_size: int = 512,
    entropy_coef: float = 0.05,
    init_from: Optional[str] = None,
    **kwargs,
) -> PPOTrainer:
    """
    Convenience function to train PPO agent.
    
    Args:
        total_timesteps: Total training timesteps
        num_envs: Number of parallel environments
        learning_rate: Learning rate for optimizer (default: 1e-4)
        save_dir: Directory to save checkpoints
        opponent_type: Type of opponent for training. Options:
            - None: Self-play training
            - "rulebased": Train against RuleBasedAgent
            - "bidding": Self-play with bidding (all positions)
            - "focused": 100% vs RuleBased (recommended)
        reward_mode: Reward shaping mode. Options:
            - "sparse": +1/-1 at game end only
            - "shaped": Intermediate rewards based on points won
            - "dense": Full reward shaping with bonuses
        use_lstm: Whether to use LSTM memory (default: True)
        hidden_size: Hidden layer size (default: 512)
        entropy_coef: Entropy coefficient for exploration (default: 0.05)
        init_from: Path to imitation model to initialize from
    """
    config = PPOConfig(
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        learning_rate=learning_rate,
        save_dir=save_dir,
        opponent_type=opponent_type,
        reward_mode=reward_mode,
        use_lstm=use_lstm,
        hidden_size=hidden_size,
        entropy_coef=entropy_coef,
        **kwargs,
    )
    
    trainer = PPOTrainer(config)
    
    # Load imitation model weights if specified
    if init_from:
        print(f"Initializing from imitation model: {init_from}")
        checkpoint = torch.load(init_from, map_location=trainer.device)
        # Load only matching keys (imitation model might be smaller)
        model_dict = trainer.network.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint["network_state_dict"].items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        trainer.network.load_state_dict(model_dict)
        print(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} layers from imitation model")
    
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    # Quick training run
    train_ppo(total_timesteps=100_000, num_envs=4)
