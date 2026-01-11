"""PettingZoo AEC environment for Schafkopf Sauspiel."""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional

import numpy as np
from gymnasium import spaces

from .cards import (
    FULL_DECK,
    TRUMP_STRENGTH,
    card_value,
    is_trump,
    lead_context,
    rank_of,
    suit_of,
    trump_rank,
    RANK_PRIORITY,
)
from .features import featurize_hand

# Card to index mapping
CARD_TO_IDX = {card: idx for idx, card in enumerate(FULL_DECK)}
IDX_TO_CARD = {idx: card for card, idx in CARD_TO_IDX.items()}

# PettingZoo import - we'll use a lightweight implementation that doesn't require the full package
try:
    from pettingzoo import AECEnv
    from pettingzoo.utils import agent_selector
    PETTINGZOO_AVAILABLE = True
except ImportError:
    PETTINGZOO_AVAILABLE = False
    AECEnv = object  # Fallback for type hints


class SchafkopfEnv(AECEnv if PETTINGZOO_AVAILABLE else object):
    """
    PettingZoo AEC Environment for Schafkopf Sauspiel.
    
    This environment implements the Sauspiel (partner game) variant where:
    - 4 players each receive 8 cards from a 32-card deck
    - Player 0 is always the declarer
    - The declarer calls an ace to determine their hidden partner
    - Teams try to win tricks and accumulate points (61+ to win)
    
    Observation Space:
        - own_hand: [32] binary - which cards the player holds
        - played_cards: [32] binary - which cards have been played
        - current_trick: [4, 32] one-hot - cards in current trick by position
        - trick_winner_history: [8] int - who won each trick
        - points_won: [4] float - normalized points per player
        - game_info: [8] float - trump count, game phase, position, etc.
    
    Action Space:
        - Discrete(32) - index of card to play (masked to legal actions)
    
    Rewards:
        - Sparse: +1/-1 at game end for winning/losing team
        - Can be configured for shaped rewards (points per trick)
    """
    
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "schafkopf_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        reward_mode: str = "sparse",  # "sparse", "shaped", or "dense"
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Track running point differential for dense rewards
        self._prev_declarer_points = 0
        
        # Agent setup
        self.possible_agents = ["player_0", "player_1", "player_2", "player_3"]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}
        
        # Game state (initialized in reset)
        self._hands: List[List[str]] = []
        self._played_cards: List[str] = []
        self._current_trick: List[tuple] = []  # List of (player_idx, card)
        self._trick_history: List[List[tuple]] = []
        self._points: List[int] = [0, 0, 0, 0]
        self._trick_number: int = 0
        self._leader: int = 0
        self._called_ace: str = ""
        self._partner_idx: Optional[int] = None
        self._declarer_idx: int = 0
        
        # PettingZoo state
        self.agents: List[str] = []
        self._agent_selector: Any = None
        self.agent_selection: str = ""
        self.rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.infos: Dict[str, dict] = {}
        self._cumulative_rewards: Dict[str, float] = {}
        
    @functools.lru_cache(maxsize=4)
    def observation_space(self, agent: str) -> spaces.Dict:
        """Return the observation space for an agent."""
        return spaces.Dict({
            # Cards in hand (32 binary)
            "own_hand": spaces.MultiBinary(32),
            # Cards that have been played (32 binary)
            "played_cards": spaces.MultiBinary(32),
            # Current trick cards by position (4 players x 32 cards one-hot)
            "current_trick": spaces.MultiBinary((4, 32)),
            # Points won by each player (normalized 0-1)
            "points": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
            # Game info: [trick_num, position_in_trick, is_declarer, is_partner_known,
            #             trump_count, hand_strength, color_aces, called_ace_suit]
            "game_info": spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32),
            # Legal action mask
            "action_mask": spaces.MultiBinary(32),
        })
    
    @functools.lru_cache(maxsize=4)
    def action_space(self, agent: str) -> spaces.Discrete:
        """Return the action space - discrete choice of 32 possible cards."""
        return spaces.Discrete(32)
    
    def _get_legal_actions(self, player_idx: int) -> List[int]:
        """Get indices of legal cards for the given player."""
        hand = self._hands[player_idx]
        if not hand:
            return []
        
        # If leading, any card is legal
        if not self._current_trick:
            return [CARD_TO_IDX[card] for card in hand]
        
        # Must follow lead context if possible
        lead_card = self._current_trick[0][1]
        lead_ctx = lead_context(lead_card)
        
        if lead_ctx == "TRUMP":
            # Must play trump if possible
            trumps = [c for c in hand if is_trump(c)]
            if trumps:
                return [CARD_TO_IDX[c] for c in trumps]
        else:
            # Must follow suit if possible (non-trump cards of that suit)
            same_suit = [c for c in hand if suit_of(c) == lead_ctx and not is_trump(c)]
            if same_suit:
                return [CARD_TO_IDX[c] for c in same_suit]
        
        # Can play anything if can't follow
        return [CARD_TO_IDX[card] for card in hand]
    
    def _get_action_mask(self, player_idx: int) -> np.ndarray:
        """Return binary mask of legal actions."""
        mask = np.zeros(32, dtype=np.int8)
        legal = self._get_legal_actions(player_idx)
        mask[legal] = 1
        return mask
    
    def _encode_observation(self, player_idx: int) -> Dict[str, np.ndarray]:
        """Encode the current game state as an observation for the given player."""
        # Own hand
        own_hand = np.zeros(32, dtype=np.int8)
        for card in self._hands[player_idx]:
            own_hand[CARD_TO_IDX[card]] = 1
        
        # Played cards
        played_cards = np.zeros(32, dtype=np.int8)
        for card in self._played_cards:
            played_cards[CARD_TO_IDX[card]] = 1
        
        # Current trick
        current_trick = np.zeros((4, 32), dtype=np.int8)
        for p_idx, card in self._current_trick:
            current_trick[p_idx, CARD_TO_IDX[card]] = 1
        
        # Points (normalized by 120 total)
        points = np.array(self._points, dtype=np.float32) / 120.0
        
        # Game info
        hand_features = featurize_hand(self._hands[player_idx]) if self._hands[player_idx] else None
        trump_count = hand_features.trump_count if hand_features else 0
        color_aces = hand_features.color_aces if hand_features else 0
        hand_strength = (hand_features.trump_strength_sum / 100.0) if hand_features else 0
        
        # Called ace suit encoded (0=Acorn, 1=Leaf, 2=Bell, 3=unknown)
        called_suit = 3
        if self._called_ace:
            suit = suit_of(self._called_ace)
            called_suit = {"Acorn": 0, "Leaf": 1, "Bell": 2}.get(suit, 3)
        
        game_info = np.array([
            self._trick_number / 7.0,  # Normalize by max tricks
            len(self._current_trick) / 3.0,  # Position in trick
            1.0 if player_idx == self._declarer_idx else 0.0,
            1.0 if self._partner_idx is not None else 0.0,
            trump_count / 8.0,
            hand_strength,
            color_aces / 3.0,
            called_suit / 3.0,
        ], dtype=np.float32)
        
        return {
            "own_hand": own_hand,
            "played_cards": played_cards,
            "current_trick": current_trick,
            "points": points,
            "game_info": game_info,
            "action_mask": self._get_action_mask(player_idx),
        }
    
    def observe(self, agent: str) -> Dict[str, np.ndarray]:
        """Return observation for the specified agent."""
        player_idx = self.agent_name_mapping[agent]
        return self._encode_observation(player_idx)
    
    def _deal_cards(self) -> None:
        """Shuffle and deal 8 cards to each player."""
        deck = FULL_DECK.copy()
        self.rng.shuffle(deck)
        self._hands = [list(deck[i*8:(i+1)*8]) for i in range(4)]
    
    def _choose_called_ace(self) -> str:
        """Declarer chooses which ace to call (for partner)."""
        declarer_hand = self._hands[self._declarer_idx]
        # Call an ace the declarer doesn't have
        candidates = [f"A_{s}" for s in ("Acorn", "Leaf", "Bell") if f"A_{s}" not in declarer_hand]
        if not candidates:
            return "A_Acorn"  # Fallback (rare edge case)
        return str(self.rng.choice(candidates))
    
    def _find_partner(self) -> Optional[int]:
        """Find which player holds the called ace."""
        for idx, hand in enumerate(self._hands):
            if idx != self._declarer_idx and self._called_ace in hand:
                return idx
        return None
    
    def _evaluate_hand_strength(self, player_idx: int) -> float:
        """
        Evaluate hand strength for bidding decision.
        Returns a score from 0-1, where higher = stronger hand.
        
        A hand is strong for declaring if it has:
        - Many trumps (4+ is good, 6+ is excellent)
        - High trump strength (Obers and Unters)
        - Color aces to cash
        - Long suits to run
        """
        hand = self._hands[player_idx]
        features = featurize_hand(hand)
        
        # Trump count score (0-6+ trumps -> 0-1)
        trump_score = min(features.trump_count / 6.0, 1.0)
        
        # Trump strength score (sum of strengths, max ~100 for best hand)
        strength_score = min(features.trump_strength_sum / 80.0, 1.0)
        
        # Color aces (0-3 aces -> bonus)
        ace_score = features.color_aces / 3.0
        
        # Combined score (weighted average)
        score = (
            0.4 * trump_score +
            0.4 * strength_score +
            0.2 * ace_score
        )
        
        return score
    
    def _run_bidding(self) -> int:
        """
        Simple bidding: each player evaluates their hand, 
        strongest hand becomes declarer.
        
        Returns the index of the winning bidder (declarer).
        """
        scores = []
        for i in range(4):
            score = self._evaluate_hand_strength(i)
            # Add small random tiebreaker
            score += self.rng.random() * 0.01
            scores.append((score, i))
        
        # Sort by score descending
        scores.sort(reverse=True)
        
        # Player with best hand declares
        # But only if their score is above a threshold (simulates "passing")
        best_score, best_player = scores[0]
        
        # Minimum threshold to declare (prevents forcing weak hands to play)
        if best_score < 0.35:
            # All hands are weak - player 0 is forced to declare (Ramsch would be better but complex)
            return 0
        
        return best_player
    
    def _beats(self, challenger: str, incumbent: str, lead_ctx: str) -> bool:
        """Check if challenger card beats the incumbent card."""
        if is_trump(challenger):
            if not is_trump(incumbent):
                return True
            return trump_rank(challenger) < trump_rank(incumbent)
        
        if is_trump(incumbent):
            return False
        
        if lead_ctx == "TRUMP":
            return False
        
        if suit_of(challenger) == suit_of(incumbent) == lead_ctx:
            return RANK_PRIORITY[rank_of(challenger)] < RANK_PRIORITY[rank_of(incumbent)]
        
        if suit_of(challenger) == lead_ctx and suit_of(incumbent) != lead_ctx:
            return True
        
        return False
    
    def _resolve_trick(self) -> int:
        """Determine trick winner and award points. Returns winner index."""
        if not self._current_trick:
            raise RuntimeError("No cards in current trick")
        
        lead_ctx = lead_context(self._current_trick[0][1])
        winner_idx, winner_card = self._current_trick[0]
        
        for player_idx, card in self._current_trick[1:]:
            if self._beats(card, winner_card, lead_ctx):
                winner_idx, winner_card = player_idx, card
        
        # Calculate trick points
        trick_points = sum(card_value(card) for _, card in self._current_trick)
        self._points[winner_idx] += trick_points
        
        # Move trick cards to played pile
        for _, card in self._current_trick:
            self._played_cards.append(card)
        
        self._trick_history.append(list(self._current_trick))
        self._current_trick = []
        
        return winner_idx
    
    def _award_trick_rewards(self, winner_idx: int, prev_points: List[int]) -> None:
        """
        Award intermediate rewards after each trick (for dense/shaped mode).
        
        Reward structure:
        1. Point differential: reward proportional to points gained vs opponent
        2. High-value capture bonus: extra reward for capturing aces/tens
        3. Trick win bonus: small bonus for winning any trick
        4. Progress toward 61: bonus as declarer team approaches winning threshold
        """
        # Calculate current declarer team points
        curr_declarer_pts = self._points[self._declarer_idx]
        if self._partner_idx is not None:
            curr_declarer_pts += self._points[self._partner_idx]
        
        prev_declarer_pts = prev_points[self._declarer_idx]
        if self._partner_idx is not None:
            prev_declarer_pts += prev_points[self._partner_idx]
        
        # Points gained by each team this trick
        declarer_gained = curr_declarer_pts - prev_declarer_pts
        opponent_gained = sum(self._points) - sum(prev_points) - declarer_gained
        
        # Trick info
        trick_cards = self._trick_history[-1]
        trick_points = sum(card_value(c) for _, c in trick_cards)
        
        # Check for high-value cards captured (aces and tens)
        high_value_captured = sum(1 for _, c in trick_cards 
                                   if rank_of(c) in ('A', '10'))
        
        # Winner on declarer team?
        winner_is_declarer = (winner_idx == self._declarer_idx or 
                              winner_idx == self._partner_idx)
        
        # Award rewards to each agent
        for agent in self.possible_agents:
            idx = self.agent_name_mapping[agent]
            is_declarer_team = (idx == self._declarer_idx or idx == self._partner_idx)
            
            reward = 0.0
            
            if self.reward_mode == "dense":
                # --- DENSE REWARD SHAPING ---
                
                # 1. Point differential reward (scaled by 120 total points)
                if is_declarer_team:
                    reward += (declarer_gained - opponent_gained) / 120.0
                else:
                    reward += (opponent_gained - declarer_gained) / 120.0
                
                # 2. High-value capture bonus (when your team wins)
                if winner_is_declarer == is_declarer_team:
                    reward += high_value_captured * 0.02
                
                # 3. Trick win bonus (small constant)
                if winner_is_declarer == is_declarer_team:
                    reward += 0.01
                
                # 4. Progress toward 61 (declarer) or denying 61 (opponent)
                if is_declarer_team:
                    # Bonus for getting closer to 61
                    if curr_declarer_pts >= 61 and prev_declarer_pts < 61:
                        reward += 0.1  # Crossed winning threshold
                else:
                    # Bonus for keeping declarer below 61
                    opponent_pts = 120 - curr_declarer_pts
                    prev_opponent_pts = 120 - prev_declarer_pts
                    if opponent_pts >= 61 and prev_opponent_pts < 61:
                        reward += 0.1
            
            else:  # "shaped" mode - simpler rewards
                # Just point-based shaping
                if is_declarer_team:
                    reward += declarer_gained / 120.0
                else:
                    reward += opponent_gained / 120.0
            
            self.rewards[agent] += reward
    
    def _calculate_final_rewards(self) -> Dict[str, float]:
        """Calculate rewards at game end."""
        declarer_points = self._points[self._declarer_idx]
        if self._partner_idx is not None:
            declarer_points += self._points[self._partner_idx]
        
        declarer_wins = declarer_points >= 61
        
        rewards = {}
        for agent in self.possible_agents:
            idx = self.agent_name_mapping[agent]
            is_declarer_team = (idx == self._declarer_idx or idx == self._partner_idx)
            
            if self.reward_mode == "sparse":
                if is_declarer_team:
                    rewards[agent] = 1.0 if declarer_wins else -1.0
                else:
                    rewards[agent] = -1.0 if declarer_wins else 1.0
            else:
                # Shaped/Dense: final bonus based on margin
                margin = declarer_points - 60  # Positive if declarer won
                if is_declarer_team:
                    rewards[agent] = (declarer_points - 60) / 60.0
                else:
                    rewards[agent] = (60 - declarer_points) / 60.0
        
        return rewards
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> None:
        """Reset the environment for a new game."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Check options for bidding mode
        use_bidding = True
        if options and options.get("fixed_declarer") is not None:
            # Force a specific declarer (for backward compatibility)
            use_bidding = False
            forced_declarer = options["fixed_declarer"]
        
        # Reset game state
        self._deal_cards()
        self._played_cards = []
        self._current_trick = []
        self._trick_history = []
        self._points = [0, 0, 0, 0]
        self._trick_number = 0
        self._prev_declarer_points = 0
        
        # Determine declarer through bidding or forced
        if use_bidding:
            self._declarer_idx = self._run_bidding()
        else:
            self._declarer_idx = forced_declarer
        
        self._leader = self._declarer_idx
        
        # Choose called ace and find partner
        self._called_ace = self._choose_called_ace()
        self._partner_idx = self._find_partner()
        
        # Reset PettingZoo state
        self.agents = self.possible_agents.copy()
        self._agent_selector = _SimpleAgentSelector(self._get_turn_order())
        self.agent_selection = self._agent_selector.next()
        
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
    
    def _get_turn_order(self) -> List[str]:
        """Get the order of agents for the current trick."""
        order = []
        for offset in range(4):
            player_idx = (self._leader + offset) % 4
            order.append(self.possible_agents[player_idx])
        return order
    
    def step(self, action: int) -> None:
        """Execute one step - play a card."""
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection
        player_idx = self.agent_name_mapping[agent]
        
        # Validate action
        legal_actions = self._get_legal_actions(player_idx)
        if action not in legal_actions:
            # Invalid action - pick random legal action
            action = self.rng.choice(legal_actions)
        
        # Play the card
        card = IDX_TO_CARD[action]
        self._hands[player_idx].remove(card)
        self._current_trick.append((player_idx, card))
        
        # Check if trick is complete
        if len(self._current_trick) == 4:
            # Save pre-trick state for dense rewards
            prev_points = self._points.copy()
            
            winner_idx = self._resolve_trick()
            self._leader = winner_idx
            self._trick_number += 1
            
            # Dense rewards: give intermediate feedback every trick
            if self.reward_mode in ("shaped", "dense"):
                self._award_trick_rewards(winner_idx, prev_points)
            
            # Check if game is over
            if self._trick_number >= 8:
                final_rewards = self._calculate_final_rewards()
                for ag in self.agents:
                    self.rewards[ag] = final_rewards[ag]
                    self.terminations[ag] = True
            else:
                # Set up next trick
                self._agent_selector = _SimpleAgentSelector(self._get_turn_order())
        
        # Accumulate rewards
        self._accumulate_rewards()
        
        # Move to next agent
        if not all(self.terminations.values()):
            self.agent_selection = self._agent_selector.next()
    
    def _was_dead_step(self, action: int) -> None:
        """Handle step for terminated agent - clear agent from list if all done."""
        # If all agents are terminated, clear the agents list
        if all(self.terminations.values()):
            self.agents = []
        elif self.terminations[self.agent_selection]:
            # Remove this agent from active list and move to next
            if self.agent_selection in self.agents:
                self.agents.remove(self.agent_selection)
            if self.agents:
                self.agent_selection = self._agent_selector.next()
    
    def _accumulate_rewards(self) -> None:
        """Accumulate rewards for all agents."""
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]
            self.rewards[agent] = 0.0
    
    def render(self) -> Optional[str]:
        """Render the current game state."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            lines = []
            lines.append(f"=== Trick {self._trick_number + 1}/8 ===")
            lines.append(f"Declarer: Player {self._declarer_idx}, Called: {self._called_ace}")
            lines.append(f"Points: {self._points}")
            lines.append(f"Current trick: {self._current_trick}")
            lines.append(f"Current player: {self.agent_selection}")
            
            for i, hand in enumerate(self._hands):
                marker = "*" if self.possible_agents[i] == self.agent_selection else " "
                lines.append(f"{marker} Player {i}: {sorted(hand)}")
            
            output = "\n".join(lines)
            if self.render_mode == "human":
                print(output)
            return output
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        pass
    
    # Additional utility methods for training
    
    def get_game_result(self) -> Dict[str, Any]:
        """Get detailed game result after termination."""
        declarer_points = self._points[self._declarer_idx]
        if self._partner_idx is not None:
            declarer_points += self._points[self._partner_idx]
        
        return {
            "declarer_idx": self._declarer_idx,
            "partner_idx": self._partner_idx,
            "called_ace": self._called_ace,
            "declarer_points": declarer_points,
            "opponent_points": 120 - declarer_points,
            "win": declarer_points >= 61,
            "schneider": declarer_points >= 91 or declarer_points <= 30,
            "schwarz": declarer_points == 120 or declarer_points == 0,
            "points_per_player": self._points.copy(),
        }


class _SimpleAgentSelector:
    """Simple agent selector without PettingZoo dependency."""
    
    def __init__(self, agent_order: List[str]):
        self.agent_order = agent_order
        self._current_idx = 0
    
    def next(self) -> str:
        """Return current agent and advance to next."""
        agent = self.agent_order[self._current_idx % len(self.agent_order)]
        self._current_idx += 1
        return agent
    
    def reset(self) -> None:
        self._current_idx = 0


# Convenience function to create environment
def make_env(render_mode: Optional[str] = None, **kwargs) -> SchafkopfEnv:
    """Create a Schafkopf environment instance."""
    return SchafkopfEnv(render_mode=render_mode, **kwargs)


# For compatibility with gymnasium's env checker
def env(**kwargs) -> SchafkopfEnv:
    """Gymnasium-style env creation."""
    return make_env(**kwargs)
