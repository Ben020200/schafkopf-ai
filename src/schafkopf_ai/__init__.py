"""Core package for Schafkopf Sucherb (Sauspiel) data mining and simulation."""

from .cards import SUITS, RANKS, FULL_DECK, POINT_VALUES, TRUMP_ORDER
from .analysis import collect_hand_statistics, run_game_batch, estimate_win_rate
from .env import SchafkopfEnv, make_env
from .ppo import PPOTrainer, PPOConfig, train_ppo, SchafkopfNetwork

__all__ = [
    "SUITS",
    "RANKS",
    "FULL_DECK",
    "POINT_VALUES",
    "TRUMP_ORDER",
    "collect_hand_statistics",
    "run_game_batch",
    "estimate_win_rate",
    "SchafkopfEnv",
    "make_env",
    "PPOTrainer",
    "PPOConfig",
    "train_ppo",
    "SchafkopfNetwork",
]
