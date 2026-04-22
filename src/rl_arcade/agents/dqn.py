"""DQN agent (CleanRL-style, single file).

Reference: Mnih et al., *Human-level control through deep reinforcement learning*,
Nature 518 (2015); Hessel et al., *Rainbow* (AAAI 2018).

P0 status: **stub**. The full implementation lands in P2 (see docs/PLAN.md).
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_arcade.config import DQNConfig


@dataclass(slots=True)
class DQNTrainOutput:
    run_id: str
    final_eval_reward: float
    total_timesteps: int
    wandb_url: str | None = None


def train_dqn(
    env_id: str,
    total_timesteps: int,
    cfg: DQNConfig,
    seed: int = 1,
) -> DQNTrainOutput:  # pragma: no cover — stub
    """Train DQN on ``env_id``."""
    raise NotImplementedError(
        "train_dqn is a P0 stub. See docs/PLAN.md P2 for the real implementation."
    )
