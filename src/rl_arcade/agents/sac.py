"""SAC agent (CleanRL-style, single file).

Reference: Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep
Reinforcement Learning with a Stochastic Actor* (ICML 2018).

P0 status: **stub**. The full implementation lands in P3 (see docs/PLAN.md).
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_arcade.config import SACConfig


@dataclass(slots=True)
class SACTrainOutput:
    run_id: str
    final_eval_reward: float
    total_timesteps: int
    wandb_url: str | None = None


def train_sac(
    env_id: str,
    total_timesteps: int,
    cfg: SACConfig,
    seed: int = 1,
) -> SACTrainOutput:  # pragma: no cover — stub
    """Train SAC on ``env_id``."""
    raise NotImplementedError(
        "train_sac is a P0 stub. See docs/PLAN.md P3 for the real implementation."
    )
