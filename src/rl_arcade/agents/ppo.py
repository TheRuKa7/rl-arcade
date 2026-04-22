"""PPO agent (CleanRL-style, single file).

Reference: Schulman et al., *Proximal Policy Optimization Algorithms* (2017),
arXiv:1707.06347. This module follows the structure of CleanRL's ``ppo.py``
(Huang et al., JMLR 2022).

P0 status: **stub**. The full implementation lands in P1 (see docs/PLAN.md).
The stub exists so the CLI can import it and tests can assert the symbols
are wired up correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_arcade.config import PPOConfig


@dataclass(slots=True)
class PPOTrainOutput:
    """Result of a PPO training run."""

    run_id: str
    final_eval_reward: float
    total_timesteps: int
    wandb_url: str | None = None


def train_ppo(
    env_id: str,
    total_timesteps: int,
    cfg: PPOConfig,
    seed: int = 1,
) -> PPOTrainOutput:  # pragma: no cover — stub
    """Train PPO on ``env_id`` for ``total_timesteps`` env steps.

    Parameters
    ----------
    env_id:
        Gymnasium env id (``"CartPole-v1"``, ``"LunarLander-v3"``, …).
    total_timesteps:
        Total number of environment steps summed across parallel envs.
    cfg:
        PPO hyperparameters (see :class:`rl_arcade.config.PPOConfig`).
    seed:
        RNG seed.
    """
    raise NotImplementedError(
        "train_ppo is a P0 stub. See docs/PLAN.md P1 for the real implementation."
    )
