"""Environment registry and wrapper helpers.

In P0 we ship a thin discovery layer that groups env ids by regime. The real
wrappers (Atari preprocessing, framestack, reward clipping) arrive in P2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EnvSpec:
    env_id: str
    regime: str  # "classic-control" | "atari" | "mujoco" | "multi-agent" | "minigrid"
    recommended_algo: str


REGISTRY: tuple[EnvSpec, ...] = (
    EnvSpec("CartPole-v1", "classic-control", "ppo"),
    EnvSpec("LunarLander-v3", "classic-control", "ppo"),
    EnvSpec("Acrobot-v1", "classic-control", "ppo"),
    EnvSpec("ALE/Pong-v5", "atari", "dqn"),
    EnvSpec("ALE/Breakout-v5", "atari", "dqn"),
    EnvSpec("ALE/SpaceInvaders-v5", "atari", "dqn"),
    EnvSpec("Pendulum-v1", "mujoco", "sac"),
    EnvSpec("HalfCheetah-v5", "mujoco", "sac"),
    EnvSpec("Hopper-v5", "mujoco", "sac"),
)


def list_envs(regime: str | None = None) -> list[EnvSpec]:
    """Return all registered envs, optionally filtered by regime."""
    if regime is None:
        return list(REGISTRY)
    return [spec for spec in REGISTRY if spec.regime == regime]
