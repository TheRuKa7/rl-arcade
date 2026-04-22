"""Runtime configuration for rl-arcade.

All hyperparameters are centralised here and overridable by env vars prefixed
``RL_ARCADE_``.  CLI flags (typer) also feed into the same settings object so
the config precedence is:

    CLI flag  >  environment variable  >  default

Docstrings on each field should cite the paper / section the default came from,
or the CleanRL reference, so nothing is a magic number.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PPOConfig(BaseSettings):
    """CleanRL-style PPO defaults.

    See Schulman et al. 2017, arXiv:1707.06347 and CleanRL's ``ppo.py`` reference.
    """

    model_config = SettingsConfigDict(env_prefix="RL_ARCADE_PPO_", extra="ignore")

    num_envs: int = Field(default=4, description="Parallel vectorised envs")
    n_steps: int = Field(default=128, description="Rollout length per env (Schulman 2017, Table 5)")
    update_epochs: int = Field(default=4, description="Number of PPO epochs per rollout")
    num_minibatches: int = Field(default=4)
    learning_rate: float = Field(default=2.5e-4, description="Adam LR; linearly annealed to 0")
    gamma: float = Field(default=0.99)
    gae_lambda: float = Field(default=0.95)
    clip_coef: float = Field(default=0.2, description="PPO clip epsilon")
    ent_coef: float = Field(default=0.01)
    vf_coef: float = Field(default=0.5)
    max_grad_norm: float = Field(default=0.5)
    clip_vloss: bool = Field(default=True)
    norm_adv: bool = Field(default=True)
    anneal_lr: bool = Field(default=True)


class DQNConfig(BaseSettings):
    """DQN + Rainbow-lite defaults (Mnih 2015, Hessel 2018)."""

    model_config = SettingsConfigDict(env_prefix="RL_ARCADE_DQN_", extra="ignore")

    buffer_size: int = Field(default=1_000_000)
    learning_starts: int = Field(default=80_000)
    batch_size: int = Field(default=32)
    train_frequency: int = Field(default=4)
    target_network_frequency: int = Field(default=1_000)
    learning_rate: float = Field(default=1e-4)
    gamma: float = Field(default=0.99)
    start_e: float = Field(default=1.0, description="Initial epsilon for eps-greedy")
    end_e: float = Field(default=0.01)
    exploration_fraction: float = Field(default=0.10)
    double_q: bool = Field(default=True)


class SACConfig(BaseSettings):
    """SAC defaults (Haarnoja 2018)."""

    model_config = SettingsConfigDict(env_prefix="RL_ARCADE_SAC_", extra="ignore")

    buffer_size: int = Field(default=1_000_000)
    learning_starts: int = Field(default=5_000)
    batch_size: int = Field(default=256)
    policy_lr: float = Field(default=3e-4)
    q_lr: float = Field(default=1e-3)
    gamma: float = Field(default=0.99)
    tau: float = Field(default=0.005, description="Polyak update rate")
    autotune_alpha: bool = Field(default=True)
    alpha: float = Field(default=0.2, description="Used only if autotune_alpha=False")


class AppConfig(BaseSettings):
    """Top-level runtime config."""

    model_config = SettingsConfigDict(env_prefix="RL_ARCADE_", extra="ignore")

    seed: int = Field(default=1, description="Global RNG seed")
    device: str = Field(default="auto", description="'auto' | 'cpu' | 'cuda' | 'cuda:N'")
    deterministic: bool = Field(default=False, description="Force deterministic CUDA kernels")
    runs_dir: str = Field(default="runs", description="Root directory for checkpoints + manifests")
    wandb_project: str = Field(default="rl-arcade")
    wandb_entity: str | None = Field(default=None)
    wandb_mode: str = Field(default="online", description="'online' | 'offline' | 'disabled'")

    ppo: PPOConfig = Field(default_factory=PPOConfig)
    dqn: DQNConfig = Field(default_factory=DQNConfig)
    sac: SACConfig = Field(default_factory=SACConfig)


def load_config() -> AppConfig:
    """Build a fresh :class:`AppConfig` honouring env-var overrides."""
    return AppConfig()
