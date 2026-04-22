"""Sanity oracle: run Stable-Baselines3 PPO on the same env as our CleanRL-style PPO.

Purpose: if our curve disagrees with SB3's by more than noise, one of us has a bug.
SB3 is battle-tested, so the bug is almost certainly ours — this script helps find it.

Not part of CI; run manually when you change ``agents/ppo.py``::

    uv run python scripts/sanity_sb3.py --env CartPole-v1 --steps 50000

Requires: ``pip install stable-baselines3`` (not a default dep).
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="SB3 PPO sanity oracle")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {exc}. Install with: pip install stable-baselines3 gymnasium"
        ) from exc

    env = gym.make(args.env)
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=1)
    model.learn(total_timesteps=args.steps)

    # Quick eval
    obs, _ = env.reset(seed=args.seed)
    total = 0.0
    for _ in range(1_000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        total += float(reward)
        if term or trunc:
            break
    print(f"[sb3-sanity] {args.env} eval return: {total:.1f}")


if __name__ == "__main__":
    main()
