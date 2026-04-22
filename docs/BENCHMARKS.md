# BENCHMARKS — reference curves and validation methodology

The RL literature is full of "my PPO beat their PPO" claims that dissolve on
re-testing.  This repo takes the opposite stance: we reproduce known results,
not invent new ones.  This doc describes *how we know* our implementations work.

## Reference sources

| Env                  | Reference                                                                            | Expected reward              |
|----------------------|--------------------------------------------------------------------------------------|------------------------------|
| CartPole-v1 (PPO)    | CleanRL reports, Gymnasium docs                                                      | 500 within 50k steps         |
| LunarLander-v3 (PPO) | CleanRL reports                                                                      | ≥ 200 within 500k steps      |
| Pong-v5 (DQN)        | Mnih 2015 + Rainbow (Hessel 2018)                                                    | ≥ +18 within 5 M frames      |
| Pendulum-v1 (SAC)    | Haarnoja 2018                                                                        | ≥ -200 within 50k steps      |
| HalfCheetah-v5 (SAC) | Haarnoja 2018 Table 1                                                                | ≥ 4000 within 1 M steps      |
| Connect Four (self-play PPO) | Our reference — self-play vs uniform random                                   | ≥ 95 % winrate after 500k steps |

All reference curves live under `artefacts/reference/` as CSV: `(step, reward_mean, reward_std)`
over 5 seeds.

## Validation modes

### 1. CI sanity check (fast)

Runs on every push.  Wall-clock budget: ≤ 3 min CPU.

- Boot the CLI (`--dry-run`), assert config parses.
- Train PPO on CartPole for 10k steps with 1 seed.
- Assert episode reward > 150 at the end (fails if PPO is broken).

### 2. Nightly (manual trigger)

Full reference runs for CartPole and LunarLander on 5 seeds.  Compare against
stored reference curves using a 2-sigma tolerance band.

### 3. Pre-release (manual GPU run)

Full Atari / MuJoCo reference runs.  Uploaded to wandb; linked in README.

## Anti-tricks

Deliberately avoided — these inflate benchmarks without improving the algorithm:

- Learning-rate sweeps hidden in `config.yaml`
- Env-specific reward shaping not disclosed
- "Best-of-5-seeds" cherry-picking
- Shorter eval episodes than training
- Ignoring wall-clock and reporting only sample complexity
