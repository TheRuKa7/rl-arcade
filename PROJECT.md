# rl-arcade

## Vision
Deep-RL lab that trains game-playing agents (PPO/DQN/SAC on Gymnasium + Atari +
MuJoCo + PettingZoo), logs reproducibly to wandb, and ships trained policies as
ONNX artefacts playable in the browser.

## Problem
Most public RL repos are CartPole notebooks. The portfolio gap is an *engineering*
substrate — reproducibility, reference curves, exportable policies — that proves
the author can ship RL, not just run a tutorial.

## Target Users
- Recruiters / hiring managers validating Rushil's ML depth
- ML engineers looking for a readable PPO/DQN/SAC reference
- Future-Rushil extending the stack with DreamerV3 / offline RL / RLHF bridges

## Tech Stack
| Layer          | Choice                                      | Why                                              |
|----------------|---------------------------------------------|--------------------------------------------------|
| Framework      | PyTorch 2.5                                 | Dominant research framework, torch.compile       |
| Envs           | Gymnasium 1.0, ALE-py, PettingZoo, MiniGrid | Maintained Gym fork, Farama standard             |
| Algorithms     | CleanRL-style single-file PPO/DQN/SAC       | Readable, auditable; SB3 as oracle only          |
| Observability  | Weights & Biases + TensorBoard              | Industry default + offline fallback              |
| Packaging      | uv + Python 3.13 + Typer CLI                | Fast, ergonomic                                  |
| Serving        | ONNX Runtime + FastAPI                      | Portable policies, REST inference                |
| Infra          | Docker (CPU baseline), GitHub Actions       | Rented GPU (A10G) for heavy training             |

## Domains Involved
- [x] ai-development
- [x] game-development
- [x] architecture-design
- [ ] web-development (P4 demo only)

## Current Phase
`BUILD` — P0 scaffold complete; P1 (PPO on classic control) is next.

## Status
P0 delivered: CLI skeleton, config, env registry, agent stubs, run-manifest
utilities, tests, Docker, CI, docs.  `uv run rl-arcade --help` works; `train
--dry-run` parses and exits cleanly.

## Key Decisions
| Decision | Choice | Rationale | Date |
|----------|--------|-----------|------|
| Framework  | CleanRL-style over SB3 | Readability + portfolio signal; SB3 used as cross-check oracle | 2026-04-22 |
| Env API    | Gymnasium 1.0          | Gym is deprecated; Farama is the active fork                   | 2026-04-22 |
| Python     | 3.13                   | Matches workspace default; modern typing features              | 2026-04-22 |
| CLI        | Typer                  | Consistent with the rest of the portfolio                      | 2026-04-22 |
| Seeding    | Full (torch/np/env) + deterministic flag | RL reproducibility is famously fragile       | 2026-04-22 |
