# PLAN — rl-arcade rollout

Phased from scaffold to production-shaped portfolio artefact.  Each phase has an
acceptance bar; we don't advance until it's green.

---

## P0 — Scaffold (this commit)

**Goal:** clone → `uv sync` → `rl-arcade --help` works.  CI green.

Deliverables:
- Python package `rl_arcade` (pyproject + src/ layout)
- Typer CLI skeleton: `train`, `replay`, `export`, `runs list`
- Config system (pydantic-settings) with env overrides
- Smoke test that boots the CLI
- GitHub Actions CI: ruff + mypy + pytest
- Dockerfile (CPU baseline) + docker-compose with TensorBoard service
- Docs: README, RESEARCH, PLAN, THINK, ARCHITECTURE

**Exit:** CI green; `uv run rl-arcade train --env CartPole-v1 --algo ppo --dry-run` returns a parsed config without errors.

---

## P1 — PPO on classic control (~1 week)

**Goal:** CleanRL-style PPO solves `CartPole-v1` and `LunarLander-v3`.

Scope:
- `src/rl_arcade/agents/ppo.py` — single-file PPO (~400 LoC).  Clipped objective,
  GAE-λ, orthogonal init, value clipping, entropy bonus, linear LR schedule.
- Vectorised envs via `gymnasium.vector.SyncVectorEnv` (then `AsyncVectorEnv`).
- Training loop with:
  - wandb logging (reward, episode length, KL, clipfrac, loss components)
  - local run manifest JSON (git SHA, hparams, env version)
  - checkpoint every N updates
- `rl-arcade replay` CLI that loads checkpoint and re-rolls 10 episodes.

Acceptance:
- CartPole-v1 hits 500 reward in <50k steps, 3 min CPU.  CI runs a 10k-step
  sanity check and asserts reward ≥ 150 (catches most regressions).
- LunarLander-v3 hits ≥ 200 reward in 500k steps on CPU in ≤ 20 min.
- wandb report link in README.

---

## P2 — DQN + Atari (~2 weeks)

**Goal:** DQN with Rainbow-lite extensions solves `ALE/Pong-v5`.

Scope:
- `src/rl_arcade/agents/dqn.py` — CleanRL-style DQN with:
  - experience replay (uniform → prioritised)
  - target network, polyak update
  - double Q-learning
  - n-step returns (optional)
  - NoisyNets (stretch)
- Atari wrapper stack: grayscale, 84×84, frame-stack=4, sticky actions, clip reward,
  episodic life, NoOp reset.  Vendored from stable-baselines3 wrappers for
  correctness (attribution preserved).
- CNN torso (Nature DQN architecture).
- GPU-aware: auto-detect CUDA, fall back to CPU (slow but functional).

Acceptance:
- Pong-v5: learns to beat the built-in opponent (+15 reward) in ~3 M steps,
  ~4 h on a single RTX 3090.
- Reference curve checked into `artefacts/reference/dqn_pong_v5.csv`.
- Short-horizon CI sanity (50k steps) asserts reward gradient is non-negative.

---

## P3 — SAC (continuous) + self-play (~2 weeks)

**Goal:** SAC on MuJoCo, + PPO self-play on PettingZoo.

Scope:
- `src/rl_arcade/agents/sac.py` — twin Q, entropy auto-tune, replay.
- SAC on `Pendulum-v1`, `HalfCheetah-v5`.
- PettingZoo adapter for `connect_four_v3` and `pong_v3`.
- Self-play loop: league with snapshot opponents, simple Elo tracker.
- MiniGrid adapter (`DoorKey-8x8`) as exploration testbed.

Acceptance:
- Pendulum solved (return ≥ -200) in 50k steps.
- HalfCheetah-v5 reaches 4k+ return in 1 M steps.
- connect_four self-play agent beats a uniform-random opponent ≥ 95 %
  of games after 500k env steps.

---

## P4 — Export + web demo (~1 week)

**Goal:** Recruiter visits a URL and watches the agent play.

Scope:
- `rl-arcade export <run_id> --format onnx` — converts actor network to ONNX,
  verifies round-trip parity on random obs.
- FastAPI inference shim: `POST /policy/{run_id}/act` accepts obs, returns action.
  Dockerised, ≤ 200 ms p95 latency on CPU.
- (Stretch) WASM demo: onnxruntime-web + HTML canvas; visitor drives env in browser.
- Demo site hosted on Cloudflare Pages or Vercel; linked from GitHub README.

Acceptance:
- `uv run pytest -k onnx` passes (round-trip within 1e-6).
- `docker compose up inference` → curl returns actions in < 200 ms CPU.
- Public URL live with a CartPole agent (stretch: Breakout agent).

---

## P5 — Optional: DreamerV3 / offline RL / RLHF essay

Stretch goals, each self-contained:

- **DreamerV3** — implement RSSM + actor-critic in a separate module.  Train on
  DMC Walker.  Mostly an educational exercise; the repo's interview value is
  *showing* model-based RL exists.
- **Offline RL** — CQL or IQL on D4RL datasets; one notebook comparing online
  vs offline performance on HalfCheetah.
- **RLHF essay** — `docs/RLHF.md` drawing the line from PPO here to PPO in
  ChatGPT.  Code reference: trl's `PPOTrainer`.

---

## Milestone schedule (if I were shipping solo)

| Phase | Wall-clock (evenings) | Compute needed |
|-------|-----------------------|----------------|
| P0    | Day 0                 | None           |
| P1    | Week 1                | CPU only       |
| P2    | Weeks 2–3             | 1× RTX 3090 for ~6 h total |
| P3    | Weeks 4–5             | 1× RTX 3090 for ~12 h total |
| P4    | Week 6                | CPU            |
| P5    | Optional              | Varies         |

---

## Non-goals

- **Beating SOTA** — we match reference curves, we don't advance them.
- **Distributed training** — noted, not built.  IMPALA / Ape-X are out of scope.
- **Robotics hardware** — simulator only.
- **Vendoring Atari ROMs** — users supply ROMs via `ale-import-roms` per Farama's policy.
