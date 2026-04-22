# rl-arcade

**Deep reinforcement-learning lab for game-playing agents.** Train PPO / DQN / SAC on
Gymnasium and Atari, benchmark against reference curves, and ship trained policies as
ONNX artefacts playable in the browser.

> Portfolio differentiator: most public RL repos ship a CartPole notebook.  `rl-arcade`
> ships the *reproducibility layer* — pinned seeds, reference learning curves, wandb
> dashboards, and an ONNX export path — so recruiters can *replay* the agent, not just
> read about it.

---

## Highlights

- **Single-file-agent first.** PPO/DQN implementations follow the
  [CleanRL](https://github.com/vwxyzjn/cleanrl) philosophy (one file ≈ 400 LoC) so the
  algorithm is *readable*, not buried under framework abstractions.
- **Classic control → Atari → self-play.** Same entry point (`rl-arcade train`) scales
  from `CartPole-v1` (3 min on CPU) to `ALE/Breakout-v5` (8 h on a single GPU) to
  PettingZoo multi-agent envs.
- **Reproducible.** Every run logs to Weights & Biases *and* a local JSON manifest with
  git SHA, seed, env version, and hparams.  `rl-arcade replay <run_id>` rehydrates it.
- **Portable policies.** Export trained actors to ONNX; a tiny WASM runner (future) will
  let recruiters drive the agent in-browser.
- **Honest baselines.** Every algorithm ships with a reference curve from the original
  paper — if our run doesn't match within noise, CI flags it.

---

## Stack

| Layer         | Choice                                           | Why                                                          |
|---------------|--------------------------------------------------|--------------------------------------------------------------|
| Framework     | PyTorch 2.5                                      | Dominant research framework, torch.compile for speedups      |
| Envs          | Gymnasium 1.0, ALE-py, PettingZoo, MiniGrid      | Gymnasium is the maintained fork of OpenAI Gym               |
| Algorithms    | CleanRL-style PPO/DQN (vendored); SB3 for sanity | Single-file agents are auditable; SB3 confirms we're correct |
| Observability | Weights & Biases, TensorBoard                    | wandb is the industry default; TB for offline               |
| Packaging     | uv, Python 3.13, Typer CLI                       | Fast installs, ergonomic CLI                                 |
| Serving       | ONNX Runtime, FastAPI inference shim             | Policy-as-a-microservice for demo site                       |
| Infra         | Docker + CUDA 12.4 base, GitHub Actions CPU CI   | GPU jobs run locally / on rented A10G                        |

---

## Docs

- [RESEARCH.md](docs/RESEARCH.md) — algorithm landscape (PPO, DQN, SAC, MuZero, Dreamer, IMPALA), env benchmarks, tooling survey
- [PLAN.md](docs/PLAN.md) — P0 scaffold → P1 PPO/DQN → P2 Atari → P3 self-play → P4 ONNX export + web demo
- [THINK.md](docs/THINK.md) — why CleanRL over SB3, cost model, failure modes, interview narrative
- [ARCHITECTURE.md](ARCHITECTURE.md) — component diagram, data flow, reproducibility contract
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — reference curves we're matching and how we validate

---

## Quickstart

```bash
# install
uv sync --extra dev

# smoke test on CartPole (3 minutes, CPU)
uv run rl-arcade train --env CartPole-v1 --algo ppo --total-timesteps 50000

# list runs
uv run rl-arcade runs list

# replay a trained policy
uv run rl-arcade replay <run_id> --episodes 5 --render

# export to ONNX
uv run rl-arcade export <run_id> --format onnx --out artefacts/policy.onnx
```

---

## Status

**Phase: P0 — scaffolding complete.** PPO single-file agent, CartPole smoke test, CI,
ONNX export stub all wired.  Atari + self-play land in P2/P3.  See
[docs/PLAN.md](docs/PLAN.md).

---

## License

MIT — see [LICENSE](LICENSE).  Gymnasium, CleanRL, and Stable-Baselines3 are MIT.
Atari ROMs are licensed per [ALE's ROM policy](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
and **not** vendored here.
