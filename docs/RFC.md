# RFC-001 — rl-arcade reproducibility substrate

**Author:** Rushil Kaul · **Status:** Draft · **Target release:** P1–P3

## 1. Summary

Pin the **run manifest**, **agent contract**, and **replay/export** paths so the
rest of the repo can grow algorithms without breaking the substrate. Keep agents
single-file per CleanRL's ethos. Enforce reference curves in CI.

## 2. Context

`RESEARCH.md` covers algorithm choices, env regimes, and framework comparison.
`PRD.md` owns goals. This RFC pins APIs so adding new agents doesn't churn
the foundation.

## 3. Detailed design

### 3.1 Agent contract (minimal)

Each agent is a module that exposes a `train_<algo>` function returning a
typed output. No base class required — keeping single-file readability.

```python
# src/rl_arcade/agents/ppo.py
@dataclass(slots=True)
class PPOTrainOutput:
    run_id: str
    final_eval_reward: float
    total_timesteps: int
    wandb_url: str | None = None

def train_ppo(env_id: str, total_timesteps: int, cfg: PPOConfig, seed: int = 1) -> PPOTrainOutput: ...
```

Dispatch in `__main__.py`:

```python
match algo:
    case "ppo": out = train_ppo(env, total_timesteps, cfg.ppo, seed)
    case "dqn": out = train_dqn(env, total_timesteps, cfg.dqn, seed)
    case "sac": out = train_sac(env, total_timesteps, cfg.sac, seed)
```

### 3.2 Run manifest

Written next to the checkpoint in `runs/{run_id}/manifest.json`:

```json
{
  "run_id": "2026-04-22T18:14:03Z-7f3a",
  "algo": "ppo",
  "env_id": "CartPole-v1",
  "seed": 1,
  "total_timesteps": 50000,
  "hparams": { "clip_coef": 0.2, "learning_rate": 0.00025, "...": "..." },
  "dependencies": { "torch": "2.5.0", "gymnasium": "1.0.0", "ale-py": "0.10.1", "cuda": "12.4" },
  "git_sha": "7f3ab2c...",
  "created_at": "2026-04-22T18:14:03Z",
  "wandb_url": "https://wandb.ai/.../runs/abc123",
  "eval_reward_mean": 498.4,
  "eval_reward_std": 2.1
}
```

Canonical implementation in `src/rl_arcade/training/manifest.py`.

### 3.3 Reference curve CI

For each (algo, env) pair we commit a curve:

```
artefacts/reference/ppo_cartpole_v1.csv   # (step, reward_mean, reward_std) over 5 seeds
```

CI runs a short-horizon sanity (10k-50k steps) and compares the **endpoint**
against a guard:

```yaml
- name: Sanity CartPole PPO
  run: uv run rl-arcade train --env CartPole-v1 --algo ppo --total-timesteps 10000 --seed 1
- name: Assert reward >= 150
  run: uv run python scripts/assert_ref.py --run-latest --env CartPole-v1 --min-reward 150
```

### 3.4 Replay contract

```
rl-arcade replay <run_id> [--episodes N] [--render] [--deterministic]
```

- Loads manifest; validates installed deps match (warn if minor mismatch).
- Re-seeds everything per manifest.
- Rolls N eval episodes; asserts mean reward within the stored ±2σ band.
- Prints a compact report; non-zero exit if band miss.

### 3.5 ONNX export

```python
# src/rl_arcade/export/onnx.py
def export_actor_onnx(ckpt_path: Path, out: Path) -> ExportInfo:
    model = load_actor(ckpt_path)
    model.eval()
    obs_sample = sample_obs(ckpt_path.parent / "manifest.json")
    torch.onnx.export(model, obs_sample, out, opset_version=17, ...)
    verify_roundtrip(model, out, obs_sample, atol=1e-6)
```

CI task verifies a CartPole actor exports cleanly.

### 3.6 Serving shim

`apps/inference/` (P3):

```
POST /policy/{run_id}/act
  body: {"obs": [0.1, -0.2, ...]}
  200:  {"action": 1, "action_logits": [..]}
```

Stateless; uses onnxruntime on CPU. Target p95 < 200 ms on commodity CPU.

### 3.7 Determinism modes

- Default: seeded but non-strict (fast).
- `--deterministic`: sets `torch.use_deterministic_algorithms(True)`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Documented perf cost.
- Manifest records whether deterministic was on.

### 3.8 Environment pins

`pyproject.toml` pins:

- `gymnasium==1.0.*`
- `ale-py==0.10.*`
- `torch>=2.5,<3.0`
- `onnx>=1.17`

Each release tag bumps minors deliberately; a PR that bumps these must also
refresh reference curves.

## 4. Alternatives considered

| Alt | Why not |
|-----|---------|
| Shared `Trainer` base class | Hides the algo; harms single-file readability |
| Ray RLlib | Over-scale for the portfolio thesis |
| Stable-Baselines3 as primary | We use it as oracle only |
| Hydra configs | Typer + pydantic-settings are enough |

## 5. Tradeoffs

- Single-file agents duplicate boilerplate (logging, eval loop). Accept the cost.
- Reference CSVs are produced on a GPU host manually; CI never re-trains at
  full scale. The sanity horizon is small and deliberate.
- ONNX export limits us to actor networks that are onnx-exportable; any non-standard
  op needs a manual rewrite.

## 6. Rollout plan

1. P1 wk 1: Ship real PPO on CartPole + reference curve + CI sanity.
2. P1 wk 2: Ship PPO on LunarLander; replay CLI; determinism flag.
3. P2 wk 3: Add Atari preprocessing + DQN; Pong reference.
4. P2 wk 4: Add ONNX export + roundtrip test.
5. P3 wk 5–6: Add SAC + serving shim + (stretch) web demo.

## 7. Open questions

- Should we maintain separate reference curves for `--deterministic` vs default?
  Leaning: just default (deterministic is for demonstration, not benchmarking).
- Per-algo config YAML vs pydantic class? Stay with pydantic for now; YAML if
  sweep tooling matures.
- Web demo hosting: Cloudflare Pages vs Vercel? Lean Cloudflare (lower bandwidth cost).
