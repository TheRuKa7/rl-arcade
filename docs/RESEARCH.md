# RESEARCH — Deep RL for Games (2026 landscape)

> Scope: pick a small, defensible stack for a *portfolio-grade* RL repo that (a) trains
> agents on recognisable games, (b) is reproducible, (c) teaches the reader how
> modern RL actually works, and (d) ships a demo artefact.

---

## 1. Algorithm landscape

### 1.1 Model-free, on-policy

| Algorithm | Year | Core idea | Strengths | Weaknesses | Use in this repo |
|-----------|------|-----------|-----------|------------|------------------|
| **PPO** (Schulman et al. 2017) | 2017 | Clipped surrogate objective, multiple epochs per rollout | Stable, tunable, dominates competitive benchmarks and LLM RLHF | Sample-inefficient vs off-policy | **Yes — default** |
| TRPO | 2015 | KL-constrained natural gradient | Theoretically grounded | Complex, superseded by PPO | No |
| A2C / A3C | 2016 | Sync / async actor-critic | Simple baseline | Noisier than PPO | Reference only |
| IMPALA | 2018 | V-trace off-policy correction, distributed | Scales to thousands of actors | Infra-heavy | Noted for P3 scaling |

### 1.2 Model-free, off-policy

| Algorithm | Year | Strengths | Weaknesses | Use in this repo |
|-----------|------|-----------|------------|------------------|
| **DQN** (Mnih et al. 2015) + Rainbow extensions (double, duelling, PER, n-step, distributional, noisy) | 2015–2018 | Works on discrete action Atari; classic result | Brittle on continuous control | **Yes — Atari track** |
| **SAC** (Haarnoja 2018) | 2018 | SOTA for continuous control, entropy-regularised | Needs twin critics, more hparams | **Yes — MuJoCo track** |
| TD3 | 2018 | Deterministic, delayed updates | Weaker than SAC on stochastic envs | Optional |
| DDPG | 2016 | Actor-critic for continuous | Unstable | No (superseded) |

### 1.3 Model-based

| Algorithm | Year | Strengths | Weaknesses | Use in this repo |
|-----------|------|-----------|------------|------------------|
| MuZero / EfficientZero | 2019/2021 | SOTA sample efficiency, learns a world model | Very complex, compute-heavy | Documented, not implemented |
| Dreamer V3 (Hafner 2023) | 2023 | One set of hparams across 150 tasks; learns latent world | Requires implementing RSSM | **Stretch goal in P4** |
| AlphaZero / AlphaGo Zero | 2017/2018 | Self-play + MCTS for board games | Game-specific | Documented for self-play discussion |

### 1.4 Why PPO + DQN + SAC (and not "just one")

These three cover the **three canonical regimes** recruiters expect you to know:

- **PPO** — discrete *and* continuous, parallel rollouts, policy gradient, the algorithm
  behind RLHF / ChatGPT fine-tuning.  Knowing it well is table stakes.
- **DQN (+ Rainbow)** — value-based, replay buffer, off-policy, target networks.  The
  algorithmic lineage every ML interviewer asks about.
- **SAC** — entropy-regularised, twin Q-networks, the state of the art for continuous
  control.  Opens the door to robotics-adjacent demos (MuJoCo, Isaac).

Three algorithms × three regimes = a lab where we can answer *any* "compare X and Y"
interview question with working code.

---

## 2. Environment choice

### 2.1 Discrete, low-dim (smoke tests)

| Env                    | Why                                              | Training budget       |
|------------------------|--------------------------------------------------|-----------------------|
| `CartPole-v1`          | 4-dim obs, 2 actions; solved in minutes on CPU   | 50k steps, ~3 min CPU |
| `LunarLander-v3`       | Slightly richer reward shape, still CPU-friendly | 500k steps, ~20 min   |
| `Acrobot-v1`           | Sparse-ish reward, tests exploration             | 200k steps            |

### 2.2 Atari (headline visuals)

| Env                                         | Why |
|---------------------------------------------|-----|
| `ALE/Breakout-v5` via Gymnasium + ALE-py    | The canonical Atari demo; Rainbow paper benchmark |
| `ALE/Pong-v5`                               | Cheapest Atari game to solve |
| `ALE/SpaceInvaders-v5`                      | Visually striking for demo videos |

Preprocessing follows the standard Atari wrapper stack (grayscale, 84×84, frame-stack=4,
sticky actions, clip reward to ±1, episodic life).

### 2.3 Continuous control (SAC demo)

| Env                          | Why                                         |
|------------------------------|---------------------------------------------|
| `Pendulum-v1`                | 3-dim obs, cheap SAC smoke test             |
| `Hopper-v5` (MuJoCo)         | Classic continuous-control benchmark        |
| `HalfCheetah-v5`             | Higher-dim, shows SAC's sample efficiency   |

### 2.4 Multi-agent / self-play (stretch)

| Env                          | Why                                         |
|------------------------------|---------------------------------------------|
| PettingZoo `connect_four_v3` | Simple two-player board game, self-play     |
| PettingZoo `pong_v3`         | Two-paddle, classic self-play demo          |
| MiniGrid `DoorKey-8x8`       | Sparse reward, tests intrinsic motivation   |

### 2.5 Why not the big ones?

- **StarCraft II / Dota** — compute out of scope (hundreds of GPU-years).
- **Isaac Lab / robotics** — interesting but needs NVIDIA Omniverse; bumps setup cost.
- **Minecraft / MineRL** — recent work (Voyager, DreamerV3) impressive but heavy.

Covered in RESEARCH so the reader knows they exist; not in scope for the repo.

---

## 3. Framework choice: CleanRL vs SB3 vs RLlib

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **CleanRL** (single-file agents) | Readable ~400 LoC per algo; widely cited; W&B integration native; pinned reference curves | Not a package — you copy files | **Primary — we vendor CleanRL-style files with attribution** |
| **Stable-Baselines3** | Battle-tested, easy API (`model.learn()`), great for validation | Abstractions hide algorithmic detail; worse for teaching | **Secondary — used only as an "am I correct?" oracle** |
| **Ray RLlib** | Scales to clusters, supports many algos | Heavy dependency, config-heavy | Out of scope |
| **torchrl** | Official PyTorch RL library, composable | Young API, less-polished examples | Monitor, not adopt |
| **Tianshou** | Clean API, good perf | Smaller community | Not selected |

**Decision:** CleanRL-style single-file agents in `src/rl_arcade/agents/`, with
`scripts/sanity_sb3.py` running SB3 on the same env for cross-check.

---

## 4. Reproducibility patterns

Mandatory for every run:

1. **Seeded everything** — `torch.manual_seed`, `np.random.seed`, `env.reset(seed=...)`,
   `torch.backends.cudnn.deterministic = True` (with documented perf hit).
2. **Environment pin** — `gymnasium==1.0.*`, `ale-py==0.10.*`.  RL is famously
   version-sensitive; an Atari wrapper change broke half the Rainbow leaderboard in 2023.
3. **Run manifest** — JSON with git SHA, CLI args, hparams, env version, timestamp,
   host GPU model, wandb run URL.  Stored next to the checkpoint.
4. **Replay CLI** — `rl-arcade replay <run_id>` loads checkpoint + manifest,
   runs N episodes, records a video, re-verifies the reward.
5. **Reference curves** — each algorithm × env pair has a known-good learning curve
   stored as CSV in `artefacts/reference/`.  CI runs a short training job and checks
   we're within a tolerance band.

---

## 5. Observability

| Tool | Use |
|------|-----|
| **Weights & Biases** | Primary — logs, system metrics, video rollouts, hparam sweeps |
| **TensorBoard** | Secondary — works offline, useful in air-gapped environments |
| **Rich console** | Pretty progress bars; optional flag |
| **wandb reports** | Shareable write-ups; linked from each run's README |

---

## 6. Deployment / demo

The classic RL failure mode is "model rots in a checkpoint nobody can run".  Plan:

1. **ONNX export** — trained actor → ONNX graph.  Test with `onnxruntime` in CI.
2. **FastAPI inference shim** — POST observation, receive action.  Dockerised.
3. **WASM runner (P4)** — onnxruntime-web + canvas visualisation of the env so
   visitors can *play against* the agent in the browser.  Nice recruiter touch.

---

## 7. Open research questions the repo gestures at

- **Sample efficiency** — can we match MuZero-level efficiency with cheap hardware?
- **Generalisation** — does PPO trained on `DoorKey-8x8` transfer to `DoorKey-16x16`?
- **Offline RL** — CQL / IQL on D4RL datasets; future `offline-rl` module.
- **RLHF** — PPO is the bridge to LLM post-training.  A side-note explaining
  how the same math powers ChatGPT's fine-tuning is a great portfolio hook.

---

## 8. Key references

- Schulman et al., "Proximal Policy Optimization Algorithms" (2017), arXiv:1707.06347
- Mnih et al., "Human-level control through deep reinforcement learning", Nature 518 (2015)
- Haarnoja et al., "Soft Actor-Critic", ICML 2018
- Hessel et al., "Rainbow: Combining Improvements in Deep RL" (2018)
- Schrittwieser et al., "MuZero" (2019)
- Hafner et al., "DreamerV3" (2023), arXiv:2301.04104
- Huang et al., "CleanRL: High-quality Single-file Implementations of DRL Algorithms",
  JMLR 2022
- Farama Foundation, Gymnasium docs (https://gymnasium.farama.org/, accessed 2026-04)
