# DFD — rl-arcade

## Level 0 — Context

```mermaid
flowchart LR
  U[ML eng / learner]
  ENV[Gymnasium / ALE / MuJoCo / PettingZoo]
  WB[Weights & Biases]
  CI[CI runner]
  HOST[GPU host<br/>local or rented]
  SERV[Inference shim<br/>FastAPI]
  BR[Browser demo<br/>onnxruntime-web]
  RLA((rl-arcade))

  U -- CLI --> RLA
  RLA -- step/reset --> ENV
  ENV -- obs/reward --> RLA
  RLA -- logs / videos --> WB
  CI -- short-horizon sanity --> RLA
  HOST -- GPU --> RLA
  RLA -- ONNX --> SERV
  SERV -- actions --> BR
```

## Level 1 — Training + export

```mermaid
flowchart TD
  subgraph CLI
    TC[rl-arcade train]
    RC[rl-arcade replay]
    EC[rl-arcade export]
    LC[rl-arcade runs list]
  end

  subgraph Cfg
    CF[pydantic config<br/>per-algo defaults]
  end

  subgraph Envs
    VE[Vectorised envs<br/>Sync/AsyncVectorEnv]
  end

  subgraph Agents
    PPO[ppo.py]
    DQN[dqn.py]
    SAC[sac.py]
  end

  subgraph Training
    LOOP[Training loop]
    REP[Replay buffer<br/>DQN/SAC]
    LOG[Logger<br/>wandb + tensorboard]
    CKPT[Checkpoint writer]
    MAN[Manifest writer]
  end

  subgraph Artefacts
    RUNS[[runs/<run_id>/]]
    REF[[artefacts/reference/*.csv]]
    ONX[[artefacts/*.onnx]]
  end

  TC --> CF
  CF --> LOOP
  LOOP --> VE
  LOOP --> PPO & DQN & SAC
  PPO & DQN & SAC --> LOOP
  DQN & SAC --> REP
  LOOP --> LOG --> WB[Weights & Biases]
  LOOP --> CKPT --> RUNS
  LOOP --> MAN --> RUNS
  RC --> RUNS
  EC --> CKPT
  EC --> ONX
  LC --> RUNS
```

## Level 2 — PPO training iteration

```mermaid
sequenceDiagram
  autonumber
  participant T as Train loop
  participant V as VectorEnv
  participant A as Actor-Critic
  participant L as Logger
  loop collect rollout
    T->>V: step(action)
    V-->>T: obs, reward, done
    T->>A: value(obs); logprob(action)
  end
  T->>T: compute GAE advantages + returns
  loop update_epochs
    T->>T: minibatch iterate rollout
    T->>A: policy ratio r_t = pi_new / pi_old
    T->>A: clipped surrogate loss + value loss + entropy bonus
    T->>A: grad step + clip norm
  end
  T->>L: log KL, clipfrac, losses, reward
  T->>T: anneal lr, increment step
```

## Level 2 — Replay from manifest

```mermaid
sequenceDiagram
  autonumber
  participant U as User
  participant R as replay cmd
  participant M as Manifest
  participant D as Deps checker
  participant P as Policy
  participant E as Env
  U->>R: rl-arcade replay <run_id>
  R->>M: read manifest.json
  M-->>R: env_id, seed, eval_reward_mean, deps
  R->>D: verify installed deps match minor
  D-->>R: ok (or warn)
  R->>E: make(env_id, seed)
  R->>P: load(ckpt.pt)
  loop N episodes
    P->>E: act(obs)
    E-->>P: obs, reward, done
  end
  R-->>U: mean_reward vs stored ±2σ band
```

## Level 2 — ONNX export + serving

```mermaid
sequenceDiagram
  autonumber
  participant U as User
  participant X as export cmd
  participant C as Checkpoint
  participant O as torch.onnx
  participant V as onnxruntime
  participant S as Inference shim
  participant B as Browser
  U->>X: rl-arcade export <run_id> --format onnx
  X->>C: load actor
  X->>O: torch.onnx.export
  O-->>X: policy.onnx
  X->>V: verify roundtrip (torch vs onnxruntime, atol 1e-6)
  V-->>X: ok
  X-->>U: artefacts/policy.onnx
  U->>S: docker compose up inference
  B->>S: POST /policy/<id>/act {obs}
  S->>V: session.run(obs)
  V-->>S: action_logits
  S-->>B: {action, action_logits}
```

## Data stores

| Store | Purpose | Retention |
|-------|---------|-----------|
| `runs/<run_id>/` | Manifest, checkpoint, evaluation stats | Until purged |
| `artefacts/reference/` | Reference CSVs for (algo × env) | Versioned in git |
| `artefacts/<run_id>/policy.onnx` | Portable policy | Ephemeral |
| wandb | Experiment-tracking store | Managed |

## Trust boundaries

```mermaid
flowchart LR
  subgraph Local["Local / GPU host"]
    CLI
    TRAIN[Training loop]
    ENV[(Env sim)]
    STORE[(runs/ artefacts/)]
  end
  subgraph Services["External SaaS"]
    WB[wandb]
  end
  subgraph Deploy["Deployment (P3)"]
    API[Inference API]
    CDN[CDN for demo]
  end
  CLI --> TRAIN --> ENV
  TRAIN --> STORE
  TRAIN --> WB
  STORE --> API
  CDN --> API
```

## Invariants

- `manifest.json` is written atomically at end-of-training; partial runs are
  clearly marked `status="incomplete"`.
- Reference-curve CSV filenames encode `(algo, env_id, n_seeds)`.
- ONNX export runs a roundtrip test before being considered valid.
- No training code path uploads raw data to a third party without explicit
  wandb config (wandb_mode=disabled is respected).
- Seeds are recorded in the manifest; a rerun with the same manifest +
  deterministic flag must reproduce within band.
