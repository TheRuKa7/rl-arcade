# PRD — rl-arcade

**Owner:** Rushil Kaul · **Status:** P0 scaffold complete · **Last updated:** 2026-04-22

## 1. TL;DR

A Deep-RL reference lab with **CleanRL-style single-file agents (PPO/DQN/SAC)**,
reproducible run manifests, reference-curve CI, and an ONNX export path that
ends in a browser-playable demo.

## 2. Problem

Public RL repos are either (a) cluttered frameworks that obscure the algorithm
or (b) notebook-grade demos with no reproducibility story. `rl-arcade` fills
the middle: readable code and production-grade repro, no SOTA claims.

## 3. Goals

| G | Goal | Measure |
|---|------|---------|
| G1 | Readable single-file algorithms | Each algo ≤ ~400 LoC with inline citations |
| G2 | Reference-matching curves | Each (algo × env) pair ships a CSV curve we hit in CI |
| G3 | Reproducibility | Seeded everything; manifest per run; replay CLI works a year later |
| G4 | Cross-family coverage | PPO (on-policy), DQN (off-policy), SAC (continuous) all implemented |
| G5 | Deployable policy | ONNX export + FastAPI inference shim + (stretch) web demo |

## 4. Non-goals

- SOTA benchmark numbers
- Distributed training (IMPALA / Ape-X)
- Robotics / sim-to-real
- Novel algorithmic research

## 5. Users & stakeholders

See `USECASES.md` P1–P5.

## 6. Functional requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| F1 | `rl-arcade train` with `--env --algo --total-timesteps --seed` | P0 |
| F2 | `rl-arcade replay <run_id>` loads checkpoint + manifest | P1 |
| F3 | `rl-arcade export <run_id> --format onnx` | P2 |
| F4 | `rl-arcade runs list` / inspect | P0 |
| F5 | PPO single-file (classic control) | P1 |
| F6 | DQN + Rainbow-lite (Atari) | P2 |
| F7 | SAC (continuous control) | P3 |
| F8 | PettingZoo self-play (stretch) | P3 |
| F9 | wandb logging + local JSON manifest | P1 |
| F10 | CI sanity runs for PPO (short horizon) | P1 |
| F11 | Reference-curve CSV per algo × env | P1 |
| F12 | SB3 cross-check script | P1 |

## 7. Non-functional requirements

| Category | Requirement |
|----------|-------------|
| Reproducibility | `--deterministic` flag produces same curves; manifest records env versions |
| Performance | CartPole solved in <5 min CPU; Atari Pong in <8 h on an A10G |
| Cost | Zero-cost CI (no GPU); GPU jobs runnable locally or rented |
| Docs | Every hparam has a citation (paper or CleanRL reference) |
| Packaging | Families as extras (`[atari]`, `[mujoco]`, `[onnx]`, `[serve]`) |

## 8. Success metrics

- **Primary:** reference-curve pass/fail in CI for every (algo × env).
- **Secondary:** number of seeds tested per benchmark curve.
- **Educational:** README Lighthouse-style "time-to-first-train" for a newcomer.

## 9. Milestones

| Phase | Deliverable | ETA |
|-------|-------------|-----|
| P0 | Scaffold, CLI, config, run-manifest, tests | shipped |
| P1 | PPO on CartPole + LunarLander, reference curves, wandb, CI sanity | +1 week |
| P2 | DQN + Atari preprocess + Pong reference | +3 weeks |
| P3 | SAC + MuJoCo + PettingZoo self-play + ONNX export + web demo | +6 weeks |
| P4 | Stretch: DreamerV3 module, offline-RL module, RLHF essay | +10 weeks |

## 10. Dependencies

- Gymnasium 1.0, ALE-py, MuJoCo, PettingZoo
- PyTorch 2.5, wandb
- ONNX, onnxruntime (for export + web)
- Optional SB3 for the oracle

## 11. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Silent PPO bug (plausible curves) | Med | Wrong credibility | SB3 oracle script + reference-curve CI |
| Env version drift | Cert. | Benchmark shift | Pin gymnasium/ale-py to exact minors |
| Reader runs code 1 year later and fails | Cert. | Rep risk | Run manifest + pinned deps + `--deterministic` mode |
| Atari ROM licensing confusion | Med | Legal | Explicit "we don't ship ROMs" in docs |
| GPU cost for full benchmarks | Med | $ | Rent on demand; document expected cost |

## 12. Open questions

- Should we ship a minimal `trainer/` class as a shared utility or keep every agent
  truly single-file? Lean: keep single-file for readability.
- Whether to run full nightly reference curves on a self-hosted GPU. Expensive;
  start with manual pre-release runs.
- Add a `rl-arcade compare <run_a> <run_b>` CLI for diffing two runs? Nice-to-have.
