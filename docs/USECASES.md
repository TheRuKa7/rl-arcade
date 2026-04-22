# USECASES — rl-arcade

End-to-end flows for a reproducible RL lab. The emphasis is on the *engineering
substrate* — reference curves, run manifests, ONNX export — so trained policies
don't rot in a drawer.

## 1. Personas

| ID | Persona | Context | Primary JTBD |
|----|---------|---------|--------------|
| P1 | **ML interviewer (Priya)** | Evaluating Rushil's RL depth in a live session | "Ask him to explain PPO and see working code" |
| P2 | **Self-learner (Alex)** | New to RL, wants a readable PPO reference | "Walk me through a single-file PPO that actually works" |
| P3 | **Games-AI researcher (Maya)** | Wants a cheap substrate for ablations | "Swap my reward function in; keep everything else stable" |
| P4 | **RLHF-curious LLM eng (Jordan)** | Wants the bridge from classical PPO to RLHF | "Show me the same PPO equation as `trl`" |
| P5 | **Recruiter / non-technical (Sam)** | Visits the repo page, 3 min budget | "Let me see a trained agent play right in the browser" |

## 2. Jobs-to-be-done

JTBD-1. **Readable algorithm reference** (CleanRL-style single-file).
JTBD-2. **Reproducible experiments** — seeded, versioned, replayable.
JTBD-3. **Reference curves** — match a known benchmark, catch regressions in CI.
JTBD-4. **Cross-check oracle** — SB3 sanity script for "is my PPO correct?".
JTBD-5. **Portable policy** — ONNX export, browser demo.
JTBD-6. **Bridge to RLHF** — tie classical PPO to LLM post-training via docs.

## 3. End-to-end flows

### Flow A — Priya grills Rushil on PPO

1. In interview, Priya asks "walk me through PPO."
2. Rushil opens `agents/ppo.py`, points to clip loss, GAE, entropy term.
3. Priya asks "when does DQN beat PPO?"; Rushil answers with `docs/THINK.md` talking points.
4. Priya asks about RLHF; Rushil shows `docs/RLHF.md` mapping PPO → InstructGPT.

### Flow B — Alex learns PPO by running it

1. `uv sync --extra dev`, then `rl-arcade train --env CartPole-v1 --algo ppo`.
2. Watches reward climb in wandb; CartPole solved in 3 minutes on CPU.
3. Reads `agents/ppo.py` top-to-bottom; each hparam has a citation.
4. Runs `scripts/sanity_sb3.py` to confirm curves match SB3's.

### Flow C — Maya runs an ablation

1. Forks the repo; changes reward shaping in her env wrapper.
2. `rl-arcade train --env her-env --algo ppo --total-timesteps 2_000_000`.
3. wandb compare-runs shows her shaped vs baseline.
4. Re-runs with `--deterministic` for reproducibility.

### Flow D — Jordan maps PPO to RLHF

1. Opens `docs/RLHF.md`; sees line-by-line correspondence.
2. Notes the KL penalty as the "only" addition to RLHF-PPO.
3. Cites the repo in an internal doc on RLHF implementation.

### Flow E — Sam views the browser demo

1. Visits the deployed demo page (P4 stretch goal).
2. Sees a CartPole agent playing inside a canvas.
3. Clicks "Breakout" (if reached) and watches the agent score.
4. Bookmarks for later.

### Flow F — Contributor adds SAC

1. Implements `agents/sac.py` to the single-file-agent contract.
2. Adds `src/rl_arcade/agents/__init__.py` dispatch.
3. Ships a reference curve CSV under `artefacts/reference/sac_pendulum.csv`.
4. CI nightly runs 20k-step sanity check on Pendulum-v1.

## 4. Acceptance scenarios

```gherkin
Scenario: CartPole PPO reaches 500 reward in under 50k steps
  Given the scaffold on a CPU-only runner
  When I run rl-arcade train --env CartPole-v1 --algo ppo --total-timesteps 50000
  Then the final evaluation reward is >= 500
  And the total wall-clock is <= 5 minutes

Scenario: CI sanity catches a PPO regression
  Given the short-horizon CI budget (10k steps)
  When a PR introduces a bug that breaks PPO
  Then the sanity run's end reward is below the guard threshold (150)
  And CI fails with a pointer to the reference curve

Scenario: Run manifest is sufficient for replay
  Given a completed run at runs/<run_id>/
  When I run rl-arcade replay <run_id> --episodes 5
  Then the eval reward is within 2 sigma of the stored eval_reward_mean

Scenario: ONNX export round-trips
  Given a trained actor
  When I run rl-arcade export <run_id> --format onnx
  Then onnxruntime inference on random observations matches torch inference within 1e-6
```

## 5. Non-use-cases

- Beating SOTA on Atari or MuJoCo (match references; don't advance)
- Distributed / large-scale training (IMPALA, Ape-X) — noted, not built
- Robotics simulators (Isaac) — out of scope
- Offline RL — on the roadmap as a separate module
