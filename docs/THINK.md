# THINK — rationale, tradeoffs, risks, interview framing

A portfolio repo exists to answer one question in an interview:
**"walk me through an RL project and defend your decisions."**

This doc is the defence.

---

## 1. Why RL at all, and why *this* framing

Rushil's portfolio covers perception (iDAS), forecasting (quanta-forecast), retrieval
(doc-rag), agents (pm-copilot), generative 3D (splat-studio), and SaaS infra
(mern-devsuite).  The one "how does the model *act* in the world" discipline missing is
sequential decision-making.  RL closes that gap *and* links directly to RLHF — the
bridge between classical ML and modern LLM post-training.

The framing is deliberately not "I beat SOTA on Atari."  It's:
**"I built a reproducibility substrate for RL experiments, trained three canonical
algorithms on three regimes, and exposed the trained policies as deployable artefacts."**

That's the PM-flavoured engineering-manager story: infra, rigour, deployability.

---

## 2. CleanRL over Stable-Baselines3 — why

| Criterion | CleanRL-style | SB3 |
|-----------|---------------|-----|
| Readability | ~400 LoC per algo, all in one file | abstracted across base classes |
| Portfolio signal | "I can implement PPO from the paper" | "I can call `.learn()`" |
| Debuggability | step through with a debugger; no magic | heavier indirection |
| Correctness guarantee | we write it, we own bugs | battle-tested |
| Perf | equivalent; both use torch | equivalent |

Picking CleanRL is a *capability signal*.  But we still install SB3 — as an oracle.
`scripts/sanity_sb3.py` runs SB3's PPO on the same env; if our curve disagrees
with SB3's by more than noise, one of us is wrong and we investigate.

**Interview soundbite:** *"I wrote PPO myself to prove I understand the algorithm,
but I cross-check against SB3 so I don't ship a plausible-looking bug."*

---

## 3. Why Gymnasium, not Gym

OpenAI Gym was deprecated in 2022; Farama Foundation forked it as **Gymnasium** and
has shipped breaking-change releases that matter (env registration, `step` 5-tuple,
wrapper redesign, ALE integration).  Pinning to `gymnasium==1.0.*` is the default
for any RL repo written in 2024+.  Using `gym` in 2026 is a red flag.

---

## 4. Hyperparameter philosophy

- **No magic numbers without a comment** — every hparam in `config.py` has a docstring
  linking to the paper section or ablation where we got it.
- **Start from CleanRL defaults** — they're curated and known-good.
- **Change one thing at a time** — runs are tagged with a `diff_from_default` field
  in the wandb config so we can bisect regressions.

---

## 5. The three things that go wrong in RL (and how we mitigate)

### 5.1 Silent non-learning

**Symptom:** reward curve is flat, no error is raised.
**Mitigations:**
- Short-horizon CI sanity check (10–50k steps) asserts positive reward gradient.
- wandb dashboards flag zero-KL (policy collapsed) and entropy=0 (deterministic).
- Reference curves — if our curve is flat but the reference isn't, bug is ours, not RL's.

### 5.2 Non-reproducibility

**Symptom:** "it worked on my machine", or the same seed gives different curves.
**Mitigations:**
- Full seeding (torch, numpy, env, `CUBLAS_WORKSPACE_CONFIG`).
- `torch.use_deterministic_algorithms(True)` behind a `--deterministic` flag
  (documented perf hit).
- Run manifest: git SHA + CLI args + env version → reproduce from manifest.
- CUDA / cudnn version pinned in Docker; rental GPU type recorded.

### 5.3 Environment-version drift

**Symptom:** Atari Breakout scores drop 20 % after a `gymnasium` point release.
**Mitigations:**
- Pin `gymnasium` and `ale-py` to exact minor versions.
- Reference curves tagged with the env version that produced them.
- CI asserts installed env versions match the manifest.

---

## 6. Costs

| Item | Quantity | Unit cost | Notes |
|------|----------|-----------|-------|
| CPU CI (GitHub-hosted) | ~5 min/push | free up to 2k min/mo | smoke tests only |
| GPU training (A10G rental) | ~20 h for P2/P3 | ~$0.50/h on Runpod | one-time per repo state |
| wandb | Personal plan | free for public projects | enough for portfolio |
| Cloudflare Pages / Vercel | 1 static site | free tier | demo hosting |
| Domain (optional) | 1 | ~$12/yr | e.g. rl-arcade.rushilkaul.dev |

All-in: < $20 to reach P3, < $50 if we splurge on continuous experiments.

---

## 7. What *isn't* in the repo and why

- **StarCraft / Dota / full-scale MineRL** — compute budget runs into thousands of
  GPU-hours.  Portfolio-inappropriate.  Documented in RESEARCH so a reviewer knows
  we know they exist.
- **Ray RLlib / distributed training** — the repo's portfolio value is readability,
  not scale.  We mention IMPALA / Ape-X in RESEARCH for completeness.
- **Novel algorithms** — we implement known algorithms carefully.  Original research
  is a PhD, not a portfolio line.
- **Unity ML-Agents** — interesting but bolts on a whole game engine; out of scope.

---

## 8. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Gymnasium 1.x API drifts further | Med | Med | Pinned; upgrade deliberately, retest references |
| Silent bug matches "plausible" curves | Med | High | SB3 cross-check; reference curve CI |
| Compute budget blown on debugging | Med | Med | Always start on CartPole (3 min), escalate only when sane |
| ALE ROM licensing confusion | Low | Med | Docs are explicit: we don't ship ROMs |
| Recruiter runs it, it doesn't work | Low | High | Dev container + `make quickstart` + pinned deps |

---

## 9. Interview talking points

### "Explain PPO like I'm a smart intern."

PPO runs multiple epochs of SGD on the same rollout, but clips the ratio between
new and old policy probabilities so the step can't be too large.  That's it.
The magic of PPO is not the math — it's the *engineering*: GAE for advantage
estimation, orthogonal init, linear LR schedule, value-function clipping, advantage
normalisation.  You can drop any one and performance tanks.

### "When is DQN better than PPO?"

Discrete actions, single-env offline-ish setups, and when you want sample efficiency
to dominate wall-clock.  PPO is better when you can parallelise environments cheaply
(robotics sims, game engines) or when action space is continuous.

### "How does this connect to LLMs?"

RLHF uses PPO to update an LLM against a reward model.  The exact clipped-surrogate
objective in our `ppo.py` is the same objective `trl`'s `PPOTrainer` uses — just
with the environment being "sample text, get a reward-model score" instead of
"step CartPole".  Interview-level fluency in PPO therefore pays dividends in *both*
classical RL *and* LLM post-training.

### "What would you change if you had a team?"

Prioritised: (a) port to IMPALA for 100×-scale throughput, (b) DreamerV3 for
sample efficiency on harder games, (c) an offline-RL track using D4RL so we can
do research without burning GPU hours.  In that order.

### "What's the PM-flavoured lesson?"

Most "AI wow" demos break when you try to reproduce them.  The scarce resource
isn't SOTA numbers — it's *trust*.  This repo trades sexy for reliable: pinned
versions, reference curves in CI, a replay command that actually runs a year later.
That's the kind of thinking that ships products.

---

## 10. Honest disclaimers

- **Not original research.** Algorithms are faithful implementations of published work.
- **Performance is reference-matching.** If you need SOTA Atari numbers, use MuZero.
- **Not a robotics solution.** Sim-to-real gap is a real problem and it is not solved here.
- **Some Atari envs have reward sparsity** that toy budgets won't crack; Montezuma's
  Revenge is deliberately *not* on our benchmark list.
