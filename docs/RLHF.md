# Note: PPO here and PPO in ChatGPT

A short essay (living doc) drawing the line from the PPO in this repo to the PPO
that fine-tunes large language models.

## The same objective

Both classical-RL PPO (this repo) and RLHF-PPO (e.g. `trl.PPOTrainer`) optimise:

```
L(θ) = E_t [ min( r_t(θ) · A_t,  clip(r_t(θ), 1-ε, 1+ε) · A_t ) ]
```

where `r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)` is the ratio between new and
old policy probabilities. Identical equation, different state/action spaces.

## What changes in RLHF

| Element      | Classical RL           | RLHF                                             |
|--------------|------------------------|--------------------------------------------------|
| State `s_t`  | env observation         | prompt + generated prefix                        |
| Action `a_t` | env action              | next token                                       |
| Reward       | env reward              | reward-model score on full generation            |
| Policy `π`   | small MLP / CNN         | multi-billion-parameter LLM                      |
| Rollout      | `env.step`              | autoregressive sampling                          |
| Extra term   | —                       | KL penalty vs SFT policy (stop reward hacking)   |

The KL penalty and the reward-model formulation are *the* RLHF additions.
Everything else is the PPO in `agents/ppo.py`.

## Why this matters for a portfolio

If an interviewer asks "how does ChatGPT learn from human feedback?", you can
point to `agents/ppo.py`, explain each term, and then describe the two additions
(reward model + KL penalty).  Bridging classical RL to LLM post-training with
one file of code is a strong signal.

## Pointers

- Ouyang et al., "Training language models to follow instructions with human
  feedback" (InstructGPT), 2022 — the canonical RLHF-PPO paper.
- `trl` library — https://github.com/huggingface/trl; `PPOTrainer` is the
  RLHF cousin of what we implement.
- DPO (Rafailov et al. 2023) — shows you can skip PPO entirely; a useful
  *counterpoint* to discuss.
