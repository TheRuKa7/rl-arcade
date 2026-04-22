# rl-arcade — web demo

A minimal Next.js 15 site that loads ONNX policies exported by
`rl-arcade export` and runs them against a JS-ported CartPole environment
in your browser. No server required after first load.

## Stack

- Next.js 15 (App Router, Turbopack) + React 19
- `onnxruntime-web` (WASM backend) for policy inference
- Vanilla-JS port of Gymnasium's CartPole-v1 dynamics
- Canvas 2D renderer (no WebGL dependency)
- Tailwind + Lucide

## Routes

| Path          | Purpose                                       |
|---------------|-----------------------------------------------|
| `/`           | Landing page, demo cards                      |
| `/cartpole`   | Live interactive CartPole-v1 with PPO actor   |
| `/lunar`      | Placeholder for LunarLander (Box2D port TBD)  |

## Run

```bash
cd web-demo
pnpm install
pnpm dev                        # http://localhost:3002
```

On first `pnpm install`, a copy-webpack-plugin step in `next.config.mjs`
copies `onnxruntime-web` WASM binaries to `public/ort/`.

## Shipping a policy

1. Train: `rl-arcade train --env CartPole-v1 --algo ppo --total-timesteps 50000`
2. Export:
   ```bash
   rl-arcade export <run_id> --format onnx \
     --out web-demo/public/models/cartpole_ppo.onnx
   ```
3. The export command runs a `torch` vs `onnxruntime` roundtrip check
   (atol 1e-6) before writing — see `RFC.md` §3.5.
4. `pnpm dev` and hit `/cartpole`.

## Actor contract

The model must:

- accept a float32 tensor `[1, obs_dim]`
- return logits `[1, n_actions]`

Greedy argmax → action. That matches `agents/ppo.py`'s discrete actor output.

## Deploy

Push to Vercel or Cloudflare Pages. The `.onnx` files are served as static
assets; the WASM runtime streams on demand. CartPole policies are ~10–50 KB
so Lighthouse-style "time-to-play" stays under a few seconds.
