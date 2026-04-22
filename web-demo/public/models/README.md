# public/models/

Export policies here via `rl-arcade export <run_id> --format onnx
--out web-demo/public/models/<env>_<algo>.onnx`.

Expected layout:

```
public/models/
├── cartpole_ppo.onnx      # PolicyRunner loads this for /cartpole
└── lunar_ppo.onnx         # (when Box2D port lands)
```

Each actor must accept a float32 observation tensor of shape `[1, obsDim]`
and return logits of shape `[1, nActions]`. That's the default output of
`agents/ppo.py`'s discrete actor.
