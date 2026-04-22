"use client";

import { Pause, Play, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { CartPoleCanvas } from "@/components/cartpole-canvas";
import {
  CartPoleState,
  resetCartPole,
  stepCartPole,
  toObs,
} from "@/lib/cartpole-env";
import { PolicyRunner } from "@/lib/policy";

const MODEL_URL = "/models/cartpole_ppo.onnx";
const TICK_MS = 30; // ~33 Hz; env dt is 20 ms but rendering faster feels floaty

export default function CartPolePage() {
  const [state, setState] = useState<CartPoleState>(() => resetCartPole());
  const [running, setRunning] = useState(false);
  const [steps, setSteps] = useState(0);
  const [episodes, setEpisodes] = useState(0);
  const [bestReward, setBestReward] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const policyRef = useRef<PolicyRunner | null>(null);
  const stateRef = useRef(state);
  const stepRef = useRef(0);
  stateRef.current = state;
  stepRef.current = steps;

  useEffect(() => {
    const p = new PolicyRunner();
    p.load(MODEL_URL)
      .then(() => {
        policyRef.current = p;
        setLoaded(true);
      })
      .catch((e) => setError(e.message));
    return () => p.dispose();
  }, []);

  const tick = useCallback(async () => {
    const p = policyRef.current;
    if (!p) return;
    const obs = toObs(stateRef.current);
    const action = (await p.act(obs)) as 0 | 1;
    const { state: next, done } = stepCartPole(stateRef.current, action);
    setState(next);
    setSteps((n) => {
      const nn = n + 1;
      if (done) {
        setBestReward((b) => (nn > b ? nn : b));
        setEpisodes((e) => e + 1);
        setState(resetCartPole());
        return 0;
      }
      if (nn >= 500) {
        // Perfect episode under Gymnasium's CartPole-v1 cap.
        setBestReward((b) => (nn > b ? nn : b));
        setEpisodes((e) => e + 1);
        setState(resetCartPole());
        return 0;
      }
      return nn;
    });
  }, []);

  useEffect(() => {
    if (!running || !loaded) return;
    const id = setInterval(tick, TICK_MS);
    return () => clearInterval(id);
  }, [running, loaded, tick]);

  return (
    <main className="mx-auto max-w-4xl p-6 space-y-4">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">CartPole-v1 · PPO</h1>
          <p className="text-sm text-muted">
            Actor network running on WASM. Env dynamics match Gymnasium reference.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setRunning((r) => !r)}
            disabled={!loaded}
            className="inline-flex items-center gap-1 rounded-md bg-primary text-white px-3 py-1.5 text-sm disabled:opacity-40"
          >
            {running ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            {running ? "Pause" : loaded ? "Play" : "Loading…"}
          </button>
          <button
            onClick={() => {
              setState(resetCartPole());
              setSteps(0);
            }}
            className="inline-flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-sm"
          >
            <RotateCcw className="h-4 w-4" /> Reset
          </button>
        </div>
      </header>

      {error ? (
        <div className="rounded-xl border border-danger/50 bg-danger/10 p-3 text-sm text-danger">
          {error} — drop a policy at <code>public{MODEL_URL}</code>.
        </div>
      ) : null}

      <CartPoleCanvas state={state} />

      <section className="grid grid-cols-3 gap-3">
        <Card label="Current steps" value={`${steps}`} />
        <Card label="Episodes" value={`${episodes}`} />
        <Card label="Best episode" value={`${bestReward}`} tone="success" />
      </section>
    </main>
  );
}

function Card({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "success";
}) {
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <div className="text-xs uppercase text-muted tracking-wide">{label}</div>
      <div
        className={`mt-1 text-2xl font-semibold tabular-nums ${tone === "success" ? "text-success" : ""}`}
      >
        {value}
      </div>
    </div>
  );
}
