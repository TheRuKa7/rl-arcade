"use client";

import * as ort from "onnxruntime-web";

// Serve ORT wasm from /ort (see next.config.mjs CopyPlugin).
ort.env.wasm.wasmPaths = "/ort/";

/**
 * Wraps an exported policy actor. The ONNX model is expected to accept a
 * float32 observation tensor of shape `[1, obsDim]` and produce action
 * logits of shape `[1, nActions]` (for discrete envs). Continuous policies
 * return mean / stddev — those aren't needed for the two demo envs.
 */
export class PolicyRunner {
  private session: ort.InferenceSession | null = null;

  async load(modelUrl: string): Promise<void> {
    if (this.session) return;
    this.session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }

  /** Greedy argmax over logits. */
  async act(obs: Float32Array): Promise<number> {
    if (!this.session) throw new Error("policy not loaded");
    const input = new ort.Tensor("float32", obs, [1, obs.length]);
    const inputName = this.session.inputNames[0];
    const outputName = this.session.outputNames[0];
    const feeds: Record<string, ort.Tensor> = { [inputName]: input };
    const out = await this.session.run(feeds);
    const logits = out[outputName].data as Float32Array;
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }

  dispose(): void {
    // ort.InferenceSession has no close in web build; drop the ref.
    this.session = null;
  }
}
