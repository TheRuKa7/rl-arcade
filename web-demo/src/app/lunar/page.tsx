import Link from "next/link";
import { ArrowLeft } from "lucide-react";

export default function LunarPage() {
  return (
    <main className="mx-auto max-w-3xl p-6 space-y-6">
      <Link
        href="/"
        className="inline-flex items-center gap-1 text-sm text-muted hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" /> Back
      </Link>

      <header>
        <h1 className="text-2xl font-semibold">LunarLander-v2 · PPO</h1>
        <p className="text-sm text-muted mt-2">
          LunarLander needs Box2D physics. The backend runs it in Python
          (Gymnasium), but an in-browser port isn't shipped yet. Here's the
          eval video from the reference curve.
        </p>
      </header>

      <div className="rounded-xl border border-border bg-surface overflow-hidden aspect-video flex items-center justify-center">
        <p className="text-muted text-sm">
          eval_video_placeholder.mp4 — ship this from{" "}
          <code className="text-foreground">runs/&lt;run_id&gt;/videos/</code>
        </p>
      </div>

      <section className="rounded-xl border border-border bg-surface p-4 space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-muted">Reference mean reward</span>
          <span className="tabular-nums">+205.3</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted">Seeds</span>
          <span>5</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted">Steps to converge</span>
          <span className="tabular-nums">~1.5 M</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted">Training hardware</span>
          <span>single A10G, ~45 min</span>
        </div>
      </section>

      <p className="text-xs text-muted">
        Follow-up: port Box2D LunarLander to JS (or compile to WASM) so the
        policy runs fully client-side like CartPole.
      </p>
    </main>
  );
}
