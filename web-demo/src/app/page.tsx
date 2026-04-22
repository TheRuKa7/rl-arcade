import Link from "next/link";
import { ArrowRight, Cpu, Rocket } from "lucide-react";

export default function Home() {
  return (
    <main className="mx-auto max-w-4xl p-8 space-y-8">
      <section>
        <h1 className="text-3xl font-semibold">Policies in your browser</h1>
        <p className="text-muted mt-2 max-w-2xl">
          Every agent in rl-arcade trains in PyTorch, exports to ONNX with a
          roundtrip test, and runs here on WebAssembly — no server, no GPU,
          no network after first load. Click a demo to watch.
        </p>
      </section>

      <section className="grid md:grid-cols-2 gap-4">
        <Link
          href="/cartpole"
          className="rounded-2xl border border-border bg-surface p-5 hover:border-primary transition"
        >
          <div className="flex items-center gap-2 text-primary">
            <Cpu className="h-5 w-5" />
            <span className="text-xs uppercase font-semibold tracking-wide">PPO</span>
          </div>
          <h2 className="text-xl font-semibold mt-2">CartPole-v1</h2>
          <p className="text-muted text-sm mt-1">
            Classic-control baseline. ≤50k steps on CPU, solved (500 reward) by
            the reference curve.
          </p>
          <div className="mt-4 text-sm text-foreground inline-flex items-center gap-1">
            Run demo <ArrowRight className="h-4 w-4" />
          </div>
        </Link>

        <Link
          href="/lunar"
          className="rounded-2xl border border-border bg-surface p-5 hover:border-primary transition"
        >
          <div className="flex items-center gap-2 text-primary">
            <Rocket className="h-5 w-5" />
            <span className="text-xs uppercase font-semibold tracking-wide">PPO</span>
          </div>
          <h2 className="text-xl font-semibold mt-2">LunarLander-v2</h2>
          <p className="text-muted text-sm mt-1">
            Coming online — physics sim runs server-side; this page links to
            the recorded eval video until the Box2D port lands.
          </p>
          <div className="mt-4 text-sm text-foreground inline-flex items-center gap-1">
            Learn more <ArrowRight className="h-4 w-4" />
          </div>
        </Link>
      </section>

      <section className="text-sm text-muted border-t border-border pt-4">
        How it works: <code className="text-foreground">rl-arcade export</code>{" "}
        runs <code className="text-foreground">torch.onnx.export</code> with a
        1e-6 roundtrip check, uploads to <code>/public/models/</code>, and{" "}
        <code>onnxruntime-web</code> (wasm backend) evaluates the actor against
        a vanilla-JS port of the env.
      </section>
    </main>
  );
}
