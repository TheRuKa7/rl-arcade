"use client";

import { useEffect, useRef } from "react";
import type { CartPoleState } from "@/lib/cartpole-env";
import { THETA_THRESHOLD_RADIANS, X_THRESHOLD } from "@/lib/cartpole-env";

export function CartPoleCanvas({ state }: { state: CartPoleState }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;

    const W = c.width;
    const H = c.height;
    ctx.clearRect(0, 0, W, H);

    // background
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, W, H);

    // track
    const trackY = H * 0.75;
    ctx.strokeStyle = "#2a2a2a";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, trackY);
    ctx.lineTo(W, trackY);
    ctx.stroke();

    const worldToScreen = (x: number): number =>
      (W / 2) + (x / X_THRESHOLD) * (W * 0.35);

    const cartX = worldToScreen(state.x);
    const cartW = 60;
    const cartH = 28;

    // cart
    ctx.fillStyle = "hsl(200 85% 60%)";
    ctx.fillRect(cartX - cartW / 2, trackY - cartH / 2, cartW, cartH);

    // pole
    const poleLen = 120;
    const poleX2 = cartX + Math.sin(state.theta) * poleLen;
    const poleY2 = trackY - cartH / 2 - Math.cos(state.theta) * poleLen;
    ctx.strokeStyle = "#f5f5f5";
    ctx.lineWidth = 6;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(cartX, trackY - cartH / 2);
    ctx.lineTo(poleX2, poleY2);
    ctx.stroke();

    // axle
    ctx.fillStyle = "#1e1e1e";
    ctx.beginPath();
    ctx.arc(cartX, trackY - cartH / 2, 5, 0, Math.PI * 2);
    ctx.fill();

    // danger markers at theta threshold
    ctx.fillStyle = "hsla(0 84% 60% / 0.4)";
    ctx.fillRect(0, 0, 10, H);
    ctx.fillRect(W - 10, 0, 10, H);

    // HUD
    ctx.fillStyle = "hsl(240 4% 60%)";
    ctx.font = "12px monospace";
    ctx.fillText(`x=${state.x.toFixed(2)}  θ=${state.theta.toFixed(3)}`, 14, 20);
    ctx.fillText(
      `|θ|<${THETA_THRESHOLD_RADIANS.toFixed(3)}  |x|<${X_THRESHOLD}`,
      14,
      36,
    );
  }, [state]);

  return (
    <canvas
      ref={ref}
      width={720}
      height={360}
      className="w-full rounded-xl border border-border bg-background"
    />
  );
}
