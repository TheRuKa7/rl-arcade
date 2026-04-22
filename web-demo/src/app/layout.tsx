import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "rl-arcade — browser demo",
  description: "Watch CleanRL-style policies play in your browser via onnxruntime-web",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background text-foreground">
        <nav className="border-b border-border px-6 py-3 flex items-center gap-4">
          <Link href="/" className="font-semibold">
            rl-arcade
          </Link>
          <Link href="/cartpole" className="text-sm text-muted hover:text-foreground">
            CartPole
          </Link>
          <Link href="/lunar" className="text-sm text-muted hover:text-foreground">
            LunarLander
          </Link>
          <a
            href="https://github.com/TheRuKa7/rl-arcade"
            target="_blank"
            rel="noreferrer"
            className="ml-auto text-sm text-muted hover:text-foreground"
          >
            GitHub →
          </a>
        </nav>
        {children}
      </body>
    </html>
  );
}
