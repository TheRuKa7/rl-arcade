"""rl-arcade CLI entry point.

Subcommands:

- ``train``    launch a training run (P0: dry-run only; P1 wires real PPO)
- ``replay``   roll a trained checkpoint, optionally render
- ``export``   convert a checkpoint to ONNX or TorchScript
- ``runs``     list / inspect local runs

All heavy imports (torch, gymnasium) are lazy so ``--help`` is snappy and CI
smoke tests don't pay the import cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rl_arcade.config import load_config

app = typer.Typer(
    name="rl-arcade",
    help="Deep RL lab for game-playing agents.",
    no_args_is_help=True,
    add_completion=False,
)
runs_app = typer.Typer(help="Inspect saved training runs.")
app.add_typer(runs_app, name="runs")


@app.command()
def train(
    env: Annotated[str, typer.Option(help="Gymnasium env id, e.g. CartPole-v1")] = "CartPole-v1",
    algo: Annotated[str, typer.Option(help="ppo | dqn | sac")] = "ppo",
    total_timesteps: Annotated[int, typer.Option(min=1)] = 50_000,
    seed: Annotated[int, typer.Option()] = 1,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Parse config and exit without training.")
    ] = False,
) -> None:
    """Launch a training run."""
    cfg = load_config()
    cfg.seed = seed

    typer.echo(f"[rl-arcade] algo={algo} env={env} total_timesteps={total_timesteps:,} seed={seed}")
    typer.echo(f"[rl-arcade] device={cfg.device} runs_dir={cfg.runs_dir} wandb={cfg.wandb_mode}")

    if algo not in {"ppo", "dqn", "sac"}:
        raise typer.BadParameter(f"Unknown algo: {algo!r}. Expected one of: ppo, dqn, sac")

    if dry_run:
        typer.echo("[rl-arcade] --dry-run set; config parsed OK. Exiting.")
        raise typer.Exit(code=0)

    typer.echo(
        "[rl-arcade] P0 scaffold: training loop not yet wired. "
        "Track https://github.com/TheRuKa7/rl-arcade/blob/main/docs/PLAN.md for P1."
    )
    raise typer.Exit(code=0)


@app.command()
def replay(
    run_id: Annotated[str, typer.Argument(help="Run id (folder name under runs/)")],
    episodes: Annotated[int, typer.Option(min=1)] = 5,
    render: Annotated[bool, typer.Option("--render", help="Open a render window.")] = False,
) -> None:
    """Replay a trained policy from a saved run."""
    typer.echo(f"[rl-arcade] replay run_id={run_id} episodes={episodes} render={render}")
    typer.echo("[rl-arcade] P0 scaffold: replay not yet wired; arrives in P1.")
    raise typer.Exit(code=0)


@app.command()
def export(
    run_id: Annotated[str, typer.Argument(help="Run id")],
    fmt: Annotated[str, typer.Option("--format", help="onnx | torchscript")] = "onnx",
    out: Annotated[Path, typer.Option(help="Output path")] = Path("artefacts/policy.onnx"),
) -> None:
    """Export a trained policy to a portable format."""
    if fmt not in {"onnx", "torchscript"}:
        raise typer.BadParameter(f"Unknown format: {fmt!r}")
    typer.echo(f"[rl-arcade] export run_id={run_id} format={fmt} out={out}")
    typer.echo("[rl-arcade] P0 scaffold: export not yet wired; arrives in P4.")
    raise typer.Exit(code=0)


@runs_app.command("list")
def runs_list() -> None:
    """List local training runs."""
    cfg = load_config()
    runs_dir = Path(cfg.runs_dir)
    if not runs_dir.exists():
        typer.echo(f"[rl-arcade] no runs directory at {runs_dir.resolve()}")
        raise typer.Exit(code=0)
    found = sorted(p.name for p in runs_dir.iterdir() if p.is_dir())
    if not found:
        typer.echo(f"[rl-arcade] {runs_dir.resolve()} is empty")
        raise typer.Exit(code=0)
    for name in found:
        typer.echo(name)


if __name__ == "__main__":
    app()
