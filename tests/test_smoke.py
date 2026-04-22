"""Smoke tests — fast, no heavy deps. Run in CI on every push."""

from __future__ import annotations

from typer.testing import CliRunner

from rl_arcade import __version__
from rl_arcade.__main__ import app
from rl_arcade.config import AppConfig, load_config
from rl_arcade.envs import EnvSpec, list_envs
from rl_arcade.training.manifest import RunManifest


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_load_config_returns_appconfig() -> None:
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.ppo.clip_coef == 0.2
    assert cfg.dqn.buffer_size == 1_000_000
    assert cfg.sac.gamma == 0.99


def test_cli_help_exits_cleanly() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "rl-arcade" in result.stdout.lower()


def test_cli_train_dry_run_parses_and_exits() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["train", "--env", "CartPole-v1", "--algo", "ppo", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "CartPole-v1" in result.stdout


def test_cli_train_rejects_unknown_algo() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["train", "--algo", "bogus", "--dry-run"])
    assert result.exit_code != 0


def test_env_registry_has_representative_entries() -> None:
    all_envs = list_envs()
    env_ids = {e.env_id for e in all_envs}
    assert "CartPole-v1" in env_ids
    assert "ALE/Pong-v5" in env_ids
    assert "HalfCheetah-v5" in env_ids

    atari = list_envs(regime="atari")
    assert all(isinstance(spec, EnvSpec) for spec in atari)
    assert all(spec.recommended_algo == "dqn" for spec in atari)


def test_run_manifest_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest = RunManifest(
        run_id="test-001",
        algo="ppo",
        env_id="CartPole-v1",
        seed=1,
        total_timesteps=50_000,
        hparams={"learning_rate": 2.5e-4, "clip_coef": 0.2},
        dependencies={"torch": "2.5.0", "gymnasium": "1.0.0"},
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)
    assert path.exists()

    loaded = RunManifest.read(path)
    assert loaded.run_id == "test-001"
    assert loaded.algo == "ppo"
    assert loaded.hparams["clip_coef"] == 0.2
