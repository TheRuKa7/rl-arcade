"""Run manifest: the single source of truth about how a training run was produced.

Every run writes a JSON manifest next to its checkpoint containing:

- git SHA
- CLI args that invoked the run
- resolved hparams
- key dependency versions (torch, gymnasium, ale-py, CUDA)
- GPU model (if applicable)
- wandb run URL
- wall-clock + env-step counts

This lets ``rl-arcade replay <run_id>`` rehydrate a run with zero ambiguity.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass(slots=True)
class RunManifest:
    """Serialisable record of a single training run."""

    run_id: str
    algo: str
    env_id: str
    seed: int
    total_timesteps: int
    hparams: dict[str, Any]
    dependencies: dict[str, str]
    git_sha: str = field(default_factory=_git_sha)
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    )
    wandb_url: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> RunManifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)
