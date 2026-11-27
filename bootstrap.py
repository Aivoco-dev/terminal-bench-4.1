"""Checkout the published benchmark revision and keep results local."""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent


def require_tools(*names):
    missing = [name for name in names if not shutil.which(name)]
    if missing:
        raise RuntimeError("Install the required host tools: " + ", ".join(missing))


def ensure_env():
    path = REPO / ".env"
    if not path.exists():
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return
        with os.fdopen(fd, "w") as file:
            file.write((REPO / ".env.example").read_text())


def result_directory(*, benchmark, harness, release, tasks, workers, continual_learning, **options):
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    path = REPO / "results" / stamp
    path.mkdir(parents=True)
    metadata = {
        "benchmark": benchmark,
        "harness": harness,
        "release": release,
        "model": "vispark/vision-large",
        "continual_learning": continual_learning,
        "started_utc": now.isoformat(),
        "task_count": len(tasks),
        "tasks": tasks,
        "workers": workers,
        **options,
    }
    (path / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{benchmark} | {len(tasks)} tasks | {harness} | {workers} worker(s)", flush=True)
    print(f"Results: {path}", flush=True)
    return path
