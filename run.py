import argparse
import json
import os
import subprocess
import sys

from bootstrap import REPO, ensure_env, require_tools, result_directory
from cl_adapter import load_key, read_env


def main():
    settings = read_env(REPO)
    parser = argparse.ArgumentParser(description="Terminal-Bench 4.1 with Vision continual learning")
    parser.add_argument("--workers", type=int, default=settings.get("BENCH_WORKERS") or "1")
    parser.add_argument("--task", action="append", help="Exact task ID. Repeat to select several tasks.")
    parser.add_argument("--plan", action="store_true", help="Show the selected task list without inference")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    known = set((REPO / "tasks.txt").read_text().splitlines())
    if args.task and (len(args.task) != len(set(args.task)) or not set(args.task) <= known):
        parser.error("Unknown Terminal-Bench task")
    if len(known) != 66:
        raise RuntimeError("Expected the 66-task Terminal-Bench release")
    selected = args.task or sorted(known)
    if args.plan:
        print(json.dumps({
            "dataset": "terminal-bench/terminal-bench@4.1.0", "harness": "Harbor + Vision CL",
            "workers": args.workers, "task_count": len(selected), "tasks": selected,
        }, indent=2))
        return 0
    ensure_env()
    load_key(REPO)
    require_tools("uvx", "docker")
    if subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        raise RuntimeError("Start the Docker engine and check that docker info succeeds")
    output = result_directory(
        benchmark="Terminal-Bench", harness="Harbor + Vision CL",
        release="terminal-bench/terminal-bench@4.1.0", tasks=selected, workers=args.workers,
        harbor_version="0.26.0",
    )
    command = [
        "uvx", "--from", "harbor==0.26.0", "harbor", "run",
        "--dataset", "terminal-bench/terminal-bench@4.1.0", "--agent", "agent:VisionAgent",
        "--model", "vision-large", "--n-concurrent", str(args.workers), "--jobs-dir", str(output),
    ]
    for task in args.task or []:
        command.extend(["--include-task-name", task])
    env = os.environ | {"PYTHONPATH": str(REPO)}
    return subprocess.run(command, cwd=REPO, env=env).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Benchmark startup or run failed: {error}", file=sys.stderr)
        raise SystemExit(1)
