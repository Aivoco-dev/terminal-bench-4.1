# Terminal-Bench 4.1

**66 public tasks · Harbor 0.26.0 · Vision Large with continual learning**

## Start

1. Install Python 3.9+, uv, and Docker. To run the GPU tasks, use a Linux host with a compatible NVIDIA GPU and container runtime.
2. Set `VISPARK_LAB_API` in this repository’s `.env`. If you cloned the repo, copy `.env.example` to `.env` or open the launcher once to create it.
3. Double-click `Start.command` or run `./start`. The launcher resolves the pinned Harbor version and official 4.1 dataset and starts all 66 tasks.

```sh
./start --plan                 # List the full set without API calls
./start --task music-harmony   # Run one task
./start --workers 4            # Run the full set with four workers
```

Set `BENCH_WORKERS` in `.env` to change the default for double-click runs. Harbor's official verifier grades each sandbox. Vision's memory carries forward within a trial and resets before the next trial.

## Results

Open `results/<run>/run.json` to see the release, settings, and task IDs. The same directory contains Harbor’s job output. The agent logs contain the per-trial Vision usage and tool trace.

[Official Terminal-Bench instructions](https://www.tbench.ai/run)
