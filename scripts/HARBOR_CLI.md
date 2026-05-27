# Harbor 0.8.0 CLI cheat sheet — Terminal-Bench 2.0 on this rig

Discovered 2026-05-24 by Phase A driver. Captures the actual flag names so the
plan's templated commands resolve.

## Dataset

The T-Bench 2.0 task set is a **package** dataset (`org/name`), not a registry
dataset. Bare `terminal-bench-2` is **not found**.

```
harbor download terminal-bench/terminal-bench-2 --cache   # 89 tasks → ~/.cache/harbor/tasks/packages/terminal-bench/<task>/<hash>/
```

For `harbor run`, pass the dataset as `--dataset terminal-bench/terminal-bench-2`
(the slash triggers Harbor's `is_package()` path, which uses the package
dataset client; bare name triggers the legacy registry client and fails).

## Flag mapping (plan §0.3 placeholders → verified flags)

| Question | Verified flag(s) | Notes |
|---|---|---|
| Subset / N-task selector | `--n-tasks` / `-l` INT | Caps after `-i`/`-x` filters. Pass `-l 5` for smoke. |
| Task name filter | `--include-task-name` / `-i` (glob, repeatable); `--exclude-task-name` / `-x` | For targeted re-runs. |
| Output dir | `--jobs-dir` / `-o` PATH | Default `jobs/`. Harbor writes `<jobs_dir>/<job_name>/result.json` + `<trial_name>/` dirs. |
| Job name | `--job-name` TEXT | Default = timestamp. Use deterministic name per leg for cleaner adapter input. |
| **Concurrency** | `--n-concurrent` / `-n` INT | **DEFAULT IS 4 — MUST EXPLICITLY SET TO 1** for single-resident-model rule. |
| Resume / retry | `harbor job resume <job_dir>` (separate subcommand); also `--max-retries` / `-r` on `run` | `--max-retries 0` (default) is fine for a single LM Studio backend; retries don't fix model errors. |
| Auto-confirm host prompts | `--yes` / `-y` | Required in detached driver — otherwise Harbor blocks on "use host environment?" prompt. |
| Quiet progress display | `--quiet` / `-q` | Use in driver scripts (no TTY). |
| Environment | `--env docker` (default) | linux/amd64 via Rosetta — also needs `DOCKER_DEFAULT_PLATFORM=linux/amd64` env var. |
| Agent | `--agent terminus-2` / `-a` | **Default is `oracle`** (runs the task's reference solution; useful only as chain-health baseline, NOT as model evaluator). `terminus-2` is T-Bench's canonical agent for measuring a model. |
| Model | `--model openai/<lm-studio-id>` / `-m` | LiteLLM routing via `OPENAI_API_BASE`. The `openai/` prefix is required. |
| Disable Hub upload | (default = no upload) | Don't pass `--upload`. |

## Required env (in driver script before `harbor run`)

```bash
export OPENAI_API_BASE="http://127.0.0.1:1234/v1"
export OPENAI_API_KEY="lm-studio"             # dummy; LM Studio doesn't validate
export DOCKER_DEFAULT_PLATFORM=linux/amd64
```

## Output schema

`<jobs_dir>/<job_name>/result.json` is a serialized
`harbor.models.job.result.JobResult`:

```jsonc
{
  "id": "<uuid>",
  "started_at": "<iso>",
  "finished_at": "<iso>",            // null until job ends
  "n_total_trials": 89,
  "stats": {
    "n_completed_trials": 89,
    "n_errored_trials": 0,
    "evals": {
      "terminus-2__openai/<model>__terminal-bench/terminal-bench-2": {
        "n_trials": 89,
        "n_errors": 0,
        "reward_stats": {
          "reward": {                 // primary T-Bench reward key
            "1.0": ["trial_name_a", ...],   // PASSED trials
            "0.0": ["trial_name_b", ...]    // FAILED trials
          }
        },
        "pass_at_k": {"1": 0.NN}      // = score
      }
    }
  },
  "trial_results": [
    {
      "task_name": "<task>",
      "trial_name": "<trial>",
      "verifier_result": {"rewards": {"reward": 1.0}},
      "exception_info": null,         // or {"exception_type": "...", ...}
      "agent_execution": {"started_at": "...", "finished_at": "..."},
      "verifier": {"started_at": "...", "finished_at": "..."}
    }
  ]
}
```

Adapter `scripts/harbor_to_summary.py` reads this and produces canonical
`tbench_<model-slug>_<ts>_summary.json` + `.jsonl` files in
`benchmarks/runs/`, matching the `m4max_charts.py` loader contract.

## Per-trial directory

`<job_dir>/<trial_name>/` contains:
- `agent/` — agent stdout/logs (from `/logs/agent` mount in container)
- `verifier/` — verifier output, including `reward.txt` and `reward.json`
- `artifacts/` — any task-defined artifacts copied out

## Canonical run command (Phase A pattern)

```bash
export OPENAI_API_BASE="http://127.0.0.1:1234/v1"
export OPENAI_API_KEY="lm-studio"
export DOCKER_DEFAULT_PLATFORM=linux/amd64

harbor run \
  --dataset terminal-bench/terminal-bench-2 \
  --agent terminus-2 \
  --model "openai/<lm-studio-id>" \
  --env docker \
  -n 1 \              # CRITICAL: single-resident-model
  -y \                # auto-confirm host env access
  --quiet \           # no TTY in nohup
  --jobs-dir .bench-logs/tbench-runs \
  --job-name <model-short>_<date>
```

For the §1 smoke test, add `-l 5` to cap at 5 tasks.
