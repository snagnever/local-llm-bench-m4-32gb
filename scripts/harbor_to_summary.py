#!/usr/bin/env python3
"""Post-process a Harbor T-Bench job into m4max_charts.py canonical summary.

Reads <job_dir>/result.json (Harbor's JobResult) and per-trial verifier files.
Source of truth for score is stats.evals[*].reward_stats — Harbor's periodic
result.json writes exclude trial_results (final write does too, in 0.8.0).

Writes:
  - benchmarks/runs/tbench_<slug>_<ts>_summary.json   (auto-discovered by m4max_charts)
  - benchmarks/runs/tbench_<slug>_<ts>.jsonl          (one row per trial: task, status, reward, elapsed, exc_type)
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RUNS = SCRIPT_DIR.parent / "benchmarks" / "runs"
REWARD_KEY = "reward"  # T-Bench primary metric


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def parse_iso(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def elapsed_seconds(start, end):
    if start and end:
        return (parse_iso(end) - parse_iso(start)).total_seconds()
    return None


def trial_dir_elapsed(trial_dir: Path) -> float | None:
    agent_dir = trial_dir / "agent"
    if not agent_dir.is_dir():
        return None
    episodes = list(agent_dir.glob("episode-*"))
    if not episodes:
        return None
    try:
        first = min(os.path.getmtime(p) for p in episodes)
        last = max(os.path.getmtime(p) for p in episodes)
        return last - first
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir", type=Path)
    ap.add_argument("--model-label", type=str, default=None)
    ap.add_argument("--lm-studio-id", type=str, default=None)
    args = ap.parse_args()

    job_dir: Path = args.job_dir.expanduser().resolve()
    result_path = job_dir / "result.json"
    if not result_path.exists():
        print(f"ERR: {result_path} missing", file=sys.stderr)
        sys.exit(1)

    data = json.loads(result_path.read_text())
    started = data.get("started_at")
    finished = data.get("finished_at")
    n_total = int(data.get("n_total_trials", 0))
    stats = data.get("stats", {}) or {}
    evals = stats.get("evals", {}) or {}

    # Pick the (only) eval entry
    if not evals:
        print(f"ERR: no evals in {result_path}", file=sys.stderr)
        sys.exit(1)
    eval_key, eval_data = next(iter(evals.items()))

    # Source of truth: reward_stats[REWARD_KEY] maps value → [trial_names]
    reward_stats = (eval_data.get("reward_stats", {}) or {}).get(REWARD_KEY, {}) or {}
    # values come back as strings in JSON when keys are floats — normalize
    passed_trials, failed_trials = [], []
    for val_key, names in reward_stats.items():
        try:
            val = float(val_key)
        except (TypeError, ValueError):
            continue
        (passed_trials if val >= 1.0 else failed_trials).extend(names)

    correct = len(passed_trials)
    failed = len(failed_trials)
    total = n_total or (correct + failed)

    # Exceptions per trial
    exc_stats = eval_data.get("exception_stats", {}) or {}
    exc_lookup = {}
    for exc_type, names in exc_stats.items():
        for n in names:
            exc_lookup[n] = exc_type

    # Decode LM studio id from eval key — "<agent>__<model>__<dataset>"
    lm_id = args.lm_studio_id
    if not lm_id:
        parts = eval_key.split("__")
        if len(parts) >= 3:
            lm_id = parts[1]
    lm_id = lm_id or "unknown"

    label = args.model_label or slugify(lm_id)
    ts = (parse_iso(started) or datetime.utcnow()).strftime("%Y-%m-%d_%H%M%S")
    out_stem = f"tbench_{label}_{ts}"

    elapsed_s = elapsed_seconds(started, finished) or 0.0
    score = correct / total if total else 0.0

    summary = {
        "run_name": out_stem,
        "benchmark": "tbench",
        "model": lm_id,
        "score": round(score, 4),
        "score_pct": f"{score * 100:.1f}%",
        "correct": correct,
        "total": total,
        "elapsed_s": round(elapsed_s, 1),
        "elapsed_min": round(elapsed_s / 60, 1),
        "harbor_output_dir": str(job_dir),
        "timestamp_start": started,
        "timestamp_end": finished,
    }

    # Per-trial rows (status + per-trial agent-dir elapsed estimate)
    rows = []
    for trial_name in sorted(passed_trials + failed_trials):
        status = "PASS" if trial_name in passed_trials else "FAIL"
        exc_type = exc_lookup.get(trial_name)
        if exc_type and status == "FAIL":
            status = "ERROR"
        rows.append({
            "trial_name": trial_name,
            "task_name": trial_name.split("__")[0],
            "status": status,
            "exception_type": exc_type,
            "elapsed_s": (
                trial_dir_elapsed(job_dir / trial_name)
                if (job_dir / trial_name).is_dir() else None
            ),
        })

    RUNS.mkdir(parents=True, exist_ok=True)
    summary_path = RUNS / f"{out_stem}_summary.json"
    jsonl_path = RUNS / f"{out_stem}.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2))
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {summary_path}")
    print(f"Wrote {jsonl_path}  ({len(rows)} trials)")
    print(f"Score: {score * 100:.1f}%  ({correct}/{total})  elapsed {elapsed_s / 60:.1f} min")


if __name__ == "__main__":
    main()
