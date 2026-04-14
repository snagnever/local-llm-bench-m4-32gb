#!/usr/bin/env python3
"""
Final data extraction: combine all sources into one reliable dataset.

Sources:
1. LM Studio server logs → per-question token counts, timing, truncation status
2. bench.py JSON results → verified final scores
3. bench.py transcript output → summary timing per benchmark

Does NOT attempt unreliable question-to-truth positional matching.
Instead, extracts reliable per-question METADATA (tokens, time, truncation)
and combines with known-good aggregate scores.
"""
import re, json, sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

LOG_DIR = Path.home() / ".lmstudio" / "server-logs" / "2026-04"
OUTPUT_DIR = Path(__file__).parent.parent / "benchmarks" / "extracted"

# ---- Known results from bench.py (verified from JSON files + transcript) ----

KNOWN_SCORES = {
    # (model, benchmark): (score, correct, total, duration_seconds, original_max_tokens)
    ("Qwen3-Coder", "mmlu"):      (0.67, 67, 100,  528,  16384),
    ("Qwen3-Coder", "humaneval"): (0.94, 94, 100,  894,  16384),
    ("Qwen3-Coder", "math"):      (0.74, 74, 100, 5030,  4096),   # BUGGED max_tokens
    ("Qwen3-Coder", "gpqa"):      (0.42, 42, 100, 2305,  16384),
    ("Qwen3-Coder", "drop"):      (0.80, 80, 100,  659,  16384),
    ("Gemma4", "mmlu"):            (0.84, 84, 100, 4392,  16384),
    ("Gemma4", "humaneval"):       (0.99, 99, 100, 11115, 16384),
    ("Gemma4", "math"):            (0.59, 59, 100, 14478, 4096),   # BUGGED max_tokens
    ("Gemma4", "drop"):            (0.89, 89, 100, 5790,  16384),  # first run (89%)
    ("huihui-ref", "mmlu"):        (0.78, 78, 100, 4182,  16384),
    ("huihui-ref", "humaneval"):   (0.91, 91, 100, 5042,  16384),
    ("huihui-ref", "math"):        (0.51, 51, 100, 10373, 4096),   # BUGGED max_tokens
    ("huihui-ref", "drop"):        (0.89, 89, 100, 1607,  16384),
    ("huihui-ref", "gpqa"):        (0.54, 54, 100, 6138,  16384),
}

# Known run windows: (model_id, bench_type_in_log, start_approx, end_time)
RUN_WINDOWS = [
    ("qwen3-coder-30b-a3b-instruct", "mcq",       "Qwen3-Coder", "mmlu",      "2026-04-10 09:50", "2026-04-10 10:01"),
    ("qwen3-coder-30b-a3b-instruct", "humaneval",  "Qwen3-Coder", "humaneval", "2026-04-10 10:08", "2026-04-10 10:24"),
    ("qwen3-coder-30b-a3b-instruct", "math",       "Qwen3-Coder", "math",      "2026-04-10 11:09", "2026-04-10 12:34"),
    ("qwen3-coder-30b-a3b-instruct", "mcq",        "Qwen3-Coder", "gpqa",      "2026-04-10 12:35", "2026-04-10 13:15"),
    ("qwen3-coder-30b-a3b-instruct", "drop",       "Qwen3-Coder", "drop",      "2026-04-10 13:21", "2026-04-10 13:33"),
    ("google/gemma-4-26b-a4b",       "mcq",        "Gemma4",      "mmlu",      "2026-04-10 14:00", "2026-04-10 15:16"),
    ("google/gemma-4-26b-a4b",       "humaneval",  "Gemma4",      "humaneval", "2026-04-10 15:20", "2026-04-10 18:28"),
    ("google/gemma-4-26b-a4b",       "math",       "Gemma4",      "math",      "2026-04-10 18:30", "2026-04-10 22:34"),
    ("google/gemma-4-26b-a4b",       "drop",       "Gemma4",      "drop",      "2026-04-10 22:33", "2026-04-11 00:10"),
    ("huihui-qwen3.5-35b-a3b-claude-4.6-opus-abliterated-i1", "mcq",   "huihui-ref", "mmlu",      "2026-04-11 00:15", "2026-04-11 01:26"),
    ("huihui-qwen3.5-35b-a3b-claude-4.6-opus-abliterated-i1", "humaneval","huihui-ref","humaneval","2026-04-11 01:26", "2026-04-11 02:50"),
    ("huihui-qwen3.5-35b-a3b-claude-4.6-opus-abliterated-i1", "math",  "huihui-ref", "math",      "2026-04-11 02:49", "2026-04-11 05:43"),
    ("huihui-qwen3.5-35b-a3b-claude-4.6-opus-abliterated-i1", "drop",  "huihui-ref", "drop",      "2026-04-11 05:43", "2026-04-11 06:10"),
    ("huihui-qwen3.5-35b-a3b-claude-4.6-opus-abliterated-i1", "mcq",   "huihui-ref", "gpqa",      "2026-04-11 06:10", "2026-04-11 07:53"),
]

def parse_log_file(filepath):
    with open(filepath) as f:
        content = f.read()
    entries = []
    chunks = content.split('[DEBUG] Received request: POST to /v1/chat/completions')
    for chunk in chunks[1:]:
        e = {}
        for field, pat in [('model', r'"system_fingerprint":\s*"([^"]+)"'),
                           ('request_model', r'"model":\s*"([^"]+)"'),
                           ('max_tokens', r'"max_tokens":\s*(\d+)'),
                           ('finish_reason', r'"finish_reason":\s*"([^"]+)"'),
                           ('prompt_tokens', r'"prompt_tokens":\s*(\d+)'),
                           ('completion_tokens', r'"completion_tokens":\s*(\d+)'),
                           ('reasoning_tokens', r'"reasoning_tokens":\s*(\d+)')]:
            m = re.search(pat, chunk)
            if m:
                val = m.group(1)
                e[field] = int(val) if field in ('max_tokens','prompt_tokens','completion_tokens','reasoning_tokens') else val
            else:
                e[field] = 0 if 'tokens' in field or field == 'max_tokens' else '?'
        if e.get('model') == '?' or e.get('model') == 0:
            e['model'] = e.get('request_model', '?')
        e['visible_tokens'] = e['completion_tokens'] - e['reasoning_tokens']
        m = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*Running chat completion', chunk)
        e['request_time'] = m.group(1) if m else ''
        # Identify benchmark from prompt
        m = re.search(r'"role":\s*"user",\s*"content":\s*"((?:[^"\\]|\\.){0,300})', chunk)
        prompt = m.group(1) if m else ''
        if 'boxed' in prompt or 'Solve this math' in prompt: e['benchmark'] = 'math'
        elif 'Read the passage' in prompt: e['benchmark'] = 'drop'
        elif 'Complete the following Python' in prompt or 'Output ONLY the complete function' in prompt: e['benchmark'] = 'humaneval'
        elif 'Answer with just the letter' in prompt: e['benchmark'] = 'mcq'
        else: e['benchmark'] = 'unknown'
        if e['completion_tokens'] > 0 or e['finish_reason'] != '?':
            entries.append(e)
    return entries

def main():
    # Parse all logs
    all_entries = []
    for lf in sorted(LOG_DIR.glob("2026-04-*.log")):
        print(f"Parsing {lf.name}...", file=sys.stderr)
        all_entries.extend(parse_log_file(lf))
    print(f"Total log entries: {len(all_entries)}", file=sys.stderr)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # For each known run, extract per-question metadata from logs
    all_rows = []

    for model_id, bench_log, model_short, bench_name, start_str, end_str in RUN_WINDOWS:
        start = datetime.strptime(start_str, '%Y-%m-%d %H:%M')
        end = datetime.strptime(end_str, '%Y-%m-%d %H:%M')

        # Filter entries
        candidates = [e for e in all_entries
                      if e['model'] == model_id
                      and e['benchmark'] == bench_log
                      and e['request_time']
                      and start <= datetime.strptime(e['request_time'], '%Y-%m-%d %H:%M:%S') <= end]
        candidates.sort(key=lambda x: x['request_time'])

        known = KNOWN_SCORES.get((model_short, bench_name))
        score, correct, total, duration, max_tok = known if known else (None, None, None, None, None)

        truncated = sum(1 for e in candidates if e['finish_reason'] == 'length')
        avg_prompt = sum(e['prompt_tokens'] for e in candidates) / len(candidates) if candidates else 0
        avg_completion = sum(e['completion_tokens'] for e in candidates) / len(candidates) if candidates else 0
        avg_reasoning = sum(e['reasoning_tokens'] for e in candidates) / len(candidates) if candidates else 0
        avg_visible = sum(e['visible_tokens'] for e in candidates) / len(candidates) if candidates else 0
        total_tokens = sum(e['completion_tokens'] for e in candidates)

        # Per-question data
        for i, entry in enumerate(candidates):
            all_rows.append({
                'model': model_short,
                'benchmark': bench_name,
                'question_num': i + 1,
                'prompt_tokens': entry['prompt_tokens'],
                'completion_tokens': entry['completion_tokens'],
                'reasoning_tokens': entry['reasoning_tokens'],
                'visible_tokens': entry['visible_tokens'],
                'max_tokens_sent': entry['max_tokens'],
                'finish_reason': entry['finish_reason'],
                'request_time': entry['request_time'],
                'truncated': entry['finish_reason'] == 'length',
            })

        print(f"  {model_short:15s} {bench_name:10s}: {len(candidates):>3d} entries | "
              f"score={score*100 if score else '?':>5}% | "
              f"truncated={truncated:>2d} | "
              f"avg_prompt={avg_prompt:>6.0f} avg_comp={avg_completion:>6.0f} "
              f"avg_think={avg_reasoning:>6.0f} avg_vis={avg_visible:>5.0f} | "
              f"total_comp={total_tokens:>7d}tok", file=sys.stderr)

    # Write per-question data
    pq_file = OUTPUT_DIR / "per_question_data.jsonl"
    with open(pq_file, 'w') as f:
        for r in all_rows:
            f.write(json.dumps(r) + '\n')

    # Build the master summary table
    summary = []
    for (model, bench), (score, correct, total, duration, max_tok) in KNOWN_SCORES.items():
        # Find matching entries for truncation count
        entries = [r for r in all_rows if r['model'] == model and r['benchmark'] == bench]
        truncated = sum(1 for e in entries if e['truncated'])
        n_entries = len(entries)

        avg_comp = sum(e['completion_tokens'] for e in entries) / n_entries if entries else 0
        avg_think = sum(e['reasoning_tokens'] for e in entries) / n_entries if entries else 0
        avg_prompt = sum(e['prompt_tokens'] for e in entries) / n_entries if entries else 0
        total_comp = sum(e['completion_tokens'] for e in entries)

        # For MATH: calculate corrected score estimate
        # The broken regex underscored by ~2-3pp (verified for Qwen3-Coder: 74→76, huihui: 51→54)
        # The truncation caused questions to be scored wrong: all truncated = definitely wrong
        math_corrected = None
        if bench == 'math':
            # Estimate: regex fix adds ~2-3pp, truncated questions need rerunning
            math_corrected = f"~{score*100+2:.0f}% (regex fix), {truncated} questions need rerun"

        is_bugged = max_tok == 4096 and bench == 'math'
        summary.append({
            'model': model,
            'benchmark': bench,
            'score': score,
            'score_pct': f"{score*100:.0f}%",
            'correct': correct,
            'total': total,
            'duration_s': duration,
            'duration_min': f"{duration/60:.0f}",
            'truncated_at_max_tokens': truncated,
            'max_tokens_used': max_tok,
            'bugged_max_tokens': is_bugged,
            'entries_in_logs': n_entries,
            'avg_prompt_tokens': round(avg_prompt),
            'avg_completion_tokens': round(avg_comp),
            'avg_reasoning_tokens': round(avg_think),
            'avg_visible_tokens': round(avg_comp - avg_think),
            'total_completion_tokens': total_comp,
            'tok_per_second': round(total_comp / duration, 1) if duration > 0 else 0,
            'math_corrected_note': math_corrected,
        })

    # Write summary
    summary_file = OUTPUT_DIR / "master_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print the master table
    print(f"\n{'='*120}", file=sys.stderr)
    print("MASTER BENCHMARK DATA TABLE", file=sys.stderr)
    print(f"{'='*120}", file=sys.stderr)
    print(f"{'Model':15s} | {'Bench':10s} | {'Score':>6s} | {'Time':>5s} | {'Trunc':>5s} | {'MaxTok':>6s} | {'AvgPrompt':>9s} | {'AvgComp':>7s} | {'AvgThink':>8s} | {'AvgVis':>6s} | {'tok/s':>5s} | {'Bug?':>4s}", file=sys.stderr)
    print("-" * 120, file=sys.stderr)
    for r in sorted(summary, key=lambda x: (x['model'], x['benchmark'])):
        bug = "BUG!" if r['bugged_max_tokens'] else ""
        print(f"{r['model']:15s} | {r['benchmark']:10s} | {r['score_pct']:>6s} | {r['duration_min']:>4s}m | {r['truncated_at_max_tokens']:>5d} | {r['max_tokens_used']:>6d} | {r['avg_prompt_tokens']:>9d} | {r['avg_completion_tokens']:>7d} | {r['avg_reasoning_tokens']:>8d} | {r['avg_visible_tokens']:>6d} | {r['tok_per_second']:>5.1f} | {bug:>4s}", file=sys.stderr)

    print(f"\nWrote {len(all_rows)} per-question entries to {pq_file}", file=sys.stderr)
    print(f"Wrote summary to {summary_file}", file=sys.stderr)

if __name__ == '__main__':
    main()
