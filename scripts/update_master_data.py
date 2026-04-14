#!/usr/bin/env python3
"""
Update master_summary.json, master_results.csv, and per_question_data.jsonl
with the new MATH and GPQA rerun data.

Marks old entries as superseded and adds new ones.
"""
import json, csv
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent.parent / "benchmarks"
RUNS_DIR = BASE / "runs"
EXTRACTED_DIR = BASE / "extracted"

# The 100q runs we want to use (filter by name)
VALID_RUNS = {
    'math_google_gemma-4-26b-a4b_20260411_135639': ('Gemma4', 'math', True),
    'math_huihui-qwen3.5-35b-a3b-claude-4.6-opus-abliterated-i1_20260411_194200': ('huihui-ref', 'math', True),
    'math_qwen3-coder-30b-a3b-instruct_20260412_063123': ('Qwen3-Coder', 'math', True),
    # Gemma GPQA 100q: combined from 10q probe + 90q main run
    'gpqa_gemma4_combined_100q': ('Gemma4', 'gpqa', True),
}

# Previous benchmark scores to preserve (these were verified valid)
PRESERVED_SCORES = {
    ("Qwen3-Coder", "mmlu"):      {'score': 0.67, 'correct': 67, 'total': 100, 'duration_s': 528,  'source': 'original'},
    ("Qwen3-Coder", "humaneval"): {'score': 0.94, 'correct': 94, 'total': 100, 'duration_s': 894,  'source': 'original'},
    ("Qwen3-Coder", "gpqa"):      {'score': 0.42, 'correct': 42, 'total': 100, 'duration_s': 2305, 'source': 'original'},
    ("Qwen3-Coder", "drop"):      {'score': 0.80, 'correct': 80, 'total': 100, 'duration_s': 659,  'source': 'original'},
    ("Gemma4", "mmlu"):            {'score': 0.84, 'correct': 84, 'total': 100, 'duration_s': 4392, 'source': 'original'},
    ("Gemma4", "humaneval"):       {'score': 0.99, 'correct': 99, 'total': 100, 'duration_s': 11115,'source': 'original'},
    ("Gemma4", "drop"):            {'score': 0.89, 'correct': 89, 'total': 100, 'duration_s': 5790, 'source': 'original'},
    # Gemma4 GPQA now from new run, not preserved
    ("huihui-ref", "mmlu"):        {'score': 0.78, 'correct': 78, 'total': 100, 'duration_s': 4182, 'source': 'original'},
    ("huihui-ref", "humaneval"):   {'score': 0.91, 'correct': 91, 'total': 100, 'duration_s': 5042, 'source': 'original'},
    ("huihui-ref", "drop"):        {'score': 0.89, 'correct': 89, 'total': 100, 'duration_s': 1607, 'source': 'original'},
    ("huihui-ref", "gpqa"):        {'score': 0.54, 'correct': 54, 'total': 100, 'duration_s': 6138, 'source': 'original'},
}

def load_new_run(run_name, model_short, benchmark, is_100q):
    """Load summary and per-question data from a new run."""
    # Combined Gemma GPQA run uses different filenames
    if run_name == 'gpqa_gemma4_combined_100q':
        summary_file = RUNS_DIR / "gpqa_gemma4_combined_100q.json"
        jsonl_file = RUNS_DIR / "gpqa_gemma4_combined_100q.jsonl"
    else:
        summary_file = RUNS_DIR / f"{run_name}_summary.json"
        jsonl_file = RUNS_DIR / f"{run_name}.jsonl"

    with open(summary_file) as f:
        summary = json.load(f)

    entries = []
    with open(jsonl_file) as f:
        for line in f:
            entries.append(json.loads(line))

    # For combined runs without full summary fields, synthesize them
    if 'score_pct' not in summary:
        summary['score_pct'] = f"{summary['score']*100:.1f}%"
    if 'elapsed_s' not in summary and 'elapsed_s' in summary:
        pass  # already there
    if 'model_config' not in summary:
        summary['model_config'] = {'context_length': 65536}
    if 'max_tokens' not in summary:
        summary['max_tokens'] = 32768

    # Compute stats
    truncated = sum(1 for e in entries if e['finish_reason'] == 'length')
    errors = sum(1 for e in entries if e['error'])
    total_prompt_tok = sum(e['prompt_tokens'] for e in entries)
    total_comp_tok = sum(e['completion_tokens'] for e in entries)
    total_reason_tok = sum(e['reasoning_tokens'] for e in entries)
    total_vis_tok = sum(e['visible_tokens'] for e in entries)
    n = len(entries)

    return {
        'model': model_short,
        'model_id': summary['model'],
        'benchmark': benchmark,
        'score': summary['score'],
        'score_pct': summary['score_pct'],
        'correct': summary['correct'],
        'total': summary['total'],
        'duration_s': summary['elapsed_s'],
        'duration_min': summary['elapsed_min'],
        'truncated_at_max_tokens': truncated,
        'errors': errors,
        'max_tokens_used': summary['max_tokens'],
        'context_length': summary['model_config'].get('context_length') if summary.get('model_config') else None,
        'bugged_max_tokens': False,
        'entries_in_logs': n,
        'avg_prompt_tokens': round(total_prompt_tok / n) if n else 0,
        'avg_completion_tokens': round(total_comp_tok / n) if n else 0,
        'avg_reasoning_tokens': round(total_reason_tok / n) if n else 0,
        'avg_visible_tokens': round(total_vis_tok / n) if n else 0,
        'total_completion_tokens': total_comp_tok,
        'tok_per_second': round(total_comp_tok / summary['elapsed_s'], 1) if summary['elapsed_s'] else 0,
        'source': 'new_run',
        'run_name': run_name,
        'is_100q': is_100q,
        'math_corrected_note': None,
    }, entries

def main():
    # Load old master_summary to preserve non-rerun entries
    old_summary_file = EXTRACTED_DIR / "master_summary.json"
    old_data = []
    if old_summary_file.exists():
        with open(old_summary_file) as f:
            old_data = json.load(f)

    # New summary: start with preserved scores + new run data
    new_summary = []

    # Add preserved scores (from original runs we didn't rerun)
    for (model, bench), info in PRESERVED_SCORES.items():
        # Find old entry for average tokens, etc.
        old_entry = next((e for e in old_data if e['model'] == model and e['benchmark'] == bench), {})
        new_summary.append({
            'model': model,
            'benchmark': bench,
            'score': info['score'],
            'score_pct': f"{info['score']*100:.0f}%",
            'correct': info['correct'],
            'total': info['total'],
            'duration_s': info['duration_s'],
            'duration_min': f"{info['duration_s']/60:.0f}",
            'truncated_at_max_tokens': old_entry.get('truncated_at_max_tokens', 0),
            'max_tokens_used': old_entry.get('max_tokens_used', 16384),
            'context_length': None,
            'bugged_max_tokens': False,
            'entries_in_logs': old_entry.get('entries_in_logs', 100),
            'avg_prompt_tokens': old_entry.get('avg_prompt_tokens', 0),
            'avg_completion_tokens': old_entry.get('avg_completion_tokens', 0),
            'avg_reasoning_tokens': old_entry.get('avg_reasoning_tokens', 0),
            'avg_visible_tokens': old_entry.get('avg_visible_tokens', 0),
            'total_completion_tokens': old_entry.get('total_completion_tokens', 0),
            'tok_per_second': old_entry.get('tok_per_second', 0),
            'math_corrected_note': None,
            'source': 'original',
            'is_100q': True,
        })

    # Add new MATH + GPQA runs
    new_per_question_entries = []
    for run_name, (model, bench, is_100q) in VALID_RUNS.items():
        print(f"Loading {run_name}...")
        summary_entry, entries = load_new_run(run_name, model, bench, is_100q)
        new_summary.append(summary_entry)
        # Tag per-question entries with model short name
        for e in entries:
            e['model_short'] = model
            new_per_question_entries.append(e)

    # Write new master_summary.json
    with open(EXTRACTED_DIR / "master_summary.json", 'w') as f:
        json.dump(new_summary, f, indent=2)
    print(f"\nWrote master_summary.json with {len(new_summary)} entries")

    # Write new master_results.csv
    with open(EXTRACTED_DIR / "master_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Model', 'Benchmark', 'Score%', 'Correct', 'Total', 'Duration_min',
                    'Truncated', 'MaxTokensSent', 'ContextLen', 'Bugged', 'Source',
                    'Is100q', 'AvgPromptTok', 'AvgCompletionTok', 'AvgThinkingTok', 'AvgVisibleTok',
                    'TotalCompletionTok', 'EffectiveTokPerSec'])
        for r in sorted(new_summary, key=lambda x: (x['model'], x['benchmark'])):
            w.writerow([
                r['model'], r['benchmark'], r['score_pct'], r['correct'], r['total'],
                r['duration_min'], r['truncated_at_max_tokens'], r['max_tokens_used'],
                r.get('context_length') or '',
                'YES' if r['bugged_max_tokens'] else '',
                r.get('source', '?'),
                'YES' if r.get('is_100q') else 'NO',
                r['avg_prompt_tokens'], r['avg_completion_tokens'],
                r['avg_reasoning_tokens'], r['avg_visible_tokens'],
                r['total_completion_tokens'], r['tok_per_second'],
            ])
    print(f"Wrote master_results.csv")

    # Write new per-question data (from new runs only — old data was positionally unreliable)
    with open(EXTRACTED_DIR / "per_question_data_v2.jsonl", 'w') as f:
        for e in new_per_question_entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    print(f"Wrote per_question_data_v2.jsonl with {len(new_per_question_entries)} entries")

    # Print summary table
    print(f"\n{'='*110}")
    print(f"{'Model':15s} | {'Benchmark':10s} | {'Score':>6s} | {'Time':>5s} | {'Trunc':>5s} | {'MaxTok':>6s} | {'Source':>10s} | {'N':>4s}")
    print("-" * 110)
    for r in sorted(new_summary, key=lambda x: (x['model'], x['benchmark'])):
        src = r.get('source', '?')[:10]
        n_str = f"{r['total']}" + ("" if r.get('is_100q') else "*")
        print(f"{r['model']:15s} | {r['benchmark']:10s} | {r['score_pct']:>6s} | {r['duration_min']:>4}m | "
              f"{r['truncated_at_max_tokens']:>5d} | {r['max_tokens_used']:>6d} | {src:>10s} | {n_str:>4s}")
    print("\n* = probe run (not full 100q)")

if __name__ == '__main__':
    main()
