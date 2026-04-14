#!/usr/bin/env python3
"""
Build a combined per-question timing file that uses:
- NEW data (from bench2.py runs) for MATH (all 3 models) and GPQA (Gemma probe)
- OLD data (from LM Studio log extraction) for MMLU, HumanEval, DROP, and original GPQA runs

The old per_question_timing.jsonl has token counts and timestamps extracted from
LM Studio server logs — those measurements are valid even though the MATH grading
was broken. We only replace the MATH entries with the new correctly-graded data.
"""
import json
from pathlib import Path

BASE = Path(__file__).parent.parent / "benchmarks"

# Load old timing data (from LM Studio log extraction)
old_file = BASE / "extracted" / "per_question_timing.jsonl"
old_entries = []
with open(old_file) as f:
    for line in f:
        old_entries.append(json.loads(line))

print(f"Loaded {len(old_entries)} old timing entries")

# Map old model IDs to short names
MODEL_MAP = {
    'qwen3-coder-30b-a3b-instruct': 'Qwen3-Coder',
    'google/gemma-4-26b-a4b': 'Gemma4',
    'huihui-qwen3.5-35b-a3b-claude-4.6-opus-abliterated-i1': 'huihui-ref',
}

# Tag with short name and filter out old MATH (we have better data now)
# Also filter out old Gemma GPQA (we have new probe data)
combined = []
for e in old_entries:
    model_short = MODEL_MAP.get(e['model'], e['model'])
    bench = e['bench']
    # Skip old MATH for our 3 models (replaced by new bench2.py runs)
    if bench == 'math' and model_short in MODEL_MAP.values():
        continue
    e['model_short'] = model_short
    combined.append(e)

print(f"Kept {len(combined)} old entries after removing MATH")

# Now load NEW data from bench2.py runs
new_runs = [
    ('math_google_gemma-4-26b-a4b_20260411_135639.jsonl', 'Gemma4'),
    ('math_huihui-qwen3.5-35b-a3b-claude-4.6-opus-abliterated-i1_20260411_194200.jsonl', 'huihui-ref'),
    ('math_qwen3-coder-30b-a3b-instruct_20260412_063123.jsonl', 'Qwen3-Coder'),
    # Gemma GPQA: combined 100q (probe + main run)
    ('gpqa_gemma4_combined_100q.jsonl', 'Gemma4'),
]

added_new = 0
for run_file, model_short in new_runs:
    path = BASE / "runs" / run_file
    if not path.exists():
        print(f"WARNING: {run_file} not found")
        continue
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            # Convert to timing entry format
            combined.append({
                'model': e['model'],
                'model_short': model_short,
                'bench': e['benchmark'],
                'max_tokens': e['max_tokens_sent'],
                'comp': e['completion_tokens'],
                'prompt': e['prompt_tokens'],
                'think': e['reasoning_tokens'],
                'vis': e['visible_tokens'],
                'req_time': e['timestamp'],
                'last_ts': e['timestamp'],  # bench2.py doesn't track separate last timestamp
                'finish': e['finish_reason'],
                'elapsed_s': e['elapsed_s'],
                'actual_tok_s': e['tok_s'],
                'correct': e['correct'],
                'question_num': e['question_num'],
                'gpu_temp_c': e.get('gpu_temp_c'),
                'source': 'bench2',
            })
            added_new += 1

print(f"Added {added_new} new entries")
print(f"Total combined: {len(combined)}")

# Report by model/benchmark
from collections import defaultdict
counts = defaultdict(int)
for e in combined:
    counts[(e['model_short'], e['bench'])] += 1
for key, n in sorted(counts.items()):
    print(f"  {key[0]:15s} {key[1]:10s}: {n}")

# Write combined file
out_file = BASE / "extracted" / "per_question_timing_combined.jsonl"
with open(out_file, 'w') as f:
    for e in combined:
        f.write(json.dumps(e, default=str) + '\n')
print(f"\nWrote {out_file}")
