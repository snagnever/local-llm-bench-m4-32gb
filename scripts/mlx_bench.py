#!/usr/bin/env python3
"""
MLX benchmark harness — mirrors bench2.py CLI and logging exactly.
Uses mlx_vlm Python API directly (no server).

Same logging schema, same console format, same pre-flight checks, same
per-question JSONL output. Only difference: loads model via mlx_vlm.load
instead of hitting LM Studio's HTTP API.

Usage:
  python3 mlx_bench.py math --examples 100 --model deadbydawn101/gemma-4-21b-REAP-Tool-Calling-mlx-4bit
  python3 mlx_bench.py mmlu --examples 100 --model deadbydawn101/...
"""
import time, json, sys, os, subprocess, re, random, argparse, gc
from datetime import datetime
from pathlib import Path

# ---- Config ----
# CRITICAL: Gemma 4 requires temperature=1.0 with top_p/top_k sampling per its generation_config.json.
# mlx_vlm defaults to temperature=0.0 (greedy) which causes thinking loops on hard questions.
# See research/WORKLOG_NIGHT.md for the investigation.
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 64
SEED = 42
COOLDOWN_GPU_TEMP = 60  # Wait for GPU to drop below this (°C) before every question

# ---- Directories ----
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / "benchmarks" / "runs"

# ---- System monitoring (same as bench2.py) ----

def get_system_state():
    state = {'gpu_temp_c': None, 'cpu_temp_c': None, 'process_ram_mb': 0, 'swap_used_mb': 0}
    try:
        proc = subprocess.Popen(["macmon", "pipe"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        line = proc.stdout.readline().decode().strip()
        proc.kill(); proc.wait()
        if line:
            d = json.loads(line)
            gpu = round(d.get('temp', {}).get('gpu_temp_avg', 0))
            cpu = round(d.get('temp', {}).get('cpu_temp_avg', 0))
            state['gpu_temp_c'] = gpu if 5 <= gpu <= 120 else None
            state['cpu_temp_c'] = cpu if 5 <= cpu <= 120 else None
    except: pass
    try:
        r = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True)
        m = re.search(r'used\s*=\s*([0-9.]+)M', r.stdout)
        if m: state['swap_used_mb'] = round(float(m.group(1)))
    except: pass
    try:
        r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        for line in r.stdout.split('\n'):
            if 'python' in line.lower() and 'mlx_bench' in line:
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        mem_kb = int(parts[5])
                        if mem_kb > state['process_ram_mb'] * 1024:
                            state['process_ram_mb'] = round(mem_kb / 1024)
                    except: pass
    except: pass
    return state

def get_hardware_info():
    info = {}
    try:
        r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        info['total_ram_gb'] = round(int(r.stdout.strip()) / 1024**3)
    except: pass
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        info['cpu'] = r.stdout.strip()
    except: pass
    try:
        r = subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True)
        info['macos_version'] = r.stdout.strip()
    except: pass
    try:
        r = subprocess.run(["uname", "-m"], capture_output=True, text=True)
        info['arch'] = r.stdout.strip()
    except: pass
    return info

# ---- Answer extraction (same as bench2.py) ----

def extract_boxed(text):
    idx = text.find('\\boxed{')
    if idx == -1: return None
    start = idx + 7; depth = 1; pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == '{': depth += 1
        elif text[pos] == '}': depth -= 1
        pos += 1
    return text[start:pos-1] if depth == 0 else None

def normalize_math(s):
    s = re.sub(r'\s+', '', s)
    def expand_frac(m):
        after = m.group(1)
        if after.startswith('{'):
            return '\\frac' + after
        if len(after) >= 2:
            return f'\\frac{{{after[0]}}}{{{after[1]}}}' + after[2:]
        return '\\frac' + after
    s = re.sub(r'\\frac([^{\\].*?)(?=\\|$|\+|-|\)|\]|,|\s|=)', expand_frac, s)
    s = re.sub(r'\\frac([^{])([^{])', r'\\frac{\1}{\2}', s)
    s = s.replace('\\left', '').replace('\\right', '')
    s = re.sub(r'\\(?:text|mathrm)\{([^}]*)\}', r'\1', s)
    return s.rstrip('.')

def extract_answer_letter(text):
    if not text: return '?'
    text = text.strip()
    for region in [text[-100:], text[:20], text[-30:]]:
        for pat in [r'[Aa]nswer[:\s]+([A-D])', r'\(([A-D])\)', r'\b([A-D])\b']:
            m = re.search(pat, region)
            if m: return m.group(1).upper()
    return '?'

# ---- Dataset loaders (same as bench2.py) ----

def load_math_questions(n):
    from datasets import load_dataset, concatenate_datasets
    subjects = ['algebra','counting_and_probability','geometry','intermediate_algebra',
                'number_theory','prealgebra','precalculus']
    ds = concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s, split="test") for s in subjects])
    random.seed(SEED)
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    questions = []
    for i, idx in enumerate(indices):
        row = ds[idx]
        expected = extract_boxed(row['solution'])
        if expected is None:
            m = re.search(r'\\boxed\{([^}]+)\}', row['solution'])
            expected = m.group(1) if m else row['solution'][-30:]
        questions.append({
            'question_num': i + 1,
            'dataset_idx': idx,
            'level': row.get('level', '?'),
            'type': row.get('type', '?'),
            'problem': row['problem'],
            'expected': expected,
            'user_prompt': f"Solve this math problem. Put your final answer in \\boxed{{}}.\n\n{row['problem']}",
        })
    return questions

def load_mmlu_questions(n):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    random.seed(SEED)
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    questions = []
    for i, idx in enumerate(indices):
        row = ds[idx]
        prompt = f"{row['question']}\n"
        for j, c in enumerate(row['choices']):
            prompt += f"({chr(65+j)}) {c}\n"
        prompt += "Answer with just the letter."
        questions.append({
            'question_num': i + 1,
            'dataset_idx': idx,
            'subject': row.get('subject', '?'),
            'expected': "ABCD"[row['answer']],
            'user_prompt': prompt,
        })
    return questions

def load_drop_questions(n):
    from datasets import load_dataset
    ds = load_dataset("ucinlp/drop", split="validation")
    random.seed(SEED)
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    questions = []
    for i, idx in enumerate(indices):
        row = ds[idx]
        questions.append({
            'question_num': i + 1,
            'dataset_idx': idx,
            'expected_answers': row['answers_spans']['spans'],
            'user_prompt': f"Read the passage and answer the question. Give ONLY the answer, nothing else.\n\nPassage: {row['passage']}\n\nQuestion: {row['question']}",
        })
    return questions

def load_humaneval_questions(n):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    random.seed(SEED)
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    questions = []
    for i, idx in enumerate(indices):
        row = ds[idx]
        questions.append({
            'question_num': i + 1,
            'dataset_idx': idx,
            'entry_point': row['entry_point'],
            'test_code': row['test'],
            'user_prompt': f"Complete the following Python function. Output ONLY the complete function, no explanation.\n\n{row['prompt']}",
        })
    return questions

def load_gpqa_questions(n):
    from datasets import load_dataset
    ds = load_dataset("fingertap/GPQA-Diamond", split="test")
    random.seed(SEED)
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    questions = []
    for i, idx in enumerate(indices):
        row = ds[idx]
        questions.append({
            'question_num': i + 1,
            'dataset_idx': idx,
            'expected': row['answer'].strip(),
            'user_prompt': f"{row['question']}\nAnswer with just the letter.",
        })
    return questions

LOADERS = {
    'math': load_math_questions,
    'mmlu': load_mmlu_questions,
    'drop': load_drop_questions,
    'humaneval': load_humaneval_questions,
    'gpqa': load_gpqa_questions,
}

# ---- Grading (same as bench2.py) ----

def grade(benchmark, question, response_text, finish_reason):
    if finish_reason == 'length':
        return False, None, 'truncated'
    if benchmark == 'math':
        got = extract_boxed(response_text)
        if got is None:
            m = re.search(r'\\boxed\{([^}]+)\}', response_text)
            got = m.group(1) if m else None
        if got is None:
            return False, None, 'no_boxed'
        got_n = normalize_math(str(got))
        exp_n = normalize_math(str(question['expected']))
        return got_n == exp_n, got, 'correct' if got_n == exp_n else 'wrong'
    elif benchmark in ('mmlu', 'gpqa'):
        got = extract_answer_letter(response_text)
        exp = question['expected']
        return got == exp, got, 'correct' if got == exp else f'wrong(exp={exp},got={got})'
    elif benchmark == 'drop':
        resp_lower = response_text.lower().strip()
        answers = question['expected_answers']
        correct = any(a.lower() in resp_lower for a in answers)
        return correct, response_text[:80], 'correct' if correct else 'wrong'
    elif benchmark == 'humaneval':
        code = response_text
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        import tempfile
        try:
            full_code = code + "\n" + question['test_code'] + f"\ncheck({question['entry_point']})"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                f.flush()
                result = subprocess.run([sys.executable, f.name],
                                      capture_output=True, text=True, timeout=10)
                ok = result.returncode == 0
                os.unlink(f.name)
                return ok, question['entry_point'], 'pass' if ok else 'fail'
        except Exception as e:
            return False, question['entry_point'], f'exec_error: {str(e)[:50]}'
    return None, None, 'ungraded'

# ---- MLX inference ----

def make_call(model, processor, tokenizer, user_prompt, max_tokens, enable_thinking=True):
    """Single MLX inference call. Returns dict matching api_call format."""
    from mlx_vlm import generate as mlx_generate
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=enable_thinking
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    start = time.time()
    try:
        out = mlx_generate(model, processor, prompt, max_tokens=max_tokens,
                          temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K)
        elapsed = time.time() - start
        content = out.text if hasattr(out, 'text') else str(out)
        prompt_tokens = getattr(out, 'prompt_tokens', 0)
        completion_tokens = getattr(out, 'generation_tokens', 0)
        gen_tps = getattr(out, 'generation_tps', 0)
        peak_mem = getattr(out, 'peak_memory', 0)

        # mlx_vlm doesn't separate reasoning_tokens — they're in content
        # Try to extract reasoning span if present (Gemma 4 channel markers)
        think_tokens = 0
        if '<|channel>thought' in content and '<channel|>' in content:
            think_part = content.split('<|channel>thought', 1)[1]
            if '<channel|>' in think_part:
                think_text = think_part.split('<channel|>', 1)[0]
                think_tokens = len(tokenizer.encode(think_text))

        # finish_reason — mlx_vlm doesn't expose it directly, infer from token count
        finish_reason = 'length' if completion_tokens >= max_tokens - 5 else 'stop'

        return {
            'content': content,
            'finish_reason': finish_reason,
            'elapsed': elapsed,
            'prompt_tokens': prompt_tokens,
            'comp_tokens': completion_tokens,
            'think_tokens': think_tokens,
            'visible_tokens': completion_tokens - think_tokens,
            'tok_s': gen_tps,
            'peak_mem_gb': peak_mem,
            'error': None,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            'content': '', 'finish_reason': 'error', 'elapsed': elapsed,
            'prompt_tokens': 0, 'comp_tokens': 0, 'think_tokens': 0,
            'visible_tokens': 0, 'tok_s': 0, 'peak_mem_gb': 0, 'error': str(e),
        }

def get_model_config_static(model_id):
    """Build a static model config from model name (no API)."""
    return {
        'model_id': model_id,
        'architecture': 'gemma4' if 'gemma' in model_id.lower() else 'unknown',
        'quantization': '4-bit MLX',
        'engine': 'mlx_vlm',
    }

# ---- Prior-run scan for auto-skip ----

def load_prior_entries(benchmark, model_id, output_dir):
    """Scan prior JSONL logs for same (benchmark, model). Return {question_num: entry}
    keeping only successful entries (finish_reason != 'error'). Latest entry wins on dup."""
    prior = {}
    safe_model = model_id.replace('/', '_')
    # Look at both mlx_ and any other prefix that might collide — match on content, not filename.
    for path in sorted(output_dir.glob(f"mlx_{benchmark}_{safe_model}_*.jsonl")):
        try:
            with open(path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    if entry.get('benchmark') != benchmark: continue
                    if entry.get('model') != model_id: continue
                    if entry.get('finish_reason') == 'error': continue
                    qnum = entry.get('question_num')
                    if qnum is None: continue
                    prior[qnum] = entry  # later file wins
        except Exception:
            continue
    return prior

# ---- Main runner (mirrors bench2.py.run_benchmark) ----

def run_benchmark(benchmark, questions, model_id, model, processor, tokenizer,
                  only_questions=None, max_tokens=32768, enable_thinking=True,
                  force_rerun=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_model = model_id.replace('/', '_')
    run_name = f"mlx_{benchmark}_{safe_model}_{timestamp}"
    log_file = OUTPUT_DIR / f"{run_name}.jsonl"
    summary_file = OUTPUT_DIR / f"{run_name}_summary.json"

    model_config = get_model_config_static(model_id)
    hardware = get_hardware_info()
    initial_state = get_system_state()

    if only_questions:
        only_set = set(only_questions)
        questions = [q for q in questions if q['question_num'] in only_set]
        print(f"Running {len(questions)} specific questions: {sorted(only_set)}")

    # Auto-skip questions already completed in prior runs (unless --force or --only)
    prior_entries = {}
    if not force_rerun and not only_questions:
        prior_entries = load_prior_entries(benchmark, model_id, OUTPUT_DIR)
        if prior_entries:
            skipped_nums = sorted(prior_entries.keys())
            before = len(questions)
            questions = [q for q in questions if q['question_num'] not in prior_entries]
            print(f"  Auto-skip: {before - len(questions)} questions already completed for "
                  f"{model_id} / {benchmark} (use --force to rerun all)")
            if skipped_nums[:10]:
                print(f"    skipping: {skipped_nums[:10]}{'...' if len(skipped_nums) > 10 else ''}")

    n = len(questions)
    # Seed running tallies from prior successful entries so the live score reflects cumulative.
    correct_count = sum(1 for e in prior_entries.values() if e.get('correct'))
    total_count = len(prior_entries)
    carried_over = len(prior_entries)

    print(f"\n{'='*110}")
    print(f"BENCHMARK: {benchmark.upper()} | Model: {model_id} | Questions: {n}")
    print(f"Engine: mlx_vlm Python API")
    print(f"Params: temp={TEMPERATURE}, max_tokens={max_tokens}, seed={SEED}, enable_thinking={enable_thinking}")
    print(f"System: RAM={hardware.get('total_ram_gb','?')}GB, CPU={hardware.get('cpu','?')}")
    print(f"State:  GPU={initial_state['gpu_temp_c']}°C, swap={initial_state['swap_used_mb']}MB, "
          f"process_ram={initial_state['process_ram_mb']}MB")
    if carried_over:
        prior_score = correct_count / carried_over * 100
        print(f"Carried over: {carried_over} prior entries "
              f"({correct_count}/{carried_over} correct = {prior_score:.0f}%)")
    print(f"Log: {log_file}")
    print(f"{'='*110}")
    sys.stdout.flush()

    start_time = time.time()

    def _ready(s):
        return s['gpu_temp_c'] is not None and s['gpu_temp_c'] <= COOLDOWN_GPU_TEMP

    with open(log_file, 'w') as logf:
        for qi, q in enumerate(questions):
            qnum = q['question_num']
            state = get_system_state()

            # Thermal safety: require a VALID reading ≤ COOLDOWN_GPU_TEMP before each question.
            # Never resume on a None reading — macmon drops readings occasionally and we don't
            # want to proceed blind during thermal stress.
            if not _ready(state):
                t = state['gpu_temp_c']
                t_str = f"{t}°C" if t is not None else "None (no reading)"
                print(f"  ⏸ GPU at {t_str} — waiting for valid reading ≤{COOLDOWN_GPU_TEMP}°C")
                while True:
                    time.sleep(15)
                    cool_state = get_system_state()
                    if _ready(cool_state):
                        print(f"  ✓ GPU at {cool_state['gpu_temp_c']}°C — resuming")
                        state = cool_state
                        break
                    ct = cool_state['gpu_temp_c']
                    ct_str = f"{ct}°C" if ct is not None else "None"
                    print(f"  ... GPU {ct_str}, still waiting")

            r = make_call(model, processor, tokenizer, q['user_prompt'], max_tokens, enable_thinking)
            total_count += 1
            state_after = get_system_state()

            is_correct, extracted, reason = grade(benchmark, q, r['content'], r['finish_reason'])
            if is_correct:
                correct_count += 1

            entry = {
                'run_name': run_name,
                'benchmark': benchmark,
                'model': model_id,
                'engine': 'mlx_vlm',
                'question_num': qnum,
                'question_of': n,
                'dataset_idx': q.get('dataset_idx'),
                'timestamp': datetime.now().isoformat(),
                'level': q.get('level'),
                'type': q.get('type'),
                'subject': q.get('subject'),
                'entry_point': q.get('entry_point'),
                'max_tokens_sent': max_tokens,
                'temperature': TEMPERATURE,
                'enable_thinking': enable_thinking,
                'prompt_tokens': r['prompt_tokens'],
                'completion_tokens': r['comp_tokens'],
                'reasoning_tokens': r['think_tokens'],
                'visible_tokens': r['visible_tokens'],
                'total_tokens': r['prompt_tokens'] + r['comp_tokens'],
                'finish_reason': r['finish_reason'],
                'elapsed_s': round(r['elapsed'], 2),
                'tok_s': round(r['tok_s'], 1),
                'correct': is_correct,
                'extracted_answer': str(extracted) if extracted else None,
                'expected_answer': str(q.get('expected', q.get('expected_answers', ''))),
                'grade_reason': reason,
                'gpu_temp_c': state['gpu_temp_c'],
                'cpu_temp_c': state['cpu_temp_c'],
                'process_ram_mb': state['process_ram_mb'],
                'swap_used_mb': state['swap_used_mb'],
                'gpu_temp_after_c': state_after['gpu_temp_c'],
                'swap_after_mb': state_after['swap_used_mb'],
                'mlx_peak_memory_gb': r['peak_mem_gb'],
                'model_arch': model_config.get('architecture'),
                'model_quant': model_config.get('quantization'),
                'raw_response': r['content'] if r['content'] else '',
                'error': r['error'],
            }
            logf.write(json.dumps(entry, ensure_ascii=False) + '\n')
            logf.flush()

            status = "OK" if is_correct else ("FAIL" if is_correct is False else "???")
            temp_str = f"{state_after['gpu_temp_c']}°C" if state_after['gpu_temp_c'] else "?"
            ram_str = f"mem={r['peak_mem_gb']:.1f}G sw={state['swap_used_mb']}M"
            detail = ''
            if benchmark == 'math':
                detail = f" | {q.get('level','?')}"
            elif benchmark in ('mmlu', 'gpqa'):
                detail = f" | exp={q.get('expected','?')} got={extracted}"
            elif benchmark == 'drop':
                detail = f" | exp={str(q.get('expected_answers',['?']))[:30]}"
            elif benchmark == 'humaneval':
                detail = f" | {q.get('entry_point','?')}"

            trunc_flag = " TRUNC!" if r['finish_reason'] == 'length' else ""
            err_flag = " ERR!" if r['error'] else ""
            score_so_far = correct_count / total_count * 100

            print(f"  {qi+1:>3}/{n} Q{qnum:<3d} | {r['elapsed']:6.1f}s | "
                  f"p={r['prompt_tokens']:>4d} c={r['comp_tokens']:>5d} "
                  f"(t={r['think_tokens']:>5d}+v={r['visible_tokens']:>4d}) "
                  f"{r['tok_s']:>5.1f}t/s | {temp_str:>5s} {ram_str} | "
                  f"{status:5}{trunc_flag}{err_flag}{detail} | "
                  f"score: {score_so_far:.0f}%")
            sys.stdout.flush()
            gc.collect()

    elapsed_total = time.time() - start_time
    score = correct_count / total_count if total_count > 0 else 0
    truncated = sum(1 for q in open(log_file) if '"finish_reason": "length"' in q)
    final_state = get_system_state()

    summary = {
        'run_name': run_name,
        'benchmark': benchmark,
        'model': model_id,
        'engine': 'mlx_vlm',
        'score': score,
        'score_pct': f"{score*100:.1f}%",
        'correct': correct_count,
        'total': total_count,
        'carried_over': carried_over,
        'fresh_run': total_count - carried_over,
        'truncated': truncated,
        'elapsed_s': round(elapsed_total),
        'elapsed_min': round(elapsed_total / 60, 1),
        'max_tokens': max_tokens,
        'temperature': TEMPERATURE,
        'seed': SEED,
        'enable_thinking': enable_thinking,
        'model_config': model_config,
        'hardware': hardware,
        'initial_state': initial_state,
        'final_state': final_state,
        'timestamp_start': datetime.fromtimestamp(start_time).isoformat(),
        'timestamp_end': datetime.now().isoformat(),
        'log_file': str(log_file),
        'only_questions': only_questions,
        'questions_run': [q['question_num'] for q in questions],
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    fresh_run = total_count - carried_over
    print(f"\n  >>> {benchmark.upper()}: {score*100:.0f}% ({correct_count}/{total_count}) in {elapsed_total:.0f}s ({elapsed_total/60:.1f}m)"
          f" [fresh={fresh_run}, carried={carried_over}]")
    if truncated > 0:
        print(f"  >>> {truncated} questions truncated at {max_tokens} tokens")
    print(f"  >>> GPU: {initial_state['gpu_temp_c']}°C → {final_state['gpu_temp_c']}°C")
    print(f"  >>> RAM: {initial_state['process_ram_mb']}MB | Swap: {initial_state['swap_used_mb']}MB → {final_state['swap_used_mb']}MB")
    print(f"  >>> Log: {log_file}")
    print(f"  >>> Summary: {summary_file}")
    return score, summary

# ---- CLI ----

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MLX benchmark harness mirroring bench2.py")
    parser.add_argument("benchmark", choices=["mmlu", "math", "humaneval", "gpqa", "drop"])
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--model", required=True, help="HF model ID for mlx_vlm.load")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--only", type=str, help="Comma-separated question numbers")
    parser.add_argument("--thinking", choices=['on','off'], default='on')
    parser.add_argument("--force", action="store_true",
                        help="Rerun questions even if already completed in prior runs "
                             "(default: auto-skip completed)")
    args = parser.parse_args()

    only = None
    if args.only:
        only = [int(x.strip()) for x in args.only.split(',')]

    if args.max_tokens is None:
        args.max_tokens = {'math': 32768, 'gpqa': 32768, 'humaneval': 16384,
                           'mmlu': 8192, 'drop': 8192}[args.benchmark]

    enable_thinking = (args.thinking == 'on')

    print(f"Loading {args.benchmark} dataset...")
    questions = LOADERS[args.benchmark](args.examples)
    print(f"Loaded {len(questions)} questions")

    print(f"\n--- Pre-flight ---")
    state = get_system_state()
    hw = get_hardware_info()
    print(f"  Model: {args.model}")
    print(f"  Engine: mlx_vlm Python API")
    print(f"  Hardware: {hw.get('cpu','?')}, {hw.get('total_ram_gb','?')}GB RAM, macOS {hw.get('macos_version','?')}")
    print(f"  GPU: {state['gpu_temp_c']}°C | CPU: {state['cpu_temp_c']}°C" if state['gpu_temp_c'] else "  Temps: ?")
    print(f"  Swap: {state['swap_used_mb']}MB")
    print(f"  max_tokens: {args.max_tokens} | thinking: {args.thinking}")

    if state['gpu_temp_c'] and state['gpu_temp_c'] > 60:
        print(f"  WARNING: GPU is warm ({state['gpu_temp_c']}°C). Consider waiting for cooldown.")

    print(f"\n  Loading model... ", end=''); sys.stdout.flush()
    t0 = time.time()
    from mlx_vlm import load
    model, processor = load(args.model)
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    print(f"{time.time()-t0:.1f}s")

    print(f"  Warmup (2+2)... ", end=''); sys.stdout.flush()
    wr = make_call(model, processor, tokenizer, "What is 2+2? Just the number.",
                   max_tokens=200, enable_thinking=enable_thinking)
    print(f"'{wr['content'].strip()[:30]}' | p={wr['prompt_tokens']} c={wr['comp_tokens']} "
          f"(t={wr['think_tokens']}+v={wr['visible_tokens']}) {wr['tok_s']:.1f}t/s | {wr['finish_reason']}")
    if '4' not in wr['content']:
        print(f"  WARNING: Warmup response doesn't contain '4': '{wr['content'][:50]}'")

    score, summary = run_benchmark(
        args.benchmark, questions, args.model, model, processor, tokenizer,
        only_questions=only, max_tokens=args.max_tokens,
        enable_thinking=enable_thinking, force_rerun=args.force,
    )
