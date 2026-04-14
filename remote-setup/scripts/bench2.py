#!/usr/bin/env python3
"""
Benchmark harness v2 — with complete logging.

Every question saves: prompt_tokens, completion_tokens, reasoning_tokens,
visible_tokens, finish_reason, elapsed, tok/s, gpu_temp, correct/wrong,
expected answer, extracted answer, raw response text.

Output:
  - Per-question JSONL log (one line per question, machine-readable)
  - Summary JSON (aggregate scores)
  - Real-time console output (human-readable)

Usage:
  python3 bench2.py math --examples 100
  python3 bench2.py math --examples 100 --only 4,13,15,26   # rerun specific questions
  python3 bench2.py gpqa --examples 100 --model google/gemma-4-26b-a4b
"""
import time, json, sys, os, subprocess, re, random, argparse
from datetime import datetime
from urllib.request import urlopen, Request
from pathlib import Path

# ---- Config ----
# Supports LMSTUDIO_URL env var to point at a remote LM Studio instance
# Default: local LM Studio on port 1234
BASE_URL = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
TEMPERATURE = 0
MAX_TOKENS = 32768
TIMEOUT = 1800  # 30 min per question
SEED = 42

# ---- Directories ----
SCRIPT_DIR = Path(__file__).parent
# Output goes next to this script (remote-setup/scripts/runs/)
OUTPUT_DIR = SCRIPT_DIR.parent / "runs"

# ---- API ----

def api_call(messages, max_tokens=MAX_TOKENS, model="local"):
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }).encode()
    req = Request(f"{BASE_URL}/chat/completions", data=payload,
                  headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read())
        elapsed = time.time() - start
        choice = data["choices"][0]
        usage = data.get("usage", {})
        comp = usage.get("completion_tokens", 0)
        think = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        prompt_tok = usage.get("prompt_tokens", 0)
        tok_s = comp / elapsed if elapsed > 0 else 0
        return {
            "content": choice["message"].get("content", ""),
            "finish_reason": choice.get("finish_reason", "?"),
            "elapsed": elapsed,
            "prompt_tokens": prompt_tok,
            "comp_tokens": comp,
            "think_tokens": think,
            "visible_tokens": comp - think,
            "tok_s": tok_s,
            "error": None,
            "model_id": data.get("model", model),
        }
    except Exception as e:
        return {
            "content": "", "finish_reason": "error", "elapsed": time.time() - start,
            "prompt_tokens": 0, "comp_tokens": 0, "think_tokens": 0,
            "visible_tokens": 0, "tok_s": 0, "error": str(e), "model_id": model,
        }

def get_system_state():
    """Capture full system state: GPU temp, CPU temp, RAM, swap.
    Works on macOS (macmon) and Windows (nvidia-smi + psutil)."""
    state = {'gpu_temp_c': None, 'cpu_temp_c': None, 'process_ram_mb': 0, 'swap_used_mb': 0}
    is_windows = sys.platform.startswith('win')
    is_macos = sys.platform == 'darwin'

    # GPU temp — nvidia-smi on Windows/Linux, macmon on macOS
    if is_windows or sys.platform.startswith('linux'):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                gpu = int(r.stdout.strip().split('\n')[0])
                state['gpu_temp_c'] = gpu if 5 <= gpu <= 120 else None
        except: pass
    elif is_macos:
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

    # RAM + Swap — psutil works on all platforms
    try:
        import psutil
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        state['swap_used_mb'] = round(swap.used / 1024**2)
        # LM Studio process RAM (find by name)
        max_mem = 0
        for proc in psutil.process_iter(['name', 'memory_info']):
            try:
                name = (proc.info['name'] or '').lower()
                if 'llmworker' in name or 'lm studio' in name or 'lmstudio' in name:
                    mem = proc.info['memory_info'].rss
                    if mem > max_mem:
                        max_mem = mem
            except: pass
        state['process_ram_mb'] = round(max_mem / 1024**2)
    except ImportError:
        # Fall back to macOS-only sysctl if psutil not installed
        if is_macos:
            try:
                r = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True)
                m = re.search(r'used\s*=\s*([0-9.]+)M', r.stdout)
                if m: state['swap_used_mb'] = round(float(m.group(1)))
            except: pass
            try:
                r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
                for line in r.stdout.split('\n'):
                    if 'llmworker' in line.lower() or ('node' in line and '.lmstudio' in line):
                        parts = line.split()
                        if len(parts) >= 6:
                            try:
                                mem_kb = int(parts[5])
                                if mem_kb > state['process_ram_mb'] * 1024:
                                    state['process_ram_mb'] = round(mem_kb / 1024)
                            except: pass
            except: pass
    except: pass

    return state

def get_model_config():
    """Get loaded model config from LM Studio API."""
    try:
        import requests as req
        # Strip /v1 suffix to get base URL for the /api/v1 endpoint
        base = BASE_URL.replace("/v1", "") if BASE_URL.endswith("/v1") else BASE_URL
        r = req.get(f"{base}/api/v1/models", timeout=5)
        for m in r.json().get('models', []):
            if m.get('loaded_instances'):
                inst = m['loaded_instances'][0]
                return {
                    'model_id': m['key'],
                    'architecture': m.get('architecture', '?'),
                    'quantization': m.get('quantization', {}).get('name', '?'),
                    'bits_per_weight': m.get('quantization', {}).get('bits_per_weight', '?'),
                    'size_bytes': m.get('size_bytes', 0),
                    'size_gb': round(m.get('size_bytes', 0) / 1024**3, 1),
                    'params': m.get('params_string', '?'),
                    'context_length': inst.get('config', {}).get('context_length', '?'),
                    'flash_attention': inst.get('config', {}).get('flash_attention', '?'),
                    'eval_batch_size': inst.get('config', {}).get('eval_batch_size', '?'),
                    'num_experts': inst.get('config', {}).get('num_experts', '?'),
                }
        return None
    except:
        return None

def get_hardware_info():
    """Get static hardware info. Cross-platform."""
    import platform
    info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'arch': platform.machine(),
    }
    try:
        import psutil
        info['total_ram_gb'] = round(psutil.virtual_memory().total / 1024**3)
        info['cpu_count'] = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    info['cpu'] = platform.processor() or '?'

    # GPU info via nvidia-smi (Windows/Linux)
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            info['gpu'] = r.stdout.strip().split('\n')[0]
    except: pass
    return info

# ---- Answer extraction ----

def extract_boxed(text):
    """Extract content from \\boxed{}, handling nested braces."""
    idx = text.find('\\boxed{')
    if idx == -1: return None
    start = idx + 7; depth = 1; pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == '{': depth += 1
        elif text[pos] == '}': depth -= 1
        pos += 1
    return text[start:pos-1] if depth == 0 else None

def normalize_math(s):
    """Normalize a LaTeX math expression for comparison.
    Handles: whitespace, \\frac shorthand, \\left/\\right, \\text{}, \\mathrm{}."""
    s = re.sub(r'\s+', '', s)
    # Normalize \frac without braces: \frac12 → \frac{1}{2}, \fracab → \frac{a}{b}
    # But \frac{1}{2} stays as is
    def expand_frac(m):
        after = m.group(1)
        # Already has braces
        if after.startswith('{'):
            return '\\frac' + after
        # Two single chars: \frac12 → \frac{1}{2}
        if len(after) >= 2:
            return f'\\frac{{{after[0]}}}{{{after[1]}}}' + after[2:]
        return '\\frac' + after
    s = re.sub(r'\\frac([^{\\].*?)(?=\\|$|\+|-|\)|\]|,|\s|=)', expand_frac, s)
    # Also handle the simple case: \fracXY where X,Y are single non-brace chars
    s = re.sub(r'\\frac([^{])([^{])', r'\\frac{\1}{\2}', s)
    # Remove \left and \right (they're just sizing hints)
    s = s.replace('\\left', '').replace('\\right', '')
    # Remove \text{} and \mathrm{} wrappers
    s = re.sub(r'\\(?:text|mathrm)\{([^}]*)\}', r'\1', s)
    # Remove trailing period
    s = s.rstrip('.')
    return s

def extract_answer_letter(text):
    if not text: return "?"
    text = text.strip()
    for region in [text[-100:], text[:20], text[-30:]]:
        for pat in [r'[Aa]nswer[:\s]+([A-D])', r'\(([A-D])\)', r'\b([A-D])\b']:
            m = re.search(pat, region)
            if m: return m.group(1).upper()
    return "?"

# ---- Dataset loaders ----

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
            'messages': [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Solve this math problem. Put your final answer in \\boxed{{}}.\n\n{row['problem']}"},
            ],
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
            'messages': [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{row['question']}\nAnswer with just the letter."},
            ],
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
            'messages': [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
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
            'messages': [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Read the passage and answer the question. Give ONLY the answer, nothing else.\n\nPassage: {row['passage']}\n\nQuestion: {row['question']}"},
            ],
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
            'messages': [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Complete the following Python function. Output ONLY the complete function, no explanation.\n\n{row['prompt']}"},
            ],
        })
    return questions

LOADERS = {
    'math': load_math_questions,
    'gpqa': load_gpqa_questions,
    'mmlu': load_mmlu_questions,
    'drop': load_drop_questions,
    'humaneval': load_humaneval_questions,
}

# ---- Grading ----

def grade(benchmark, question, response_text, finish_reason):
    """Grade a single response. Returns (correct: bool, extracted: str, reason: str)."""
    if finish_reason == 'length':
        return False, None, f'truncated'

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
            return False, question['entry_point'], f'exec_error: {e}'

    return None, None, 'ungraded'

# ---- Main runner ----

def run_benchmark(benchmark, questions, model_id, only_questions=None, max_tokens=MAX_TOKENS):
    """Run benchmark questions and log everything."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_model = model_id.replace('/', '_')
    run_name = f"{benchmark}_{safe_model}_{timestamp}"
    log_file = OUTPUT_DIR / f"{run_name}.jsonl"
    summary_file = OUTPUT_DIR / f"{run_name}_summary.json"

    # Capture run-level metadata
    model_config = get_model_config()
    hardware = get_hardware_info()
    initial_state = get_system_state()

    # Filter to specific questions if --only specified
    if only_questions:
        only_set = set(only_questions)
        questions = [q for q in questions if q['question_num'] in only_set]
        print(f"Running {len(questions)} specific questions: {sorted(only_set)}")

    n = len(questions)
    correct_count = 0
    total_count = 0

    print(f"\n{'='*110}")
    print(f"BENCHMARK: {benchmark.upper()} | Model: {model_id} | Questions: {n}")
    print(f"Params: temp={TEMPERATURE}, max_tokens={max_tokens}, seed={SEED}")
    if model_config:
        print(f"Config: ctx={model_config.get('context_length','?')}, quant={model_config.get('quantization','?')}, "
              f"size={model_config.get('size_gb','?')}GB, arch={model_config.get('architecture','?')}, "
              f"experts={model_config.get('num_experts','?')}")
    print(f"System: RAM={hardware.get('total_ram_gb','?')}GB, CPU={hardware.get('cpu','?')}")
    print(f"State:  GPU={initial_state['gpu_temp_c']}°C, swap={initial_state['swap_used_mb']}MB, "
          f"process_ram={initial_state['process_ram_mb']}MB")
    print(f"Log: {log_file}")
    print(f"{'='*110}")
    sys.stdout.flush()

    start_time = time.time()

    with open(log_file, 'w') as logf:
        for qi, q in enumerate(questions):
            qnum = q['question_num']

            # Get system state before question
            state = get_system_state()

            # Send to API
            r = api_call(q['messages'], max_tokens=max_tokens, model=model_id)
            total_count += 1

            # Get system state after question
            state_after = get_system_state()

            # Grade
            is_correct, extracted, reason = grade(benchmark, q, r['content'], r['finish_reason'])
            if is_correct:
                correct_count += 1

            # Build log entry — everything we could ever need
            entry = {
                # Run identity
                'run_name': run_name,
                'benchmark': benchmark,
                'model': model_id,
                'question_num': qnum,
                'question_of': n,
                'dataset_idx': q.get('dataset_idx'),
                'timestamp': datetime.now().isoformat(),
                # Question metadata
                'level': q.get('level'),
                'type': q.get('type'),
                'subject': q.get('subject'),
                'entry_point': q.get('entry_point'),
                # API request params
                'max_tokens_sent': max_tokens,
                'temperature': TEMPERATURE,
                'context_length': model_config.get('context_length') if model_config else None,
                # API response — token accounting
                'prompt_tokens': r['prompt_tokens'],
                'completion_tokens': r['comp_tokens'],
                'reasoning_tokens': r['think_tokens'],
                'visible_tokens': r['visible_tokens'],
                'total_tokens': r['prompt_tokens'] + r['comp_tokens'],
                'finish_reason': r['finish_reason'],
                # Timing
                'elapsed_s': round(r['elapsed'], 2),
                'tok_s': round(r['tok_s'], 1),
                # Grading
                'correct': is_correct,
                'extracted_answer': str(extracted) if extracted else None,
                'expected_answer': str(q.get('expected', q.get('expected_answers', ''))),
                'grade_reason': reason,
                # System state (before question)
                'gpu_temp_c': state['gpu_temp_c'],
                'cpu_temp_c': state['cpu_temp_c'],
                'process_ram_mb': state['process_ram_mb'],
                'swap_used_mb': state['swap_used_mb'],
                # System state (after question)
                'gpu_temp_after_c': state_after['gpu_temp_c'],
                'swap_after_mb': state_after['swap_used_mb'],
                # Model config
                'model_arch': model_config.get('architecture') if model_config else None,
                'model_quant': model_config.get('quantization') if model_config else None,
                'model_size_gb': model_config.get('size_gb') if model_config else None,
                'model_params': model_config.get('params') if model_config else None,
                # Raw response (full — for audit and re-grading)
                'raw_response': r['content'] if r['content'] else '',
                'error': r['error'],
            }

            # Write to log file
            logf.write(json.dumps(entry, ensure_ascii=False) + '\n')
            logf.flush()

            # Console output
            status = "OK" if is_correct else ("FAIL" if is_correct is False else "???")
            temp_str = f"{state_after['gpu_temp_c']}°C" if state_after['gpu_temp_c'] else "?"
            ram_str = f"ram={state['process_ram_mb']}M sw={state['swap_used_mb']}M"
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
            score_so_far = correct_count / total_count * 100

            print(f"  {qi+1:>3}/{n} Q{qnum:<3d} | {r['elapsed']:6.1f}s | "
                  f"p={r['prompt_tokens']:>4d} c={r['comp_tokens']:>5d} "
                  f"(t={r['think_tokens']:>5d}+v={r['visible_tokens']:>4d}) "
                  f"{r['tok_s']:>5.1f}t/s | {temp_str:>5s} {ram_str} | "
                  f"{status:5}{trunc_flag}{detail} | "
                  f"score: {score_so_far:.0f}%")
            sys.stdout.flush()

    elapsed_total = time.time() - start_time
    score = correct_count / total_count if total_count > 0 else 0
    truncated = sum(1 for q in open(log_file) if '"finish_reason": "length"' in q)
    final_state = get_system_state()

    # Summary — includes everything needed to reproduce
    summary = {
        'run_name': run_name,
        'benchmark': benchmark,
        'model': model_id,
        'score': score,
        'score_pct': f"{score*100:.1f}%",
        'correct': correct_count,
        'total': total_count,
        'truncated': truncated,
        'elapsed_s': round(elapsed_total),
        'elapsed_min': round(elapsed_total / 60, 1),
        # Params
        'max_tokens': max_tokens,
        'temperature': TEMPERATURE,
        'seed': SEED,
        'timeout_s': TIMEOUT,
        # Model config
        'model_config': model_config,
        # Hardware
        'hardware': hardware,
        # System state
        'initial_state': initial_state,
        'final_state': final_state,
        # Run info
        'timestamp_start': datetime.fromtimestamp(start_time).isoformat(),
        'timestamp_end': datetime.now().isoformat(),
        'log_file': str(log_file),
        'only_questions': only_questions,
        'questions_run': [q['question_num'] for q in questions],
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  >>> {benchmark.upper()}: {score*100:.0f}% ({correct_count}/{total_count}) in {elapsed_total:.0f}s ({elapsed_total/60:.1f}m)")
    if truncated > 0:
        print(f"  >>> {truncated} questions truncated at {max_tokens} tokens")
    print(f"  >>> GPU: {initial_state['gpu_temp_c']}°C → {final_state['gpu_temp_c']}°C")
    print(f"  >>> RAM: {initial_state['process_ram_mb']}MB | Swap: {initial_state['swap_used_mb']}MB → {final_state['swap_used_mb']}MB")
    print(f"  >>> Log: {log_file}")
    print(f"  >>> Summary: {summary_file}")

    return score, summary

# ---- CLI ----

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark harness v2 with full logging")
    parser.add_argument("benchmark", choices=["mmlu", "math", "humaneval", "gpqa", "drop"])
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--model", type=str, default="local")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--only", type=str, help="Comma-separated question numbers to run (1-indexed)")
    args = parser.parse_args()

    # Parse --only
    only = None
    if args.only:
        only = [int(x.strip()) for x in args.only.split(',')]

    # Load questions
    print(f"Loading {args.benchmark} dataset...")
    loader = LOADERS[args.benchmark]
    questions = loader(args.examples)
    print(f"Loaded {len(questions)} questions")

    # Detect model if "local"
    model_id = args.model
    if model_id == "local":
        try:
            import requests
            r = requests.get("http://127.0.0.1:1234/api/v1/models", timeout=5)
            for m in r.json().get('models', []):
                if m.get('loaded_instances'):
                    model_id = m['key']
                    break
        except:
            pass

    # Pre-flight check
    print(f"\n--- Pre-flight ---")
    state = get_system_state()
    mcfg = get_model_config()
    hw = get_hardware_info()
    print(f"  Model: {model_id}")
    if mcfg:
        print(f"  Config: ctx={mcfg.get('context_length','?')}, quant={mcfg.get('quantization','?')}, "
              f"size={mcfg.get('size_gb','?')}GB, arch={mcfg.get('architecture','?')}")
    print(f"  Hardware: {hw.get('cpu','?')}, {hw.get('total_ram_gb','?')}GB RAM, macOS {hw.get('macos_version','?')}")
    print(f"  GPU: {state['gpu_temp_c']}°C | CPU: {state['cpu_temp_c']}°C" if state['gpu_temp_c'] else "  Temps: ?")
    print(f"  RAM: {state['process_ram_mb']}MB process | Swap: {state['swap_used_mb']}MB")
    print(f"  max_tokens: {args.max_tokens}")

    # Warmup
    print(f"  Warmup...", end=' ')
    sys.stdout.flush()
    wr = api_call([{"role": "user", "content": "What is 2+2? Just the number."}],
                  max_tokens=200, model=model_id)
    print(f"'{wr['content'].strip()[:20]}' | p={wr['prompt_tokens']} c={wr['comp_tokens']} "
          f"(t={wr['think_tokens']}+v={wr['visible_tokens']}) {wr['tok_s']:.1f}t/s | {wr['finish_reason']}")
    if '4' not in wr['content']:
        print(f"  WARNING: Warmup response doesn't contain '4': '{wr['content'][:50]}'")

    # Run
    score, summary = run_benchmark(
        args.benchmark, questions, model_id,
        only_questions=only, max_tokens=args.max_tokens
    )
