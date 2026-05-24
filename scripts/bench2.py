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
  python3 bench2.py livecodebench --examples 50 --lcb-version release_v6

Benchmarks: mmlu, math, humaneval, gpqa, drop, livecodebench
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
# 30 min per question default; override via BENCH_TIMEOUT env var (seconds).
# Slow dense thinking models (e.g. qwen3.6-27b at ~20 t/s) can need ~55 min
# to spend the full 65 536 max_tokens budget. Without a longer timeout,
# urlopen aborts mid-flight and the next request queues behind the still-
# running LM Studio inference, cascading into back-to-back timeouts and
# wedging the run.
TIMEOUT = int(os.environ.get("BENCH_TIMEOUT", "1800"))
SEED = 42

# ---- Directories ----
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / "benchmarks" / "runs"

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
    """Capture full system state: GPU temp, CPU temp, RAM, swap."""
    state = {'gpu_temp_c': None, 'cpu_temp_c': None, 'process_ram_mb': 0, 'swap_used_mb': 0}

    # GPU + CPU temp from macmon
    try:
        proc = subprocess.Popen(["macmon", "pipe"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        line = proc.stdout.readline().decode().strip()
        proc.kill(); proc.wait()
        if line:
            d = json.loads(line)
            gpu = round(d.get('temp', {}).get('gpu_temp_avg', 0))
            cpu = round(d.get('temp', {}).get('cpu_temp_avg', 0))
            # Filter out bad readings (sensor glitches return negative or impossibly high values)
            state['gpu_temp_c'] = gpu if 5 <= gpu <= 120 else None
            state['cpu_temp_c'] = cpu if 5 <= cpu <= 120 else None
    except: pass

    # Swap
    try:
        r = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True)
        m = re.search(r'used\s*=\s*([0-9.]+)M', r.stdout)
        if m: state['swap_used_mb'] = round(float(m.group(1)))
    except: pass

    # LM Studio process RAM
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

    return state

def get_model_config():
    """Get loaded model config from LM Studio API."""
    try:
        import requests as req
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
    """Get static hardware info."""
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

# LiveCodeBench prompts (mirrors lcb_runner/prompts/code_generation.py shape)
LCB_SYS_FUNCTIONAL = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests."
)
LCB_INSTR_FUNCTIONAL = (
    "### Question:\n{question}\n\n### Format:\nYou will use the following starter code "
    "to write the solution and enclose your code within delimiters.\n```python\n{starter}\n```\n\n"
    "### Answer: (use the provided format with backticks)"
)
LCB_INSTR_STDIN = (
    "### Question:\n{question}\n\n### Format: read the inputs from stdin solve the problem "
    "and write the answer to stdout (do not directly test on the sample inputs). "
    "Enclose your code within delimiters as follows.\n"
    "```python\n# YOUR CODE HERE\n```\n\n"
    "### Answer: (use the provided format with backticks)"
)

def _decode_lcb_private_tests(raw):
    """Private tests in LiveCodeBench are stored either as raw JSON or as
    base64-encoded zlib-compressed JSON/pickle. Try both."""
    import base64, zlib, pickle
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        decoded = base64.b64decode(raw.encode("utf-8") if isinstance(raw, str) else raw)
        decompressed = zlib.decompress(decoded)
        try:
            return json.loads(decompressed)
        except Exception:
            return json.loads(pickle.loads(decompressed))
    except Exception as e:
        print(f"  WARN: could not decode private tests: {e}", file=sys.stderr)
        return []

LCB_VERSION_FILES = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}

def load_livecodebench_questions(n, version="release_v6"):
    """Load LiveCodeBench code-generation problems. `version` is one of
    release_v1..release_v6 (rolling time windows). v6 covers problems through
    ~April 2025 — the contamination-resistant choice for current models.

    Loads JSONL shards directly from the HF Hub (the dataset uses a script-based
    loader that the modern `datasets` library no longer accepts)."""
    from huggingface_hub import hf_hub_download
    files = LCB_VERSION_FILES[version]
    rows = []
    for fname in files:
        path = hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            filename=fname,
            repo_type="dataset",
        )
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    random.seed(SEED)
    indices = random.sample(range(len(rows)), min(n, len(rows)))
    questions = []
    for i, idx in enumerate(indices):
        row = rows[idx]
        public_tests = json.loads(row.get('public_test_cases') or '[]')
        private_tests = _decode_lcb_private_tests(row.get('private_test_cases'))
        all_tests = public_tests + private_tests
        # Determine test type from first test (LCB problems are homogeneous)
        testtype = all_tests[0].get('testtype', 'stdin') if all_tests else 'stdin'
        # For functional tests, function name lives in metadata
        metadata = {}
        try:
            metadata = json.loads(row.get('metadata') or '{}')
        except Exception:
            pass
        fn_name = metadata.get('func_name')
        starter = row.get('starter_code') or ''

        if testtype == 'functional' or starter:
            instr = LCB_INSTR_FUNCTIONAL.format(
                question=row['question_content'], starter=starter
            )
        else:
            instr = LCB_INSTR_STDIN.format(question=row['question_content'])

        questions.append({
            'question_num': i + 1,
            'dataset_idx': idx,
            'question_id': row.get('question_id', f'lcb_{idx}'),
            'platform': row.get('platform', '?'),
            'difficulty': row.get('difficulty', '?'),
            'contest_date': row.get('contest_date', '?'),
            'testtype': testtype,
            'starter_code': starter,
            'fn_name': fn_name,
            'test_cases': all_tests,
            'lcb_version': version,
            'messages': [
                {"role": "system", "content": LCB_SYS_FUNCTIONAL},
                {"role": "user", "content": instr},
            ],
        })
    return questions

LOADERS = {
    'math': load_math_questions,
    'gpqa': load_gpqa_questions,
    'mmlu': load_mmlu_questions,
    'drop': load_drop_questions,
    'humaneval': load_humaneval_questions,
    'livecodebench': load_livecodebench_questions,
}

# ---- Code extraction & LiveCodeBench test runner ----

def extract_python_code(text):
    """Extract Python code from a model response. Returns the largest
    code block, or the raw text if no fences found."""
    if not text:
        return ''
    # Prefer ```python ... ``` blocks
    blocks = re.findall(r'```(?:python|py)?\s*\n?(.*?)```', text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()

LCB_TIMEOUT_S = 12  # per-test timeout

def _run_lcb_functional_test(code, test_case, fn_name, starter_code):
    """Run a single functional test. `test_case.input` holds one JSON-encoded
    argument per line; `test_case.output` is the JSON-encoded expected return."""
    raw_input = test_case.get('input', '')
    expected = (test_case.get('output') or '').strip()

    # Each non-empty line of input is one positional arg, JSON-parsed
    arg_lines = [ln for ln in raw_input.split('\n') if ln.strip() != '']
    # Use raw JSON literals; let the runner parse them in-process for fidelity
    args_repr = '[' + ', '.join(arg_lines) + ']' if arg_lines else '[]'

    if 'class Solution' in starter_code or 'class Solution' in code:
        invoke = f"Solution().{fn_name}(*_args)"
    else:
        invoke = f"{fn_name}(*_args)"

    runner = (
        f"{code}\n\n"
        f"import json as _json, sys as _sys\n"
        f"_args = {args_repr}\n"
        f"_result = {invoke}\n"
        f"_sys.stdout.write(_json.dumps(_result))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, '-c', runner],
            capture_output=True, text=True, timeout=LCB_TIMEOUT_S,
        )
        if proc.returncode != 0:
            return False, f'rc={proc.returncode}: {proc.stderr[:120]}'
        got = proc.stdout.strip()
        # Try structured equality first
        try:
            if json.loads(got) == json.loads(expected):
                return True, 'ok'
        except Exception:
            pass
        return (got == expected), 'ok' if got == expected else f'mismatch'
    except subprocess.TimeoutExpired:
        return False, 'timeout'
    except Exception as e:
        return False, f'exec_error: {e}'

def _run_lcb_stdin_test(code, test_case):
    """Run a single stdin/stdout test."""
    test_input = test_case.get('input', '')
    expected = (test_case.get('output') or '').strip()
    try:
        proc = subprocess.run(
            [sys.executable, '-c', code],
            input=test_input, capture_output=True, text=True,
            timeout=LCB_TIMEOUT_S,
        )
        if proc.returncode != 0:
            return False, f'rc={proc.returncode}: {proc.stderr[:120]}'
        got_lines = [ln.rstrip() for ln in proc.stdout.strip().split('\n')]
        exp_lines = [ln.rstrip() for ln in expected.split('\n')]
        return (got_lines == exp_lines), 'ok' if got_lines == exp_lines else 'mismatch'
    except subprocess.TimeoutExpired:
        return False, 'timeout'
    except Exception as e:
        return False, f'exec_error: {e}'

def grade_livecodebench(question, response_text):
    """Score a LiveCodeBench problem. Pass@1: all test cases must pass."""
    code = extract_python_code(response_text)
    if not code:
        return False, None, 'no_code'

    tests = question.get('test_cases') or []
    if not tests:
        return False, question['question_id'], 'no_tests'

    testtype = question.get('testtype', 'stdin')
    fn_name = question.get('fn_name')
    starter = question.get('starter_code', '')

    for idx, tc in enumerate(tests):
        tc_type = tc.get('testtype', testtype)
        if tc_type == 'functional':
            if not fn_name:
                return False, question['question_id'], 'no_fn_name'
            ok, reason = _run_lcb_functional_test(code, tc, fn_name, starter)
        else:
            ok, reason = _run_lcb_stdin_test(code, tc)
        if not ok:
            return False, question['question_id'], f'fail_test{idx}:{reason}'
    return True, question['question_id'], 'pass'

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

    elif benchmark == 'livecodebench':
        return grade_livecodebench(question, response_text)

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
                # LiveCodeBench-specific
                'lcb_question_id': q.get('question_id'),
                'lcb_platform': q.get('platform'),
                'lcb_difficulty': q.get('difficulty'),
                'lcb_testtype': q.get('testtype'),
                'lcb_version': q.get('lcb_version'),
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
            elif benchmark == 'livecodebench':
                detail = f" | {q.get('platform','?')}/{q.get('difficulty','?')} {q.get('question_id','?')}"

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
    parser.add_argument("benchmark", choices=["mmlu", "math", "humaneval", "gpqa", "drop", "livecodebench"])
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--model", type=str, default="local")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--only", type=str, help="Comma-separated question numbers to run (1-indexed)")
    parser.add_argument("--lcb-version", type=str, default="release_v6",
                        choices=["release_v1","release_v2","release_v3","release_v4","release_v5","release_v6"],
                        help="LiveCodeBench release window (default release_v6 — through ~Apr 2025)")
    args = parser.parse_args()

    # Parse --only
    only = None
    if args.only:
        only = [int(x.strip()) for x in args.only.split(',')]

    # Load questions
    print(f"Loading {args.benchmark} dataset...")
    loader = LOADERS[args.benchmark]
    if args.benchmark == 'livecodebench':
        questions = loader(args.examples, version=args.lcb_version)
        print(f"Loaded {len(questions)} questions (LiveCodeBench {args.lcb_version})")
    else:
        questions = loader(args.examples)
        print(f"Loaded {len(questions)} questions")

    # Detect model if "local"
    model_id = args.model
    if model_id == "local":
        try:
            import requests
            base = BASE_URL.replace("/v1", "") if BASE_URL.endswith("/v1") else BASE_URL
            r = requests.get(f"{base}/api/v1/models", timeout=5)
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
