#!/usr/bin/env python3
"""
⚠️ DEPRECATED — use bench2.py instead ⚠️

This is the original benchmark harness. It has known bugs:
  1. run_math() had max_tokens=4096 hardcoded, truncating thinking models
  2. Broken \boxed{} regex that couldn't handle nested braces
  3. No per-question detail logging

All three issues were fixed in bench2.py. See results/AUDIT_REPORT.md
for details. This file is kept only for historical reference.

Original docstring: Benchmark harness with real-time per-question output.
Shows: question #, time, tokens, tok/s, temperature, answer preview.

Usage: python3 bench.py <benchmark> [--examples N] [--model MODEL_ID]
Benchmarks: mmlu, math, humaneval, gpqa, drop
"""
import time, json, sys, os, subprocess
from urllib.request import urlopen, Request

# Config
BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = os.environ.get("LMSTUDIO_MODEL", "local")
TEMPERATURE = 0
MAX_TOKENS = 32768

def api_call(messages, max_tokens=MAX_TOKENS):
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }).encode()
    req = Request(f"{BASE_URL}/chat/completions", data=payload,
                  headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urlopen(req, timeout=1800) as resp:
            data = json.loads(resp.read())
        elapsed = time.time() - start
        choice = data["choices"][0]["message"]
        usage = data.get("usage", {})
        comp = usage.get("completion_tokens", 0)
        think = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        tok_s = comp / elapsed if elapsed > 0 else 0
        return {
            "content": choice.get("content", ""),
            "elapsed": elapsed,
            "comp_tokens": comp,
            "think_tokens": think,
            "tok_s": tok_s,
            "error": None,
        }
    except Exception as e:
        return {"content": "", "elapsed": time.time() - start, "comp_tokens": 0, "think_tokens": 0, "tok_s": 0, "error": str(e)}

def get_temp():
    try:
        # macmon pipe outputs one JSON line per second; wait for 2 lines to ensure we get data
        proc = subprocess.Popen(["macmon", "pipe"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        import select
        lines = []
        for _ in range(3):  # read up to 3 lines (3 seconds)
            line = proc.stdout.readline().decode().strip()
            if line:
                lines.append(line)
                break
        proc.kill()
        proc.wait()
        if lines:
            d = json.loads(lines[-1])
            gpu = d['temp']['gpu_temp_avg']
            # Flag throttling
            flag = " THROTTLE!" if gpu >= 95 else (" HOT" if gpu >= 85 else "")
            return f"{gpu:.0f}°C{flag}"
    except:
        pass
    return "?"

def extract_answer_letter(text):
    """Extract A/B/C/D from model response. Checks end first (after thinking), then start."""
    if not text:
        return "?"
    text = text.strip()
    import re
    # Check last 100 chars first (where the answer usually is after thinking)
    tail = text[-100:]
    # Look for patterns like "Answer: A", "(A)", "The answer is A", just "A" on its own line
    for pattern in [r'[Aa]nswer[:\s]+([A-D])', r'\(([A-D])\)', r'\b([A-D])\b']:
        match = re.search(pattern, tail)
        if match:
            return match.group(1).upper()
    # Check first 20 chars (model might answer directly)
    head = text[:20]
    for pattern in [r'^([A-D])\b', r'^\(([A-D])\)', r'^[Aa]nswer[:\s]+([A-D])']:
        match = re.search(pattern, head)
        if match:
            return match.group(1).upper()
    # Last resort: find any standalone letter in the last 30 chars
    match = re.search(r'\b([A-D])\b', text[-30:])
    if match:
        return match.group(1).upper()
    return "?"

def load_mmlu(n):
    """Load MMLU questions from the dataset."""
    try:
        from simple_evals.mmlu_eval import MMLUEval
        evaluator = MMLUEval(num_examples=n)
        return evaluator
    except Exception as e:
        print(f"Error loading MMLU: {e}")
        return None

def run_mmlu(n):
    print(f"Loading MMLU dataset ({n} questions)...")
    sys.stdout.flush()

    # Use datasets library directly
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")

    # Sample n questions
    import random
    random.seed(42)
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    correct = 0
    total = 0
    results = []

    for i, idx in enumerate(indices):
        row = ds[idx]
        q = row["question"]
        choices = row["choices"]
        answer_idx = row["answer"]  # 0-3
        answer_letter = "ABCD"[answer_idx]
        subject = row.get("subject", "?")

        prompt = f"{q}\n"
        for j, c in enumerate(choices):
            prompt += f"({chr(65+j)}) {c}\n"
        prompt += "Answer with just the letter."

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        r = api_call(messages, max_tokens=MAX_TOKENS)
        total += 1

        got = extract_answer_letter(r["content"])

        is_correct = got == answer_letter
        if is_correct:
            correct += 1

        temp = get_temp()
        status = "OK" if is_correct else "WRONG"
        preview = r["content"].replace("\n", " ")[:30] if r["content"] else str(r.get("error","?"))[:30]

        print(f"  {i+1:>3}/{n} | {r['elapsed']:5.1f}s | {r['comp_tokens']:>4}tok {r['tok_s']:>5.1f}t/s | {temp:>5} | {status:5} | {subject[:15]:15} | {preview}")
        sys.stdout.flush()

        results.append({"idx": idx, "correct": is_correct, "elapsed": r["elapsed"], "tokens": r["comp_tokens"]})

    score = correct / total if total > 0 else 0
    return score, results

def extract_boxed(text):
    """Extract content from \\boxed{}, handling nested braces."""
    idx = text.find('\\boxed{')
    if idx == -1:
        return None
    start = idx + 7
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == '{': depth += 1
        elif text[pos] == '}': depth -= 1
        pos += 1
    return text[start:pos-1] if depth == 0 else None

def run_math(n):
    from datasets import load_dataset
    # Load all math subjects
    from datasets import concatenate_datasets
    subjects = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    all_ds = [load_dataset("EleutherAI/hendrycks_math", s, split="test") for s in subjects]
    ds = concatenate_datasets(all_ds)

    import random, re
    random.seed(42)
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    correct = 0
    total = 0

    for i, idx in enumerate(indices):
        row = ds[idx]
        problem = row["problem"]
        solution = row["solution"]

        match = extract_boxed(solution)
        if match is None:
            match_obj = re.search(r'\\boxed\{([^}]+)\}', solution)
            match = match_obj.group(1) if match_obj else None
        expected = match if match else solution[-20:]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Solve this math problem. Put your final answer in \\boxed{{}}.\n\n{problem}"},
        ]

        r = api_call(messages, max_tokens=MAX_TOKENS)
        total += 1

        # Check if answer contains the expected value (handle nested braces)
        got = extract_boxed(r["content"]) if r["content"] else None
        if got is None:
            got_match = re.search(r'\\boxed\{([^}]+)\}', r["content"]) if r["content"] else None
            got = got_match.group(1) if got_match else "?"
        # Normalize whitespace for comparison
        got_norm = re.sub(r'\s+', '', str(got))
        exp_norm = re.sub(r'\s+', '', str(expected))
        is_correct = got_norm == exp_norm
        if is_correct:
            correct += 1

        temp = get_temp()
        status = "OK" if is_correct else "WRONG"
        level = row.get("level", "?")

        print(f"  {i+1:>3}/{n} | {r['elapsed']:5.1f}s | {r['comp_tokens']:>4}tok {r['tok_s']:>5.1f}t/s | {temp:>5} | {status:5} | {level}")
        sys.stdout.flush()

    return correct / total if total > 0 else 0, []

def run_gpqa(n):
    from datasets import load_dataset
    ds = load_dataset("fingertap/GPQA-Diamond", split="test")

    import random
    random.seed(42)
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    correct = 0
    total = 0

    for i, idx in enumerate(indices):
        row = ds[idx]
        question = row["question"]  # already includes choices
        answer_letter = row["answer"].strip()  # "A", "B", "C", or "D"

        prompt = f"{question}\nAnswer with just the letter."

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        r = api_call(messages, max_tokens=MAX_TOKENS)
        total += 1

        got = extract_answer_letter(r["content"])

        is_correct = got == answer_letter
        if is_correct:
            correct += 1

        temp = get_temp()
        status = "OK" if is_correct else "WRONG"

        print(f"  {i+1:>3}/{n} | {r['elapsed']:5.1f}s | {r['comp_tokens']:>4}tok {r['tok_s']:>5.1f}t/s | {temp:>12} | {status:5} | exp={answer_letter} got={got}")
        sys.stdout.flush()

    return correct / total if total > 0 else 0, []

def run_drop(n):
    from datasets import load_dataset
    ds = load_dataset("ucinlp/drop", split="validation")

    import random
    random.seed(42)
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    correct = 0
    total = 0

    for i, idx in enumerate(indices):
        row = ds[idx]
        passage = row["passage"]
        question = row["question"]
        answers = row["answers_spans"]["spans"]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Read the passage and answer the question. Give ONLY the answer, nothing else.\n\nPassage: {passage}\n\nQuestion: {question}"},
        ]

        r = api_call(messages, max_tokens=MAX_TOKENS)
        total += 1

        # Simple check: any expected answer appears in response
        response_lower = r["content"].lower().strip() if r["content"] else ""
        is_correct = any(a.lower() in response_lower for a in answers)
        if is_correct:
            correct += 1

        temp = get_temp()
        status = "OK" if is_correct else "WRONG"
        expected_preview = answers[0][:20] if answers else "?"
        got_preview = r["content"].replace("\n", " ")[:20] if r["content"] else "error"

        print(f"  {i+1:>3}/{n} | {r['elapsed']:5.1f}s | {r['comp_tokens']:>4}tok {r['tok_s']:>5.1f}t/s | {temp:>5} | {status:5} | exp={expected_preview} got={got_preview}")
        sys.stdout.flush()

    return correct / total if total > 0 else 0, []

def run_humaneval(n):
    """HumanEval - generate code and execute tests to verify correctness."""
    from datasets import load_dataset
    import tempfile, textwrap

    ds = load_dataset("openai/openai_humaneval", split="test")

    import random
    random.seed(42)
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    correct = 0
    total = 0

    for i, idx in enumerate(indices):
        row = ds[idx]
        prompt = row["prompt"]
        test_code = row["test"]
        entry_point = row["entry_point"]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Complete the following Python function. Output ONLY the complete function, no explanation.\n\n{prompt}"},
        ]

        r = api_call(messages, max_tokens=MAX_TOKENS)
        total += 1

        # Extract code from response
        code = r["content"] if r["content"] else ""
        # Strip markdown fences if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        # Try to execute the code + tests
        is_correct = False
        try:
            # Combine generated code with test cases
            full_code = code + "\n" + test_code + f"\ncheck({entry_point})"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                f.flush()
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True, text=True, timeout=10
                )
                is_correct = result.returncode == 0
                os.unlink(f.name)
        except Exception:
            is_correct = False

        if is_correct:
            correct += 1

        temp = get_temp()
        status = "PASS" if is_correct else "FAIL"

        print(f"  {i+1:>3}/{n} | {r['elapsed']:5.1f}s | {r['comp_tokens']:>4}tok {r['tok_s']:>5.1f}t/s | {temp:>12} | {status:4} | {entry_point}")
        sys.stdout.flush()

    return correct / total if total > 0 else 0, []


# ---- Main ----
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("benchmark", choices=["mmlu", "math", "humaneval", "gpqa", "drop", "all"])
parser.add_argument("--examples", type=int, default=20)
parser.add_argument("--model", type=str, default=None)
args = parser.parse_args()

if args.model:
    MODEL = args.model
    os.environ["LMSTUDIO_MODEL"] = args.model

benchmarks = ["mmlu", "math", "humaneval", "gpqa", "drop"] if args.benchmark == "all" else [args.benchmark]

print(f"\n{'='*80}")
print(f"Benchmark: {', '.join(benchmarks)} | Model: {MODEL} | Examples: {args.examples}")
print(f"Params: temp={TEMPERATURE}, max_tokens={MAX_TOKENS}")
print(f"{'='*80}")

all_scores = {}

for bench in benchmarks:
    print(f"\n--- {bench.upper()} ({args.examples}q) --- {time.strftime('%H:%M:%S')}")

    start = time.time()
    if bench == "mmlu":
        score, _ = run_mmlu(args.examples)
    elif bench == "math":
        score, _ = run_math(args.examples)
    elif bench == "humaneval":
        score, _ = run_humaneval(args.examples)
    elif bench == "gpqa":
        score, _ = run_gpqa(args.examples)
    elif bench == "drop":
        score, _ = run_drop(args.examples)
    elapsed = time.time() - start

    all_scores[bench] = score
    print(f"  >>> {bench.upper()}: {score*100:.0f}% in {elapsed:.0f}s")

    # Temp check between benchmarks
    temp = get_temp()
    print(f"  >>> Temperature: {temp}")

print(f"\n{'='*80}")
print(f"SUMMARY: {MODEL}")
for bench, score in all_scores.items():
    print(f"  {bench:12s}: {score*100:.0f}%")
print(f"{'='*80}\n")

# Save results to JSON
results_dir = "results/quality"
os.makedirs(results_dir, exist_ok=True)
safe_model = MODEL.replace("/", "_").replace(" ", "_")
results_file = os.path.join(results_dir, f"{safe_model}_{args.examples}q_{time.strftime('%Y%m%d_%H%M%S')}.json")
with open(results_file, "w") as f:
    json.dump({"model": MODEL, "examples": args.examples, "scores": {k: v for k, v in all_scores.items()}, "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
print(f"Results saved to: {results_file}")
