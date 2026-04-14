#!/usr/bin/env python3
"""
Quick speed/quality probe script for any OpenAI-compatible endpoint.
Tests a small set of benchmark questions and reports tok/s, quality, etc.

Usage:
  python3 probe.py --url http://localhost:8080 --model gemma-4-21b-reap --name "Gemma 4 REAP MLX"
"""
import argparse, json, time, re, sys, random
from urllib.request import urlopen, Request
from datetime import datetime
from pathlib import Path

def api_call(base_url, model, messages, max_tokens=4096, temperature=0, timeout=300):
    url = f"{base_url}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        elapsed = time.time() - start
        choice = data["choices"][0]
        usage = data.get("usage", {})
        comp = usage.get("completion_tokens", 0)
        think = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0) if isinstance(usage.get("completion_tokens_details"), dict) else 0
        prompt_tok = usage.get("prompt_tokens", 0)
        tok_s = comp / elapsed if elapsed > 0 else 0
        return {
            "content": choice["message"].get("content", ""),
            "reasoning_content": choice["message"].get("reasoning_content", ""),
            "finish_reason": choice.get("finish_reason", "?"),
            "elapsed": elapsed,
            "prompt_tokens": prompt_tok,
            "comp_tokens": comp,
            "think_tokens": think,
            "visible_tokens": comp - think,
            "tok_s": tok_s,
            "error": None,
        }
    except Exception as e:
        return {"content": "", "finish_reason": "error", "elapsed": time.time() - start,
                "prompt_tokens": 0, "comp_tokens": 0, "think_tokens": 0,
                "visible_tokens": 0, "tok_s": 0, "error": str(e), "reasoning_content": ""}

def extract_boxed(text):
    idx = text.find('\\boxed{')
    if idx == -1:
        return None
    start = idx + 7; depth = 1; pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == '{': depth += 1
        elif text[pos] == '}': depth -= 1
        pos += 1
    return text[start:pos-1] if depth == 0 else None

def normalize_math(s):
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'\\frac([^{])([^{])', r'\\frac{\1}{\2}', s)
    s = s.replace('\\left', '').replace('\\right', '')
    s = re.sub(r'\\(?:text|mathrm)\{([^}]*)\}', r'\1', s)
    return s.rstrip('.')

def extract_letter(text):
    if not text: return '?'
    for region in [text[-100:], text[:20], text[-30:]]:
        for pat in [r'[Aa]nswer[:\s]+([A-D])', r'\(([A-D])\)', r'\b([A-D])\b']:
            m = re.search(pat, region)
            if m: return m.group(1).upper()
    return '?'

def build_questions(n_per_bench=3):
    """Build a small probe set: 3 MMLU + 3 MATH + 3 DROP + 2 GPQA + 2 HumanEval"""
    from datasets import load_dataset, concatenate_datasets
    questions = []

    # MMLU (short, fast)
    print("Loading MMLU...", file=sys.stderr)
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    random.seed(42); idx = random.sample(range(len(mmlu)), 100)
    for i in range(n_per_bench):
        row = mmlu[idx[i]]
        prompt = f"{row['question']}\n"
        for j, c in enumerate(row['choices']):
            prompt += f"({chr(65+j)}) {c}\n"
        prompt += "Answer with just the letter."
        questions.append({
            'bench': 'mmlu', 'q_num': i+1,
            'expected': "ABCD"[row['answer']],
            'subject': row.get('subject', '?'),
            'messages': [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            'max_tokens': 4096,
        })

    # MATH (3 questions — easy, medium, hard based on sample)
    print("Loading MATH...", file=sys.stderr)
    subs = ['algebra','counting_and_probability','geometry','intermediate_algebra','number_theory','prealgebra','precalculus']
    math_ds = concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s, split="test") for s in subs])
    random.seed(42); midx = random.sample(range(len(math_ds)), 100)
    for i in range(n_per_bench):
        row = math_ds[midx[i]]
        exp = extract_boxed(row['solution'])
        if exp is None:
            m = re.search(r'\\boxed\{([^}]+)\}', row['solution'])
            exp = m.group(1) if m else row['solution'][-30:]
        questions.append({
            'bench': 'math', 'q_num': i+1,
            'expected': exp, 'level': row.get('level', '?'),
            'messages': [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Solve this math problem. Put your final answer in \\boxed{{}}.\n\n{row['problem']}"},
            ],
            'max_tokens': 16384,
        })

    # DROP
    print("Loading DROP...", file=sys.stderr)
    drop = load_dataset("ucinlp/drop", split="validation")
    random.seed(42); didx = random.sample(range(len(drop)), 100)
    for i in range(n_per_bench):
        row = drop[didx[i]]
        questions.append({
            'bench': 'drop', 'q_num': i+1,
            'expected_answers': row['answers_spans']['spans'],
            'messages': [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Read the passage and answer the question. Give ONLY the answer, nothing else.\n\nPassage: {row['passage']}\n\nQuestion: {row['question']}"},
            ],
            'max_tokens': 4096,
        })

    # GPQA (2 questions — harder benchmark)
    print("Loading GPQA...", file=sys.stderr)
    gpqa = load_dataset("fingertap/GPQA-Diamond", split="test")
    random.seed(42); gidx = random.sample(range(len(gpqa)), 100)
    for i in range(2):
        row = gpqa[gidx[i]]
        questions.append({
            'bench': 'gpqa', 'q_num': i+1,
            'expected': row['answer'].strip(),
            'messages': [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{row['question']}\nAnswer with just the letter."},
            ],
            'max_tokens': 16384,
        })

    return questions

def grade(q, response):
    bench = q['bench']
    content = response['content']
    if response['finish_reason'] == 'length':
        return False, None, 'truncated'
    if bench == 'math':
        got = extract_boxed(content)
        if got is None:
            m = re.search(r'\\boxed\{([^}]+)\}', content)
            got = m.group(1) if m else None
        if got is None:
            return False, None, 'no_boxed'
        got_n = normalize_math(str(got))
        exp_n = normalize_math(str(q['expected']))
        return got_n == exp_n, got, 'correct' if got_n == exp_n else 'wrong'
    elif bench in ('mmlu', 'gpqa'):
        got = extract_letter(content)
        return got == q['expected'], got, 'correct' if got == q['expected'] else f'wrong(exp={q["expected"]},got={got})'
    elif bench == 'drop':
        resp = content.lower().strip()
        ok = any(a.lower() in resp for a in q['expected_answers'])
        return ok, content[:50], 'correct' if ok else 'wrong'
    return None, None, 'ungraded'

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="Base URL (e.g., http://localhost:8080)")
    p.add_argument("--model", required=True, help="Model ID to send in requests")
    p.add_argument("--name", default=None, help="Friendly name for this config")
    p.add_argument("--out", default=None, help="Output log file")
    p.add_argument("--n", type=int, default=3, help="Questions per benchmark type")
    args = p.parse_args()

    name = args.name or args.model
    out_path = args.out or str(Path(__file__).resolve().parent.parent / "results" / "probes" / f"probe_{name.replace('/','_').replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    questions = build_questions(args.n)
    print(f"\n{'='*90}")
    print(f"PROBE: {name}")
    print(f"URL: {args.url}  |  Model: {args.model}")
    print(f"Questions: {len(questions)}  |  Log: {out_path}")
    print(f"{'='*90}")

    # Warmup
    print("Warmup...", end=' '); sys.stdout.flush()
    w = api_call(args.url, args.model, [{"role":"user","content":"What is 2+2? Just the number."}], max_tokens=200)
    if w['error']:
        print(f"FAIL: {w['error']}")
        sys.exit(1)
    print(f"'{w['content'].strip()[:30]}' | p={w['prompt_tokens']} c={w['comp_tokens']} (t={w['think_tokens']}+v={w['visible_tokens']}) {w['tok_s']:.1f}t/s | {w['finish_reason']}")

    results = []
    start = time.time()
    with open(out_path, 'w') as logf:
        for i, q in enumerate(questions):
            r = api_call(args.url, args.model, q['messages'], max_tokens=q['max_tokens'])
            correct, extracted, reason = grade(q, r)
            entry = {
                'config_name': name,
                'url': args.url, 'model': args.model,
                'benchmark': q['bench'], 'q_num': q['q_num'],
                'prompt_tokens': r['prompt_tokens'],
                'completion_tokens': r['comp_tokens'],
                'reasoning_tokens': r['think_tokens'],
                'visible_tokens': r['visible_tokens'],
                'tok_s': round(r['tok_s'], 2),
                'elapsed_s': round(r['elapsed'], 2),
                'finish_reason': r['finish_reason'],
                'correct': correct,
                'extracted': str(extracted) if extracted else None,
                'expected': str(q.get('expected', q.get('expected_answers', ''))),
                'reason': reason,
                'error': r['error'],
                'response_preview': r['content'][:300] if r['content'] else '',
            }
            logf.write(json.dumps(entry, ensure_ascii=False) + '\n')
            results.append(entry)

            status = 'OK' if correct else ('FAIL' if correct is False else '??')
            trunc = ' TRUNC' if r['finish_reason'] == 'length' else ''
            print(f"  {i+1:>2}/{len(questions)} {q['bench']:9s} Q{q['q_num']} | {r['elapsed']:6.1f}s | p={r['prompt_tokens']:>4d} c={r['comp_tokens']:>5d} (t={r['think_tokens']:>5d}+v={r['visible_tokens']:>4d}) {r['tok_s']:>5.1f}t/s | {status:4s}{trunc} | {reason}")

    total = time.time() - start
    n_correct = sum(1 for r in results if r['correct'])
    # Per benchmark
    by_bench = {}
    for r in results:
        b = r['benchmark']
        by_bench.setdefault(b, {'correct':0,'total':0,'tok_s':[],'comp':[]})
        if r['correct']: by_bench[b]['correct'] += 1
        by_bench[b]['total'] += 1
        by_bench[b]['tok_s'].append(r['tok_s'])
        by_bench[b]['comp'].append(r['completion_tokens'])

    # Average tok/s (excluding zeros from errors)
    non_zero_toks = [r['tok_s'] for r in results if r['tok_s'] > 0]
    avg_toks = sum(non_zero_toks) / len(non_zero_toks) if non_zero_toks else 0
    median_toks = sorted(non_zero_toks)[len(non_zero_toks)//2] if non_zero_toks else 0

    print(f"\n{'='*90}")
    print(f"TOTALS: {n_correct}/{len(results)} = {n_correct/len(results)*100:.0f}% in {total:.0f}s")
    print(f"Speed: mean={avg_toks:.1f} tok/s, median={median_toks:.1f} tok/s")
    print(f"Per benchmark:")
    for b, d in by_bench.items():
        avg = sum(d['tok_s'])/len(d['tok_s']) if d['tok_s'] else 0
        avg_comp = sum(d['comp'])/len(d['comp']) if d['comp'] else 0
        print(f"  {b:10s}: {d['correct']}/{d['total']} | avg tok/s: {avg:.1f} | avg comp: {avg_comp:.0f}")
    print(f"Log: {out_path}")

if __name__ == '__main__':
    main()
