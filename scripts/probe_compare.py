#!/usr/bin/env python3
"""
Quick comparison probe — tests any endpoint on pre-cached MATH/MMLU/GPQA questions.
No dataset imports; uses pre-built /tmp/*_questions_seed42.json.

Usage:
  python3 probe_compare.py --url http://127.0.0.1:8080 --model ukint-vs/gemma-4-21b-a4b-it-REAP-MLX-4bit --name REAP_MLX --bench math --start 1 --n 5
  python3 probe_compare.py --url http://127.0.0.1:1234 --model google/gemma-4-26b-a4b --name Gemma_LMS --bench math --start 1 --n 5 --reasoning off
"""
import argparse, json, time, re, sys
from urllib.request import urlopen, Request
from pathlib import Path
from datetime import datetime

def extract_boxed(text):
    idx = text.find('\\boxed{')
    if idx == -1: return None
    start = idx + 7; depth = 1; pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == '{': depth += 1
        elif text[pos] == '}': depth -= 1
        pos += 1
    return text[start:pos-1] if depth == 0 else None

def norm_math(s):
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

def api_call(url, model, messages, max_tokens, timeout=900, extra_body=None):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if extra_body:
        payload.update(extra_body)
    start = time.time()
    req = Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            d = json.loads(resp.read())
        elapsed = time.time() - start
        u = d.get('usage', {})
        choice = d['choices'][0]
        # Detect server format
        if 'completion_tokens' in u:  # OpenAI / LM Studio format
            comp = u.get('completion_tokens', 0)
            think = u.get('completion_tokens_details', {}).get('reasoning_tokens', 0) if isinstance(u.get('completion_tokens_details'), dict) else 0
            prompt_tok = u.get('prompt_tokens', 0)
            tok_s = comp / elapsed if elapsed else 0
            return {
                'content': choice['message'].get('content', ''),
                'finish': choice.get('finish_reason', '?'),
                'elapsed': elapsed,
                'prompt_tok': prompt_tok,
                'comp_tok': comp,
                'think_tok': think,
                'vis_tok': comp - think,
                'tok_s': tok_s,
                'peak_mem_gb': None,
            }
        else:  # mlx_vlm format
            comp = u.get('output_tokens', 0)
            prompt_tok = u.get('input_tokens', 0)
            gen_tps = u.get('generation_tps', 0)
            peak = u.get('peak_memory', None)
            return {
                'content': choice['message'].get('content', ''),
                'finish': choice.get('finish_reason', '?'),
                'elapsed': elapsed,
                'prompt_tok': prompt_tok,
                'comp_tok': comp,
                'think_tok': 0,  # not reported by mlx_vlm
                'vis_tok': comp,
                'tok_s': gen_tps if gen_tps else (comp / elapsed if elapsed else 0),
                'peak_mem_gb': peak,
            }
    except Exception as e:
        return {'error': str(e), 'elapsed': time.time() - start,
                'content': '', 'finish': 'error', 'prompt_tok': 0, 'comp_tok': 0,
                'think_tok': 0, 'vis_tok': 0, 'tok_s': 0, 'peak_mem_gb': None}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--bench", required=True, choices=['math', 'mmlu', 'gpqa'])
    p.add_argument("--start", type=int, default=1, help="First question number")
    p.add_argument("--n", type=int, default=5, help="Number of questions")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--reasoning", default=None, choices=['on', 'off'], help="Add reasoning parameter to request (for LM Studio)")
    p.add_argument("--enable-thinking", action='store_true', help="Add enable_thinking=True (for mlx_vlm)")
    p.add_argument("--timeout", type=int, default=900, help="Timeout per question in seconds")
    args = p.parse_args()

    name = args.name or f"{args.model}_{args.bench}"
    out_path = Path(__file__).resolve().parent.parent / "results" / "probes" / f"probe_{name.replace('/','_').replace(' ','_')}_{args.bench}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load pre-cached questions
    qfile = f"/tmp/{args.bench}_questions_seed42.json"
    with open(qfile) as f:
        all_qs = json.load(f)

    # Default max_tokens per benchmark
    if args.max_tokens is None:
        args.max_tokens = {'math': 16384, 'mmlu': 4096, 'gpqa': 16384}[args.bench]

    # Build extra_body
    extra_body = None
    if args.reasoning:
        extra_body = {"reasoning": args.reasoning}
    elif args.enable_thinking:
        extra_body = {"enable_thinking": True}

    # Header
    print(f"{'='*100}")
    print(f"PROBE: {name} | {args.bench.upper()} Q{args.start}-{args.start+args.n-1}")
    print(f"URL: {args.url} | Model: {args.model}")
    print(f"max_tokens: {args.max_tokens} | reasoning: {args.reasoning or 'default'} | enable_thinking: {args.enable_thinking}")
    print(f"Log: {out_path}")
    print(f"{'='*100}")

    # Warmup
    print("Warmup... ", end=''); sys.stdout.flush()
    w = api_call(args.url, args.model,
                 [{"role":"user","content":"What is 2+2? Just the number."}],
                 max_tokens=200, timeout=120, extra_body=extra_body)
    if w.get('error'):
        print(f"FAIL: {w['error']}")
        sys.exit(1)
    print(f"'{w['content'].strip()[:20]}' | p={w['prompt_tok']} c={w['comp_tok']} t={w['think_tok']} v={w['vis_tok']} {w['tok_s']:.1f}t/s | {w['finish']}")

    # Run questions
    results = []
    with open(out_path, 'w') as logf:
        for i in range(args.n):
            q_idx = args.start + i - 1
            if q_idx >= len(all_qs): break
            q = all_qs[q_idx]

            # Build messages based on benchmark
            if args.bench == 'math':
                user_prompt = f"Solve this math problem. Put your final answer in \\boxed{{}}.\n\n{q['problem']}"
            else:
                user_prompt = q['prompt']

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt},
            ]

            r = api_call(args.url, args.model, messages, max_tokens=args.max_tokens,
                        timeout=args.timeout, extra_body=extra_body)

            # Grade
            if args.bench == 'math':
                if r['finish'] == 'length':
                    correct = False; got = None; reason = 'truncated'
                elif r.get('error'):
                    correct = False; got = None; reason = f"error: {r['error'][:50]}"
                else:
                    got = extract_boxed(r['content'])
                    if got is None:
                        m = re.search(r'\\boxed\{([^}]+)\}', r['content'])
                        got = m.group(1) if m else None
                    if got is None:
                        correct = False; reason = 'no_boxed'
                    else:
                        correct = norm_math(str(got)) == norm_math(str(q['expected']))
                        reason = 'correct' if correct else 'wrong'
            else:
                if r.get('error'):
                    correct = False; got = None; reason = f"error: {r['error'][:50]}"
                else:
                    got = extract_letter(r['content'])
                    correct = got == q['expected']
                    reason = 'correct' if correct else f"wrong(exp={q['expected']},got={got})"

            entry = {
                'name': name, 'benchmark': args.bench, 'q_num': q['q_num'],
                'level': q.get('level'), 'subject': q.get('subject'),
                'elapsed_s': round(r['elapsed'], 2),
                'prompt_tokens': r['prompt_tok'],
                'completion_tokens': r['comp_tok'],
                'reasoning_tokens': r['think_tok'],
                'visible_tokens': r['vis_tok'],
                'tok_s': round(r['tok_s'], 2),
                'finish_reason': r['finish'],
                'peak_mem_gb': r['peak_mem_gb'],
                'correct': correct,
                'expected': q['expected'],
                'extracted': str(got) if got else None,
                'reason': reason,
                'max_tokens_sent': args.max_tokens,
                'reasoning_mode': args.reasoning,
                'enable_thinking': args.enable_thinking,
                'error': r.get('error'),
                'response_tail': r['content'][-200:] if r['content'] else '',
            }
            logf.write(json.dumps(entry, ensure_ascii=False) + '\n')
            logf.flush()
            results.append(entry)

            status = 'OK' if correct else 'FAIL'
            trunc = ' TRUNC' if r['finish'] == 'length' else ''
            err = ' ERROR' if r.get('error') else ''
            mem = f" mem={r['peak_mem_gb']:.1f}GB" if r['peak_mem_gb'] else ""
            print(f"  Q{q['q_num']:>3d} {q.get('level', q.get('subject','?'))[:10]:10s} | {r['elapsed']:6.1f}s | p={r['prompt_tok']:>4d} c={r['comp_tok']:>5d} t={r['think_tok']:>5d} v={r['vis_tok']:>5d} {r['tok_s']:>5.1f}t/s{mem} | {status:5s}{trunc}{err}")

    # Summary
    n_correct = sum(1 for r in results if r['correct'])
    n_truncated = sum(1 for r in results if r['finish_reason'] == 'length')
    n_errors = sum(1 for r in results if r.get('error'))
    total_time = sum(r['elapsed_s'] for r in results)
    total_comp = sum(r['completion_tokens'] for r in results)
    total_think = sum(r['reasoning_tokens'] for r in results)

    print(f"\n{'='*100}")
    print(f"SUMMARY: {n_correct}/{len(results)} correct | {n_truncated} truncated | {n_errors} errors")
    print(f"Time: {total_time:.1f}s total, {total_time/len(results):.1f}s avg per question" if results else "No results")
    print(f"Tokens: {total_comp} total ({total_think} thinking, {total_comp-total_think} visible)")
    if results:
        tok_ss = [r['tok_s'] for r in results if r['tok_s'] > 0]
        if tok_ss:
            print(f"Speed: mean={sum(tok_ss)/len(tok_ss):.1f} tok/s, median={sorted(tok_ss)[len(tok_ss)//2]:.1f} tok/s")
    print(f"Log: {out_path}")

if __name__ == '__main__':
    main()
