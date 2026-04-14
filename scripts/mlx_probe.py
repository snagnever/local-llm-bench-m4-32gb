#!/usr/bin/env python3
"""
Direct Python probe using mlx_vlm library (no server).
Loads model once, runs N questions, measures precise timings.
Tests both thinking ON and OFF modes.

Usage:
  python3 mlx_probe.py --model ukint-vs/gemma-4-21b-a4b-it-REAP-MLX-4bit --thinking on --bench math --n 5
"""
import argparse, json, time, re, sys, gc
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--thinking", choices=['on', 'off'], default='off')
    p.add_argument("--bench", required=True, choices=['math', 'mmlu', 'gpqa', 'drop'])
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--kv-bits", type=int, default=None, help="KV cache quantization bits")
    p.add_argument("--kv-quant-scheme", choices=['uniform', 'turboquant'], default=None)
    p.add_argument("--thinking-budget", type=int, default=None, help="Max thinking tokens before forced response")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    if args.max_tokens is None:
        args.max_tokens = {'math': 16384, 'mmlu': 4096, 'gpqa': 16384, 'drop': 4096}[args.bench]

    name = args.name or f"{args.model.split('/')[-1]}_{args.thinking}"
    out_path = Path(__file__).resolve().parent.parent / "results" / "probes" / f"mlx_{name}_{args.bench}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{'='*100}")
    print(f"MLX-VLM PROBE: {name}")
    print(f"Model: {args.model}")
    print(f"Benchmark: {args.bench} Q{args.start}-{args.start+args.n-1}")
    print(f"Thinking: {args.thinking} | max_tokens: {args.max_tokens}")
    if args.kv_bits:
        print(f"KV quant: {args.kv_bits} bits, scheme: {args.kv_quant_scheme}")
    print(f"Log: {out_path}")
    print(f"{'='*100}")

    # Load questions
    qfile = f"/tmp/{args.bench}_questions_seed42.json"
    with open(qfile) as f:
        all_qs = json.load(f)

    # Import and load model
    print(f"\nImporting mlx_vlm... ", end=''); sys.stdout.flush()
    t0 = time.time()
    from mlx_vlm import load, generate
    import_time = time.time() - t0
    print(f"{import_time:.1f}s")

    print(f"Loading model... ", end=''); sys.stdout.flush()
    t0 = time.time()
    model, processor = load(args.model)
    load_time = time.time() - t0
    print(f"{load_time:.1f}s")

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    # Warmup
    print(f"\nWarmup (2+2)... ", end=''); sys.stdout.flush()
    t0 = time.time()
    messages = [{"role": "user", "content": "What is 2+2? Just the number."}]
    enable_thinking = (args.thinking == 'on')
    try:
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=enable_thinking
        )
    except TypeError:
        # Some tokenizers don't accept enable_thinking
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    gen_kwargs = {"max_tokens": 200, "temperature": 0.0}
    if args.kv_bits:
        gen_kwargs["kv_bits"] = args.kv_bits
    if args.kv_quant_scheme:
        gen_kwargs["kv_quant_scheme"] = args.kv_quant_scheme

    out = generate(model, processor, prompt, **gen_kwargs)
    warm_time = time.time() - t0
    print(f"{warm_time:.1f}s")
    # Extract text - out is a GenerationResult object
    warm_text = out.text if hasattr(out, 'text') else str(out)
    print(f"  Response: {warm_text[:100]!r}")
    if hasattr(out, 'input_tokens'):
        print(f"  Input tokens: {out.input_tokens}, Output tokens: {out.output_tokens}")
        print(f"  Prompt tps: {getattr(out, 'prompt_tps', 0):.1f}, Gen tps: {getattr(out, 'generation_tps', 0):.1f}")
    if hasattr(out, 'peak_memory'):
        print(f"  Peak memory: {out.peak_memory:.1f} GB")

    # Run questions
    results = []
    total_start = time.time()
    with open(out_path, 'w') as logf:
        for i in range(args.n):
            q_idx = args.start + i - 1
            if q_idx >= len(all_qs): break
            q = all_qs[q_idx]

            if args.bench == 'math':
                user_prompt = f"Solve this math problem. Put your final answer in \\boxed{{}}.\n\n{q['problem']}"
            elif args.bench == 'drop':
                user_prompt = f"Read the passage and answer the question. Give ONLY the answer, nothing else.\n\nPassage: {q['passage']}\n\nQuestion: {q['question']}"
            else:
                user_prompt = q['prompt']

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

            gen_kwargs = {"max_tokens": args.max_tokens, "temperature": 0.0}
            if args.kv_bits:
                gen_kwargs["kv_bits"] = args.kv_bits
            if args.kv_quant_scheme:
                gen_kwargs["kv_quant_scheme"] = args.kv_quant_scheme

            t0 = time.time()
            try:
                out = generate(model, processor, prompt, **gen_kwargs)
                elapsed = time.time() - t0
                content = out.text if hasattr(out, 'text') else str(out)
                input_tokens = getattr(out, 'prompt_tokens', 0)
                output_tokens = getattr(out, 'generation_tokens', 0)
                prompt_tps = getattr(out, 'prompt_tps', 0)
                gen_tps = getattr(out, 'generation_tps', 0)
                peak_mem = getattr(out, 'peak_memory', 0)
                error = None
            except Exception as e:
                elapsed = time.time() - t0
                content = ''
                input_tokens = output_tokens = 0
                prompt_tps = gen_tps = peak_mem = 0
                error = str(e)

            # Grade
            if args.bench == 'math':
                if error:
                    correct, got, reason = False, None, f"error: {error[:50]}"
                else:
                    got = extract_boxed(content)
                    if got is None:
                        m = re.search(r'\\boxed\{([^}]+)\}', content)
                        got = m.group(1) if m else None
                    if got is None:
                        correct, reason = False, 'no_boxed'
                    else:
                        correct = norm_math(str(got)) == norm_math(str(q['expected']))
                        reason = 'correct' if correct else 'wrong'
            elif args.bench == 'drop':
                if error:
                    correct, got, reason = False, None, f"error: {error[:50]}"
                else:
                    resp = content.lower().strip()
                    answers = q['expected_answers']
                    correct = any(a.lower() in resp for a in answers)
                    got = content[:80]
                    reason = 'correct' if correct else 'wrong'
            else:
                if error:
                    correct, got, reason = False, None, f"error: {error[:50]}"
                else:
                    got = extract_letter(content)
                    correct = got == q['expected']
                    reason = 'correct' if correct else f"wrong(exp={q['expected']},got={got})"

            expected_display = q.get('expected', q.get('expected_answers', ''))
            entry = {
                'name': name, 'benchmark': args.bench, 'q_num': q['q_num'],
                'level': q.get('level'), 'subject': q.get('subject'),
                'elapsed_s': round(elapsed, 2),
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'prompt_tps': round(prompt_tps, 2),
                'generation_tps': round(gen_tps, 2),
                'peak_memory_gb': round(peak_mem, 2),
                'correct': correct,
                'expected': str(expected_display),
                'extracted': str(got) if got else None,
                'reason': reason,
                'max_tokens_sent': args.max_tokens,
                'thinking': args.thinking,
                'kv_bits': args.kv_bits,
                'kv_quant_scheme': args.kv_quant_scheme,
                'error': error,
                'response_tail': content[-300:] if content else '',
                'response_full': content,
            }
            logf.write(json.dumps(entry, ensure_ascii=False) + '\n')
            logf.flush()
            results.append(entry)

            status = 'OK' if correct else 'FAIL'
            print(f"  Q{q['q_num']:>3d} {q.get('level', q.get('subject','?'))[:12]:12s} | {elapsed:6.1f}s | in={input_tokens:>4d} out={output_tokens:>5d} | prompt={prompt_tps:.0f} gen={gen_tps:.1f} t/s | mem={peak_mem:.1f}GB | {status} | {reason}")
            # Clean up between questions
            gc.collect()

    total = time.time() - total_start
    n_correct = sum(1 for r in results if r['correct'])
    print(f"\n{'='*100}")
    print(f"SUMMARY: {n_correct}/{len(results)} correct in {total:.0f}s")
    if results:
        avg_gen = sum(r['generation_tps'] for r in results if r['generation_tps']) / max(1, sum(1 for r in results if r['generation_tps']))
        avg_out = sum(r['output_tokens'] for r in results) / len(results)
        print(f"Avg gen t/s: {avg_gen:.1f} | Avg output: {avg_out:.0f} tokens")
    print(f"Log: {out_path}")

if __name__ == '__main__':
    main()
