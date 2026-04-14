#!/usr/bin/env python3
"""
Compare speed with/without TurboQuant KV cache compression on deadbydawn REAP.
Tests the same 3 MATH questions under both settings.
"""
import json, time, re, sys, argparse, gc
from pathlib import Path

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
    return s.rstrip('.')

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--turboquant", action='store_true', help="Enable TurboQuant KV cache")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=8192)
    args = p.parse_args()

    model_id = "deadbydawn101/gemma-4-21b-REAP-Tool-Calling-mlx-4bit"
    label = "deadbydawn_TurboQuant_ON" if args.turboquant else "deadbydawn_baseline"

    out_path = Path(__file__).resolve().parent.parent / "results" / "probes" / f"tq_{label}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{'='*90}")
    print(f"TURBOQUANT SPEED TEST: {label}")
    print(f"Model: {model_id}")
    print(f"Questions: MATH Q{args.start}-{args.start+args.n-1}, max_tokens={args.max_tokens}")
    print(f"{'='*90}")

    # Apply TurboQuant BEFORE loading anything else
    if args.turboquant:
        print("Patching mlx-lm with TurboQuant KV cache...")
        from turboquant_mlx.mlx_kvcache import TurboQuantKVCache
        import mlx_lm.models.cache as cache_module
        # Save original
        orig_make_cache = cache_module.make_prompt_cache
        def tq_make_cache(model, **kw):
            return [TurboQuantKVCache() for _ in range(len(model.layers))]
        cache_module.make_prompt_cache = tq_make_cache
        print("TurboQuant KV cache patched")

    print("Loading model...", end=' '); sys.stdout.flush()
    t0 = time.time()
    from mlx_vlm import load, generate
    model, processor = load(model_id)
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    load_time = time.time() - t0
    print(f"{load_time:.1f}s")

    # Load questions
    with open('/tmp/math_questions_seed42.json') as f:
        all_qs = json.load(f)

    # Warmup
    print("Warmup... ", end=''); sys.stdout.flush()
    t0 = time.time()
    messages = [{"role": "user", "content": "What is 2+2? Just the number."}]
    try:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=True)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    out = generate(model, processor, prompt, max_tokens=200, temperature=0.0)
    warm_time = time.time() - t0
    content = out.text if hasattr(out, 'text') else str(out)
    print(f"{warm_time:.1f}s | '{content[:30]!r}'")
    if hasattr(out, 'generation_tps'):
        print(f"  gen tps: {out.generation_tps:.1f} | peak mem: {getattr(out, 'peak_memory', 0):.1f}GB")

    # Run questions
    results = []
    total_start = time.time()
    with open(out_path, 'w') as logf:
        for i in range(args.n):
            q_idx = args.start + i - 1
            if q_idx >= len(all_qs): break
            q = all_qs[q_idx]

            user_prompt = f"Solve this math problem. Put your final answer in \\boxed{{}}.\n\n{q['problem']}"
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt},
            ]
            try:
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=True)
            except TypeError:
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            t0 = time.time()
            try:
                out = generate(model, processor, prompt, max_tokens=args.max_tokens, temperature=0.0)
                elapsed = time.time() - t0
                content = out.text if hasattr(out, 'text') else str(out)
                input_tok = getattr(out, 'prompt_tokens', 0)
                output_tok = getattr(out, 'generation_tokens', 0)
                prompt_tps = getattr(out, 'prompt_tps', 0)
                gen_tps = getattr(out, 'generation_tps', 0)
                peak_mem = getattr(out, 'peak_memory', 0)
                error = None
            except Exception as e:
                elapsed = time.time() - t0
                content = ''
                input_tok = output_tok = 0
                prompt_tps = gen_tps = peak_mem = 0
                error = str(e)

            # Grade
            got = extract_boxed(content)
            if got is None:
                m = re.search(r'\\boxed\{([^}]+)\}', content)
                got = m.group(1) if m else None
            correct = (got and norm_math(str(got)) == norm_math(str(q['expected']))) if not error else False

            entry = {
                'label': label,
                'turboquant': args.turboquant,
                'q_num': q['q_num'], 'level': q['level'],
                'elapsed_s': round(elapsed, 2),
                'input_tokens': input_tok, 'output_tokens': output_tok,
                'prompt_tps': round(prompt_tps, 2),
                'generation_tps': round(gen_tps, 2),
                'peak_memory_gb': round(peak_mem, 2),
                'correct': correct,
                'expected': q['expected'], 'extracted': str(got) if got else None,
                'error': error,
                'response_tail': content[-200:],
            }
            logf.write(json.dumps(entry, ensure_ascii=False) + '\n')
            results.append(entry)

            status = 'OK' if correct else 'FAIL'
            print(f"  Q{q['q_num']:>3d} {q['level']:8s} | {elapsed:6.1f}s | in={input_tok:>4d} out={output_tok:>5d} | gen={gen_tps:5.1f} t/s | mem={peak_mem:4.1f}GB | {status}")
            gc.collect()

    total = time.time() - total_start
    n_correct = sum(1 for r in results if r['correct'])
    print(f"\nSUMMARY: {n_correct}/{len(results)} correct in {total:.1f}s")
    if results:
        gen_tps_list = [r['generation_tps'] for r in results if r['generation_tps']]
        if gen_tps_list:
            print(f"Avg gen tok/s: {sum(gen_tps_list)/len(gen_tps_list):.1f}")
        print(f"Peak memory: {max(r['peak_memory_gb'] for r in results):.1f} GB")
    print(f"Log: {out_path}")

if __name__ == '__main__':
    main()
