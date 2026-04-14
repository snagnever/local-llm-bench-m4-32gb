#!/usr/bin/env python3
"""
Long-context performance test for Gemma 4 variants on Apple Silicon.
Tests prefill speed, generation speed, peak memory at various context sizes.

Uses a "needle in a haystack" approach: embeds a unique fact in long filler text
and asks the model to retrieve it. Tests both context handling and retrieval.

Usage:
  python3 long_context_test.py --model deadbydawn101/gemma-4-21b-REAP-Tool-Calling-mlx-4bit \
    --sizes 4096,8192,16384,32768,65536 --turboquant
"""
import argparse, json, time, gc, sys
from pathlib import Path
from datetime import datetime

# Pre-built filler text — a chunk that's about 200 tokens
FILLER_CHUNK = """The history of computing technology spans many centuries of human ingenuity. From the abacus
in ancient Mesopotamia to the modern silicon transistor, each generation has built upon the discoveries
of the previous. Charles Babbage's Difference Engine in the 1820s was an early mechanical computer,
though it was never fully constructed in his lifetime. Ada Lovelace, working with Babbage, is often
credited as the first computer programmer for her notes on the Analytical Engine. The 20th century
saw rapid advances: ENIAC in 1945, the transistor in 1947, integrated circuits in the late 1950s,
and microprocessors in the early 1970s. Each leap reduced size and increased performance by orders of
magnitude. The personal computer revolution of the 1980s brought computing into homes, while the internet
of the 1990s connected the world. Today's mobile devices have more processing power than entire data
centers from just a few decades ago. Machine learning and artificial intelligence represent the latest
frontier, with neural networks trained on vast datasets capable of remarkable feats."""

NEEDLE = "The secret access code for the underground vault is XENON-7592-FALCON."

def build_context(target_tokens, tokenizer):
    """Build a context of approximately target_tokens by repeating filler with the needle in the middle."""
    # Estimate tokens per chunk
    chunk_tokens = len(tokenizer.encode(FILLER_CHUNK))
    needed_chunks = max(1, target_tokens // chunk_tokens)

    # Insert needle in the middle
    half = needed_chunks // 2
    parts = []
    for i in range(needed_chunks):
        parts.append(FILLER_CHUNK)
        if i == half:
            parts.append(f"\n\nIMPORTANT NOTE: {NEEDLE}\n\n")

    context = "\n\n".join(parts)
    actual_tokens = len(tokenizer.encode(context))
    return context, actual_tokens

def build_prompt(context, tokenizer, enable_thinking=False):
    """Build the user prompt asking to retrieve the needle."""
    user_msg = f"""Read the following long document carefully:

{context}

QUESTION: What is the secret access code mentioned in the document? Answer with just the code (no explanation)."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_msg},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=enable_thinking
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    return prompt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--sizes", default="4096,8192,16384,32768",
                   help="Comma-separated context sizes to test")
    p.add_argument("--turboquant", action='store_true',
                   help="Enable TurboQuant KV cache compression")
    p.add_argument("--max-tokens", type=int, default=200,
                   help="Max output tokens (small for needle-in-haystack)")
    p.add_argument("--thinking", choices=['on', 'off'], default='off',
                   help="Enable Gemma 4 thinking mode")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    sizes = [int(s) for s in args.sizes.split(',')]
    name = args.name or f"{args.model.split('/')[-1]}_{args.thinking}{'_TQ' if args.turboquant else ''}"
    out_path = Path(__file__).resolve().parent.parent / "results" / "probes" / f"longctx_{name.replace('/','_').replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{'='*100}")
    print(f"LONG CONTEXT TEST: {name}")
    print(f"Model: {args.model}")
    print(f"Sizes: {sizes}")
    print(f"Thinking: {args.thinking} | TurboQuant: {args.turboquant} | Max output: {args.max_tokens}")
    print(f"Log: {out_path}")
    print(f"{'='*100}")

    print("\nLoading model... ", end=''); sys.stdout.flush()
    t0 = time.time()
    from mlx_vlm import load, generate
    model, processor = load(args.model)
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    print(f"{time.time()-t0:.1f}s")

    # Warmup
    print("Warmup... ", end=''); sys.stdout.flush()
    t0 = time.time()
    out = generate(model, processor, "Hello", max_tokens=10, temperature=0.0)
    print(f"{time.time()-t0:.1f}s, peak={getattr(out, 'peak_memory', 0):.1f}GB")

    # Baseline reference token count
    needle_tokens = len(tokenizer.encode(NEEDLE))
    chunk_tokens = len(tokenizer.encode(FILLER_CHUNK))
    print(f"Needle: {needle_tokens} tokens, chunk: {chunk_tokens} tokens")

    enable_thinking = (args.thinking == 'on')
    results = []

    with open(out_path, 'w') as logf:
        for size in sizes:
            print(f"\n--- Context size: {size} tokens ---")
            try:
                context, actual_tokens = build_context(size, tokenizer)
                prompt = build_prompt(context, tokenizer, enable_thinking=enable_thinking)
                actual_prompt_tokens = len(tokenizer.encode(prompt))
                print(f"  Built prompt: {actual_prompt_tokens} tokens (target {size})")

                # Build generate kwargs
                gen_kwargs = {"max_tokens": args.max_tokens, "temperature": 0.0}
                if args.turboquant:
                    gen_kwargs["kv_bits"] = 4
                    gen_kwargs["kv_quant_scheme"] = "turboquant"

                t0 = time.time()
                out = generate(model, processor, prompt, **gen_kwargs)
                elapsed = time.time() - t0

                input_tok = getattr(out, 'prompt_tokens', actual_prompt_tokens)
                output_tok = getattr(out, 'generation_tokens', 0)
                prompt_tps = getattr(out, 'prompt_tps', 0)
                gen_tps = getattr(out, 'generation_tps', 0)
                peak_mem = getattr(out, 'peak_memory', 0)
                content = out.text if hasattr(out, 'text') else str(out)

                # Check if the model found the needle
                found_needle = "XENON-7592-FALCON" in content
                # Compute prefill time roughly: total - generation_time
                gen_time = output_tok / gen_tps if gen_tps else 0
                prefill_time = elapsed - gen_time

                entry = {
                    'name': name,
                    'context_size_target': size,
                    'context_size_actual': actual_prompt_tokens,
                    'input_tokens': input_tok,
                    'output_tokens': output_tok,
                    'elapsed_s': round(elapsed, 2),
                    'prefill_time_s': round(prefill_time, 2),
                    'gen_time_s': round(gen_time, 2),
                    'prompt_tps': round(prompt_tps, 1),
                    'generation_tps': round(gen_tps, 1),
                    'peak_memory_gb': round(peak_mem, 2),
                    'found_needle': found_needle,
                    'turboquant': args.turboquant,
                    'thinking': args.thinking,
                    'response_preview': content[:200],
                    'response_tail': content[-200:],
                }
                logf.write(json.dumps(entry, ensure_ascii=False) + '\n')
                logf.flush()
                results.append(entry)

                status = "✓" if found_needle else "✗"
                print(f"  {status} | total {elapsed:6.1f}s | prefill {prefill_time:5.1f}s ({prompt_tps:.0f}t/s) | gen {gen_time:5.1f}s ({gen_tps:.1f}t/s) | peak {peak_mem:.2f}GB | out {output_tok}tok")
                if not found_needle:
                    print(f"    Response preview: {content[:150]!r}")

                gc.collect()

            except Exception as e:
                print(f"  ERROR: {e}")
                entry = {
                    'name': name, 'context_size_target': size,
                    'error': str(e), 'turboquant': args.turboquant,
                }
                logf.write(json.dumps(entry, ensure_ascii=False) + '\n')
                results.append(entry)
                gc.collect()
                # If we OOM, no point trying larger sizes
                if 'memory' in str(e).lower() or 'OOM' in str(e).upper() or 'metal' in str(e).lower():
                    print(f"  Likely OOM, stopping further sizes")
                    break

    # Summary
    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}")
    print(f"{'Context':>9s} | {'Prefill t/s':>11s} | {'Gen t/s':>8s} | {'Total':>6s} | {'Peak GB':>7s} | {'Needle':>6s}")
    for r in results:
        if 'error' in r:
            print(f"{r['context_size_target']:>9d} | ERROR: {r['error'][:50]}")
        else:
            print(f"{r['context_size_actual']:>9d} | {r['prompt_tps']:>11.0f} | {r['generation_tps']:>8.1f} | {r['elapsed_s']:>5.1f}s | {r['peak_memory_gb']:>6.2f} | {'✓' if r['found_needle'] else '✗':>6s}")

if __name__ == '__main__':
    main()
