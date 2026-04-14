#!/usr/bin/env python3
"""
Benchmark: How does context_length affect inference speed and RAM usage?

For each context_length setting:
1. Unload model, reload with new context_length
2. Wait for load, record RAM/swap
3. Warmup with trivial question (discard)
4. Run 3 fixed questions, record tok/s, tokens, elapsed
5. Record RAM/swap after generation
"""
import requests, json, time, subprocess, sys
from pathlib import Path

MODEL_KEY = "google/gemma-4-26b-a4b"
BASE_URL = "http://127.0.0.1:1234"
CONTEXT_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

# Fixed test questions (same for every context size, temp=0 for determinism)
QUESTIONS = [
    {
        "name": "MMLU (short)",
        "max_tokens": 2048,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "The longest wavelength of light that can excite an electron from the valence band to the conduction band in a particular semiconductor is 620 nm. What is the energy gap?\n(A) 1 eV\n(B) 2 eV\n(C) 3 eV\n(D) 4 eV\nAnswer with just the letter."}
        ]
    },
    {
        "name": "DROP (medium)",
        "max_tokens": 2048,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Read the passage and answer the question. Give ONLY the answer, nothing else.\n\nPassage: In the county, the population was spread out with 23.50% under the age of 18, 7.80% from 18 to 24, 28.50% from 25 to 44, 25.60% from 45 to 64, and 14.60% who were 65 years of age or older. The median age was 39 years. For every 100 females there were 95.90 males.\n\nQuestion: Which age group had the second largest population?"}
        ]
    },
    {
        "name": "MATH (thinking)",
        "max_tokens": 4096,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Solve this math problem. Put your final answer in \\boxed{}.\n\nFind the number of positive integers n less than or equal to 1000 such that n^2 - 1 is divisible by 24."}
        ]
    },
]

WARMUP = {
    "messages": [{"role": "user", "content": "What is 2+2? Just the number."}],
    "max_tokens": 100,
    "temperature": 0
}


def get_memory():
    """Get RAM and swap usage."""
    # Swap
    result = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True)
    swap_line = result.stdout.strip()
    swap_used = 0
    m = __import__('re').search(r'used\s*=\s*([0-9.]+)M', swap_line)
    if m:
        swap_used = float(m.group(1))

    # LM Studio process memory
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    lm_mem = 0
    for line in result.stdout.split('\n'):
        if 'llmworker' in line.lower() or ('lmstudio' in line.lower() and 'node' in line.lower()):
            parts = line.split()
            if len(parts) >= 6:
                try:
                    mem_kb = int(parts[5])
                    if mem_kb > lm_mem:
                        lm_mem = mem_kb
                except:
                    pass

    return {
        'swap_used_mb': swap_used,
        'lm_process_mb': round(lm_mem / 1024),
    }


def get_gpu_temp():
    """Get GPU temperature from macmon."""
    try:
        result = subprocess.run(
            ["macmon", "pipe"],
            capture_output=True, text=True, timeout=3
        )
        data = json.loads(result.stdout.strip().split('\n')[0])
        return data.get('gpu_temp', '?')
    except:
        return '?'


def unload_model():
    """Unload the current model."""
    try:
        requests.post(f"{BASE_URL}/api/v1/models/unload", json={"instance_id": MODEL_KEY}, timeout=30)
        time.sleep(2)
    except:
        pass


def load_model(context_length):
    """Load model with specific context_length."""
    payload = {
        "model": MODEL_KEY,
        "context_length": context_length,
    }
    try:
        r = requests.post(f"{BASE_URL}/api/v1/models/load", json=payload, timeout=120)
        if r.status_code != 200:
            print(f"  Load returned {r.status_code}: {r.text[:200]}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"  Load error: {e}", file=sys.stderr)
        return False

    # Wait for model to be ready
    for _ in range(60):
        try:
            r = requests.get(f"{BASE_URL}/v1/models", timeout=5)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False


def warmup():
    """Send warmup question, discard result."""
    try:
        r = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={"model": MODEL_KEY, **WARMUP},
            timeout=60
        )
        d = r.json()
        usage = d.get('usage', {})
        content = d['choices'][0]['message']['content']
        print(f"  Warmup: '{content.strip()[:30]}' | {usage.get('completion_tokens',0)}tok", file=sys.stderr)
        return True
    except Exception as e:
        print(f"  Warmup failed: {e}", file=sys.stderr)
        return False


def run_question(q):
    """Run a single question and return timing data."""
    start = time.time()
    try:
        r = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": MODEL_KEY,
                "messages": q["messages"],
                "max_tokens": q["max_tokens"],
                "temperature": 0,
            },
            timeout=600
        )
        elapsed = time.time() - start
        d = r.json()
        usage = d.get('usage', {})
        choice = d['choices'][0]

        comp = usage.get('completion_tokens', 0)
        think = usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)
        prompt = usage.get('prompt_tokens', 0)
        tok_s = comp / elapsed if elapsed > 0 else 0

        return {
            'name': q['name'],
            'elapsed': round(elapsed, 2),
            'prompt_tokens': prompt,
            'completion_tokens': comp,
            'reasoning_tokens': think,
            'visible_tokens': comp - think,
            'tok_s': round(tok_s, 1),
            'finish_reason': choice.get('finish_reason', '?'),
            'answer_preview': choice['message']['content'][:50].strip(),
        }
    except Exception as e:
        return {
            'name': q['name'],
            'error': str(e),
            'elapsed': round(time.time() - start, 2),
        }


def main():
    results = []

    print(f"{'='*100}", file=sys.stderr)
    print(f"CONTEXT LENGTH vs SPEED BENCHMARK — {MODEL_KEY}", file=sys.stderr)
    print(f"{'='*100}", file=sys.stderr)

    for ctx in CONTEXT_SIZES:
        print(f"\n--- context_length={ctx} ---", file=sys.stderr)

        # Unload
        print(f"  Unloading...", file=sys.stderr)
        unload_model()
        time.sleep(3)

        # Record memory before load
        mem_before = get_memory()
        print(f"  Memory before load: process={mem_before['lm_process_mb']}MB swap={mem_before['swap_used_mb']:.0f}MB", file=sys.stderr)

        # Load with new context
        print(f"  Loading with context_length={ctx}...", file=sys.stderr)
        load_start = time.time()
        if not load_model(ctx):
            print(f"  FAILED to load model!", file=sys.stderr)
            continue
        load_time = time.time() - load_start
        print(f"  Loaded in {load_time:.1f}s", file=sys.stderr)

        # Wait for memory to settle
        time.sleep(5)

        # Record memory after load
        mem_after = get_memory()
        gpu_temp = get_gpu_temp()
        print(f"  Memory after load: process={mem_after['lm_process_mb']}MB swap={mem_after['swap_used_mb']:.0f}MB gpu={gpu_temp}°C", file=sys.stderr)

        # Warmup
        if not warmup():
            print(f"  Warmup failed, skipping context={ctx}", file=sys.stderr)
            continue

        # Run questions
        row = {
            'context_length': ctx,
            'load_time_s': round(load_time, 1),
            'ram_mb': mem_after['lm_process_mb'],
            'swap_mb': round(mem_after['swap_used_mb']),
            'gpu_temp': gpu_temp,
        }

        for q in QUESTIONS:
            print(f"  Running: {q['name']}...", file=sys.stderr, end=' ')
            result = run_question(q)
            if 'error' in result:
                print(f"ERROR: {result['error']}", file=sys.stderr)
            else:
                print(f"{result['tok_s']} tok/s | {result['completion_tokens']}tok ({result['reasoning_tokens']}think+{result['visible_tokens']}vis) | {result['elapsed']}s | \"{result['answer_preview'][:30]}\"", file=sys.stderr)

            row[f"{q['name']}_tok_s"] = result.get('tok_s', 0)
            row[f"{q['name']}_comp_tok"] = result.get('completion_tokens', 0)
            row[f"{q['name']}_think_tok"] = result.get('reasoning_tokens', 0)
            row[f"{q['name']}_vis_tok"] = result.get('visible_tokens', 0)
            row[f"{q['name']}_elapsed"] = result.get('elapsed', 0)
            row[f"{q['name']}_finish"] = result.get('finish_reason', 'error')

        # Final memory check
        mem_final = get_memory()
        row['swap_after_mb'] = round(mem_final['swap_used_mb'])

        results.append(row)

    # Print summary table
    print(f"\n{'='*120}", file=sys.stderr)
    print(f"{'Context':>8s} | {'RAM':>7s} | {'Swap':>6s} | {'Load':>5s} | {'MMLU tok/s':>10s} | {'DROP tok/s':>10s} | {'MATH tok/s':>10s} | {'MMLU tok':>8s} | {'DROP tok':>8s} | {'MATH tok':>8s}", file=sys.stderr)
    print("-" * 120, file=sys.stderr)
    for r in results:
        print(f"{r['context_length']:>8d} | {r['ram_mb']:>6d}M | {r['swap_mb']:>5d}M | {r['load_time_s']:>4.0f}s | "
              f"{r.get('MMLU (short)_tok_s', 0):>10.1f} | {r.get('DROP (medium)_tok_s', 0):>10.1f} | {r.get('MATH (thinking)_tok_s', 0):>10.1f} | "
              f"{r.get('MMLU (short)_comp_tok', 0):>8d} | {r.get('DROP (medium)_comp_tok', 0):>8d} | {r.get('MATH (thinking)_comp_tok', 0):>8d}", file=sys.stderr)

    # Save results
    out_file = str(Path(__file__).resolve().parent.parent / 'results' / 'extracted' / 'context_speed_bench.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}", file=sys.stderr)


if __name__ == '__main__':
    main()
