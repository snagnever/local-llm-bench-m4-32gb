#!/usr/bin/env python3
"""
Speed probe: 3 questions per model to assess speed and basic quality.
Follows clean-state protocol with macmon monitoring.

Usage: python3 speed_probe.py <model_id> <output_dir>
"""
import time, json, sys, os, subprocess, signal
from urllib.request import urlopen, Request
from datetime import datetime

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "local"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "results/speed_probe"
BASE_URL = "http://127.0.0.1:1234/v1"

# Sanitize model ID for filenames
SAFE_NAME = MODEL_ID.replace("/", "_").replace(" ", "_")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

QUESTIONS = [
    {
        "name": "trivial",
        "category": "sanity",
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "expected": "4",
        "max_tokens": 256,
    },
    {
        "name": "mmlu_atmosphere",
        "category": "knowledge",
        "messages": [{"role": "user", "content": "Which of the following is the correct order of the layers of the Earth's atmosphere from lowest to highest?\n(A) Troposphere, Stratosphere, Mesosphere, Thermosphere\n(B) Stratosphere, Troposphere, Mesosphere, Thermosphere\n(C) Troposphere, Mesosphere, Stratosphere, Thermosphere\n(D) Mesosphere, Troposphere, Stratosphere, Thermosphere\nAnswer with just the letter."}],
        "expected": "A",
        "max_tokens": 512,
    },
    {
        "name": "code_second_largest",
        "category": "coding",
        "messages": [{"role": "user", "content": "Write a Python function that takes a list of integers and returns the second largest element. If fewer than 2 unique elements, return None. Just the function, no explanation."}],
        "expected": None,  # manual check
        "max_tokens": 1024,
    },
]

def call_api(messages, max_tokens):
    payload = json.dumps({
        "model": MODEL_ID,
        "messages": [{"role": "system", "content": "You are a helpful assistant."}] + messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()
    req = Request(f"{BASE_URL}/chat/completions", data=payload,
                  headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        elapsed = time.time() - start
        content = data["choices"][0]["message"].get("content", "")
        usage = data.get("usage", {})
        return {
            "success": True,
            "content": content,
            "elapsed": round(elapsed, 2),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
        }
    except Exception as e:
        elapsed = time.time() - start
        return {"success": False, "error": str(e), "elapsed": round(elapsed, 2)}

def main():
    print(f"\n{'='*60}")
    print(f"Speed Probe: {MODEL_ID}")
    print(f"Time: {TIMESTAMP}")
    print(f"{'='*60}")

    # Start macmon logging
    macmon_path = os.path.join(OUTPUT_DIR, f"{SAFE_NAME}_{TIMESTAMP}_macmon.jsonl")
    macmon_proc = subprocess.Popen(
        ["macmon", "pipe"],
        stdout=open(macmon_path, "w"),
        stderr=subprocess.DEVNULL,
    )
    print(f"  macmon logging to: {os.path.basename(macmon_path)}")
    time.sleep(1)  # Let macmon start

    # Warmup (throwaway)
    print(f"  Warmup query...")
    warmup = call_api([{"role": "user", "content": "Say hello."}], 32)
    print(f"  Warmup done in {warmup['elapsed']}s")
    time.sleep(1)

    # Run questions
    results = {
        "model": MODEL_ID,
        "timestamp": TIMESTAMP,
        "base_url": BASE_URL,
        "questions": [],
    }

    total_time = 0
    for q in QUESTIONS:
        print(f"  {q['name']:25s} | ", end="", flush=True)
        r = call_api(q["messages"], q["max_tokens"])

        if r["success"]:
            answer_preview = r["content"].replace("\n", " ")[:60]
            visible_tokens = r["completion_tokens"] - r["reasoning_tokens"]
            correct = "?" if q["expected"] is None else ("OK" if q["expected"] in r["content"][:20] else "WRONG")
            print(f"{r['elapsed']:5.1f}s | {r['completion_tokens']:>4}tok ({r['reasoning_tokens']:>3}think +{visible_tokens:>3}vis) | {correct:5s} | {answer_preview}")
        else:
            print(f"{r['elapsed']:5.1f}s | ERROR: {r['error'][:50]}")

        total_time += r["elapsed"]
        results["questions"].append({**q, "result": r})

    results["total_time"] = round(total_time, 2)
    print(f"  {'TOTAL':25s} | {total_time:5.1f}s")

    # Stop macmon
    macmon_proc.send_signal(signal.SIGTERM)
    macmon_proc.wait()

    # Analyze macmon data
    try:
        with open(macmon_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        if lines:
            max_ram = max(l["memory"]["ram_usage"] for l in lines) / 1e9
            max_swap = max(l["memory"]["swap_usage"] for l in lines) / 1e9
            avg_gpu = sum(l["gpu_usage"][1] for l in lines) / len(lines) * 100
            max_power = max(l["all_power"] for l in lines)
            results["system_metrics"] = {
                "peak_ram_gb": round(max_ram, 1),
                "peak_swap_gb": round(max_swap, 1),
                "avg_gpu_pct": round(avg_gpu, 1),
                "peak_power_w": round(max_power, 1),
                "samples": len(lines),
            }
            gpu_spill = "YES - SWAPPING" if max_swap > 4 else "no"
            print(f"\n  System: RAM={max_ram:.1f}GB, Swap={max_swap:.1f}GB, GPU={avg_gpu:.0f}%, Power={max_power:.0f}W, Spill={gpu_spill}")
    except Exception as e:
        print(f"\n  macmon analysis error: {e}")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, f"{SAFE_NAME}_{TIMESTAMP}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {os.path.basename(results_path)}")
    print()

if __name__ == "__main__":
    main()
