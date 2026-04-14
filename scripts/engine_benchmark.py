#!/usr/bin/env python3
"""Quick engine benchmark: same 3 questions across different engines."""
import time, json, sys
from urllib.request import urlopen, Request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:1234/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "local"

QUESTIONS = [
    ("2+2", [{"role": "user", "content": "What is 2+2? Answer with just the number."}]),
    ("MMLU", [{"role": "user", "content": "Which of the following is the correct order of the layers of the Earths atmosphere from lowest to highest?\n(A) Troposphere, Stratosphere, Mesosphere, Thermosphere\n(B) Stratosphere, Troposphere, Mesosphere, Thermosphere\n(C) Troposphere, Mesosphere, Stratosphere, Thermosphere\n(D) Mesosphere, Troposphere, Stratosphere, Thermosphere\nAnswer with just the letter."}]),
    ("Code", [{"role": "user", "content": "Write a Python function that takes a list of integers and returns the second largest element. If fewer than 2 unique elements, return None. Just the function."}]),
    ("Reasoning", [{"role": "user", "content": "In quantum mechanics, explain the measurement problem and how Copenhagen differs from Many-Worlds. Be concise (3 sentences)."}]),
    ("Math", [{"role": "user", "content": "A train leaves station A at 60 mph. Another train leaves station B (300 miles away) at 40 mph toward A. When do they meet? Show your work briefly."}]),
]

print(f"\n{'='*60}")
print(f"Engine: {BASE_URL} | Model: {MODEL}")
print(f"{'='*60}")

total_time = 0
for name, messages in QUESTIONS:
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 2048,
    }).encode()

    req = Request(f"{BASE_URL}/chat/completions", data=payload, headers={"Content-Type": "application/json"})

    start = time.time()
    try:
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        elapsed = time.time() - start

        content = data["choices"][0]["message"].get("content", "")
        usage = data.get("usage", {})
        comp_tokens = usage.get("completion_tokens", "?")
        reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)

        answer_preview = content.replace("\n", " ")[:80]
        total_time += elapsed

        print(f"  {name:12s} | {elapsed:5.1f}s | {comp_tokens:>4} tok ({reasoning:>3} think) | {answer_preview}")
    except Exception as e:
        elapsed = time.time() - start
        total_time += elapsed
        print(f"  {name:12s} | {elapsed:5.1f}s | ERROR: {str(e)[:60]}")

print(f"  {'TOTAL':12s} | {total_time:5.1f}s")
print()
