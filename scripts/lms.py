#!/usr/bin/env python3
"""
LM Studio model loader and health checker.

Usage:
  python3 lms.py status                    # Show loaded model + memory
  python3 lms.py load <model> [--ctx N]    # Load model with context_length
  python3 lms.py unload                    # Unload current model
  python3 lms.py warmup                    # Warmup loaded model with trivial question
  python3 lms.py check                     # Full pre-flight: status + warmup + verify params
"""
import requests, json, subprocess, sys, time

BASE_URL = "http://127.0.0.1:1234"

def get_memory():
    swap_used = 0
    try:
        r = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True)
        import re
        m = re.search(r'used\s*=\s*([0-9.]+)M', r.stdout)
        if m: swap_used = float(m.group(1))
    except: pass

    lm_mem = 0
    try:
        r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        for line in r.stdout.split('\n'):
            if 'llmworker' in line.lower() or ('node' in line and '.lmstudio' in line):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        mem_kb = int(parts[5])
                        if mem_kb > lm_mem: lm_mem = mem_kb
                    except: pass
    except: pass

    return {'lm_process_mb': round(lm_mem / 1024), 'swap_used_mb': round(swap_used)}

def get_gpu_temp():
    try:
        r = subprocess.run(["macmon", "pipe"], capture_output=True, text=True, timeout=3)
        d = json.loads(r.stdout.strip().split('\n')[0])
        return round(d.get('gpu_temp_avg', d.get('temp', {}).get('gpu_temp_avg', 0)))
    except: return None

def get_loaded_model():
    try:
        r = requests.get(f"{BASE_URL}/api/v1/models", timeout=5)
        models = r.json().get('models', [])
        for m in models:
            if m.get('loaded_instances'):
                inst = m['loaded_instances'][0]
                return {
                    'id': m['key'],
                    'arch': m.get('architecture', '?'),
                    'quant': m.get('quantization', {}).get('name', '?'),
                    'size_gb': round(m.get('size_bytes', 0) / 1024**3, 1),
                    'context_length': inst.get('config', {}).get('context_length', '?'),
                    'flash_attention': inst.get('config', {}).get('flash_attention', '?'),
                }
        return None
    except: return None

def cmd_status():
    model = get_loaded_model()
    mem = get_memory()
    temp = get_gpu_temp()
    if model:
        print(f"Model:   {model['id']}")
        print(f"Arch:    {model['arch']} | Quant: {model['quant']} | Size: {model['size_gb']}GB")
        print(f"Context: {model['context_length']} | Flash: {model['flash_attention']}")
    else:
        print("Model:   None loaded")
    print(f"RAM:     {mem['lm_process_mb']}MB process | {mem['swap_used_mb']}MB swap")
    print(f"GPU:     {temp}°C" if temp else "GPU:     ?")
    return model

def cmd_load(model_id, context_length=None):
    print(f"Unloading current model...")
    try:
        # Unload all
        r = requests.get(f"{BASE_URL}/api/v1/models", timeout=5)
        for m in r.json().get('models', []):
            for inst in m.get('loaded_instances', []):
                requests.post(f"{BASE_URL}/api/v1/models/unload",
                            json={"instance_id": inst['id']}, timeout=30)
    except: pass
    time.sleep(3)

    payload = {"model": model_id}
    if context_length:
        payload["context_length"] = context_length

    print(f"Loading {model_id} (context={context_length or 'default'})...")
    start = time.time()
    try:
        r = requests.post(f"{BASE_URL}/api/v1/models/load", json=payload, timeout=120)
        elapsed = time.time() - start
        if r.status_code == 200:
            print(f"Loaded in {elapsed:.1f}s")
            time.sleep(3)
            cmd_status()
            return True
        else:
            print(f"FAILED: {r.status_code} {r.text[:200]}")
            return False
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def cmd_unload():
    try:
        r = requests.get(f"{BASE_URL}/api/v1/models", timeout=5)
        for m in r.json().get('models', []):
            for inst in m.get('loaded_instances', []):
                requests.post(f"{BASE_URL}/api/v1/models/unload",
                            json={"instance_id": inst['id']}, timeout=30)
        print("Unloaded.")
    except Exception as e:
        print(f"Error: {e}")

def cmd_warmup():
    model = get_loaded_model()
    if not model:
        print("ERROR: No model loaded")
        return False

    print(f"Warming up {model['id']}...")
    try:
        r = requests.post(f"{BASE_URL}/v1/chat/completions", json={
            "model": model['id'],
            "messages": [{"role": "user", "content": "What is 2+2? Just the number."}],
            "max_tokens": 200, "temperature": 0
        }, timeout=60)
        d = r.json()
        usage = d.get('usage', {})
        content = d['choices'][0]['message']['content'].strip()
        comp = usage.get('completion_tokens', 0)
        think = usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)
        prompt = usage.get('prompt_tokens', 0)
        print(f"  Response: \"{content[:50]}\"")
        print(f"  Tokens:   prompt={prompt} completion={comp} (think={think} vis={comp-think})")
        print(f"  Finish:   {d['choices'][0].get('finish_reason', '?')}")

        if '4' in content:
            print("  Warmup OK")
            return True
        else:
            print(f"  WARNING: Expected '4' in response, got '{content[:30]}'")
            return False
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

def cmd_check():
    print("=== PRE-FLIGHT CHECK ===")
    model = cmd_status()
    if not model:
        print("\nFAIL: No model loaded")
        return False
    print()
    ok = cmd_warmup()
    if not ok:
        print("\nFAIL: Warmup failed")
        return False
    mem = get_memory()
    temp = get_gpu_temp()
    print(f"\n  Memory: {mem['lm_process_mb']}MB process, {mem['swap_used_mb']}MB swap")
    print(f"  GPU:    {temp}°C" if temp else "  GPU:    ?")
    if temp and temp > 50:
        print(f"  WARNING: GPU is warm ({temp}°C). Consider waiting for cooldown.")
    print("\nPRE-FLIGHT: PASS")
    return True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="LM Studio model manager")
    parser.add_argument("command", choices=["status", "load", "unload", "warmup", "check"])
    parser.add_argument("model", nargs="?", help="Model ID for load command")
    parser.add_argument("--ctx", type=int, help="Context length")
    args = parser.parse_args()

    if args.command == "status": cmd_status()
    elif args.command == "load":
        if not args.model:
            print("Usage: lms.py load <model_id> [--ctx N]")
            sys.exit(1)
        cmd_load(args.model, args.ctx)
    elif args.command == "unload": cmd_unload()
    elif args.command == "warmup": cmd_warmup()
    elif args.command == "check": cmd_check()
