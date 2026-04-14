#!/usr/bin/env python3
"""
Tool-calling benchmark harness.

Runs jdhodges-style + Veerman-style tool-calling test cases against any
OpenAI-compatible endpoint (LM Studio default). Matches bench2.py logging
quality: per-question JSONL, summary JSON, temperature/t/s/RAM/swap gauges,
thermal cooldown gate, and auto-skip of completed cases.

Usage:
    python3 tool_call_bench.py --model <lmstudio_key> --suite jdhodges
    python3 tool_call_bench.py --model <lmstudio_key> --suite veerman
    python3 tool_call_bench.py --model <lmstudio_key> --suite both

"""
import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths + constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent.parent
TC_DIR = REPO_DIR / "research" / "benchmarks" / "tool_calling"
RUNS_DIR = REPO_DIR / "research" / "benchmarks" / "runs"

# Deterministic generation
TEMPERATURE = 0.0
TOP_P = 1.0
SEED = 42

# Thermal safety — match bench2.py / mlx_bench.py behaviour
# Can be disabled with --no-cooldown (in which case we still log temps but never pause).
COOLDOWN_GPU_TEMP = 60  # °C; wait for ≤ this before each question

# Default to LM Studio endpoints; overridable with --base-url
LMSTUDIO_BASE = "http://localhost:1234"
LMSTUDIO_API = f"{LMSTUDIO_BASE}/api/v1"
OPENAI_BASE_DEFAULT = f"{LMSTUDIO_BASE}/v1"

# Per-call HTTP timeout (seconds)
REQUEST_TIMEOUT = 600

# ---------------------------------------------------------------------------
# System monitoring (mirrors bench2.py)
# ---------------------------------------------------------------------------

def get_system_state() -> dict:
    state = {"gpu_temp_c": None, "cpu_temp_c": None,
             "process_ram_mb": 0, "swap_used_mb": 0, "free_pages": 0}
    try:
        proc = subprocess.Popen(["macmon", "pipe"], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
        line = proc.stdout.readline().decode().strip()
        proc.kill(); proc.wait()
        if line:
            d = json.loads(line)
            gpu = round(d.get("temp", {}).get("gpu_temp_avg", 0))
            cpu = round(d.get("temp", {}).get("cpu_temp_avg", 0))
            state["gpu_temp_c"] = gpu if 5 <= gpu <= 120 else None
            state["cpu_temp_c"] = cpu if 5 <= cpu <= 120 else None
    except Exception:
        pass
    try:
        r = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True)
        m = re.search(r"used\s*=\s*([0-9.]+)M", r.stdout)
        if m:
            state["swap_used_mb"] = round(float(m.group(1)))
    except Exception:
        pass
    try:
        r = subprocess.run(["vm_stat"], capture_output=True, text=True)
        m = re.search(r"Pages free:\s+(\d+)", r.stdout)
        if m:
            state["free_pages"] = int(m.group(1))
    except Exception:
        pass
    return state


def get_hardware_info() -> dict:
    info = {}
    try:
        r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                           capture_output=True, text=True)
        info["total_ram_gb"] = round(int(r.stdout.strip()) / 1024**3)
    except Exception:
        pass
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                           capture_output=True, text=True)
        info["cpu"] = r.stdout.strip()
    except Exception:
        pass
    try:
        r = subprocess.run(["sw_vers", "-productVersion"],
                           capture_output=True, text=True)
        info["macos_version"] = r.stdout.strip()
    except Exception:
        pass
    try:
        r = subprocess.run(["uname", "-m"], capture_output=True, text=True)
        info["arch"] = r.stdout.strip()
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Thermal gate
# ---------------------------------------------------------------------------

def _ready(state: dict) -> bool:
    return state["gpu_temp_c"] is not None and state["gpu_temp_c"] <= COOLDOWN_GPU_TEMP


def wait_for_cooldown(enabled: bool = True) -> dict:
    """Return a fresh state reading. If enabled, wait until GPU ≤ COOLDOWN_GPU_TEMP
    (with a valid reading). If disabled, return immediately — callers still log temps."""
    state = get_system_state()
    if not enabled:
        return state
    if _ready(state):
        return state
    t = state["gpu_temp_c"]
    t_str = f"{t}°C" if t is not None else "None (no reading)"
    print(f"  ⏸ GPU at {t_str} — waiting for valid reading ≤{COOLDOWN_GPU_TEMP}°C",
          flush=True)
    while True:
        time.sleep(15)
        st = get_system_state()
        if _ready(st):
            print(f"  ✓ GPU at {st['gpu_temp_c']}°C — resuming", flush=True)
            return st
        ct = st["gpu_temp_c"]
        ct_str = f"{ct}°C" if ct is not None else "None"
        print(f"  ... GPU {ct_str}, still waiting", flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_suite(suite: str) -> tuple[list, list]:
    """Return (tools, cases) for the requested suite."""
    if suite == "jdhodges":
        tools = yaml.safe_load((TC_DIR / "tool_definitions.yaml").read_text())["tools"]
        cases = yaml.safe_load((TC_DIR / "test_cases.yaml").read_text())
        return tools, cases
    if suite == "veerman":
        tools = yaml.safe_load((TC_DIR / "veerman_tools.yaml").read_text())["tools"]
        cases = yaml.safe_load((TC_DIR / "veerman_cases.yaml").read_text())
        return tools, cases
    raise ValueError(f"Unknown suite: {suite}")


# ---------------------------------------------------------------------------
# OpenAI-style request
# ---------------------------------------------------------------------------

def run_case(client: OpenAI, model: str, case: dict, tools: list) -> dict:
    """Execute one tool-calling case. Returns a dict matching bench2.py fields."""
    today = datetime.now().strftime("%Y-%m-%d")
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a helpful assistant with access to tools. "
                f"Today's date is {today}. "
                "Use tools when appropriate to answer user requests. "
                "If a request doesn't need a tool, respond directly. "
                "If required information is missing, ask for clarification. "
                "You may call multiple tools in parallel when appropriate."
            ),
        },
        {"role": "user", "content": case["prompt"]},
    ]

    t1_start = time.time()
    err = None
    t1_tool_calls, t1_text, usage = [], "", {}
    raw_msg_dict = None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=TEMPERATURE,
            top_p=TOP_P,
            seed=SEED,
            timeout=REQUEST_TIMEOUT,
        )
        t1_elapsed = time.time() - t1_start
        msg = resp.choices[0].message
        # Extract tool calls
        if msg.tool_calls:
            for tc in msg.tool_calls:
                t1_tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments_raw": tc.function.arguments,
                })
        t1_text = msg.content or ""
        raw_msg_dict = {
            "role": msg.role,
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": tc.type,
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ],
        }
        # Extract usage — OpenAI SDK parses prompt_tokens/completion_tokens/total_tokens,
        # but mlx_vlm.server returns input_tokens/output_tokens + generation_tps. Pull
        # both shapes from the raw dump so either endpoint works.
        u = resp.usage
        if u is not None:
            raw_u = {}
            try:
                raw_u = u.model_dump()
            except Exception:
                pass
            prompt_tok = (getattr(u, "prompt_tokens", None)
                          or raw_u.get("prompt_tokens")
                          or raw_u.get("input_tokens", 0) or 0)
            comp_tok = (getattr(u, "completion_tokens", None)
                        or raw_u.get("completion_tokens")
                        or raw_u.get("output_tokens", 0) or 0)
            total_tok = (getattr(u, "total_tokens", None)
                         or raw_u.get("total_tokens")
                         or (prompt_tok + comp_tok))
            usage = {
                "prompt_tokens": prompt_tok,
                "completion_tokens": comp_tok,
                "total_tokens": total_tok,
                # mlx_vlm.server extras — keep if present
                "prompt_tps": raw_u.get("prompt_tps"),
                "generation_tps": raw_u.get("generation_tps"),
                "peak_memory_gb": raw_u.get("peak_memory"),
            }
        finish_reason = getattr(resp.choices[0], "finish_reason", "?")
    except Exception as e:
        t1_elapsed = time.time() - t1_start
        err = str(e)
        finish_reason = "error"

    turn_1 = {
        "elapsed_s": round(t1_elapsed, 3),
        "tool_calls": t1_tool_calls,
        "text": t1_text,
        "usage": usage,
        "finish_reason": finish_reason,
        "error": err,
        "raw_message": raw_msg_dict,
    }

    result = {"turn_1": turn_1}

    # Multi-turn: only if case requests it AND turn 1 produced tool calls.
    if case.get("multi_turn") and t1_tool_calls and not err:
        stub = case.get("turn_2_stub_result", '{"status": "ok"}')
        # Replay with assistant message + stubbed tool results
        messages_t2 = list(messages) + [raw_msg_dict]
        for tc in t1_tool_calls:
            messages_t2.append({
                "role": "tool",
                "tool_call_id": tc.get("id", "call_stub"),
                "content": stub,
            })
        t2_start = time.time()
        t2_err = None
        try:
            resp2 = client.chat.completions.create(
                model=model,
                messages=messages_t2,
                tools=tools,
                tool_choice="auto",
                temperature=TEMPERATURE,
                top_p=TOP_P,
                seed=SEED,
                timeout=REQUEST_TIMEOUT,
            )
            t2_elapsed = time.time() - t2_start
            msg2 = resp2.choices[0].message
            t2_tool_calls = []
            if msg2.tool_calls:
                for tc in msg2.tool_calls:
                    t2_tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments_raw": tc.function.arguments,
                    })
            t2_text = msg2.content or ""
            u2 = resp2.usage
            usage2 = {}
            if u2 is not None:
                usage2 = {
                    "prompt_tokens": getattr(u2, "prompt_tokens", 0),
                    "completion_tokens": getattr(u2, "completion_tokens", 0),
                    "total_tokens": getattr(u2, "total_tokens", 0),
                }
            t2_finish = getattr(resp2.choices[0], "finish_reason", "?")
        except Exception as e:
            t2_elapsed = time.time() - t2_start
            t2_err = str(e)
            t2_tool_calls, t2_text, usage2, t2_finish = [], "", {}, "error"
        result["turn_2"] = {
            "elapsed_s": round(t2_elapsed, 3),
            "tool_calls": t2_tool_calls,
            "text": t2_text,
            "usage": usage2,
            "finish_reason": t2_finish,
            "error": t2_err,
        }

    return result


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut",
    "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii",
    "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine",
    "md": "maryland", "ma": "massachusetts", "mi": "michigan",
    "mn": "minnesota", "ms": "mississippi", "mo": "missouri",
    "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico",
    "ny": "new york", "nc": "north carolina", "nd": "north dakota",
    "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
    "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming",
}


def _canonicalize(s: str) -> str:
    return s.strip().lower()


def _expand_location(loc: str) -> str:
    parts = [p.strip() for p in loc.split(",")]
    expanded = [US_STATES.get(p, p) for p in parts]
    return ", ".join(expanded)


def _locations_match(expected: str, actual: str) -> bool:
    e = _expand_location(_canonicalize(expected))
    a = _expand_location(_canonicalize(actual))
    return e.split(",")[0].strip() == a.split(",")[0].strip()


def _datetimes_match(expected: str, actual: str) -> bool:
    e_d = re.sub(r"[^\d]", "", expected)
    a_d = re.sub(r"[^\d]", "", actual)
    if e_d == a_d:
        return True
    # Prefix match on date + hour+minute
    return e_d[:8] == a_d[:8] and e_d[8:12] == a_d[8:12]


def _text_similar(expected: str, actual: str) -> bool:
    kw = set(expected.lower().split()) - {"the", "a", "an", "to", "for", "and", "or"}
    aw = set(actual.lower().split())
    if not kw:
        return True
    overlap = len(kw & aw)
    return overlap / len(kw) >= 0.6


def _values_match(key: str, expected, actual) -> bool:
    if actual is None:
        return False
    if isinstance(expected, str) and isinstance(actual, str):
        e, a = _canonicalize(expected), _canonicalize(actual)
        if key in ("location", "city"):
            return _locations_match(e, a)
        if key in ("datetime", "start", "end", "start_date", "end_date", "time"):
            return _datetimes_match(expected, actual)
        if key in ("title", "subject", "query", "pattern", "body"):
            return _text_similar(e, a)
        return e == a
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(expected - actual) < 0.01
    if isinstance(expected, list) and isinstance(actual, list):
        return {str(x).lower() for x in expected} == {str(x).lower() for x in actual}
    if isinstance(expected, str):
        return str(actual).lower() == expected.lower()
    return expected == actual


def _match_tool_calls(expected_list: list, actual_list: list) -> tuple[bool, dict]:
    breakdown = {
        "expected_count": len(expected_list),
        "actual_count": len(actual_list),
        "tools": [],
    }
    all_pass = True
    matched = set()
    for exp in expected_list:
        best_i = None
        for i, act in enumerate(actual_list):
            if i in matched:
                continue
            if act.get("name") == exp["name"]:
                best_i = i
                break
        if best_i is None:
            breakdown["tools"].append({"pass": False,
                                       "reason": f"no_match_for_{exp['name']}"})
            all_pass = False
            continue
        matched.add(best_i)
        act = actual_list[best_i]
        # Parse args
        raw = act.get("arguments_raw", "{}")
        try:
            act_args = json.loads(raw) if isinstance(raw, str) else raw
            schema_valid = True
        except json.JSONDecodeError:
            schema_valid = False
            act_args = {}
        tool_r = {
            "name": exp["name"],
            "schema_valid": schema_valid,
            "pass": False,
        }
        if not schema_valid:
            tool_r["reason"] = "invalid_json_args"
            breakdown["tools"].append(tool_r)
            all_pass = False
            continue
        exp_args = exp.get("args") or {}
        if not exp_args:
            tool_r["pass"] = True
            breakdown["tools"].append(tool_r)
            continue
        arg_results = {}
        all_args_ok = True
        for k, ev in exp_args.items():
            if ev is None:
                arg_results[k] = {"match": k in act_args,
                                  "expected": "any",
                                  "actual": act_args.get(k)}
                continue
            av = act_args.get(k)
            ok = _values_match(k, ev, av)
            arg_results[k] = {"match": ok, "expected": ev, "actual": av}
            if not ok:
                all_args_ok = False
        tool_r["args"] = arg_results
        tool_r["pass"] = all_args_ok
        if not all_args_ok:
            failed = [k for k, v in arg_results.items() if not v["match"]]
            tool_r["reason"] = f"arg_mismatch: {','.join(failed)}"
            all_pass = False
        breakdown["tools"].append(tool_r)
    if len(actual_list) > len(expected_list):
        breakdown["extra_calls"] = len(actual_list) - len(expected_list)
        all_pass = False
    return all_pass, breakdown


def score_case(case: dict, result: dict) -> dict:
    turn_1 = result.get("turn_1", {})
    if turn_1.get("error"):
        return {"pass": False,
                "reason": f"request_error: {turn_1['error'][:120]}",
                "breakdown": {}}

    expected = case.get("expected_tools") or []
    actual_calls = turn_1.get("tool_calls", [])

    # "No tool expected" cases (edge cases / restraint).
    if not expected:
        no_call = len(actual_calls) == 0
        return {
            "pass": no_call,
            "reason": None if no_call else f"called_tool_when_not_needed: {actual_calls[0].get('name') if actual_calls else '?'}",
            "breakdown": {"no_tool_correct": no_call,
                          "clarified": no_call if case.get("clarify") else None},
        }

    if not actual_calls:
        return {"pass": False, "reason": "no_tool_called",
                "breakdown": {"tool_called": False}}

    # Multi-turn — requires turn 1 tool match AND turn 2 tool call.
    if case.get("multi_turn"):
        t1_pass, t1_bd = _match_tool_calls(expected, actual_calls)
        t2 = result.get("turn_2", {})
        t2_expected = case.get("turn_2_expected_tools") or []
        t2_actual = t2.get("tool_calls", [])
        if t2.get("error"):
            return {"pass": False,
                    "reason": f"turn_2_error: {t2['error'][:120]}",
                    "breakdown": {"turn_1": t1_bd}}
        if t2_expected:
            t2_pass, t2_bd = _match_tool_calls(t2_expected, t2_actual)
        else:
            t2_pass = True
            t2_bd = {"flexible": True}
        overall = t1_pass and t2_pass
        reason = None
        if not t1_pass:
            reason = "turn_1_tool_mismatch"
        elif not t2_pass:
            reason = "turn_2_tool_mismatch"
        return {"pass": overall, "reason": reason,
                "breakdown": {"turn_1": t1_bd, "turn_2": t2_bd}}

    # Single-turn tool matching.
    ok, bd = _match_tool_calls(expected, actual_calls)
    return {"pass": ok,
            "reason": None if ok else "tool_mismatch",
            "breakdown": bd}


# ---------------------------------------------------------------------------
# Auto-skip: scan prior run files for same (model, suite)
# ---------------------------------------------------------------------------

def load_prior_case_ids(model_slug: str, suite: str, run_prefix: str = "toolcall") -> dict:
    """Return {case_id: entry} from prior successful runs for this exact run_prefix."""
    prior = {}
    pattern = f"{run_prefix}_{suite}_{model_slug}_*.jsonl"
    for path in sorted(RUNS_DIR.glob(pattern)):
        try:
            with open(path) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    if e.get("suite") != suite:
                        continue
                    if e.get("model_slug") != model_slug:
                        continue
                    # Skip errors — they're retry-worthy
                    if e.get("finish_reason") == "error":
                        continue
                    cid = e.get("case_id")
                    if cid:
                        prior[cid] = e
        except Exception:
            continue
    return prior


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", s)


def run_bench(model_lmskey: str, suite: str, force: bool = False,
              only_ids: list | None = None,
              base_url: str = OPENAI_BASE_DEFAULT,
              cooldown_enabled: bool = True,
              run_prefix: str = "toolcall") -> tuple[float, dict]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = slugify(model_lmskey)
    run_name = f"{run_prefix}_{suite}_{model_slug}_{timestamp}"
    log_file = RUNS_DIR / f"{run_name}.jsonl"
    summary_file = RUNS_DIR / f"{run_name}_summary.json"

    tools, cases = load_suite(suite)
    hardware = get_hardware_info()
    initial_state = get_system_state()

    if only_ids:
        wanted = set(only_ids)
        cases = [c for c in cases if c.get("id") in wanted]

    # Auto-skip — keyed by prefix so LM Studio runs don't collide with mlx_vlm runs
    prior = {} if force else load_prior_case_ids(model_slug, suite, run_prefix)
    carried = 0
    if prior:
        before = len(cases)
        cases = [c for c in cases if c.get("id") not in prior]
        carried = before - len(cases)

    # Prior entries use score_pass as the top-level field (see how we write them
    # below). Older versions used score.pass — keep the fallback for safety.
    def _prior_passed(e):
        if "score_pass" in e:
            return bool(e["score_pass"])
        return bool(e.get("score", {}).get("pass", False))
    passed_carried = sum(1 for e in prior.values() if _prior_passed(e))
    correct_count = passed_carried
    total_count = carried

    n_fresh = len(cases)

    print(f"\n{'='*110}", flush=True)
    print(f"TOOL_CALL | Suite: {suite.upper()} | Model: {model_lmskey} | Fresh: {n_fresh} | Carried: {carried}", flush=True)
    print(f"Params: temp={TEMPERATURE}, top_p={TOP_P}, seed={SEED}", flush=True)
    print(f"Endpoint: {base_url} | Cooldown gate: {'ON (≤60°C)' if cooldown_enabled else 'OFF'} | Run prefix: {run_prefix}", flush=True)
    print(f"System: RAM={hardware.get('total_ram_gb','?')}GB, CPU={hardware.get('cpu','?')}", flush=True)
    print(f"State:  GPU={initial_state['gpu_temp_c']}°C, swap={initial_state['swap_used_mb']}MB", flush=True)
    if carried:
        print(f"Auto-skip: {carried} prior entries ({passed_carried}/{carried} correct "
              f"= {passed_carried/max(carried,1)*100:.0f}%)", flush=True)
    print(f"Log: {log_file}", flush=True)
    print(f"{'='*110}", flush=True)
    sys.stdout.flush()

    client = OpenAI(base_url=base_url, api_key="lm-studio",
                    timeout=REQUEST_TIMEOUT)

    start_time = time.time()

    with open(log_file, "w") as logf:
        for qi, case in enumerate(cases, 1):
            cid = case.get("id", f"case_{qi}")
            state = wait_for_cooldown(enabled=cooldown_enabled)

            r = run_case(client, model_lmskey, case, tools)
            total_count += 1
            state_after = get_system_state()

            score = score_case(case, r)
            if score["pass"]:
                correct_count += 1

            t1 = r.get("turn_1", {})
            t2 = r.get("turn_2")
            usage = t1.get("usage", {}) or {}
            comp_tok = usage.get("completion_tokens", 0) or 0
            prompt_tok = usage.get("prompt_tokens", 0) or 0
            elapsed_s = t1.get("elapsed_s", 0) or 0
            tok_s = (comp_tok / elapsed_s) if (comp_tok and elapsed_s) else 0
            # mlx_vlm.server reports its own decode tps; prefer it when present.
            gen_tps_engine = usage.get("generation_tps")
            prefill_tps_engine = usage.get("prompt_tps")
            peak_mem_gb_engine = usage.get("peak_memory_gb")

            entry = {
                "run_name": run_name,
                "suite": suite,
                "model": model_lmskey,
                "model_slug": model_slug,
                "case_id": cid,
                "category": case.get("category"),
                "split": case.get("split"),
                "prompt": case.get("prompt"),
                "expected_tools": case.get("expected_tools"),
                "multi_turn": bool(case.get("multi_turn")),
                "timestamp": datetime.now().isoformat(),

                # Turn 1
                "turn_1_elapsed_s": elapsed_s,
                "turn_1_tool_calls": t1.get("tool_calls"),
                "turn_1_text": t1.get("text"),
                "turn_1_finish_reason": t1.get("finish_reason"),
                "prompt_tokens": prompt_tok,
                "completion_tokens": comp_tok,
                "total_tokens": usage.get("total_tokens", 0),
                "tok_s": round(tok_s, 2),
                "engine_generation_tps": gen_tps_engine,
                "engine_prompt_tps": prefill_tps_engine,
                "engine_peak_memory_gb": peak_mem_gb_engine,

                # Turn 2 (if any)
                "turn_2_elapsed_s": (t2 or {}).get("elapsed_s"),
                "turn_2_tool_calls": (t2 or {}).get("tool_calls"),
                "turn_2_text": (t2 or {}).get("text"),
                "turn_2_usage": (t2 or {}).get("usage"),

                # Scoring
                "score_pass": score["pass"],
                "score_reason": score.get("reason"),
                "score_breakdown": score.get("breakdown"),

                # System
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "seed": SEED,
                "gpu_temp_c": state["gpu_temp_c"],
                "cpu_temp_c": state["cpu_temp_c"],
                "gpu_temp_after_c": state_after["gpu_temp_c"],
                "swap_used_mb": state["swap_used_mb"],
                "swap_after_mb": state_after["swap_used_mb"],
                "free_pages": state["free_pages"],

                "error": t1.get("error"),
                "finish_reason": t1.get("finish_reason"),
                "raw_message": t1.get("raw_message"),
            }
            logf.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            logf.flush()

            status = "OK" if score["pass"] else "FAIL"
            temp_str = (f"{state_after['gpu_temp_c']}°C"
                        if state_after["gpu_temp_c"] else "?")
            reason_str = ""
            if not score["pass"] and score.get("reason"):
                reason_str = f" ({score['reason'][:40]})"
            score_so_far = correct_count / total_count * 100 if total_count else 0

            # Prefer engine-reported tps when available; fall back to wall-clock
            tps_display = gen_tps_engine if gen_tps_engine else tok_s
            tps_str = f"{tps_display:>5.1f}t/s" if tps_display else " ?t/s"
            print(
                f"  {qi:>3d}/{n_fresh} {cid:<36s} | "
                f"{elapsed_s:6.1f}s | p={prompt_tok:>4d} c={comp_tok:>4d} "
                f"{tps_str} | {temp_str:>5s} sw={state['swap_used_mb']}M | "
                f"{status:5s}{reason_str} | "
                f"score: {score_so_far:.0f}%",
                flush=True,
            )
            gc.collect()

    elapsed_total = time.time() - start_time
    score_pct = correct_count / total_count if total_count else 0
    final_state = get_system_state()

    # Rebuild summary by scanning the log file for fresh rows + prior rows
    # (score counts already merged above)
    by_category = {}
    fresh_pass = 0
    fresh_total = 0
    fresh_tok_total = 0
    fresh_time_total = 0.0
    for e in prior.values():
        cat = e.get("category", "?")
        c = by_category.setdefault(cat, {"pass": 0, "total": 0})
        c["total"] += 1
        if e.get("score_pass"):
            c["pass"] += 1
    with open(log_file) as f:
        for line in f:
            e = json.loads(line)
            cat = e.get("category", "?")
            c = by_category.setdefault(cat, {"pass": 0, "total": 0})
            c["total"] += 1
            if e.get("score_pass"):
                c["pass"] += 1
                fresh_pass += 1
            fresh_total += 1
            ct = e.get("completion_tokens", 0)
            elap = e.get("turn_1_elapsed_s", 0) or 0
            if ct and elap:
                fresh_tok_total += ct
                fresh_time_total += elap

    fresh_tps = (fresh_tok_total / fresh_time_total) if fresh_time_total else 0

    summary = {
        "run_name": run_name,
        "suite": suite,
        "model": model_lmskey,
        "model_slug": model_slug,
        "score": score_pct,
        "score_pct": f"{score_pct*100:.1f}%",
        "correct": correct_count,
        "total": total_count,
        "carried_over": carried,
        "fresh_run": fresh_total,
        "by_category": by_category,
        "fresh_tok_weighted_tps": round(fresh_tps, 2),
        "elapsed_s": round(elapsed_total),
        "elapsed_min": round(elapsed_total / 60, 1),
        "temperature": TEMPERATURE,
        "seed": SEED,
        "hardware": hardware,
        "initial_state": initial_state,
        "final_state": final_state,
        "timestamp_start": datetime.fromtimestamp(start_time).isoformat(),
        "timestamp_end": datetime.now().isoformat(),
        "log_file": str(log_file),
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(
        f"\n  >>> {suite.upper()}: {score_pct*100:.0f}% ({correct_count}/{total_count}) "
        f"in {elapsed_total:.0f}s ({elapsed_total/60:.1f}m) "
        f"[fresh={fresh_total}, carried={carried}] tok-wt {fresh_tps:.1f}t/s",
        flush=True,
    )
    print(f"  >>> GPU: {initial_state['gpu_temp_c']}°C → {final_state['gpu_temp_c']}°C",
          flush=True)
    print(f"  >>> Swap: {initial_state['swap_used_mb']}MB → {final_state['swap_used_mb']}MB",
          flush=True)
    print(f"  >>> Log: {log_file}", flush=True)
    print(f"  >>> Summary: {summary_file}", flush=True)
    return score_pct, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tool-calling benchmark harness")
    p.add_argument("--model", required=True,
                   help="Model identifier passed to the chat/completions endpoint")
    p.add_argument("--suite", choices=["jdhodges", "veerman"], required=True)
    p.add_argument("--force", action="store_true",
                   help="Rerun completed cases even if a prior run exists")
    p.add_argument("--only", type=str,
                   help="Comma-separated case_ids to run")
    p.add_argument("--base-url", default=OPENAI_BASE_DEFAULT,
                   help="OpenAI-compatible endpoint (default: LM Studio on :1234). "
                        "Use http://localhost:8080/v1 for mlx_vlm.server")
    p.add_argument("--no-cooldown", action="store_true",
                   help="Disable the 60°C thermal gate (still log temps)")
    p.add_argument("--run-prefix", default="toolcall",
                   help="File/run name prefix to separate LM Studio vs mlx_vlm results "
                        "(default: toolcall, use toolcall_mlx for mlx_vlm runs)")
    args = p.parse_args()

    only_ids = None
    if args.only:
        only_ids = [x.strip() for x in args.only.split(",")]

    print(f"Loading {args.suite} suite...", flush=True)

    print(f"\n--- Pre-flight ---", flush=True)
    state = get_system_state()
    hw = get_hardware_info()
    print(f"  Model: {args.model}", flush=True)
    print(f"  Endpoint: {args.base_url}", flush=True)
    print(f"  Cooldown gate: {'OFF' if args.no_cooldown else 'ON (≤60°C)'}", flush=True)
    print(f"  Run prefix: {args.run_prefix}", flush=True)
    print(f"  Hardware: {hw.get('cpu','?')}, {hw.get('total_ram_gb','?')}GB RAM, macOS {hw.get('macos_version','?')}", flush=True)
    print(f"  GPU: {state['gpu_temp_c']}°C | CPU: {state['cpu_temp_c']}°C", flush=True)
    print(f"  Swap: {state['swap_used_mb']}MB", flush=True)

    score, summary = run_bench(args.model, args.suite,
                               force=args.force, only_ids=only_ids,
                               base_url=args.base_url,
                               cooldown_enabled=not args.no_cooldown,
                               run_prefix=args.run_prefix)
