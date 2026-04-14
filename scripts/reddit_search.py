#!/usr/bin/env python3
"""
Reddit deep-dive search for local LLM benchmarking topics.
Uses YARS infrastructure from reddit-scraper for proxy + session management.
"""
import os
import sys
import json
import time
import random
import logging
from datetime import datetime, timezone

# --- PATH SETUP ---
YARS_SRC = os.path.join(
    os.path.expanduser("~"), "github", "reddit-scraper", "yars", "src"
)
sys.path.insert(0, YARS_SRC)

from yars.yars import YARS

# --- CONFIG ---
PROXY = "https://vicnaum1:IHgqFCZa03qNsqkkOZwL@core-residential.evomi-proxy.com:1001"
OUTPUT_DIR = os.path.join(
    os.path.expanduser("~"), "github", "livebench", "research", "raw"
)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reddit_deep_dive.md")

SUBREDDITS = ["LocalLLaMA", "MachineLearning", "LocalAI"]

QUERIES = [
    "best model 32GB Mac 2026",
    "Qwen3.5 35B A3B",
    "Gemma 4 26B A4B",
    "MLX vs llama.cpp",
    "lm-evaluation-harness local",
    "APEX quantization MoE",
    "Rapid-MLX",
    "Nemotron Cascade",
    "GLM 4.7 Flash",
    "abliterated quality loss",
    "benchmark methodology local LLM",
]

SEARCH_LIMIT = 50


def ts_to_str(ts):
    """Convert Unix timestamp to human-readable string."""
    if not ts:
        return "unknown"
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    except Exception:
        return str(ts)


def rich_search(miner, subreddit, query, limit=50, sort="new"):
    """
    Search a subreddit and return rich post metadata (score, comments, etc.).
    Uses the YARS session directly to get full post data from the search API.
    """
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    # Wrap multi-word queries in quotes for exact phrase
    if " " in query or "." in query:
        q = f'"{query}"'
    else:
        q = query

    params = {
        "q": q,
        "limit": limit,
        "sort": sort,
        "type": "link",
        "restrict_sr": "on",
        "t": "year",  # restrict to past year for recency
    }

    try:
        data = miner._make_request_with_retries(url, params=params)
        if not data or "data" not in data:
            return []
    except Exception as e:
        print(f"  [ERROR] Search failed for r/{subreddit} '{query}': {e}")
        return []

    results = []
    for post in data.get("data", {}).get("children", []):
        d = post.get("data", {})
        selftext = d.get("selftext", "") or ""
        results.append(
            {
                "title": d.get("title", ""),
                "link": f"https://www.reddit.com{d.get('permalink', '')}",
                "permalink": d.get("permalink", ""),
                "subreddit": d.get("subreddit", subreddit),
                "score": d.get("score", 0),
                "num_comments": d.get("num_comments", 0),
                "created_utc": d.get("created_utc", 0),
                "created_str": ts_to_str(d.get("created_utc")),
                "selftext_preview": selftext[:500],
                "author": d.get("author", ""),
                "query": query,
            }
        )
    return results


def fetch_full_thread(miner, permalink):
    """Fetch the full post body + top comments for a thread."""
    try:
        details = miner.scrape_post_details(permalink)
        return details
    except Exception as e:
        print(f"  [ERROR] Could not fetch thread {permalink}: {e}")
        return None


def flatten_comments(comments, depth=0, max_depth=3):
    """Flatten nested comments into readable text."""
    lines = []
    for c in comments:
        if not isinstance(c, dict):
            continue
        indent = "  " * depth
        author = c.get("author", "?")
        score = c.get("score", "?")
        body = (c.get("body", "") or "")[:600]
        if body.strip():
            lines.append(f"{indent}**u/{author}** (score: {score}):\n{indent}> {body}\n")
        if depth < max_depth and c.get("replies"):
            lines.extend(flatten_comments(c["replies"], depth + 1, max_depth))
    return lines


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Initializing YARS with proxy...")
    miner = YARS(proxy=PROXY, timeout=20)

    # ---- PHASE 1: Search all subreddits for all queries ----
    all_results = []
    seen_links = set()

    for subreddit in SUBREDDITS:
        for query in QUERIES:
            print(f"Searching r/{subreddit} for: {query}")
            results = rich_search(miner, subreddit, query, limit=SEARCH_LIMIT, sort="new")
            for r in results:
                if r["link"] not in seen_links:
                    seen_links.add(r["link"])
                    all_results.append(r)
            print(f"  -> {len(results)} results ({len(all_results)} unique total)")
            # Be polite
            time.sleep(random.uniform(1.5, 3.0))

    print(f"\n=== Total unique results: {len(all_results)} ===\n")

    # ---- Sort by relevance: score * recency ----
    # Prioritize 2026 posts, then by score+comments
    now_ts = time.time()
    for r in all_results:
        age_days = max(1, (now_ts - r["created_utc"]) / 86400) if r["created_utc"] else 999
        # Recency boost: posts from last 30 days get 10x, last 90 days get 3x
        recency = 10 if age_days < 30 else (3 if age_days < 90 else 1)
        r["relevance_score"] = (r["score"] + r["num_comments"] * 2) * recency

    all_results.sort(key=lambda x: x["relevance_score"], reverse=True)

    # ---- PHASE 2: Save initial results ----
    # Write intermediate results in case full thread fetching is slow
    with open(OUTPUT_FILE, "w") as f:
        f.write("# Reddit Deep Dive: Local LLM Benchmarking on Apple Silicon\n\n")
        f.write(f"*Search performed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n\n")
        f.write(f"**Subreddits searched:** {', '.join(f'r/{s}' for s in SUBREDDITS)}\n\n")
        f.write(f"**Total unique posts found:** {len(all_results)}\n\n")
        f.write("---\n\n")

        f.write("## All Search Results (sorted by relevance)\n\n")
        for i, r in enumerate(all_results, 1):
            f.write(f"### {i}. [{r['title']}]({r['link']})\n")
            f.write(f"- **Subreddit:** r/{r['subreddit']} | **Score:** {r['score']} | ")
            f.write(f"**Comments:** {r['num_comments']} | **Date:** {r['created_str']}\n")
            f.write(f"- **Author:** u/{r['author']} | **Query match:** \"{r['query']}\"\n")
            f.write(f"- **Relevance score:** {r['relevance_score']}\n")
            if r["selftext_preview"].strip():
                preview = r["selftext_preview"].replace("\n", "\n  > ")
                f.write(f"  > {preview}\n")
            f.write("\n")

    print(f"Initial results saved to {OUTPUT_FILE}")

    # ---- PHASE 3: Fetch full threads for top results ----
    top_n = min(15, len(all_results))
    print(f"\nFetching full content for top {top_n} threads...\n")

    thread_details = []
    for i, r in enumerate(all_results[:top_n]):
        permalink = r["permalink"]
        if not permalink:
            continue
        # Remove trailing slash for scrape_post_details
        permalink = permalink.rstrip("/")
        print(f"  [{i+1}/{top_n}] Fetching: {r['title'][:80]}...")
        details = fetch_full_thread(miner, permalink)
        if details:
            details["meta"] = r
            thread_details.append(details)
        time.sleep(random.uniform(2.0, 4.0))

    print(f"\nSuccessfully fetched {len(thread_details)} full threads.\n")

    # ---- PHASE 4: Append full thread content to output ----
    with open(OUTPUT_FILE, "a") as f:
        f.write("\n---\n\n")
        f.write("## Full Thread Content (Top Threads)\n\n")

        for i, td in enumerate(thread_details, 1):
            meta = td["meta"]
            f.write(f"### Thread {i}: {td['title']}\n")
            f.write(f"- **Link:** {meta['link']}\n")
            f.write(f"- **Score:** {meta['score']} | **Comments:** {meta['num_comments']} | ")
            f.write(f"**Date:** {meta['created_str']}\n\n")

            body = (td.get("body") or "")[:3000]
            if body.strip():
                f.write("**Post body:**\n\n")
                f.write(f"{body}\n\n")

            comments = td.get("comments", [])
            if comments:
                f.write("**Top comments:**\n\n")
                comment_lines = flatten_comments(comments, max_depth=2)
                for line in comment_lines[:40]:  # Cap at 40 comment blocks
                    f.write(f"{line}\n")

            f.write("\n---\n\n")

    print(f"Full output saved to {OUTPUT_FILE}")

    # Also dump raw JSON for later analysis
    json_out = os.path.join(OUTPUT_DIR, "reddit_deep_dive_raw.json")
    with open(json_out, "w") as f:
        json.dump(
            {
                "search_results": all_results,
                "thread_details": [
                    {
                        "title": td["title"],
                        "body": td.get("body", ""),
                        "meta": td["meta"],
                        "comments": td.get("comments", []),
                    }
                    for td in thread_details
                ],
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Raw JSON saved to {json_out}")


if __name__ == "__main__":
    main()
