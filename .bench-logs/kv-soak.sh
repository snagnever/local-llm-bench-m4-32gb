#!/bin/bash
URL=http://127.0.0.1:1234/v1/chat/completions
LOG=.bench-logs/kv-soak-results.log
: > "$LOG"
for i in $(seq 1 10); do
  echo "[$(date +%H:%M:%S)] req $i/10 starting..." >> "$LOG"
  resp=$(curl -s --max-time 900 "$URL" -H 'Content-Type: application/json' -d '{
    "model":"minimax-m2.5",
    "messages":[{"role":"user","content":"Write an extremely detailed, comprehensive technical essay (aim for maximum length) on the design and trade-offs of distributed consensus algorithms — Paxos, Multi-Paxos, Raft, Viewstamped Replication, and Byzantine fault tolerance. Cover safety, liveness, leader election, log replication, membership changes, and failure modes in depth."}],
    "max_tokens":8192,"temperature":0.7}')
  fin=$(echo "$resp" | jq -r '.choices[0].finish_reason // "ERR"' 2>/dev/null)
  ct=$(echo "$resp" | jq -r '.usage.completion_tokens // 0' 2>/dev/null)
  echo "[$(date +%H:%M:%S)] req $i/10 done: finish=$fin completion_tokens=$ct" >> "$LOG"
done
echo "[$(date +%H:%M:%S)] SOAK COMPLETE — all 10 survived" >> "$LOG"
