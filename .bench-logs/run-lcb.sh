#!/bin/bash
cd /Users/vitor/LocalProjects/local-llms/tools/local-llm-bench-m4-32gb
export LMSTUDIO_URL=http://127.0.0.1:1234/v1
exec ../../.venv/bin/python scripts/bench2.py livecodebench --examples 50 --model minimax-m2.5 --lcb-version release_v6
