#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_REPO="${MODEL_REPO:-AndeyTait/DecisionForge-0.8B}"
PORT="${PORT:-8123}"
VENV="${DECISIONFORGE_VENV:-.venv-mac-demo}"
CACHE_DIR="${DECISIONFORGE_CACHE:-$HOME/.cache/huggingface/hub}"
PYTHON_BIN="${DECISIONFORGE_PYTHON:-}"
DEMO_URL="${DECISIONFORGE_DEMO_URL:-https://decision-forge-pi.vercel.app/#demo}"

if [ "${1:-}" = "--remote" ]; then
  REMOTE_HOST="${DECISIONFORGE_REMOTE_HOST:-autodl-dsh}"
  REMOTE_PORT="${DECISIONFORGE_REMOTE_PORT:-8124}"
  echo "Opening an SSH tunnel to the remote 0.8B service..."
  ssh -o ExitOnForwardFailure=yes -N -L "$PORT:127.0.0.1:$REMOTE_PORT" "$REMOTE_HOST" &
  TUNNEL_PID=$!
  trap 'kill "$TUNNEL_PID" 2>/dev/null || true' EXIT INT TERM
  for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      open "$DEMO_URL"
      echo "Remote mode is ready. Press Ctrl-C to close the tunnel."
      wait "$TUNNEL_PID"
      exit $?
    fi
    sleep 1
  done
  echo "The remote endpoint did not become ready." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
  if [ -z "$PYTHON_BIN" ]; then
    for candidate in "$HOME/.local/bin/python3.12" "$HOME/.local/bin/python3.11" python3.12 python3.11; do
      if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN=$(command -v "$candidate")
        break
      fi
    done
  fi
  uv venv --python "${PYTHON_BIN:-3.12}" "$VENV"
fi

"$VENV/bin/python" -c 'import torch, transformers, fastapi, huggingface_hub' 2>/dev/null || \
  uv pip install --python "$VENV/bin/python" \
    'torch==2.14.0' 'transformers==5.17.0' 'safetensors>=0.5' \
    fastapi uvicorn huggingface_hub

echo "Downloading $MODEL_REPO from Hugging Face (cached after the first run)..."
if [ -z "${HF_TOKEN:-}" ]; then
  HF_TOKEN=$(security find-internet-password -s huggingface.co -w 2>/dev/null || true)
  export HF_TOKEN
fi
MODEL_DIR=$("$VENV/bin/python" scripts/fetch_hf_model.py "$MODEL_REPO" --cache-dir "$CACHE_DIR" | tail -n 1)
unset HF_TOKEN

echo "Starting DecisionForge on http://127.0.0.1:$PORT/demo"
"$VENV/bin/python" -m decisionforge.serve \
  --checkpoint-dir "$MODEL_DIR" \
  --device auto \
  --precision auto \
  --host 127.0.0.1 \
  --port "$PORT" \
  --allow-cors &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    open "$DEMO_URL"
    echo "Ready. Press Ctrl-C to stop."
    wait "$SERVER_PID"
    exit $?
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    wait "$SERVER_PID"
    exit $?
  fi
  sleep 1
done

echo "Timed out while loading the checkpoint." >&2
exit 1
