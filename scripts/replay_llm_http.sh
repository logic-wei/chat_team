#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Replay an LLM HTTP request captured in llm_http/*.json.

Usage:
  scripts/replay_llm_http.sh -f <log_file> [options]

Required:
  -f, --file PATH           Request log JSON file (for example llm_http/...-http.json)

Options:
  -n, --repeat N            Number of replay attempts (default: 1)
  -t, --max-time SEC        curl max time per request in seconds (default: 75)
  -c, --connect-timeout SEC curl connect timeout in seconds (default: 10)
  -o, --out-dir DIR         Output directory for headers/body/metrics
                            (default: /tmp/llm_http_replay_<timestamp>)
      --insecure            Add curl -k
  -h, --help                Show this help

Examples:
  scripts/replay_llm_http.sh -f llm_http/20260529-143310-866-037-mechanic_technician-agent-http.json
  scripts/replay_llm_http.sh -f llm_http/xxx-http.json -n 3 -t 90
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command not found: $1" >&2
    exit 1
  fi
}

LOG_FILE=""
REPEAT=1
MAX_TIME=75
CONNECT_TIMEOUT=10
OUT_DIR=""
INSECURE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file)
      LOG_FILE="${2:-}"
      shift 2
      ;;
    -n|--repeat)
      REPEAT="${2:-}"
      shift 2
      ;;
    -t|--max-time)
      MAX_TIME="${2:-}"
      shift 2
      ;;
    -c|--connect-timeout)
      CONNECT_TIMEOUT="${2:-}"
      shift 2
      ;;
    -o|--out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --insecure)
      INSECURE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_cmd jq
require_cmd curl

if [[ -z "$LOG_FILE" ]]; then
  echo "Error: --file is required" >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$LOG_FILE" ]]; then
  echo "Error: file not found: $LOG_FILE" >&2
  exit 2
fi

if ! [[ "$REPEAT" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --repeat must be a positive integer" >&2
  exit 2
fi

if ! [[ "$MAX_TIME" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Error: --max-time must be a positive number" >&2
  exit 2
fi

if ! [[ "$CONNECT_TIMEOUT" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Error: --connect-timeout must be a positive number" >&2
  exit 2
fi

URL="$(jq -r '.http.url // empty' "$LOG_FILE")"
METHOD="$(jq -r '.http.method // "POST"' "$LOG_FILE")"
BODY="$(jq -r '.http.body // empty' "$LOG_FILE")"

if [[ -z "$URL" ]]; then
  echo "Error: unable to read .http.url from $LOG_FILE" >&2
  exit 2
fi

if [[ -z "$BODY" ]]; then
  echo "Error: unable to read .http.body from $LOG_FILE" >&2
  exit 2
fi

AUTH="$(jq -r '.http.headers[]? | select((.name|ascii_downcase)=="authorization") | .value' "$LOG_FILE" | head -n 1)"
CONTENT_TYPE="$(jq -r '.http.headers[]? | select((.name|ascii_downcase)=="content-type") | .value' "$LOG_FILE" | head -n 1)"
ACCEPT="$(jq -r '.http.headers[]? | select((.name|ascii_downcase)=="accept") | .value' "$LOG_FILE" | head -n 1)"

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="/tmp/llm_http_replay_$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$OUT_DIR"

REQUEST_BODY_FILE="$OUT_DIR/request_body.json"
printf '%s' "$BODY" > "$REQUEST_BODY_FILE"

METRICS_FILE="$OUT_DIR/metrics.csv"
echo "run,http_code,total_s,connect_s,starttransfer_s,exit_code" > "$METRICS_FILE"

echo "Replay source: $LOG_FILE"
echo "Output dir   : $OUT_DIR"
echo "Method/URL   : $METHOD $URL"
echo "Max time     : $MAX_TIME s"
echo "Repeats      : $REPEAT"
echo

for i in $(seq 1 "$REPEAT"); do
  hdr_file="$OUT_DIR/run_${i}.headers"
  body_file="$OUT_DIR/run_${i}.body"

  echo "=== run $i/$REPEAT ==="

  curl_args=(
    -sS
    -X "$METHOD"
    "$URL"
    -H "Content-Type: ${CONTENT_TYPE:-application/json}"
    -H "Accept: ${ACCEPT:-application/json}"
    --connect-timeout "$CONNECT_TIMEOUT"
    --max-time "$MAX_TIME"
    --data-binary "@$REQUEST_BODY_FILE"
    -D "$hdr_file"
    -o "$body_file"
    -w "http_code=%{http_code} total=%{time_total} connect=%{time_connect} starttransfer=%{time_starttransfer}"
  )

  if [[ -n "$AUTH" ]]; then
    curl_args+=( -H "Authorization: $AUTH" )
  fi
  if [[ "$INSECURE" -eq 1 ]]; then
    curl_args+=( -k )
  fi

  set +e
  curl_out="$(curl "${curl_args[@]}" 2>&1)"
  rc=$?
  set -e

  http_code="$(printf '%s\n' "$curl_out" | sed -n 's/.*http_code=\([0-9][0-9][0-9]\).*/\1/p' | tail -n 1)"
  total_s="$(printf '%s\n' "$curl_out" | sed -n 's/.*total=\([0-9.]*\).*/\1/p' | tail -n 1)"
  connect_s="$(printf '%s\n' "$curl_out" | sed -n 's/.*connect=\([0-9.]*\).*/\1/p' | tail -n 1)"
  starttransfer_s="$(printf '%s\n' "$curl_out" | sed -n 's/.*starttransfer=\([0-9.]*\).*/\1/p' | tail -n 1)"

  echo "$curl_out"
  echo "curl_exit=$rc"
  echo "$i,${http_code:-000},${total_s:-0},${connect_s:-0},${starttransfer_s:-0},$rc" >> "$METRICS_FILE"

  if [[ $rc -eq 0 ]]; then
    echo "result: SUCCESS"
  else
    echo "result: FAILED"
  fi
  echo
 done

echo "Saved metrics: $METRICS_FILE"
echo "Saved files  : $OUT_DIR/run_*.headers, $OUT_DIR/run_*.body"
echo "Tip          : exit_code=28 usually means client timeout"
