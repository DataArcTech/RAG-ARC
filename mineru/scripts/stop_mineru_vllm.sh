#!/usr/bin/env bash
set -euo pipefail

# Stop MinerU server(s) and/or vLLM server(s) started in background.
#
# Primary mechanism:
# - vLLM pool script writes pidfiles:   /tmp/vllm_<port>.pid
# - (optional) you may also have pidfiles: /tmp/mineru_<port>.pid
#
# Why this script exists:
# - vLLM uses multiple processes (frontend + EngineCore). Killing only the LISTEN pid
#   (e.g. via lsof) may leave GPU-resident worker processes alive. Here we try to kill
#   the whole process group for robustness.

usage() {
  cat <<'EOF'
Usage:
  stop_mineru_vllm.sh [options]

Options:
  --vllm-ports <csv>     Stop vLLM by ports, e.g. 30000,30001
  --mineru-ports <csv>   Stop MinerU server by ports, e.g. 8897,8898,8899
  --kill-enginecore-gpus <csv>
                          Kill leftover vLLM EngineCore processes on specific GPUs, e.g. 5,6
  --all-pidfiles         Stop any /tmp/vllm_*.pid and /tmp/mineru_*.pid found
  --clean                Remove /tmp/vllm_*.{pid,log} and /tmp/mineru_*.{pid,log}
  -n, --dry-run          Print what would be killed, but do not kill
  -h, --help             Show help

Examples:
  # Stop a vLLM pool started by start_vllm_pool.sh
  ./scripts/stop_mineru_vllm.sh --vllm-ports 30000,30001 --clean

  # Stop 3 MinerU servers
  ./scripts/stop_mineru_vllm.sh --mineru-ports 8897,8898,8899 --clean

  # Stop everything that has pidfiles under /tmp (safe default for our scripts)
  ./scripts/stop_mineru_vllm.sh --all-pidfiles --clean

  # If you already killed the vLLM LISTEN pid but GPU memory is still occupied,
  # force-kill EngineCore on the GPUs used by that vLLM instance:
  ./scripts/stop_mineru_vllm.sh --kill-enginecore-gpus 5,6
EOF
}

DRY_RUN=0
CLEAN=0
ALL_PIDFILES=0
VLLM_PORTS=""
MINERU_PORTS=""
ENGINECORE_GPUS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vllm-ports) VLLM_PORTS="${2:-}"; shift 2;;
    --mineru-ports) MINERU_PORTS="${2:-}"; shift 2;;
    --kill-enginecore-gpus) ENGINECORE_GPUS="${2:-}"; shift 2;;
    --all-pidfiles) ALL_PIDFILES=1; shift;;
    --clean) CLEAN=1; shift;;
    -n|--dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

_log() { echo "[$(date +'%F %T')] $*"; }

kill_pgid_or_pid() {
  local pid="$1"
  if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
    return 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  local pgid=""
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  if [[ -n "$pgid" && "$pgid" =~ ^[0-9]+$ ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      _log "DRY kill -TERM -- -${pgid} (pgid for pid=${pid})"
    else
      _log "kill -TERM -- -${pgid} (pgid for pid=${pid})"
      kill -TERM -- "-${pgid}" 2>/dev/null || true
    fi
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      if [[ "$DRY_RUN" == "1" ]]; then
        _log "DRY kill -KILL -- -${pgid} (pgid for pid=${pid})"
      else
        _log "kill -KILL -- -${pgid} (pgid for pid=${pid})"
        kill -KILL -- "-${pgid}" 2>/dev/null || true
      fi
    fi
    return 0
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    _log "DRY kill -KILL ${pid}"
  else
    _log "kill -KILL ${pid}"
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

stop_by_pidfile() {
  local pidfile="$1"
  if [[ ! -f "$pidfile" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  pid="${pid//$'\n'/}"
  if [[ -z "$pid" ]]; then
    return 0
  fi
  _log "pidfile=${pidfile} pid=${pid}"
  kill_pgid_or_pid "$pid"
}

stop_vllm_port() {
  local port="$1"
  port="$(echo "$port" | xargs)"
  [[ -z "$port" ]] && return 0
  _log "Stopping vLLM port=${port}"

  # 1) pidfile from our pool script
  stop_by_pidfile "/tmp/vllm_${port}.pid"

  # 2) any 'vllm serve ... --port <port>' processes
  while read -r pid cmd; do
    [[ -z "$pid" ]] && continue
    _log "matched pid=${pid} cmd=${cmd}"
    kill_pgid_or_pid "$pid"
  done < <(ps -eo pid=,cmd= | rg -S "vllm serve" | rg -S " --port ${port}(\\s|$)" | awk '{pid=$1; $1=""; sub(/^ /,""); print pid, $0}')

  # 3) anything listening on that port (frontends)
  if command -v lsof >/dev/null 2>&1; then
    while read -r pid; do
      [[ -z "$pid" ]] && continue
      _log "LISTEN pid=${pid} (via lsof)"
      kill_pgid_or_pid "$pid"
    done < <(lsof -t -nP -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)
  fi
}

stop_mineru_port() {
  local port="$1"
  port="$(echo "$port" | xargs)"
  [[ -z "$port" ]] && return 0
  _log "Stopping MinerU server port=${port}"

  stop_by_pidfile "/tmp/mineru_${port}.pid"

  while read -r pid cmd; do
    [[ -z "$pid" ]] && continue
    _log "matched pid=${pid} cmd=${cmd}"
    kill_pgid_or_pid "$pid"
  done < <(ps -eo pid=,cmd= | rg -S "mineru_main.py server" | rg -S " --port ${port}(\\s|$)" | awk '{pid=$1; $1=""; sub(/^ /,""); print pid, $0}')

  if command -v lsof >/dev/null 2>&1; then
    while read -r pid; do
      [[ -z "$pid" ]] && continue
      _log "LISTEN pid=${pid} (via lsof)"
      kill_pgid_or_pid "$pid"
    done < <(lsof -t -nP -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)
  fi
}

kill_enginecore_on_gpus() {
  local gpus="$1"
  gpus="$(echo "$gpus" | xargs)"
  [[ -z "$gpus" ]] && return 0
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    _log "nvidia-smi not found; cannot kill EngineCore by GPU."
    return 0
  fi
  _log "Killing VLLM::EngineCore on GPUs: ${gpus}"
  local lines=""
  if ! lines="$(nvidia-smi -i "${gpus}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"; then
    lines=""
  fi
  if [[ -z "$lines" ]]; then
    _log "No compute-app entries found (or NVML error)."
    return 0
  fi
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local pid name mem
    pid="$(echo "$line" | awk -F',' '{print $1}' | xargs)"
    name="$(echo "$line" | awk -F',' '{print $2}' | xargs)"
    mem="$(echo "$line" | awk -F',' '{print $3}' | xargs)"
    if [[ "$name" == "VLLM::EngineCore" ]]; then
      _log "EngineCore pid=${pid} used_mem=${mem}"
      kill_pgid_or_pid "$pid"
    fi
  done <<< "$lines"
}

if [[ "$ALL_PIDFILES" == "1" ]]; then
  for f in /tmp/vllm_*.pid /tmp/mineru_*.pid; do
    [[ -f "$f" ]] || continue
    stop_by_pidfile "$f"
  done
fi

if [[ -n "$VLLM_PORTS" ]]; then
  IFS=',' read -ra ports <<< "$VLLM_PORTS"
  for p in "${ports[@]}"; do
    stop_vllm_port "$p"
  done
fi

if [[ -n "$MINERU_PORTS" ]]; then
  IFS=',' read -ra ports <<< "$MINERU_PORTS"
  for p in "${ports[@]}"; do
    stop_mineru_port "$p"
  done
fi

if [[ -n "$ENGINECORE_GPUS" ]]; then
  kill_enginecore_on_gpus "$ENGINECORE_GPUS"
fi

if [[ "$CLEAN" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    _log "DRY clean /tmp/vllm_*.{pid,log} /tmp/mineru_*.{pid,log}"
  else
    rm -f /tmp/vllm_*.pid /tmp/vllm_*.log /tmp/mineru_*.pid /tmp/mineru_*.log 2>/dev/null || true
  fi
fi

_log "Done."
