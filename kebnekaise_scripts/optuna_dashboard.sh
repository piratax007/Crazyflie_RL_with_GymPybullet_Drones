#!/bin/bash -l

set -euo pipefail

BASE_DIR="/proj/nobackup/kebne_ltu_rmm/Fausto"
REPO_DIR="$BASE_DIR/Crazyflie_RL_with_GymPybullet_Drones"
VENV_ACTIVATE="$BASE_DIR/optuna_venv/bin/activate"
DB_DIR="$REPO_DIR/results/optuna_db"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

usage() {
  echo "Usage: $(basename "$0") <db-name-or-path>"
  exit 1
}

resolve_db_file() {
  local input="$1"
  if [[ -f "$input" ]]; then
    readlink -f "$input"
  else
    local name="${input%.db}.db"
    readlink -f "$DB_DIR/$name"
  fi
}

load_env() {
  module load GCC/12.3.0 OpenMPI/4.1.5 tensorboard/2.15.1 Optuna/3.5.0
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
}

run_dashboard() {
  local db_file="$1"
  cd "$REPO_DIR"
  echo "DB: $db_file"
  echo "Dashboard: http://$HOST:$PORT (Ctrl-C to stop)"
  optuna-dashboard "sqlite:////$db_file" --host "$HOST" --port "$PORT"
}

main() {
  [[ $# -ge 1 ]] || usage
  load_env
  local db_file
  db_file="$(resolve_db_file "$1")"
  [[ -f "$db_file" ]] || { echo "Error: DB not fount -> $db_file" >&2; exit 2; }
  run_dashboard "$db_file"
}

main "$@"