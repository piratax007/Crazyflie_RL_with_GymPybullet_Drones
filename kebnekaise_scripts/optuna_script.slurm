#!/usr/bin/env bash
#SBATCH -A hpc2n2025-242
#SBATCH -J optuna-ppo-cf
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks=10                        # parallel Optuna workers (processes)
#SBATCH --cpus-per-task=4                 # CPU cores per worker
#SBATCH --mem=0                           # full node memory (or set e.g. 64G)
# #SBATCH --gres=gpu:1                    # uncomment & tune if using GPUs
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

cleanup() {
  if [ -n "${SNAP_PID:-}" ] && kill -0 "{SNAP_PID}" 2>/dev/null; then
    kill "${SNAP_PID}" 2>/dev/null || true
    wait "${SNAP_PID}" 2>/dev/null || true
  fi
  jobs -pr | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

########################################
# Repo root detection via SLURM_SUBMIT_DIR
########################################
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${SUBMIT_DIR}"
if [ -d "${SUBMIT_DIR}/python_scripts" ]; then
  REPO_ROOT="${SUBMIT_DIR}"
elif [ -d "${SUBMIT_DIR}/../python_scripts" ]; then
  REPO_ROOT="$(cd "${SUBMIT_DIR}/.." >/dev/null 2>&1 && pwd)"
else
  echo "[ERROR] Could not find repo root: expected 'python_scripts' in ${SUBMIT_DIR} or its parent." >&2
  exit 2
fi
cd "${REPO_ROOT}"
echo "Repo root: ${REPO_ROOT}"

########################################
# Kebnekaise modules
########################################
ml purge > /dev/null 2>&1
ml GCC/12.3.0 OpenMPI/4.1.5
ml gym-pybullet-drones/2.0.0-3d7b12edd4915a27e6cec9f2c0eb4b5479f7735e
ml TensorFlow/2.15.1-CUDA-12.1.1
ml Optuna/3.5.0

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

########################################
# Run config (module mode)
########################################
OPT_MODULE="python_scripts.optimize_ppo"  # entrypoint: python -m python_scripts.optimize_ppo

# Crazyflie env + study
ENV_ID="JournalStage3Euler"
STUDY_NAME="Journal_WCL_Euler"

# Global trial budget across ALL workers
TOTAL_TRIALS=100

# PPO training/eval params
TOTAL_TIMESTEPS=20000000
EVAL_FREQ=10000
N_EVAL_EPISODES=10
N_ENVS=4
POLICY="MlpPolicy"
SEED_BASE=1000

# Logging
LOG_DIR="${REPO_ROOT}/results/optuna_logs"
mkdir -p "${LOG_DIR}"

# SQLite DB on node-local scratch for safe concurrency
NODE_SCRATCH="${SNIC_TMP:-${TMPDIR:-/tmp}}"
mkdir -p "${NODE_SCRATCH}/optuna_cf_db"
DB_PATH="${NODE_SCRATCH}/optuna_cf_db/optuna_cf.db"
STORAGE_URL="sqlite:///${DB_PATH}?timeout=120"

# Where to publish a live-updating copy for the dashboard
DEST_DB="${REPO_ROOT}/results/optuna_db/optuna_cf_${SLURM_JOB_ID}.db"
mkdir -p "$(dirname "$DEST_DB")"

# Background: take consistent snapshots every 2 minutes
( while true; do
    # Create an atomic, consistent copy using sqlite3 .backup (safe while writers run)
    sqlite3 "${DB_PATH}" ".backup '${DEST_DB}.tmp'"
    mv -f "${DEST_DB}.tmp" "${DEST_DB}"
    sleep 120
  done ) >/dev/null 2>&1 &

# Optuna sampler/pruner knobs
N_STARTUP_TRIALS=5
N_EVALUATIONS=4
USE_CONSTANT_LIAR=1   # 1 -> add --tpe_constant_liar

# Export for Python pre-init
export STORAGE_URL
export STUDY_NAME

########################################
# Pre-initialize SQLite & Optuna study (avoids “table exists” races)
########################################
python3 - <<'PY'
import os, sqlite3, pathlib, optuna
storage_url = os.environ["STORAGE_URL"]
study_name  = os.environ["STUDY_NAME"]

if storage_url.startswith("sqlite:///"):
    db_spec = storage_url[len("sqlite:///"):]
    db_path = db_spec.split("?", 1)[0]
    p = pathlib.Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(str(p))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=120000;")
        conn.close()
        print(f"[pre-init] SQLite ready at {p}")
    except Exception as e:
        print(f"[pre-init][WARN] SQLite PRAGMAs failed: {e}")
optuna.create_study(storage=storage_url, study_name=study_name,
                    direction="maximize", load_if_exists=True)
print(f"[pre-init] Optuna study initialized: {study_name}")
PY

########################################
# Split trials per worker
########################################
NTASKS="${SLURM_NTASKS:-1}"
TRIALS_EACH=$(( TOTAL_TRIALS / NTASKS ))
REMAINDER=$(( TOTAL_TRIALS % NTASKS ))

echo "Workers: ${NTASKS} | Total trials: ${TOTAL_TRIALS} -> per worker: ${TRIALS_EACH} (+1 for first ${REMAINDER})"
echo "DB path: ${DB_PATH}"

PIDS=""
RANK=0
while [ "${RANK}" -lt "${NTASKS}" ]; do
  WORKER_TRIALS="${TRIALS_EACH}"
  if [ "${RANK}" -lt "${REMAINDER}" ]; then
    WORKER_TRIALS=$((TRIALS_EACH + 1))
  fi

  EXTRA_FLAG=""
  if [ "${USE_CONSTANT_LIAR}" -eq 1 ]; then
    EXTRA_FLAG="--tpe_constant_liar"
  fi

  echo "Launching worker ${RANK} with ${WORKER_TRIALS} trials..."
  # Stagger starts slightly to reduce SQLite connect bursts
  sleep $((RANK % 3))

  srun --kill-on-bad-exit=1 --exclusive -N1 -n1 \
    python3 -m "${OPT_MODULE}" \
      --env_id "${ENV_ID}" \
      --study_name "${STUDY_NAME}" \
      --storage "${STORAGE_URL}" \
      --total_timesteps "${TOTAL_TIMESTEPS}" \
      --eval_freq "${EVAL_FREQ}" \
      --n_eval_episodes "${N_EVAL_EPISODES}" \
      --n_envs "${N_ENVS}" \
      --policy "${POLICY}" \
      --log_dir "${LOG_DIR}" \
      --n_startup_trials "${N_STARTUP_TRIALS}" \
      --n_evaluations "${N_EVALUATIONS}" \
      --n_trials "${WORKER_TRIALS}" \
      --n_jobs 1 \
      --seed $((SEED_BASE + RANK)) \
      ${EXTRA_FLAG} \
    > "worker_${SLURM_JOB_ID:-job}_${RANK}.out" 2>&1 &
  PIDS="${PIDS} $!"

  RANK=$((RANK + 1))
done

wait
for pid in $PIDS; do
  wait "$pid"
done
echo "All workers finished."

# Persist DB if created
if [ -f "${DB_PATH}" ]; then
  DEST_DB="${REPO_ROOT}/optuna_cf_${SLURM_JOB_ID:-manual}.db"
  cp -f "${DB_PATH}" "${DEST_DB}" || true
  echo "Copied Optuna DB to: ${DEST_DB}"
  echo "Run:  optuna-dashboard sqlite:///${DEST_DB} --study ${STUDY_NAME}"
else
  echo "[WARN] DB file not found at ${DB_PATH} (workers may have failed early)."
fi
