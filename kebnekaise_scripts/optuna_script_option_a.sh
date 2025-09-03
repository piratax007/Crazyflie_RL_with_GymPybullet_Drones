#!/bin/bash
#SBATCH -A hpc2n2024-120
#SBATCH -J optuna-ppo-cf
#SBATCH -t 168:00:00
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
# #SBATCH --gres=gpu:1                    # uncomment and tune if using GPUs
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

ml purge > /dev/null 2>&1
ml GCC/12.3.0 OpenMPI/4.1.5
ml gym-pybullet-drones/2.0.0-3d7b12edd4915a27e6cec9f2c0eb4b5479f7735e
ml TensorFlow/2.15.1-CUDA-12.1.1
ml Optuna/3.5.0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

#############################
# User-configurable params
#############################
# Path to your optimization script (the one you run manually)
# If your repo has it as a module (python -m python_scripts.optimize_ppo),
# set OPT_SCRIPT to that file path (e.g., ${PWD}/python_scripts/optimize_ppo.py)
OPT_SCRIPT="${PWD}/python_scripts/optimize_ppo.py"

# Crazyflie env + study
ENV_ID="CLStage1S2RE2EDRQuat"
STUDY_NAME="Domain_Randomization_Quaternion"

# Global trial budget across ALL workers
TOTAL_TRIALS=200

# PPO training/eval params
TOTAL_TIMESTEPS=30000000
EVAL_FREQ=10000
N_EVAL_EPISODES=10
N_ENVS=1
POLICY="MlpPolicy"
SEED_BASE=1000

# Logging (TensorBoard etc.) to shared/project storage
LOG_DIR="${PWD}/results/optuna_logs"

# SQLite DB on node-local scratch for safe concurrency
NODE_SCRATCH="${SNIC_TMP:-${TMPDIR:-/tmp}}"
mkdir -p "${NODE_SCRATCH}/optuna_cf_db"
DB_PATH="${NODE_SCRATCH}/optuna_cf_db/optuna_cf.db"
# Increase SQLite timeout to reduce 'database is locked' errors
STORAGE_URL="sqlite:///${DB_PATH}?timeout=120"

# Optuna sampler/pruner knobs
N_STARTUP_TRIALS=10
N_EVALUATIONS=2
USE_CONSTANT_LIAR=1   # 1 to enable --tpe_constant_liar

#############################
# Derived: split trials per worker
#############################
NTASKS=${SLURM_NTASKS:-1}
TRIALS_EACH=$(( TOTAL_TRIALS / NTASKS ))
REMAINDER=$(( TOTAL_TRIALS % NTASKS ))

echo "Job ${SLURM_JOB_ID:-N/A} starting on $(hostname)"
echo "Node-local scratch: ${NODE_SCRATCH}"
echo "Optuna DB: ${DB_PATH}"
echo "Workers: ${NTASKS}  | total trials: ${TOTAL_TRIALS}  -> per worker: ${TRIALS_EACH} (+1 for first ${REMAINDER})"
echo "Storage URL: ${STORAGE_URL}"

# Start N workers; each grabs distinct trials from the same study/storage.
for RANK in $(seq 0 $((NTASKS-1))); do
  # Distribute remainder: first $REMAINDER workers do one extra trial
  WORKER_TRIALS=${TRIALS_EACH}
  if [ "${RANK}" -lt "${REMAINDER}" ]; then
    WORKER_TRIALS=$((TRIALS_EACH + 1))
  fi

  # Ensure log dir exists
  mkdir -p "${LOG_DIR}"

  # Build args — this matches your underscore-style CLI
  COMMON_ARGS=(
    --env_id "${ENV_ID}"
    --study_name "${STUDY_NAME}"
    --storage "${STORAGE_URL}"
    --total_timesteps "${TOTAL_TIMESTEPS}"
    --eval_freq "${EVAL_FREQ}"
    --n_eval_episodes "${N_EVAL_EPISODES}"
    --n_envs "${N_ENVS}"
    --policy "${POLICY}"
    --log_dir "${LOG_DIR}"
    --n_startup_trials "${N_STARTUP_TRIALS}"
    --n_evaluations "${N_EVALUATIONS}"
    --n_trials "${WORKER_TRIALS}"
    --n_jobs 1
    --seed $((SEED_BASE + RANK))
  )

  if [ "${USE_CONSTANT_LIAR}" -eq 1 ]; then
    COMMON_ARGS+=( --tpe_constant_liar )
  fi

  echo "Launching worker ${RANK} with ${WORKER_TRIALS} trials..."
  srun --exclusive -N1 -n1 \
    python3 "${OPT_SCRIPT}" "${COMMON_ARGS[@]}" \
    > "worker_${SLURM_JOB_ID:-job}_${RANK}.out" 2>&1 &

done

wait
echo "All workers finished."

# Persist the DB artifact to shared storage for dashboarding later
DEST_DB="${PWD}/optuna_cf_${SLURM_JOB_ID:-manual}.db"
cp -f "${DB_PATH}" "${DEST_DB}" || true
echo "Copied Optuna DB to: ${DEST_DB}"
echo "Run:  optuna-dashboard sqlite:///${DEST_DB} --study ${STUDY_NAME}"
