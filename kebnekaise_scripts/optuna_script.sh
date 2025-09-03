#!/usr/bin/env bash
#SBATCH -A hpc2n2024-120
#SBATCH -t 168:00:00
#SBATCH -n 1
#SBATCH -c 8

ml purge > /dev/null 2>&1
ml GCC/12.3.0 OpenMPI/4.1.5
ml gym-pybullet-drones/2.0.0-3d7b12edd4915a27e6cec9f2c0eb4b5479f7735e
ml TensorFlow/2.15.1-CUDA-12.1.1
ml Optuna/3.5.0

cd ..
echo "Optimization started at $(date)"

echo "CURRENT PATH $(pwd)"

python3 -m python_scripts.optimize_ppo --env_id 'CLStage1Sim2RealDomainRandomization' \
 --study_name 'Big_Domain_Randomization' --storage sqlite:///optuna_cf.db \
 --total_timesteps 30000000 --eval_freq 10000 --n_trials 200 \
 --tpe_constant_liar

echo "Optimization finished at $(date)"
