#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-442 -p alvis
#SBATCH -t 48:00:00
#SBATCH -n 1
#SBATCH -C NOGPU

ml purge > /dev/null 2>&1
ml GCC/12.3.0 OpenMPI/4.1.5
ml gym-pybullet-drones/2.0.0-3d7b12e
ml TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1

cd ..
echo "Training started at $(date)"

python3 -m python_scripts.execute_secuencial_learning  --environment 'EjcCLStage2' --learning-id 'EJC_Agent-B_90_CPU' \
--algorithm 'ppo' --parallel-environments 1 --seed 90 --time-steps 100000000 --stop-on-reward-threshold-flag Fase

echo "Training finished at $(date)"