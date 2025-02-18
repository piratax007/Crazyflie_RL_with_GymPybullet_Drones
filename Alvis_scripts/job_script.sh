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

python3 -m python_scripts.execute_sequential_learning  --environment 'BasicRewardStage2' --learning-id \
'EJC_stage-2_Basic-Reward_80' \
--algorithm 'ppo' --parallel-environments 4 --seed 80 --time-steps 30000000 --continuous-learning True

echo "Training finished at $(date)"