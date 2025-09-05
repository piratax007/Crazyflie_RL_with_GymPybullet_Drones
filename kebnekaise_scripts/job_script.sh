#!/usr/bin/env bash
#SBATCH -A hpc2n2024-120
#SBATCH -t 48:00:00
#SBATCH -n 1
#SBATCH -c 4

ml purge > /dev/null 2>&1
ml GCC/12.3.0 OpenMPI/4.1.5
ml gym-pybullet-drones/2.0.0-3d7b12edd4915a27e6cec9f2c0eb4b5479f7735e
ml TensorFlow/2.15.1-CUDA-12.1.1

cd ..
echo "Training started at $(date)"

echo "CURRENT PATH $(pwd)"

python3 -m python_scripts.execute_sequential_learning  --environment 'CLStage1S2RE2EDRQuat' \
 --learning-id 'First-Stage_s2r-e2e-dr-quaternion' --algorithm 'ppo' --parallel-environments 4 \
 --time-steps 30000000

echo "Training finished at $(date)"
