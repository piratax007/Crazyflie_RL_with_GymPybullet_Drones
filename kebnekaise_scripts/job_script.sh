#!/usr/bin/env bash
#SBATCH -A hpc2n2024-120
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -c 1

ml purge > /dev/null 2>&1
ml GCC/12.3.0 OpenMPI/4.1.5
ml gym-pybullet-drones/2.0.0-3d7b12edd4915a27e6cec9f2c0eb4b5479f7735e
ml TensorFlow/2.15.1-CUDA-12.1.1

cd ..
echo "Training started at $(date)"

python3 -m python_scripts.execute_sequential_learning  --environment 'EjcCLStage2' \
 --learning-id 'EJC_stage-2_90' \
--algorithm 'ppo' --parallel-environments 4 --seed 90 --time-steps 30000000 --continuous-learning True

echo "Training finished at $(date)"
