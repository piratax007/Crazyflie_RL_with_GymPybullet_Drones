#!/usr/bin/env bash
#SBATCH -A hpc2n2024-120
#SBATCH -t 00:20:00
#SBATCH -n 1
#SBATCH --job-name=Crazyflie_training

module purge
module load GCC/12.3.0 OpenMPI/4.1.5 Stable-Baselines3/2.3.2 gym-pybullet-drones/2.0.0-3d7b12e-CUDA-12.1.1
module load GCC/12.3.0 tensorboard/2.15.1

cd ..

python3 -m python_scripts.execute_secuencial_learning.py
