#!/usr/bin/env bash
#SBATCH -A hpc2n2024-120
#SBATCH -t 3-12:00:00
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gpus=2

module purge
module load GCC/12.3.0 OpenMPI/4.1.5 Stable-Baselines3/2.3.2 gym-pybullet-drones/2.0.0-3d7b12edd4915a27e6cec9f2c0eb4b5479f7735e
module load GCC/12.3.0 OpenMPI/4.1.1 tensorboard/2.15.1
module load GCC/12.3.0 OpenMPI/4.1.5 Optuna/3.5.0

cd ..

python3 -m python_scripts.fine_tuning