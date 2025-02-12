#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-442 -p alvis
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gpus-per-node=A40:1

ml purge > /dev/null 2>&1
ml GCC/12.3.0 OpenMPI/4.1.5
ml gym-pybullet-drones/2.0.0-3d7b12e
ml tensorboard/2.15.1-gfbf-2023a

cd ..

python3 -m python_scripts.execute_secuencial_learning --learning-id 'EJC_CL_Stage1' --seed 90
