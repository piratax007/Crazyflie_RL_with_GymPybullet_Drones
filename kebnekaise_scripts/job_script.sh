#!/usr/bin/env bash
#SBATCH -A hpc2n2024-120
#SBATCH -t 00:20:00
#SBATCH -n 1
#SBATCH --job-name=hello_gympybullets_in_kebnekaise

module purge
module load GCC/12.3.0 OpenMPI/4.1.5 Stable-Baselines3/2.3.2 gym-pybullet-drones/2.0.0-3d7b12edd4915a27e6cec9f2c0eb4b5479f7735e

cp -r ../environments/HoverCrazyflieSim2Real.py $TMPDIR
cp -r ../python_scripts/learning_script.py $TMPDIR
cp -r ../python_scripts/execute_secuencial_learning.py $TPMDIR

cd $TMPDIR

python3 execute_secuencial_learning.py