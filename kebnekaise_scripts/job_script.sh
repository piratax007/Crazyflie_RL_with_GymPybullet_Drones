#!/usr/bin/env bash
#SBATCH -A hpc2n2025-242
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

python3 -m python_scripts.execute_sequential_learning  --environment 'JournalStage2Euler' \
 --learning-id 'Journal_Stage2_Euler_noHPO_Noise_TargetAttitudeError075_CylinderR2H2' \
 --continuous-learning True --path-to-previous-model 'results/save-Journal_Stage1_AttitudeRewardWeight075_Euler_noHPO_Noise-12.19.2025_14.50.10' \
 --algorithm 'ppo' --parallel-environments 4 --time-steps 16000000

echo "Training finished at $(date)"
