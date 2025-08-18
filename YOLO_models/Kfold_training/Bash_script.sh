#!/bin/bash
#SBATCH --job-name=yolov5_WBF_ensemble
#SBATCH --account=project_2006327
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/wbf_ensemble_%j.log

# === Load environment ===
export PATH="/scratch/project_2006327/YOLOv5_b/my_env/bin:$PATH"
module load git

cd /scratch/project_2006327/YOLOv5_b/datasets/Lumo_PNEO_data/Kfold_training

srun python 04_WBF_ensemble.py
