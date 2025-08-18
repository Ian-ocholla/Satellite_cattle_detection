#!/bin/bash
#SBATCH --job-name=Lumo_PNEO_yolov5m_kfolds
#SBATCH --account=project_2006327
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --time=09:15:00
#SBATCH --gres=gpu:v100:1,nvme:10
#SBATCH --output=logs/Kfolds_aug_train_03_%j.log

# 1. Activate Python environment
export PATH="/scratch/project_2006327/YOLOv5_b/my_env/bin:$PATH"

# 2. Load necessary modules
module load git

# 3. Variables
PROJECT_DIR="/scratch/project_2006327/YOLOv5_b"
YOLOV5_DIR="$PROJECT_DIR/yolov5"
DATASET_DIR="$PROJECT_DIR/datasets/Lumo_PNEO_data/Kfold_training"
K_FOLD_DIR="$PROJECT_DIR/datasets/Lumo_PNEO_data/Kfold_training/kfolds_aug"
WEIGHTS="$YOLOV5_DIR/custom_aerial_weights.pt"
NUM_FOLDS=10

# 4. Go to YOLOv5 directory
cd $YOLOV5_DIR

# 5. Train each fold
echo "🚀 Starting training for each fold..."

for fold in $(seq 0 $(($NUM_FOLDS - 1))); do
    echo "🚀 Training fold $fold..."
    srun python train.py \
           --img 640 \
           --batch 16 \
           --epochs 100 \
           --data "$K_FOLD_DIR/fold${fold}_data.yaml" \
           --weights "$WEIGHTS" \
           --project "$DATASET_DIR/kfolds_aug/runs/MTL_train" \
           --name "fold${fold}_v5m" \
           --patience 10 \
           --cache
done

echo "🎉 All folds have been trained."
