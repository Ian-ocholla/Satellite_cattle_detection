#!/bin/bash
#SBATCH --job-name=yolov5_run_each_fold
#SBATCH --account=project_2006327
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/run_each_folds_%j.log

# === Load environment ===
export PATH="/scratch/project_2006327/YOLOv5_b/my_env/bin:$PATH"
module load git

##Automatically collects the best weights and run on the images, to get the txt files with predicted coordinates and confidence level

# === Define variables ===
PROJECT_DIR="/scratch/project_2006327/YOLOv5_b"
YOLOV5_DIR="$PROJECT_DIR/yolov5"
TEST_DIR="$PROJECT_DIR/datasets/Lumo_PNEO_data/Test_data_render"
DATASET_DIR="$PROJECT_DIR/datasets/Lumo_PNEO_data/Kfold_training/kfolds_aug"
FOLDS_DIR="$DATASET_DIR/runs/MTL_train"
TEST_IMAGES="$TEST_DIR/Kapiti_WV2_data/images/test"
## DATA_YAML="$YOLOV5_DIR/data/Kfold_raw_aug_val.yaml"
RESULTS_ROOT="$DATASET_DIR/runs/detect_MTL/Kapiti_WV2_data/pred_folds_iou05"


# === Go to YOLOv5 directory ===
cd "$YOLOV5_DIR"

# === List of fold names ===
FOLDS=("fold0_v5m" "fold1_v5m" "fold2_v5m" "fold3_v5m" "fold4_v5m" "fold5_v5m" "fold6_v5m" "fold7_v5m" "fold8_v5m" "fold9_v5m")

# === Loop over folds ===
for FOLD in "${FOLDS[@]}"; do
    echo "🚀 Running inference for $FOLD..."

    WEIGHT_PATH="$FOLDS_DIR/$FOLD/weights/best.pt"
    OUT_DIR="$RESULTS_ROOT/$FOLD"

    python detect.py \
        --weights "$WEIGHT_PATH" \
        --img 640 \
        --conf 0.25 \
        --iou 0.5 \
        --source "$TEST_IMAGES" \
        --save-txt \
        --save-conf \
        --nosave \
        --project "$RESULTS_ROOT" \
        --name "$FOLD" \
        --exist-ok

    echo "✅ Saved predictions for $FOLD to $OUT_DIR"
done

echo "🎉 All folds completed."
