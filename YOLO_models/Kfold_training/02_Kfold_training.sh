#!/bin/bash
#SBATCH --job-name=Lumo_PNEO_yolov5m_raw_aug_BG
#SBATCH --account=project_2006327
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --time=08:15:00
#SBATCH --gres=gpu:v100:1,nvme:10
#SBATCH --output=logs/yolov5_train_BG_%j.log

# 1. Activate Python environment
export PATH="/scratch/project_2006327/YOLOv5_b/my_env/bin:$PATH"

# 2. Load necessary modules
module load git

# 3. Variables
PROJECT_DIR="/scratch/project_2006327/YOLOv5_b"
YOLOV5_DIR="$PROJECT_DIR/yolov5"
DATASET_DIR="$PROJECT_DIR/datasets/Lumo_PNEO_data/Kfold_training"
K_FOLD_DIR="$PROJECT_DIR/datasets/Lumo_PNEO_data/Kfold_training/raw_aug_BG_train/kfold_splits"
WEIGHTS="$YOLOV5_DIR/yolov5m.pt"
NUM_FOLDS=10

# 4. Go to YOLOv5 directory
cd $YOLOV5_DIR

# 5. K-Fold Split + YAML Generation
echo "🔄 Creating K-fold splits and YAML configs..."

python <<EOF
import os
import shutil
from sklearn.model_selection import KFold

image_dir = "$DATASET_DIR/raw_aug_BG_train/images/train"
label_dir = "$DATASET_DIR/raw_aug_BG_train/labels/train"
output_dir = "$K_FOLD_DIR"
os.makedirs(output_dir, exist_ok=True)

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
kf = KFold(n_splits=$NUM_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(image_files)):
    for split, idxs in zip(['train', 'val'], [train_idx, val_idx]):
        img_out = os.path.join(output_dir, f'fold{fold}/images/{split}')
        lbl_out = os.path.join(output_dir, f'fold{fold}/labels/{split}')
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for idx in idxs:
            img = image_files[idx]
            lbl = img.rsplit('.', 1)[0] + '.txt'

            shutil.copy(os.path.join(image_dir, img), os.path.join(img_out, img))
            label_path = os.path.join(label_dir, lbl)
            dest_label_path = os.path.join(lbl_out, lbl)

            # If the label file exists, copy it
            if os.path.exists(label_path):
                shutil.copy(label_path, dest_label_path)
            else:
                # If no label file exists (e.g. background image), create an empty one
                with open(dest_label_path, 'w') as f:
                    pass  # creates an empty .txt file

    # Create YAML file
    with open(os.path.join(output_dir, f'fold{fold}_data.yaml'), 'w') as f:
        f.write(f"path: {output_dir}/fold{fold}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['C']\n")
EOF

echo "✅ Done creating folds and config files."

# 6. Train each fold
for fold in $(seq 0 $(($NUM_FOLDS - 1))); do
    echo "🚀 Training fold $fold..."
    srun python train.py \
           --img 640 \
           --batch 16 \
           --epochs 100 \
           --data "$K_FOLD_DIR/fold${fold}_data.yaml" \
           --weights "$WEIGHTS" \
           --project "$DATASET_DIR/raw_aug_BG_train/runs/train" \
           --name "fold${fold}_v5m" \
           --patience 10 \
           --cache
done

echo "🎉 All folds trained."
