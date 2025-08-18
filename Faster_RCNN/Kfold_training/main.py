import os
from dataset_utils import register_all_kfolds
from config_utils import setup_config_from_fold
from train import train_model

if __name__ == "__main__":

    #path to the kfold dataset root
    KFOLD_BASE = "/projappl/project_2006327/Detectron/2025/Kfold_dataset/train_aug_data/coco_data/Kfolds"

    #step 1: register all folds
    fold_names = register_all_kfolds(KFOLD_BASE)

    #step 2: loop through each fold
    for fold_id in sorted(fold_names.keys()):
        print(f"\n📦 Starting training for Fold {fold_id}...\n")

        train_name = fold_names[fold_id]["train"]
        val_name = fold_names[fold_id]["val"]

        # Step 3: Setup config for this fold
        cfg = setup_config_from_fold(fold_id, num_classes=1, device="cuda", 
                             base_lr=0.0005, max_iter=12000, img_size=(336, 336), use_amp=True, save_config=True)

        # Optional: update output dir per fold
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, f"fold_{fold_id}")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # step 4: Train model
        trainer = train_model(cfg)
        
        print(f"\n✅ Finished training for Fold {fold_id}. Model saved in {cfg.OUTPUT_DIR}\n")
