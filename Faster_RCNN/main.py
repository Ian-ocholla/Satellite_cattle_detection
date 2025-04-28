import os
from dataset_utils import register_datasets
from config_utils import setup_config
from train import train_model

if __name__ == "__main__":
    # Define dataset name
    dataset_name = "coco_data"

    # Register datasets
    name_ds_train, name_ds_val, name_ds_test = register_datasets(dataset_name)

    # Setup model configuration
    cfg = setup_config(name_ds_train, name_ds_val, name_ds_test)

    # Train model
    trainer = train_model(cfg)
