"""
configures detectron2 settings
"""

import os
import pickle
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Set the working directory
WORK_DIR = "/projappl/project_2006327/Detectron/2025/satellite_dataset"  # Change this to your actual project directory
os.chdir(WORK_DIR)  # Change the working directory globally
print(f"Working directory set to: {os.getcwd()}")

def setup_config(name_ds_train, name_ds_val, name_ds_test, num_classes=1, device="cuda",
                base_lr=0.0005, max_iter=15000, img_size=(336, 336), use_amp=True):
    """
    Sets up the Detectron2 configuration file for training.
    
    Arguments:
        - name_ds_train: Dataset name for training.
        - name_ds_val: Dataset name for validation.
        - name_ds_test: Dataset name for testing.
        - num_classes: Number of classes (default 1).
        - device: Device to run the model on (default "cuda").
        - base_lr: Base learning rate (default 0.00005).
        - max_iter: Maximum number of iterations (default 5000).
        - img_size: Image size for training and evaluation (default (336, 336)).
        - use_amp: Whether to enable automatic mixed precision (default True).
    
    Returns:
        - cfg: The final Detectron2 configuration object.
    """
    # Create an output directory specific to the dataset
    output_dir = os.path.join(WORK_DIR, "output/A001", name_ds_test)
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    
    output_cfg_path = os.path.join(output_dir, "cfg.pickle")
    os.makedirs(os.path.dirname(output_cfg_path), exist_ok=True)  # Ensure the directory exists

    # Load configuration
    cfg = get_cfg()
    config_file_url = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    #config_file_url = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_file_url))  

    # Set model weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file_url)

    # modifying the rpn.min_size for small objects
    cfg.MODEL.RPN.MIN_SIZE = 1
    cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 1

    #increase the feature map resolution
    """
    Adjust the stride of the backbone and FPN, smaller strides results in higher resolution feature maps
    """
    
    cfg.MODEL.FPN.MIN_LEVEL = 2  # Start using lower-level features
    cfg.MODEL.FPN.MAX_LEVEL = 5  # Maintain high-res levels
    cfg.MODEL.BACKBONE.FREEZE_AT = 2  # Keep early layers trainable
    
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 256  # Boost lowest level features
    cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 128

    #Modification in the Region Proposal Density
    cfg.MODEL.RPN_NMS_THRESH = 0.2 # lower threshold to keep more objects
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
    
    cfg.RPN_POST_NMS_TOP_N_TRAIN = 2000
    cfg.RPN_POST_NMS_TOP_N_TEST = 1000

    # Set datasets
    cfg.DATASETS.TRAIN = (name_ds_train,)
    cfg.DATASETS.TEST = (name_ds_val,)

    # Training hyperparameters
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.MIN_SIZE_TRAIN = 1024 #img_size[0] #(336,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024 #img_size[1] #336
    cfg.INPUT.MIN_SIZE_TEST = 1024 #img_size[0] #336
    cfg.INPUT.MAX_SIZE_TEST = 1024 #img_size[1] #336
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.SOLVER.GAMMA = 0.1 #change this from 0.5 to 0.1
    cfg.SOLVER.BASE_LR = base_lr #coco is 0.00025
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"  # Cosine LR decay
    
    #Add gradient clipping tostabilize training
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.AMP.ENABLED = use_amp  # Use mixed precision for stability

    #use focal loss instead of cross entropy- small object s often have low detection condifecne due to class imabalence
    cfg.MODEL.ROI_HEADS.SMOOTH_L1_BETA = 0.1
    cfg.MODEL.ROI_HEADS.LOSS_TYPE = "focal" #replace the standatd loss with focal loss

    #improving ROI pooling (feature alignment for small objects)
    ## use a finer-grained ROI align with a higher resolution (e.g 7x7  to 14x14
    cfg.MODEL.ROI_HEADS.POOLER_RESOLUTION = 14  # Default: 7

    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.NESTROV = False
    cfg.SOLVER.STEPS = (1500, 2500)  # 🔹 Now within range (≤ MAX_ITER)
    cfg.SOLVER.CHECKPOINT_PERIOD = 250
    cfg.TEST.EVAL_PERIOD = cfg.SOLVER.CHECKPOINT_PERIOD

    # anchors
    #cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[43.48403622, 29.42209169, 36.92428582, 52.30700844, 22.32369148]]
    #test anchor for small objects
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[1, 2, 4, 8]] # remove 128 and 256
    
    #Test aspect ratios
    #cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.56776786, 1.60585184, 0.95999676]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 0.5, 1.0, 2.0]]
    
    # pixels means and standard deviations
    cfg.MODEL.PIXEL_MEAN = [ 121.27864708,  125.98920839, 114.19349952]
    cfg.MODEL.PIXEL_STD = [21.4458683,  24.23267415, 31.24573202]

    # Model classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir

    # Save configuration
    with open(output_cfg_path, "wb") as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    return cfg
