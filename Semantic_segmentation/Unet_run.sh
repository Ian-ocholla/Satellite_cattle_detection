#!/bin/bash
#SBATCH --job-name=Unet_Lumo_PNEO3_2022Jan_unet_b82
#SBATCH --account=project_2008354
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=85G
#SBATCH --time=10:15:00
##SBATCH --mail-type=BEGIN 
#SBATCH --gres=gpu:v100:1,nvme:10

export PATH="/scratch/project_2008354/Cattle_UNet/myenv/bin:$PATH"

export GIT_PYTHON_REFRESH=quiet
## Distibuted training issue on HPC
export TF_ENABLE_ONEDNN_OPTS=0 

## remve verbosity of the warning that are not errors
export TF_CPP_MIN_LOG_LEVEL=2

## Disable XLA explicitly by setting the environment variable
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"


cd /scratch/project_2008354/Cattle_UNet/py_codes/Lumo_PNEO3_GPUs

srun python unet_model.py