#!/bin/bash
#SBATCH --job-name=gen_kitti_db
#SBATCH --partition=prepost
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=./log_out/gen_kitti_db_%j.out
#SBATCH --error=./log_err/gen_kitti_db_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

module purge
conda deactivate

module load pytorch-gpu/py3/1.7.0+hvd-0.21.0

set -x
srun python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval
