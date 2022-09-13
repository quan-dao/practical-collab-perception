#!/bin/bash
#SBATCH --job-name=gen_nusc_db
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=./tools/idr_log/gen_nusc_db_%j.out
#SBATCH --error=./tools/idr_log/gen_nusc_db_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1                                       # comment this to run on multi-node
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1                             # gpu_p2 has 8 GPUs/node, default has 4 GPUs/node
#SBATCH --gres=gpu:1                                    # gpu_p2 has 8 GPUs/node, default has 4 GPUs/node

#SBATCH --cpus-per-task=10                               # for gpu_p2 set to 3, for default set to 10

#SBATCH --hint=nomultithread


module purge
conda deactivate

module load pytorch-gpu/py3/1.7.0+hvd-0.21.0

set -x
srun python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval
