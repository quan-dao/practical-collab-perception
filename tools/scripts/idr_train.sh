#!/bin/bash
#SBATCH --job-name=train
#SBATCH --qos=qos_gpu-t3

##SBATCH --partition=gpu_p2
##SBATCH -C v100-32g

#SBATCH --output=idr_log/train_%j.out
#SBATCH --error=idr_log/train_%j.err

#SBATCH --time=07:00:00                                 # check the TIME limit TRAINING != EVAL

#SBATCH --nodes=1                                       # comment this to run on multi-node
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4                             # gpu_p2 has 8 GPUs/node, default has 4 GPUs/node
#SBATCH --gres=gpu:4                                    # gpu_p2 has 8 GPUs/node, default has 4 GPUs/node

#SBATCH --cpus-per-task=10                               # for gpu_p2 set to 3, for default set to 10

#SBATCH --hint=nomultithread

module purge
conda deactivate

module load pytorch-gpu/py3/1.7.0+hvd-0.21.0

set -x
srun python -u train.py --launcher slurm --cfg_file cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml --batch_size 24 \
    --sync_bn --fix_random_seed \
    --extra_tag test_div8
