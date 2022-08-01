#!/bin/bash
#SBATCH --job-name=test_pp
#SBATCH --partition=prepost
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:07:00
#SBATCH --output=idr_log/test_pp_%j.out
#SBATCH --error=idr_log/test_pp_%j.err

#SBATCH --hint=nomultithread

module purge
conda deactivate

module load pytorch-gpu/py3/1.7.0+hvd-0.21.0

set -x
srun python -u train.py --cfg_file cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml --batch_size 3 \
    --extra_tag test_pp_div8_bs3 --fix_random_seed
