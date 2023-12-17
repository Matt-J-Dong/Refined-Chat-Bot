#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu
#SBATCH --job-name=llama_test_web
#SBATCH --output=llama_test_web.out

module purge

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

singularity exec $nv \
  --overlay /scratch/mjd9571/singularity/overlay-50G-10M.ext3:rw \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c "source /ext3/env.sh; python /scratch/mjd9571/singularity/llama_test_web.py"