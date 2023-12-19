#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu
#SBATCH --job-name=llama_finetune
#SBATCH --output=llama_finetune.out

module purge

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

singularity exec --nv \
  --overlay /scratch/mjd9571/singularity/overlay-50G-10M.ext3:rw \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c "source /ext3/env.sh; autotrain llm \
--train \
--model 'abhishek/llama-2-7b-hf-small-shards' \
--project-name 'Refined_Chat_Bot' \
--data-path './' \
--text-column "text" \
--lr 2e-3 \
--batch-size 32 \
--epochs 8 \
--block-size 1024 \
--warmup-ratio 0.01 \
--lora-r 16 \
--lora-alpha 32 \
--lora-dropout 0.05 \
--weight-decay 0.01 \
--gradient-accumulation 4 \
--quantization "int4" \
--mixed-precision "fp16" \
--peft"

