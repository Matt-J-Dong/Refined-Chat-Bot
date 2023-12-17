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

singularity exec $nv \
  --overlay /scratch/mjd9571/singularity/overlay-50G-10M.ext3:rw \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c "source /ext3/env.sh; autotrain llm --train --project_name 'Refined_Chatbot' --model 'abhishek/llama-2-7b-hf-small-shards' --data_path . --use_peft --use_int4 --learning_rate 2e-4 --train_batch_size 2 --num_train_epochs 9 --trainer sft > training.log &"