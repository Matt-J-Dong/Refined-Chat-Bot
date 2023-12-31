#!/bin/bash

autotrain llm \
--train \
--model 'abhishek/llama-2-7b-hf-small-shards' \
--project-name 'Refined_Chat_Bot' \
--data-path './data' \
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
--peft
