# Models

## LLaMA:
llama_finetune.sbatch and llama_finetuned_test.py are the two relevant files.
The autotrain command was generally run through the interactive srun on HPC.


## Falcon:
saves to/loads the `adapter_config.json`, `adapter_model.bin`, `special_tokens_map.json`, `tokenizer.json`, and `tokenizer_config.json` files after finetuning.
