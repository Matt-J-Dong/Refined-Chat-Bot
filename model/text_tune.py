from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
import os 

device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("./Refined_Chat_Bot")
model = AutoModelForCausalLM.from_pretrained("./Refined_Chat_Bot")
model = model.to(device)
dialogue_dataset_path = "./harry_only_data.txt"

input_context = '''
###Prompt:
Answer as if you are Harry Potter from the novel Harry Potter and the Philosopher’s Stone.
Who is Taylor Swift?

###Assistant:
'''

# Enable gradient computation for input tensor
input_ids = tokenizer.encode(input_context, return_tensors="pt").to(device)
input_ids.requires_grad = True  # Enable gradients

# Read and preprocess the dialogue dataset
with open(dialogue_dataset_path, "r") as dialogue_file:
    dialogue_data = dialogue_file.readlines()

# Create a text dataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=dialogue_dataset_path,
    block_size=128,  # Adjust block size according to your dataset and available memory
)

# Prepare data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We are not using masked language modeling in this example
)


# Set up training arguments
training_args = TrainingArguments(
    output_dir="scratch/mjd9571/singularity/text_tune",
    overwrite_output_dir=True,
    num_train_epochs=20,  # Adjust the number of epochs as needed
    per_device_train_batch_size=4,  # Adjust batch size as needed
    gradient_accumulation_steps=2,
    save_steps=10_000,
    save_total_limit=2,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("model")
tokenizer.save_pretrained("model")