from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'NousResearch/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
from datasets import load_dataset
import os

dataset_path = '/content/drive/MyDrive/harry_only_data.txt'  # Replace with your dataset path

# Function to tokenize and format each line of dialogue
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# Load the dataset from the text file
dataset = load_dataset('text', data_files={'train': dataset_path})

# Apply the preprocessing function to tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)
trainer.train()
model.save_pretrained("./fine_tuned_model")