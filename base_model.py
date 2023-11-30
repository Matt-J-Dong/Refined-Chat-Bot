import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer #pylint: disable=import-error

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Check if CUDA (GPU support) is available and move the model to GPU if possible
if torch.cuda.is_available():
    model = model.cuda()

# Example text strings
text_examples = [
    "The quick brown fox jumps over the lazy dog",
    "In a galaxy far, far away",
    "To be or not to be, that is the question"
]

# Tokenizing and generating responses for the text examples
for text in text_examples:
    inputs = tokenizer.encode(text, return_tensors="pt")

    # Generate attention mask (1 for real tokens and 0 for padding tokens)
    attention_mask = inputs.ne(tokenizer.pad_token_id).int()

    # Move inputs and attention mask to GPU if available
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        attention_mask = attention_mask.cuda()

    outputs = model.generate(
        inputs,
        max_length=50,
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        verbose=True  # Verbose output
    )

    print(f"Input: {text}\nOutput: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")

