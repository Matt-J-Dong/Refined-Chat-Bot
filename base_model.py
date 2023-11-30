import numpy
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
harrypotter = [
    "Respond to the question in the way that Harry Potter would. The question: Explain what a LSTM is, and give an example explanation.",
    "Respond to the question in the way that Harry Potter would. The question: Who is Voldemort?",
    "Respond to the question in the way that Harry Potter would. The question: What's 1+1?",
    "Respond to the question in the way that Harry Potter would. The question: Would you like to eat with Ron tomorrow in the cafeteria?"
]
# Tokenizing and generating responses for the text examples
# for text in text_examples:
#     inputs = tokenizer.encode(text, return_tensors="pt")

#     # Move inputs to GPU if available
#     if torch.cuda.is_available():
#         inputs = inputs.cuda()

#     outputs = model.generate(
#         inputs,
#         max_length=50,
#         num_return_sequences=1,
#         #verbose=True  # Verbose output
#     )

#     print(f"Input: {text}\nOutput: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")

for text in harrypotter:
    inputs = tokenizer.encode(text, return_tensors="pt")

    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = inputs.cuda()

    outputs = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        #verbose=True  # Verbose output
    )

    print(f"Input: {text}")
    print(f"Output 1: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    #print(f"Output 2: {tokenizer.decode(outputs[1], skip_special_tokens=True)}")
