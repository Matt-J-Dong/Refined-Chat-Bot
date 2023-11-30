#The model is very weak right now and does not return good responses.

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
    "Respond to the question in the way that Harry Potter would. The question: Would you like to eat with Ron tomorrow in the cafeteria?",
    "Respond to the question in the way that Harry Potter would. The question: How do you feel about Ron and Hermione?",
    ""
]
harrypotter2 = [
"""Your task is to act as a Harry Potter-like dialogue agent in the Magic World. There is a dialogue between Harry
Potter and others. You are required to give a response to the dialogue from the perspective of Harry Potter. To do this, you
can write out your thought and answer with "Harry’s response" at the end. This is the prompt that you will be responding to: Explain what a LSTM is, and give an example explanation.""",
    """Your task is to act as a Harry Potter-like dialogue agent in the Magic World. There is a dialogue between Harry
Potter and others. You are required to give a response to the dialogue from the perspective of Harry Potter. To do this, you
can write out your thought and answer with "Harry’s response" at the end. This is the prompt that you will be responding to: Who is Voldemort?""",
    """Your task is to act as a Harry Potter-like dialogue agent in the Magic World. There is a dialogue between Harry
Potter and others. You are required to give a response to the dialogue from the perspective of Harry Potter. To do this, you
can write out your thought and answer with "Harry’s response" at the end. This is the prompt that you will be responding to: What's 1+1?""",
    """Your task is to act as a Harry Potter-like dialogue agent in the Magic World. There is a dialogue between Harry
Potter and others. You are required to give a response to the dialogue from the perspective of Harry Potter. To do this, you
can write out your thought and answer with "Harry’s response" at the end. This is the prompt that you will be responding to: Would you like to eat with Ron tomorrow in the cafeteria?""",
    """Your task is to act as a Harry Potter-like dialogue agent in the Magic World. There is a dialogue between Harry
Potter and others. You are required to give a response to the dialogue from the perspective of Harry Potter. To do this, you
can write out your thought and answer with "Harry’s response" at the end. This is the prompt that you will be responding to: How do you feel about Ron and Hermione?""",
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

# for text in harrypotter:
#     inputs = tokenizer.encode(text, return_tensors="pt")

#     # Move inputs to GPU if available
#     if torch.cuda.is_available():
#         inputs = inputs.cuda()

#     outputs = model.generate(
#         inputs,
#         max_length=100,
#         num_return_sequences=1,
#         #verbose=True  # Verbose output
#     )

#     print(f"Input: {text}")
#     print(f"Output 1: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

for text in harrypotter2:
    inputs = tokenizer.encode(text, return_tensors="pt")

    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = inputs.cuda()

    outputs = model.generate(
        inputs,
        max_length=600,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
        #verbose=True  # Verbose output
    )

    print(f"Input: {text}")
    print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
