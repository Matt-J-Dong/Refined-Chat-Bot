from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Example text strings
text_examples = [
    "The quick brown fox jumps over the lazy dog",
    "In a galaxy far, far away",
    "To be or not to be, that is the question"
]

# Tokenizing and generating responses for the text examples
for text in text_examples:
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    print(f"Input: {text}\nOutput: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")
