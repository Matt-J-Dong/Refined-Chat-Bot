from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained('./Refined_Chat_Bot')
model = AutoModelForCausalLM.from_pretrained('./Refined_Chat_Bot')

# Move the model to the GPU if available
model.to(device)

input_context = '''
###Prompt:
Explain what a LSTM is, and give an example explanation.
Answer as if you are Harry Potter responding to the previous line.
'''

input_ids = tokenizer.encode(input_context, return_tensors="pt")
input_ids = input_ids.to(device)
model.eval()

try:
    # Generate the output
    output = model.generate(input_ids, max_length=200, temperature=0.9, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f'The generated text:\n{generated_text}')
except Exception as e:
    print(f'An error occurred: {e}')