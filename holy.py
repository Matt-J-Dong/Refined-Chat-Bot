from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./Refined_Chat_Bot')
model = AutoModelForCausalLM.from_pretrained('./Refined_Chat_Bot')

input_context = '''
###Prompt:
Answer as if you are Harry Potter from the novel Harry Potter and the Philosopher’s Stone.
Who is Taylor Swift?

###Assistant:
'''

input_ids = tokenizer.encode(input_context, return_tensors="pt")
output = model.generate(input_ids, max_length=85, temperature = 0.3, num_return_sequences = 1)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)