#Currently just returns a copy of the input, this is a fail

from transformers import BartTokenizer, BartForConditionalGeneration #pylint: disable=import-error

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

# Example input texts for different tasks
texts = [
    "The Louvre Museum in Paris is home to the Mona Lisa painting.",
    "summarize: Researchers at MIT have created a new AI model that outperforms others in several tasks.",
    "translate English to French: Hello, how are you?",
    "What is the capital of France?"
]

def generate_text(input_text):
    inputs = tokenizer([input_text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=5, max_length=200, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

generated_texts = [generate_text(text) for text in texts]
num = 0
for text in generated_texts:
    num += 1
    print(f"Output # {num}")
    print(text)
