import sentencepiece
from transformers import T5Tokenizer, T5ForConditionalGeneration #pylint: disable=import-error

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Example input texts for different tasks
texts = [
    "translate English to German: The weather is nice today.",
    "summarize: The Great Wall of China is one of the greatest wonders of the world. It was built over a period of 2000 years and stretches more than 13,000 miles.",
    "translate French to English: Bonjour, comment ça va?",
    "question: What is the chemical formula for water?",
    "translate Spanish to English: ¿Dónde está la biblioteca?",
    "explain: Photosynthesis is a process used by plants to convert light into energy."
]

# Function to generate outputs for each input text
def generate_text(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_new_tokens=999, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generating outputs
generated_texts = [generate_text(text) for text in texts]

# Printing the outputs
for num, text in enumerate(generated_texts, 1):
    print(f"Output # {num}: {text}\n")

