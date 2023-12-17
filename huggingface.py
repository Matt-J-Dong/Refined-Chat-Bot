import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
texts = ["How do I get a replacement Medicare card?",
        		"What is the monthly premium for Medicare Part B?",
        		"How do I terminate my Medicare Part B (medical insurance)?",
        		"How do I sign up for Medicare?",
        		"Can I sign up for Medicare Part B if I am working and have health insurance through an employer?",
        		"How do I sign up for Medicare Part B if I already have Part A?"]
model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
ids = tokenizer(texts, padding=True, return_tensors="pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
ids = ids.to(device)	
model.eval()
with torch.no_grad():
  out = model(**ids)
last_hidden_states = out.last_hidden_state
sentence_embedding = last_hidden_states[:, 0, :]
print("Shape of the batch embedding: {}".format(sentence_embedding.shape))