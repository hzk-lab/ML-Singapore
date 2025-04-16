from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

import os
os.environ['HUGGINGFACE_HUB_TIMEOUT'] = '60'

# Load the model and tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Create a sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Sample text
text = "I'm feeling really down today. Nothing seems to help."

# Run prediction
result = sentiment_pipeline(text)

# Print result
print(result)
