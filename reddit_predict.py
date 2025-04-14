import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import numpy as np

# Load trained model and tokenizer
model_path = "./deberta-stress-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load the processed Reddit data
from reddit_preprocess import final_df  # This gives us the "text" and "source_subreddit" columns

# Convert to HuggingFace dataset
ds_reddit = Dataset.from_pandas(final_df)

# Tokenize
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

ds_reddit = ds_reddit.map(tokenize, batched=True)

# Get predictions
model.eval()
preds = []

with torch.no_grad():
    for batch in ds_reddit.iter(batch_size=16):
        input_ids = torch.tensor(batch["input_ids"])
        attention_mask = torch.tensor(batch["attention_mask"])
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        batch_preds = torch.argmax(probs, axis=1).numpy()
        preds.extend(batch_preds)

# Attach predictions to original data
final_df["predicted_stress"] = preds

# Save it
final_df.to_csv("reddit_stress_predictions.csv", index=False)

print("✅ Saved predictions to reddit_stress_predictions.csv")
