import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
# Load your global dataset CSV
df_global = pd.read_csv("mental_health.csv")
df_global = df_global[['text', 'label']].dropna()

# Convert to Hugging Face dataset format
ds_global = Dataset.from_pandas(df_global)

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small", use_fast=False)

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

ds_global = ds_global.map(tokenize, batched=True)
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-small", num_labels=2)

training_args = TrainingArguments(
    output_dir="./deberta-stress",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_global,
    eval_dataset=ds_global.shuffle(seed=42).select(range(500)),  # quick validation subset
    tokenizer=tokenizer
)

trainer.train()

trainer.save_model("./deberta-stress-model-retrain")
tokenizer.save_pretrained("./deberta-stress-model-retrain")

