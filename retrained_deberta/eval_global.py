# evaluate_model.py
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load your saved model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./deberta-stress-model-retrain-global")
tokenizer = AutoTokenizer.from_pretrained("./deberta-stress-model-retrain-global")

# Load your evaluation data
df = pd.read_csv("../mental_health.csv")[['text', 'label']].dropna()
dataset = Dataset.from_pandas(df)

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)

def compute_metrics(pred):
    predictions, labels = pred
    preds = predictions.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

metrics = trainer.evaluate(eval_dataset=dataset.select(range(500)))  # or full dataset
print(metrics)
# statistics
# {'eval_loss': 0.01881900429725647, 
#  'eval_model_preparation_time': 0.0014, 
#  'eval_accuracy': 0.994, 
#  'eval_f1': 0.994106090373281, 
#  'eval_precision': 0.9921568627450981, 
#  'eval_recall': 0.9960629921259843, 
#  'eval_runtime': 47.6793, ''
#  'eval_samples_per_second': 10.487, 
#  'eval_steps_per_second': 1.321}

