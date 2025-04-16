import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

# ---------------- Dataset ----------------
class StressDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx], 
                                padding='max_length', 
                                truncation=True, 
                                max_length=self.max_len, 
                                return_tensors='pt')
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ---------------- Load and Prepare Data ----------------
df = pd.read_csv('mental_health.csv')

# Only binary: 1 = stressful, 0 = not stressful
df['label'] = df['label'].apply(lambda x: 1 if str(x).lower() == 'stressful' else 0)

X_train, X_val, y_train, y_val = train_test_split(df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

train_dataset = StressDataset(X_train.tolist(), y_train.tolist(), tokenizer)
val_dataset = StressDataset(X_val.tolist(), y_val.tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ---------------- Model ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# ---------------- Training ----------------
epochs = 3
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

# ---------------- Evaluation ----------------
model.eval()
predictions = []
ground_truth = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        ground_truth.extend(labels.cpu().numpy())

f1 = f1_score(ground_truth, predictions)
print(f"🎯 Validation F1-macro score (binary stress detection): {f1:.4f}")

# ---------------- Save Results ----------------
val_texts = X_val.tolist()
val_df = pd.DataFrame({
    'text': val_texts,
    'true_label': ground_truth,
    'predicted_label': predictions
})
val_df.to_csv("roberta_stress_predictions.csv", index=False)
print("✅ Saved predictions to roberta_stress_predictions.csv")
