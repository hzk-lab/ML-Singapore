import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizer
from sklearn.metrics import f1_score

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example Dataset class
class CommentDataset(Dataset):
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
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        return input_ids, attention_mask, torch.tensor(self.labels[idx])

# Attention layer
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        attn_weights = torch.tanh(self.attn(lstm_out)).squeeze(-1)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
        return context, attn_weights

# LSTM model with attention
class LSTMAttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        context, attn_weights = self.attention(lstm_out)
        output = self.fc(context)
        return output, attn_weights

# Load and prepare data
df = pd.read_csv('mental_health.csv')
X = df['text'].tolist()
y = LabelEncoder().fit_transform(df['label'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

# Use BERT tokenizer for better tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = CommentDataset(X_train, y_train, tokenizer)
val_dataset = CommentDataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model setup
VOCAB_SIZE = tokenizer.vocab_size
PAD_IDX = tokenizer.pad_token_id
model = LSTMAttentionModel(VOCAB_SIZE, 128, 64, output_dim=len(set(y)), padding_idx=PAD_IDX).to(device)

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for input_ids, attn_mask, labels in tqdm(train_loader):
        input_ids, attn_mask, labels = input_ids.to(device), attn_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(input_ids, attn_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")


# Evaluation and saving to CSV
from sklearn.preprocessing import LabelEncoder

# Load original label encoder to get label names
label_encoder = LabelEncoder()
label_encoder.fit(df['label'])  # Fit on full data labels

model.eval()
results = []
texts = []

with torch.no_grad():
    for input_ids, attn_mask, labels in tqdm(val_loader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        outputs, _ = model(input_ids, attn_mask)
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

        results.extend(preds.cpu().numpy())
        texts.extend(tokenizer.batch_decode(input_ids, skip_special_tokens=True))

# Decode labels
label_encoder = LabelEncoder()
label_encoder.fit(df['label'])
decoded_true = label_encoder.inverse_transform(y_val)
decoded_pred = label_encoder.inverse_transform(results)

# Compute F1-macro
f1 = f1_score(y_val, results, average='macro')
print(f"🎯 Validation F1-macro score: {f1:.4f}")

# Save CSV
val_df = pd.DataFrame({
    'text': X_val,
    'true_label': decoded_true,
    'predicted_label': decoded_pred
})

val_df.to_csv('/home/huangzekai/桌面/ML Singpaore/ML-Singapore/data/lstm_attention_predictions.csv', index=False)
print("✅ Predictions saved to lstm_attention_predictions.csv")

