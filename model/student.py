import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

# ---------------- Config ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
PSEUDO_THRESHOLD = 0.9
COMMENT_SCORE_THRESHOLD = 1  # Only keep comments with score >= 1

# ---------------- Dataset ----------------
class SimpleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ---------------- Load Teacher ----------------
teacher_path = "/home/huangzekai/桌面/ML Singpaore/ML-Singapore/output/teacher_roberta_model"
tokenizer = RobertaTokenizer.from_pretrained(teacher_path)
teacher_model = RobertaForSequenceClassification.from_pretrained(teacher_path).to(DEVICE)
teacher_model.eval()

# ---------------- Load & Preprocess SG Data ----------------
sg_df = pd.read_csv("/home/huangzekai/桌面/ML Singpaore/ML-Singapore/raw_data/nus_comments_cleaned.csv")

# Filter based on comment_score
sg_df = sg_df[sg_df["comment_score"] >= COMMENT_SCORE_THRESHOLD]

# Combine post_title and comment_body
sg_df["text"] = sg_df["post_title"].fillna('') + " " + sg_df["comment_body"].fillna('')

# Remove empty text entries
sg_df = sg_df[sg_df["text"].str.strip().astype(bool)]

sg_texts = sg_df["text"].tolist()

# ---------------- Generate Pseudo Labels ----------------
pseudo_labels = []
confidences = []

with torch.no_grad():
    for i in tqdm(range(0, len(sg_texts), BATCH_SIZE), desc="Pseudo-labeling"):
        batch_texts = sg_texts[i:i+BATCH_SIZE]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        confs, preds = torch.max(probs, dim=1)

        pseudo_labels.extend(preds.cpu().tolist())
        confidences.extend(confs.cpu().tolist())

# ---------------- Filter High-Confidence ----------------
sg_df["pseudo_label"] = pseudo_labels
sg_df["confidence"] = confidences
filtered_df = sg_df[sg_df["confidence"] >= PSEUDO_THRESHOLD]
print(f"✨ 保留下来用于学生训练的伪标签样本: {len(filtered_df)}")

# ---------------- Train Student ----------------
X = filtered_df["text"].tolist()
y = filtered_df["pseudo_label"].tolist()

train_dataset = SimpleDataset(X, y, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

student_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(DEVICE)
optimizer = AdamW(student_model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

student_model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Student Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"✅ Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

# ---------------- Save Student Model ----------------
student_model.save_pretrained("/home/huangzekai/桌面/ML Singpaore/ML-Singapore/output/student_model_finetuned")
tokenizer.save_pretrained("/home/huangzekai/桌面/ML Singpaore/ML-Singapore/output/student_model_finetuned")
print("✅ 学生模型已保存至 student_model_finetuned")
