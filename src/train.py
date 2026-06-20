"""
Rumor Detection Model Training Script
Uses DistilBERT fine-tuned for binary rumor classification.
"""
import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
BATCH_SIZE = 16
MAX_LEN = 128
EPOCHS = 10
LR = 5e-5
WARMUP_RATIO = 0.1
MODEL_NAME = "distilbert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def clean_text(text):
    """Basic text cleaning for tweets."""
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&lt;", "<", text)
    return text.strip()


class RumorDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["text"].apply(clean_text).tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().tolist())
            trues.extend(labels.cpu().tolist())
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds)
    return acc, f1, preds, trues


def main():
    print(f"Using device: {DEVICE}")

    # ── Load data ──
    train_path = os.path.join(BASE_DIR, "train.csv")
    val_path = os.path.join(BASE_DIR, "val.csv")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    print(f"Train label distribution:\n{train_df['label'].value_counts()}")

    # ── Tokenizer ──
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    # ── Datasets & Loaders ──
    train_set = RumorDataset(train_df, tokenizer, MAX_LEN)
    val_set = RumorDataset(val_df, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # ── Model ──
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(DEVICE)

    # ── Optimizer & Scheduler ──
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training ──
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        val_acc, val_f1, _, _ = evaluate(model, val_loader)
        print(
            f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt")
            )
            print(f"  -> Best model saved (acc={best_val_acc:.4f})")

    # ── Final evaluation ──
    model.load_state_dict(
        torch.load(
            os.path.join(CHECKPOINT_DIR, "best_model.pt"),
            map_location=DEVICE,
            weights_only=True,
        )
    )
    val_acc, val_f1, preds, trues = evaluate(model, val_loader)
    print(f"\n{'='*50}")
    print(f"Final Val Accuracy: {val_acc:.4f}")
    print(f"Final Val F1: {val_f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(trues, preds, target_names=["Non-Rumor", "Rumor"]))

    # ── Save tokenizer ──
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    print(f"\nModel and tokenizer saved to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
