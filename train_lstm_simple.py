import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from models.bert_model import TextClassifierWithoutBert
from sklearn.metrics import classification_report

# Load data
processed_dir = Path('outputs/processed')
with open(processed_dir / 'goemotions_texts.json', 'r') as f:
    texts = json.load(f)
labels = np.load(processed_dir / 'goemotions_processed.npz')['labels']

print(f"Loaded {len(texts)} texts")

# Simple tokenizer (basic word-level)
class SimpleTokenizer:
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.word_to_idx = {'<pad>': 0, '<unk>': 1}
        self.idx_to_word = {0: '<pad>', 1: '<unk>'}

    def fit(self, texts):
        from collections import Counter
        words = []
        for text in texts:
            words.extend(text.lower().split())
        word_counts = Counter(words)
        for word, _ in word_counts.most_common(self.vocab_size - 2):
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

    def encode(self, text, max_length=128):
        tokens = text.lower().split()[:max_length]
        indices = [self.word_to_idx.get(token, 1) for token in tokens]
        # Pad to max_length
        indices += [0] * (max_length - len(indices))
        return torch.tensor(indices[:max_length])

# Create tokenizer and fit on training data
tokenizer = SimpleTokenizer()
train_texts = texts[:int(0.7*len(texts))]
tokenizer.fit(train_texts)

# Dataset
class SimpleTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = torch.FloatTensor(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = tokenizer.encode(self.texts[idx], self.max_length)
        return {'input_ids': input_ids, 'labels': self.labels[idx]}

# Split data
n = len(texts)
train_idx, val_idx, test_idx = int(0.7*n), int(0.85*n), n

train_dataset = SimpleTextDataset(texts[:train_idx], labels[:train_idx], tokenizer)
val_dataset = SimpleTextDataset(texts[train_idx:val_idx], labels[train_idx:val_idx], tokenizer)
test_dataset = SimpleTextDataset(texts[val_idx:], labels[val_idx:], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Model
model = TextClassifierWithoutBert(num_labels=28)
device = torch.device('cpu')
model.to(device)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

print("Training LSTM model...")
for epoch in range(3):  # Quick training
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Test
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels']

        outputs = model(input_ids)
        preds = (torch.sigmoid(outputs) > 0.5).float()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

# Calculate metrics
from utils.metrics import compute_all_multilabel_metrics
metrics = compute_all_multilabel_metrics(all_labels, all_preds)
print("\nLSTM Model Results:")
print(f"F1 (micro): {metrics['f1_micro']:.4f}")
print(f"F1 (macro): {metrics['f1_macro']:.4f}")
print(f"Exact Match: {metrics['exact_match']:.4f}")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'test_f1': metrics['f1_micro'],
    'config': {'model_type': 'lstm', 'num_labels': 28}
}, 'outputs/models/lstm_goemotions_final.pt')
print("Model saved as lstm_goemotions_final.pt")