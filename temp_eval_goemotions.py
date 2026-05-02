import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from models.bert_model import BertEmotionClassifier
from utils.data_loader import GoEmotionsDataset
from configs.config import RANDOM_SEED, BATCH_SIZE, GOEMOTIONS_CONFIG
from sklearn.metrics import accuracy_score, classification_report
from transformers import DistilBertTokenizer

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(GOEMOTIONS_CONFIG['model_name'])

# Load processed data
import json
texts = json.load(open('outputs/processed/goemotions_texts.json'))
npz = np.load('outputs/processed/goemotions_processed.npz')
labels = npz['labels']
print('Data loaded - Texts:', len(texts), 'Labels:', labels.shape)

# Split
n = len(texts)
indices = np.random.RandomState(RANDOM_SEED).permutation(n)
train_end = int(n * 0.7)
val_end = int(n * 0.85)
train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]
print('Split counts', len(train_idx), len(val_idx), len(test_idx))

# Model
ckpt_path = Path('outputs/models/bert_goemotions_final.pt')
print('Checkpoint exists:', ckpt_path.exists())
if not ckpt_path.exists():
    raise SystemExit('Missing GoEmotions checkpoint')
ckpt = torch.load(ckpt_path, map_location='cpu')
model = BertEmotionClassifier(num_labels=28)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Test dataset and loader
test_texts = texts[test_idx]
test_labels = labels[test_idx]
test_dataset = GoEmotionsDataset(test_texts, test_labels, tokenizer=tokenizer)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Evaluate
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels_batch = batch['labels']
        
        outputs = model(input_ids, attention_mask)
        preds = (torch.sigmoid(outputs) > 0.5).int()
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels_batch.cpu().numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

# Multi-label metrics
from utils.metrics import compute_all_multilabel_metrics
metrics = compute_all_multilabel_metrics(all_labels, all_preds)
print('GoEmotions Test metrics:')
print(f'F1 (micro): {metrics["f1_micro"]:.4f}')
print(f'F1 (macro): {metrics["f1_macro"]:.4f}')
print(f'Precision (micro): {metrics["precision_micro"]:.4f}')
print(f'Recall (micro): {metrics["recall_micro"]:.4f}')
print(f'Exact Match: {metrics["exact_match"]:.4f}')
