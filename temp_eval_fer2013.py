import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from models.cnn_fer_model import CNN_FER
from utils.data_loader import FER2013ChunkedDataset
from configs.config import RANDOM_SEED, BATCH_SIZE, FER2013_CONFIG
from sklearn.metrics import accuracy_score, classification_report

chunk_dir = Path('outputs/processed/fer2013_chunks')
print('FER2013 chunk dir exists:', chunk_dir.exists())
if not chunk_dir.exists():
    raise SystemExit('Missing FER2013 chunks')

# Load dataset metadata
full_dataset = FER2013ChunkedDataset(chunk_dir)
print('Total FER2013 samples:', len(full_dataset))

# split
n = len(full_dataset)
indices = np.random.RandomState(RANDOM_SEED).permutation(n)
train_end = int(n * 0.7)
val_end = int(n * 0.85)
train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]
print('Split counts', len(train_idx), len(val_idx), len(test_idx))

# model
ckpt_path = Path('outputs/models/cnn_fer_final.pt')
print('Checkpoint exists:', ckpt_path.exists())
if not ckpt_path.exists():
    raise SystemExit('Missing FER2013 checkpoint')
ckpt = torch.load(ckpt_path, map_location='cpu')
model = CNN_FER(n_classes=7)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# test loader
# use Subset on dataset directly because chunk dataset supports __getitem__
test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
all_preds, all_labels = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        outputs = model(Xb)
        preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print('FER2013 Test accuracy:', acc)
print('Classification report:')
print(classification_report(all_labels, all_preds, target_names=list(FER2013_CONFIG['labels'].values()), zero_division=0))
