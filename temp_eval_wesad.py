import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from models.cnn_lstm_model import CNN_LSTM
from utils.data_loader import WESADDataset
from configs.config import BATCH_SIZE
from sklearn.metrics import accuracy_score, classification_report

p = Path('outputs/processed/wesad_processed.npz')
print('WESAD file exists:', p.exists())
if not p.exists():
    raise SystemExit('Missing WESAD processed data')
npz = np.load(p)
X, y = npz['X'], npz['y']
print('Data shape', X.shape, y.shape)

n = len(X)
indices = np.random.RandomState(42).permutation(n)
train_end = int(n * 0.7)
val_end = int(n * 0.85)
train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]
print('Split counts', len(train_idx), len(val_idx), len(test_idx))

ckpt_path = Path('outputs/models/cnn_lstm_final.pt')
print('Checkpoint exists:', ckpt_path.exists())
if not ckpt_path.exists():
    raise SystemExit('Missing checkpoint')
ckpt = torch.load(ckpt_path, map_location='cpu')
config = ckpt.get('config', {'n_features': 5, 'n_classes': 3, 'seq_len': 42000})
model = CNN_LSTM(n_features=config['n_features'], n_classes=config['n_classes'], seq_len=config['seq_len'])
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

dataset = WESADDataset(X, y)
test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
all_preds, all_labels = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        outputs = model(Xb)
        preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print('Test accuracy:', acc)
print('Classification report:')
print(classification_report(all_labels, all_preds, target_names=['baseline', 'stress', 'amusement'], zero_division=0))
