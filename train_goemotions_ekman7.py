"""
Entrainement BERT pour GoEmotions avec 7 emotions d'Ekman (single-label).

Difference avec train_goemotions.py (preserve comme preuve unimodale):
  - num_labels = 7 (au lieu de 28)
  - Classification single-label (au lieu de multi-label)
  - Loss: CrossEntropyLoss (au lieu de BCEWithLogitsLoss)
  - Predictions: argmax (au lieu de sigmoid + threshold)
  - Mapping des labels effectue a la volee dans le DataLoader

Usage:
    python train_goemotions_ekman7.py
    python train_goemotions_ekman7.py --epochs 5 --batch-size 16
    python train_goemotions_ekman7.py --device cuda

Justification academique:
    Reduction de granularite vers 7 emotions d'Ekman selon le mapping
    officiel propose par Demszky et al. (2020). Voir Chapitre IV.4 du memoire.

Resultats attendus (selon estimation sur 7 classes):
    F1 micro: 60-65% (vs 37.38% sur 28 classes)
    F1 macro: 55-60%
    Accuracy: 65-70%
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).parent))

from configs.config import (
    BATCH_SIZE, LEARNING_RATE, EPOCHS, MODELS_DIR,
    GOEMOTIONS_CONFIG, EKMAN_EMOTIONS, RANDOM_SEED, PROCESSED_DIR
)
from preprocessing.goemotions_ekman7 import (
    convert_28_to_7_ekman, get_class_distribution, EKMAN_EMOTIONS as EK
)
from utils.data_loader import GoEmotionsChunkedDataset

try:
    from transformers import DistilBertTokenizer
    from models.bert_model import BertEmotionClassifier
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("ERREUR: transformers requis. Installation: pip install transformers")


# =============================================================================
# DATASET WRAPPER : conversion automatique 28 -> 7 (single-label)
# =============================================================================

class GoEmotionsEkman7Wrapper(Dataset):
    """
    Wrapper autour de GoEmotionsChunkedDataset qui convertit
    a la volee les labels (n, 28) multi-label en (n,) single-label Ekman.
    """

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]

        # item['labels'] est un tensor float (28,) multi-label binaire
        labels_28 = item['labels'].numpy().reshape(1, -1)  # (1, 28)
        label_7 = convert_28_to_7_ekman(labels_28, mode='single_label')[0]

        item['labels'] = torch.LongTensor([label_7])[0]  # scalaire long
        return item


def get_goemotions_ekman7_loaders(batch_size=8, tokenizer=None, max_length=128,
                                   train_ratio=0.7, val_ratio=0.15):
    """
    Cree les DataLoaders GoEmotions avec labels convertis en 7 emotions Ekman.
    """
    chunk_dir = PROCESSED_DIR / "goemotions_chunks"

    if not chunk_dir.exists():
        raise FileNotFoundError(
            f"Chunks GoEmotions non trouves dans {chunk_dir}. "
            f"Executez d'abord le preprocessing GoEmotions."
        )

    # Calculer les splits sur l'ensemble complet
    base_full = GoEmotionsChunkedDataset(chunk_dir)
    n = len(base_full)
    indices = np.random.RandomState(RANDOM_SEED).permutation(n)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Creer 3 datasets avec splits + tokenizer
    train_base = GoEmotionsChunkedDataset(
        chunk_dir, indices[:train_end].tolist(), tokenizer, max_length
    )
    val_base = GoEmotionsChunkedDataset(
        chunk_dir, indices[train_end:val_end].tolist(), tokenizer, max_length
    )
    test_base = GoEmotionsChunkedDataset(
        chunk_dir, indices[val_end:].tolist(), tokenizer, max_length
    )

    # Wrapper pour conversion 28 -> 7
    train_dataset = GoEmotionsEkman7Wrapper(train_base)
    val_dataset = GoEmotionsEkman7Wrapper(val_base)
    test_dataset = GoEmotionsEkman7Wrapper(test_base)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    print(f"GoEmotions Ekman7 - Train: {len(train_dataset)}, "
          f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


# =============================================================================
# METRIQUES SINGLE-LABEL
# =============================================================================

def compute_metrics(labels, predictions):
    """Calcule les metriques single-label (accuracy, F1)."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        classification_report, confusion_matrix
    )

    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'precision_weighted': precision,
        'recall_weighted': recall,
    }


# =============================================================================
# TRAINING / VALIDATION
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entraine pour une epoque."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(train_loader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


def validate(model, loader, criterion, device, name="Val"):
    """Evalue sur un dataset."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=name, leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics, np.array(all_preds), np.array(all_labels)


# =============================================================================
# MAIN
# =============================================================================

def train(args):
    print("="*70)
    print("ENTRAINEMENT BERT - GOEMOTIONS 7 EMOTIONS D'EKMAN (single-label)")
    print("="*70)

    if not TRANSFORMERS_AVAILABLE:
        print("ERREUR: transformers requis.")
        return 1

    # Device
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device       : {device}")

    # Hyperparametres
    batch_size = args.batch_size or 16  # Plus de marge avec 7 classes
    learning_rate = args.lr or 2e-5
    epochs = args.epochs or 5

    print(f"Batch size   : {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs       : {epochs}")
    print(f"Num labels   : 7 (Ekman)")

    # Tokenizer
    print(f"\nChargement tokenizer {GOEMOTIONS_CONFIG['model_name']}...")
    tokenizer = DistilBertTokenizer.from_pretrained(GOEMOTIONS_CONFIG['model_name'])

    # DataLoaders
    print("\nChargement des donnees (avec conversion 28->7)...")
    train_loader, val_loader, test_loader = get_goemotions_ekman7_loaders(
        batch_size=batch_size,
        tokenizer=tokenizer,
        max_length=GOEMOTIONS_CONFIG['max_length']
    )

    # Verifier la distribution des classes sur le train set
    print("\nVerification de la distribution des classes (echantillon train)...")
    sample_labels = []
    for i, batch in enumerate(train_loader):
        sample_labels.extend(batch['labels'].numpy())
        if i >= 50:  # Echantillon de 50 batchs
            break
    sample_labels = np.array(sample_labels)
    distribution = get_class_distribution(sample_labels)
    for emotion, count in distribution.items():
        print(f"  {emotion:10s}: {count} ({count/len(sample_labels)*100:.1f}%)")

    # Modele
    print(f"\nCreation du modele BERT (num_labels=7)...")
    model = BertEmotionClassifier(
        num_labels=7,
        freeze_bert=args.freeze_bert
    )
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parametres entrainables: {trainable:,} / {total:,}")

    # Loss et optimizer
    criterion = nn.CrossEntropyLoss()  # Single-label
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs
    )

    # Entrainement
    print("\nDemarrage entrainement...")
    print("-"*70)
    best_val_f1 = 0.0
    history = []

    for epoch in range(epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, _, _ = validate(model, val_loader, criterion, device, "Val")
        scheduler.step()

        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_metrics['accuracy'],
            'train_f1': train_metrics['f1_weighted'],
            'val_loss': val_loss,
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['f1_weighted'],
            'val_f1_macro': val_metrics['f1_macro'],
        }
        history.append(epoch_log)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_weighted']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_weighted']:.4f} (macro: {val_metrics['f1_macro']:.4f})")

        if val_metrics['f1_weighted'] > best_val_f1:
            best_val_f1 = val_metrics['f1_weighted']
            best_path = MODELS_DIR / "bert_goemotions_ekman7_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'val_loss': val_loss,
                'config': {
                    'num_labels': 7,
                    'ekman_emotions': EKMAN_EMOTIONS,
                    'model_name': GOEMOTIONS_CONFIG['model_name'],
                }
            }, best_path)
            print(f"  -> Meilleur modele sauvegarde: {best_path}")

    print("-"*70)
    print(f"\nEntrainement termine. Meilleur F1 (val) : {best_val_f1:.4f}")

    # Evaluation finale sur test set
    print("\n" + "="*70)
    print("EVALUATION FINALE SUR TEST SET")
    print("="*70)

    checkpoint = torch.load(MODELS_DIR / "bert_goemotions_ekman7_best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_metrics, test_preds, test_labels = validate(
        model, test_loader, criterion, device, "Test"
    )

    print(f"\nTest Metrics:")
    print(f"  Accuracy        : {test_metrics['accuracy']:.4f}")
    print(f"  F1 weighted     : {test_metrics['f1_weighted']:.4f}")
    print(f"  F1 macro        : {test_metrics['f1_macro']:.4f}")
    print(f"  F1 micro        : {test_metrics['f1_micro']:.4f}")
    print(f"  Precision (w)   : {test_metrics['precision_weighted']:.4f}")
    print(f"  Recall (w)      : {test_metrics['recall_weighted']:.4f}")

    # Rapport par classe
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nRapport de classification par emotion d'Ekman:")
    print(classification_report(test_labels, test_preds, target_names=EKMAN_EMOTIONS, zero_division=0))

    print("\nMatrice de confusion (lignes=vrais, colonnes=predits):")
    cm = confusion_matrix(test_labels, test_preds)
    header = "         " + "  ".join(f"{e[:7]:>7s}" for e in EKMAN_EMOTIONS)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {EKMAN_EMOTIONS[i][:7]:>7s}  " + "  ".join(f"{c:>7d}" for c in row))

    # Sauvegarder modele final + historique
    final_path = MODELS_DIR / "bert_goemotions_ekman7_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_metrics': test_metrics,
        'history': history,
        'config': {
            'num_labels': 7,
            'ekman_emotions': EKMAN_EMOTIONS,
            'model_name': GOEMOTIONS_CONFIG['model_name'],
        }
    }, final_path)
    print(f"\nModele final sauvegarde: {final_path}")

    # Sauvegarder un rapport JSON
    report = {
        'method': 'BERT GoEmotions 7 Ekman emotions (single-label)',
        'mapping_source': 'Demszky et al. 2020',
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'history': history,
    }
    report_path = MODELS_DIR / "bert_goemotions_ekman7_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=float)
    print(f"Rapport JSON sauvegarde : {report_path}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Train BERT GoEmotions 7 Ekman classes")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu ou cuda")
    parser.add_argument("--freeze-bert", action="store_true",
                       help="Geler les poids BERT (plus rapide mais moins performant)")
    args = parser.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
