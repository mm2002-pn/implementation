"""
Script d'entrainement pour GoEmotions (texte).
Utilise DistilBERT pour classification multi-label de 28 emotions.

Usage:
    python train_goemotions.py
    python train_goemotions.py --epochs 5 --batch-size 8
    python train_goemotions.py --freeze-bert  # Fine-tuning partiel
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from configs.config import (
    BATCH_SIZE, LEARNING_RATE, EPOCHS, MODELS_DIR,
    GOEMOTIONS_CONFIG
)
from utils.data_loader import get_goemotions_loaders
from utils.metrics import (
    compute_multilabel_f1, compute_multilabel_accuracy,
    compute_all_multilabel_metrics, MetricsTracker
)

# Verifier si transformers est disponible
try:
    from transformers import DistilBertTokenizer
    from models.bert_model import BertEmotionClassifier, TextClassifierWithoutBert
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers non installe. Installation: pip install transformers")


def train_epoch(model, train_loader, criterion, optimizer, device, use_bert=True):
    """Entraine le modele pour une epoque."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(train_loader, desc="Training", leave=False):
        if use_bert:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
        else:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)

        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping pour stabilite
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        # Predictions (sigmoid pour multi-label)
        probs = torch.sigmoid(outputs)
        all_preds.append(probs.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    f1_micro = compute_multilabel_f1(all_labels, all_preds, threshold=0.5, average='micro')

    return avg_loss, f1_micro


def validate(model, val_loader, criterion, device, use_bert=True):
    """Evalue le modele sur le set de validation."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            if use_bert:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
            else:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    f1_micro = compute_multilabel_f1(all_labels, all_preds, threshold=0.5, average='micro')
    f1_macro = compute_multilabel_f1(all_labels, all_preds, threshold=0.5, average='macro')

    return avg_loss, f1_micro, f1_macro, all_preds, all_labels


def train(args):
    """Fonction principale d'entrainement."""

    print("="*60)
    print("ENTRAINEMENT BERT - GOEMOTIONS")
    print("="*60)

    if not TRANSFORMERS_AVAILABLE and not args.no_bert:
        print("Erreur: transformers requis. Installez avec: pip install transformers")
        print("Ou utilisez --no-bert pour le modele LSTM simple.")
        return

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Hyperparametres (reduits pour BERT)
    batch_size = args.batch_size or 8  # BERT demande plus de memoire
    learning_rate = args.lr or 2e-5    # LR plus petit pour fine-tuning
    epochs = args.epochs or 5          # Moins d'epoques pour BERT

    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")

    use_bert = TRANSFORMERS_AVAILABLE and not args.no_bert

    # Tokenizer
    tokenizer = None
    if use_bert:
        print(f"\nChargement du tokenizer {GOEMOTIONS_CONFIG['model_name']}...")
        tokenizer = DistilBertTokenizer.from_pretrained(GOEMOTIONS_CONFIG['model_name'])

    # DataLoaders
    print("\nChargement des donnees...")

    import json
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer

    print("📊 Chargement des données GoEmotions...\n")

    # Charger les fichiers
    processed_dir = Path('outputs/processed')
    with open(processed_dir / 'goemotions_texts.json', 'r') as f:
        texts = json.load(f)
    labels = np.load(processed_dir / 'goemotions_processed.npz')['labels']

    print(f"✅ {len(texts)} textes chargés")

    # Dataset
    class GoEmotionsDataset(Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.texts = texts
            self.labels = torch.FloatTensor(labels)
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoded = self.tokenizer(self.texts[idx], padding='max_length',
                                    truncation=True, max_length=128, return_tensors='pt')
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'labels': self.labels[idx]
            }

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    n = len(texts)
    train_idx, val_idx = int(0.7*n), int(0.85*n)

    train_loader = DataLoader(GoEmotionsDataset(texts[:train_idx], labels[:train_idx], tokenizer), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(GoEmotionsDataset(texts[train_idx:val_idx], labels[train_idx:val_idx], tokenizer), batch_size=batch_size)
    test_loader = DataLoader(GoEmotionsDataset(texts[val_idx:], labels[val_idx:], tokenizer), batch_size=batch_size)

    print(f"✅ Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # Modele
    print("\nCreation du modele...")
    num_labels = len(GOEMOTIONS_CONFIG['emotions'])

    if use_bert:
        model = BertEmotionClassifier(
            num_labels=num_labels,
            freeze_bert=args.freeze_bert
        )
        model_name = "bert_goemotions"
    else:
        model = TextClassifierWithoutBert(num_labels=num_labels)
        model_name = "lstm_goemotions"

    model = model.to(device)
    print(f"Modele: {model_name}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametres entrainables: {trainable_params:,} / {total_params:,}")

    # Loss (BCEWithLogitsLoss pour multi-label)
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer (AdamW recommande pour BERT)
    if use_bert:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Entrainement
    best_val_f1 = 0.0

    print("\nDemarrage de l'entrainement...")
    print("-"*60)

    for epoch in range(epochs):
        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, use_bert
        )

        val_loss, val_f1_micro, val_f1_macro, _, _ = validate(
            model, val_loader, criterion, device, use_bert
        )

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, F1-micro: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, F1-micro: {val_f1_micro:.4f}, F1-macro: {val_f1_macro:.4f}")

        if val_f1_micro > best_val_f1:
            best_val_f1 = val_f1_micro
            model_path = MODELS_DIR / f"{model_name}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1_micro,
                'val_loss': val_loss
            }, model_path)
            print(f"  -> Meilleur modele sauvegarde")

    print("-"*60)
    print(f"\nEntrainement termine!")
    print(f"Meilleur F1-micro validation: {best_val_f1:.4f}")

    # Evaluation finale
    print("\n" + "="*60)
    print("EVALUATION SUR TEST SET")
    print("="*60)

    checkpoint = torch.load(MODELS_DIR / f"{model_name}_best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_f1_micro, test_f1_macro, all_preds, all_labels = validate(
        model, test_loader, criterion, device, use_bert
    )

    metrics = compute_all_multilabel_metrics(all_labels, all_preds)

    print(f"\nTest Metrics:")
    print(f"  F1-micro: {metrics['f1_micro']:.4f}")
    print(f"  F1-macro: {metrics['f1_macro']:.4f}")
    print(f"  F1-samples: {metrics['f1_samples']:.4f}")
    print(f"  Precision: {metrics['precision_micro']:.4f}")
    print(f"  Recall: {metrics['recall_micro']:.4f}")
    print(f"  Exact Match: {metrics['exact_match']:.4f}")

    # Top emotions
    print("\nPerformance par emotion (top 10):")
    emotion_names = GOEMOTIONS_CONFIG['emotions']
    per_emotion_f1 = []

    for i, emotion in enumerate(emotion_names):
        if all_labels[:, i].sum() > 0:
            f1 = compute_multilabel_f1(
                all_labels[:, i:i+1],
                all_preds[:, i:i+1],
                threshold=0.5,
                average='binary'
            )
            per_emotion_f1.append((emotion, f1, all_labels[:, i].sum()))

    per_emotion_f1.sort(key=lambda x: -x[1])
    for emotion, f1, count in per_emotion_f1[:10]:
        print(f"  {emotion:15s}: F1={f1:.4f} (n={int(count)})")

    # Sauvegarder modele final
    final_path = MODELS_DIR / f"{model_name}_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_f1': metrics['f1_micro'],
        'config': GOEMOTIONS_CONFIG if use_bert else {'model_type': 'lstm', 'num_labels': num_labels}
    }, final_path)
    print(f"\nModele final sauvegarde: {final_path}")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Entrainement BERT GoEmotions")
    parser.add_argument("--epochs", type=int, default=10, help="Nombre d'epoques")  # Augmenté
    parser.add_argument("--batch-size", type=int, default=None, help="Taille du batch")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")  # Réduit
    parser.add_argument("--freeze-bert", action="store_true", help="Geler les poids BERT")
    parser.add_argument("--no-bert", action="store_true", help="Utiliser LSTM au lieu de BERT")
    parser.add_argument("--threshold", type=float, default=0.3, help="Seuil pour classification multi-label")  # Nouveau paramètre

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
