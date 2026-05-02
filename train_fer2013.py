"""
Script d'entrainement pour FER2013 (expressions faciales).
Utilise CNN pour classifier 7 emotions.

Usage:
    python train_fer2013.py
    python train_fer2013.py --epochs 30 --batch-size 16
    python train_fer2013.py --lite  # Version allegee
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
from datetime import datetime

# Ajouter le path parent pour imports
sys.path.append(str(Path(__file__).parent))

from configs.config import (
    BATCH_SIZE, LEARNING_RATE, EPOCHS, MODELS_DIR, LOGS_DIR,
    FER2013_CONFIG
)
from models.cnn_fer_model import CNN_FER, CNN_FER_Lite, create_model
from utils.data_loader import get_fer2013_loaders
from utils.metrics import (
    compute_accuracy, compute_f1, compute_all_metrics,
    get_classification_report, MetricsTracker
)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Entraine le modele pour une epoque.

    Returns:
        train_loss, train_acc
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (X, y) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        X = X.to(device)
        y = y.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward
        loss.backward()
        optimizer.step()

        # Statistiques
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    accuracy = compute_accuracy(np.array(all_labels), np.array(all_preds))

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Evalue le modele sur le set de validation.

    Returns:
        val_loss, val_acc, val_f1
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in tqdm(val_loader, desc="Validation", leave=False):
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = compute_accuracy(np.array(all_labels), np.array(all_preds))
    f1 = compute_f1(np.array(all_labels), np.array(all_preds), average='weighted')

    return avg_loss, accuracy, f1


def train(args):
    """Fonction principale d'entrainement."""

    print("="*60)
    print("ENTRAINEMENT CNN - FER2013")
    print("="*60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Hyperparametres
    batch_size = args.batch_size or BATCH_SIZE
    learning_rate = args.lr or LEARNING_RATE
    epochs = args.epochs or EPOCHS

    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")

    # DataLoaders
    print("\nChargement des donnees...")
    train_loader, val_loader, test_loader = get_fer2013_loaders(
        batch_size=batch_size,
        use_chunks=True
    )

    if train_loader is None:
        print("Erreur: Donnees non trouvees. Executez d'abord le preprocessing.")
        return

    # Modele
    print("\nCreation du modele...")
    if args.lite:
        model = CNN_FER_Lite(n_classes=7)
        model_name = "cnn_fer_lite"
    else:
        model = CNN_FER(n_classes=7)
        model_name = "cnn_fer"

    model = model.to(device)
    print(f"Modele: {model_name}")
    print(f"Parametres: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss et Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Tracker
    tracker = MetricsTracker()
    best_val_acc = 0.0

    # Entrainement
    print("\nDemarrage de l'entrainement...")
    print("-"*60)

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validation
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)

        # Scheduler
        scheduler.step(val_loss)

        # Tracker
        tracker.update(
            train_loss, val_loss,
            train_acc, val_acc,
            epoch=epoch
        )

        # Affichage
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # Sauvegarder meilleur modele
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = MODELS_DIR / f"{model_name}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, model_path)
            print(f"  -> Meilleur modele sauvegarde: {model_path}")

    print("-"*60)
    print(f"\nEntrainement termine!")
    print(f"Meilleure accuracy validation: {best_val_acc:.4f}")

    # Evaluation finale sur test set
    print("\n" + "="*60)
    print("EVALUATION SUR TEST SET")
    print("="*60)

    # Charger meilleur modele
    checkpoint = torch.load(MODELS_DIR / f"{model_name}_best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Testing"):
            X = X.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metriques
    metrics = compute_all_metrics(all_preds, all_labels)
    target_names = list(FER2013_CONFIG['labels'].values())

    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"Test F1 (macro): {metrics['f1_macro']:.4f}")

    print("\nRapport de classification:")
    print(get_classification_report(all_labels, all_preds, target_names=target_names))

    # Sauvegarder modele final
    final_path = MODELS_DIR / f"{model_name}_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': metrics['accuracy'],
        'test_f1': metrics['f1_weighted'],
        'config': FER2013_CONFIG
    }, final_path)
    print(f"\nModele final sauvegarde: {final_path}")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Entrainement CNN FER2013")
    parser.add_argument("--epochs", type=int, default=None, help="Nombre d'epoques")
    parser.add_argument("--batch-size", type=int, default=None, help="Taille du batch")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--lite", action="store_true", help="Utiliser version allegee")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
