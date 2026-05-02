"""
Script d'evaluation des modeles et de la fusion multimodale.

Charge les 3 modeles entraines et evalue:
1. Performance individuelle de chaque modele
2. Performance de la fusion multimodale

Usage:
    python evaluate.py
    python evaluate.py --fusion-method voting
    python evaluate.py --model fer2013  # Evaluer un seul modele
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from configs.config import (
    MODELS_DIR, FIGURES_DIR,
    WESAD_CONFIG, FER2013_CONFIG, GOEMOTIONS_CONFIG,
    FUSION_CONFIG, EMOTION_MAPPING
)
from utils.data_loader import (
    get_wesad_loaders,
    get_fer2013_loaders,
    get_goemotions_loaders
)
from utils.metrics import (
    compute_all_metrics,
    compute_all_multilabel_metrics,
    get_classification_report,
    print_metrics
)
from fusion.multimodal_fusion import MultimodalFusion, map_to_common_emotions


def load_model(model_name, device='cpu'):
    """
    Charge un modele entraine.

    Args:
        model_name: 'wesad', 'fer2013', ou 'goemotions'
        device: 'cpu' ou 'cuda'

    Returns:
        model, config
    """
    if model_name == 'wesad':
        from models.cnn_lstm_model import CNN_LSTM

        model_path = MODELS_DIR / "cnn_lstm_best.pt"
        if not model_path.exists():
            model_path = MODELS_DIR / "cnn_lstm_final.pt"

        if not model_path.exists():
            print(f"Modele WESAD non trouve: {model_path}")
            return None, None

        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get('config', {'n_features': 5, 'n_classes': 3, 'seq_len': 42000})

        model = CNN_LSTM(
            n_features=config['n_features'],
            n_classes=config['n_classes'],
            seq_len=config['seq_len']
        )
        model.load_state_dict(checkpoint['model_state_dict'])

    elif model_name == 'fer2013':
        from models.cnn_fer_model import CNN_FER

        model_path = MODELS_DIR / "cnn_fer_best.pt"
        if not model_path.exists():
            model_path = MODELS_DIR / "cnn_fer_final.pt"

        if not model_path.exists():
            print(f"Modele FER2013 non trouve: {model_path}")
            return None, None

        checkpoint = torch.load(model_path, map_location=device)
        model = CNN_FER(n_classes=7)
        model.load_state_dict(checkpoint['model_state_dict'])
        config = FER2013_CONFIG

    elif model_name == 'goemotions':
        try:
            from models.bert_model import BertEmotionClassifier

            model_path = MODELS_DIR / "bert_goemotions_best.pt"
            if not model_path.exists():
                model_path = MODELS_DIR / "bert_goemotions_final.pt"

            if not model_path.exists():
                print(f"Modele GoEmotions non trouve: {model_path}")
                return None, None

            checkpoint = torch.load(model_path, map_location=device)
            model = BertEmotionClassifier(num_labels=28)
            model.load_state_dict(checkpoint['model_state_dict'])
            config = GOEMOTIONS_CONFIG

        except ImportError:
            print("transformers non installe. Impossible de charger le modele BERT.")
            return None, None

    else:
        raise ValueError(f"Modele inconnu: {model_name}")

    model = model.to(device)
    model.eval()

    return model, config


def evaluate_wesad(model, device='cpu'):
    """Evalue le modele WESAD."""
    print("\n" + "="*60)
    print("EVALUATION WESAD (Biometrique)")
    print("="*60)

    _, _, test_loader = get_wesad_loaders()

    if test_loader is None:
        print("Donnees de test non disponibles.")
        return None

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Evaluating WESAD"):
            X = X.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(y.numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = compute_all_metrics(all_labels, all_preds)
    target_names = ['baseline', 'stress', 'amusement']

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")

    print("\nRapport de classification:")
    print(get_classification_report(all_labels, all_preds, target_names=target_names))

    return all_probs, all_labels, metrics


def evaluate_fer2013(model, device='cpu'):
    """Evalue le modele FER2013."""
    print("\n" + "="*60)
    print("EVALUATION FER2013 (Visuel)")
    print("="*60)

    _, _, test_loader = get_fer2013_loaders()

    if test_loader is None:
        print("Donnees de test non disponibles.")
        return None

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Evaluating FER2013"):
            X = X.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(y.numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = compute_all_metrics(all_labels, all_preds)
    target_names = list(FER2013_CONFIG['labels'].values())

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")

    print("\nRapport de classification:")
    print(get_classification_report(all_labels, all_preds, target_names=target_names))

    return all_probs, all_labels, metrics


def evaluate_goemotions(model, device='cpu'):
    """Evalue le modele GoEmotions."""
    print("\n" + "="*60)
    print("EVALUATION GOEMOTIONS (Texte)")
    print("="*60)

    try:
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(GOEMOTIONS_CONFIG['model_name'])
    except ImportError:
        print("transformers non installe.")
        return None

    _, _, test_loader = get_goemotions_loaders(tokenizer=tokenizer)

    if test_loader is None:
        print("Donnees de test non disponibles.")
        return None

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating GoEmotions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    metrics = compute_all_multilabel_metrics(all_labels, all_probs)

    print(f"\nF1 (micro): {metrics['f1_micro']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"Precision: {metrics['precision_micro']:.4f}")
    print(f"Recall: {metrics['recall_micro']:.4f}")
    print(f"Exact Match: {metrics['exact_match']:.4f}")

    return all_probs, all_labels, metrics


def evaluate_fusion(bio_probs=None, visual_probs=None, text_probs=None,
                   method='weighted_average'):
    """
    Evalue la fusion multimodale.

    Note: Comme les datasets sont differents, on ne peut pas vraiment
    fusionner les predictions. Cette fonction montre comment la fusion
    fonctionnerait si les donnees etaient alignees.
    """
    print("\n" + "="*60)
    print(f"EVALUATION FUSION MULTIMODALE ({method})")
    print("="*60)

    fusion = MultimodalFusion(method=method)

    # Comme les datasets ne sont pas alignes, on simule
    # En pratique, il faudrait des donnees avec les 3 modalites
    print("\nNote: Les datasets ne sont pas alignes.")
    print("La fusion multimodale necessite des donnees avec les 3 modalites.")

    # Demo avec des donnees simulees
    print("\nDemo avec donnees simulees:")

    n_samples = 10

    # Simuler des predictions alignees
    bio_demo = np.random.softmax(np.random.randn(n_samples, 3))
    visual_demo = np.random.softmax(np.random.randn(n_samples, 7))
    text_demo = np.random.softmax(np.random.randn(n_samples, 28))

    # Fusionner
    fused = fusion.fuse(bio_demo, visual_demo, text_demo, map_to_common=True)
    emotions, confidence = fusion.get_final_prediction(fused)

    print(f"Predictions fusionnees:")
    for i, (emotion, conf) in enumerate(zip(emotions[:5], confidence[:5])):
        print(f"  Sample {i+1}: {emotion} (conf: {conf:.2f})")

    return fused


def main():
    parser = argparse.ArgumentParser(description="Evaluation des modeles")
    parser.add_argument("--model", type=str, default=None,
                       choices=['wesad', 'fer2013', 'goemotions'],
                       help="Evaluer un seul modele")
    parser.add_argument("--fusion-method", type=str, default='weighted_average',
                       choices=['average', 'weighted_average', 'voting', 'max'],
                       help="Methode de fusion")
    parser.add_argument("--device", type=str, default='cpu',
                       help="Device (cpu ou cuda)")

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    results = {}

    if args.model is None or args.model == 'wesad':
        model, config = load_model('wesad', device)
        if model is not None:
            results['wesad'] = evaluate_wesad(model, device)

    if args.model is None or args.model == 'fer2013':
        model, config = load_model('fer2013', device)
        if model is not None:
            results['fer2013'] = evaluate_fer2013(model, device)

    if args.model is None or args.model == 'goemotions':
        model, config = load_model('goemotions', device)
        if model is not None:
            results['goemotions'] = evaluate_goemotions(model, device)

    # Fusion (demo)
    if args.model is None:
        # Fix pour softmax numpy
        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / np.sum(e_x, axis=-1, keepdims=True)
        np.random.softmax = lambda x: softmax(x)

        evaluate_fusion(method=args.fusion_method)

    print("\n" + "="*60)
    print("EVALUATION TERMINEE")
    print("="*60)

    return results


if __name__ == "__main__":
    main()
