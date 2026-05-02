"""
Pipeline complet de reconnaissance emotionnelle multimodale.

Ce script execute tout le pipeline:
1. Preprocessing des 3 datasets
2. Entrainement des 3 modeles
3. Evaluation et fusion

Usage:
    python main.py --all              # Execute tout
    python main.py --preprocess       # Preprocessing seulement
    python main.py --train            # Entrainement seulement
    python main.py --evaluate         # Evaluation seulement
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from configs.config import (
    PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, LOGS_DIR,
    WESAD_PATH, FER2013_PATH, GOEMOTIONS_PATH
)


def run_preprocessing(datasets=None):
    """Execute le preprocessing des datasets."""
    print("\n" + "="*70)
    print("PHASE 1: PREPROCESSING")
    print("="*70)

    if datasets is None:
        datasets = ['fer2013', 'goemotions', 'wesad']

    # FER2013
    if 'fer2013' in datasets:
        print("\n--- Preprocessing FER2013 ---")
        try:
            from preprocessing.fer2013_preprocessing import preprocess_fer2013

            if FER2013_PATH.exists():
                preprocess_fer2013(save_chunks=True)
            else:
                print(f"Dataset FER2013 non trouve: {FER2013_PATH}")
        except Exception as e:
            print(f"Erreur preprocessing FER2013: {e}")

    # GoEmotions
    if 'goemotions' in datasets:
        print("\n--- Preprocessing GoEmotions ---")
        try:
            from preprocessing.goemotions_preprocessing import preprocess_goemotions

            if GOEMOTIONS_PATH.exists():
                preprocess_goemotions(save_chunks=False)
            else:
                print(f"Dataset GoEmotions non trouve: {GOEMOTIONS_PATH}")
        except Exception as e:
            print(f"Erreur preprocessing GoEmotions: {e}")

    # WESAD
    if 'wesad' in datasets:
        print("\n--- Preprocessing WESAD ---")
        try:
            from preprocessing.wesad_preprocessing import preprocess_wesad

            if WESAD_PATH.exists():
                preprocess_wesad(use_pkl=True)
            else:
                print(f"Dataset WESAD non trouve: {WESAD_PATH}")
        except Exception as e:
            print(f"Erreur preprocessing WESAD: {e}")

    print("\nPreprocessing termine!")


def run_training(models=None, epochs=None, lite=False):
    """Execute l'entrainement des modeles."""
    print("\n" + "="*70)
    print("PHASE 2: ENTRAINEMENT")
    print("="*70)

    if models is None:
        models = ['fer2013', 'goemotions', 'wesad']

    # FER2013
    if 'fer2013' in models:
        print("\n--- Entrainement FER2013 ---")
        try:
            from train_fer2013 import train

            class Args:
                def __init__(self):
                    self.epochs = epochs
                    self.batch_size = None
                    self.lr = None
                    self.lite = lite

            train(Args())
        except Exception as e:
            print(f"Erreur entrainement FER2013: {e}")

    # GoEmotions
    if 'goemotions' in models:
        print("\n--- Entrainement GoEmotions ---")
        try:
            from train_goemotions import train

            class Args:
                def __init__(self):
                    self.epochs = epochs or 5
                    self.batch_size = 8
                    self.lr = None
                    self.freeze_bert = False
                    self.no_bert = False

            train(Args())
        except Exception as e:
            print(f"Erreur entrainement GoEmotions: {e}")

    # WESAD
    if 'wesad' in models:
        print("\n--- Entrainement WESAD ---")
        try:
            from train_wesad import train

            class Args:
                def __init__(self):
                    self.epochs = epochs
                    self.batch_size = None
                    self.lr = None
                    self.lite = lite

            train(Args())
        except Exception as e:
            print(f"Erreur entrainement WESAD: {e}")

    print("\nEntrainement termine!")


def run_evaluation(fusion_method='weighted_average'):
    """Execute l'evaluation des modeles."""
    print("\n" + "="*70)
    print("PHASE 3: EVALUATION")
    print("="*70)

    try:
        from evaluate import main as evaluate_main

        class Args:
            def __init__(self):
                self.model = None
                self.fusion_method = fusion_method
                self.device = 'cpu'

        evaluate_main()
    except Exception as e:
        print(f"Erreur evaluation: {e}")

    print("\nEvaluation terminee!")


def check_requirements():
    """Verifie les dependances."""
    print("\n--- Verification des dependances ---")

    required = ['torch', 'numpy', 'pandas', 'sklearn', 'tqdm']
    optional = ['transformers', 'matplotlib', 'seaborn']

    missing_required = []
    missing_optional = []

    for pkg in required:
        try:
            __import__(pkg)
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [MANQUANT] {pkg}")
            missing_required.append(pkg)

    for pkg in optional:
        try:
            __import__(pkg)
            print(f"  [OK] {pkg} (optionnel)")
        except ImportError:
            print(f"  [MANQUANT] {pkg} (optionnel)")
            missing_optional.append(pkg)

    if missing_required:
        print(f"\nERREUR: Dependances manquantes: {missing_required}")
        print("Installation: pip install " + " ".join(missing_required))
        return False

    if missing_optional:
        print(f"\nNote: Dependances optionnelles manquantes: {missing_optional}")

    return True


def check_datasets():
    """Verifie la disponibilite des datasets."""
    print("\n--- Verification des datasets ---")

    datasets = {
        'FER2013': FER2013_PATH,
        'GoEmotions': GOEMOTIONS_PATH,
        'WESAD': WESAD_PATH
    }

    available = []
    for name, path in datasets.items():
        if path.exists():
            print(f"  [OK] {name}: {path}")
            available.append(name.lower())
        else:
            print(f"  [MANQUANT] {name}: {path}")

    return available


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline complet de reconnaissance emotionnelle multimodale"
    )

    parser.add_argument("--all", action="store_true",
                       help="Execute tout le pipeline")
    parser.add_argument("--preprocess", action="store_true",
                       help="Execute le preprocessing")
    parser.add_argument("--train", action="store_true",
                       help="Execute l'entrainement")
    parser.add_argument("--evaluate", action="store_true",
                       help="Execute l'evaluation")

    parser.add_argument("--datasets", type=str, nargs='+',
                       choices=['fer2013', 'goemotions', 'wesad'],
                       help="Datasets a traiter")
    parser.add_argument("--models", type=str, nargs='+',
                       choices=['fer2013', 'goemotions', 'wesad'],
                       help="Modeles a entrainer")

    parser.add_argument("--epochs", type=int, default=None,
                       help="Nombre d'epoques")
    parser.add_argument("--lite", action="store_true",
                       help="Utiliser versions allegees des modeles")
    parser.add_argument("--fusion-method", type=str, default='weighted_average',
                       choices=['average', 'weighted_average', 'voting', 'max'],
                       help="Methode de fusion")

    parser.add_argument("--check", action="store_true",
                       help="Verifier dependances et datasets")

    args = parser.parse_args()

    print("="*70)
    print("PIPELINE RECONNAISSANCE EMOTIONNELLE MULTIMODALE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Verification
    if args.check or args.all:
        if not check_requirements():
            return
        available_datasets = check_datasets()

        if args.check:
            return

    # Determiner quoi executer
    run_all = args.all
    run_prep = args.preprocess or run_all
    run_train_flag = args.train or run_all
    run_eval = args.evaluate or run_all

    if not (run_prep or run_train_flag or run_eval):
        parser.print_help()
        print("\nUtilisez --all pour executer tout le pipeline")
        return

    # Preprocessing
    if run_prep:
        run_preprocessing(datasets=args.datasets)

    # Entrainement
    if run_train_flag:
        run_training(
            models=args.models,
            epochs=args.epochs,
            lite=args.lite
        )

    # Evaluation
    if run_eval:
        run_evaluation(fusion_method=args.fusion_method)

    print("\n" + "="*70)
    print("PIPELINE TERMINE")
    print("="*70)


if __name__ == "__main__":
    main()
