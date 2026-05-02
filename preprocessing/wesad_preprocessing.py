"""
Preprocessing WESAD (Wearable Stress and Affect Detection)
Optimise pour 8GB RAM - Traitement par sujet et par fenetre

Signaux E4: ACC, BVP, EDA, HR, TEMP
Labels: 1=baseline, 2=stress, 3=amusement, 4=meditation
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Ajouter le path parent pour imports
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import (
    WESAD_PATH, WESAD_CONFIG, PROCESSED_DIR, RANDOM_SEED
)


def load_subject_pkl(subject_id):
    """
    Charge le fichier .pkl d'un sujet.
    Retourne None si erreur (pour economiser RAM en cas d'echec).
    """
    pkl_path = WESAD_PATH / subject_id / f"{subject_id}.pkl"

    if not pkl_path.exists():
        print(f"Fichier non trouve: {pkl_path}")
        return None

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data
    except Exception as e:
        print(f"Erreur chargement {subject_id}: {e}")
        return None


def extract_e4_features(subject_id):
    """
    Extrait les features des fichiers E4 CSV (plus leger que .pkl).
    Retourne un dict avec les signaux resamples.
    """
    e4_path = WESAD_PATH / subject_id / f"{subject_id}_E4_Data"

    if not e4_path.exists():
        return None

    signals = {}

    for signal_name in WESAD_CONFIG["e4_signals"]:
        csv_path = e4_path / f"{signal_name}.csv"
        if not csv_path.exists():
            continue

        try:
            # Lire en chunks pour economiser RAM
            chunks = []
            for chunk in pd.read_csv(csv_path, header=None, chunksize=10000):
                chunks.append(chunk.values)

            data = np.vstack(chunks)

            # Les 2 premieres lignes sont timestamp et sample_rate
            if signal_name != "IBI":
                timestamp = data[0, 0]
                sample_rate = data[1, 0]
                signal_data = data[2:].astype(np.float32)
            else:
                signal_data = data.astype(np.float32)
                sample_rate = None

            signals[signal_name] = {
                "data": signal_data,
                "sample_rate": sample_rate
            }

            # Liberer memoire
            del chunks, data

        except Exception as e:
            print(f"Erreur lecture {signal_name} pour {subject_id}: {e}")

    return signals


def create_windows(signal, labels, window_size, overlap, sample_rate):
    """
    Decoupe le signal en fenetres avec overlap.

    Args:
        signal: numpy array (n_samples, n_features)
        labels: numpy array (n_samples,)
        window_size: taille fenetre en secondes
        overlap: pourcentage overlap (0-1)
        sample_rate: frequence echantillonnage

    Returns:
        X: (n_windows, window_samples, n_features)
        y: (n_windows,) label majoritaire par fenetre
    """
    window_samples = int(window_size * sample_rate)
    step = int(window_samples * (1 - overlap))

    n_samples = len(signal)
    n_windows = (n_samples - window_samples) // step + 1

    if n_windows <= 0:
        return None, None

    X = []
    y = []

    for i in range(n_windows):
        start = i * step
        end = start + window_samples

        if end > n_samples:
            break

        window_signal = signal[start:end]
        window_labels = labels[start:end]

        # Label majoritaire dans la fenetre
        unique, counts = np.unique(window_labels, return_counts=True)
        majority_label = unique[np.argmax(counts)]

        # Garder seulement les fenetres avec labels cibles
        if majority_label in WESAD_CONFIG["target_labels"]:
            X.append(window_signal)
            y.append(majority_label)

    if len(X) == 0:
        return None, None

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def normalize_signal(signal):
    """Normalisation Z-score par canal."""
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True) + 1e-8
    return (signal - mean) / std


def process_subject_from_pkl(subject_id, window_size=60, overlap=0.5):
    """
    Traite un sujet depuis le fichier .pkl.
    Utilise les donnees chest (RespiBAN) car plus completes.

    Returns:
        X: features (n_windows, window_samples, n_features)
        y: labels (n_windows,)
    """
    print(f"\nTraitement {subject_id}...")

    # Charger pkl
    data = load_subject_pkl(subject_id)
    if data is None:
        return None, None

    try:
        # Extraire signaux chest (700 Hz pour la plupart)
        chest_data = data['signal']['chest']
        labels = data['label']

        # Signaux disponibles: ACC, ECG, EMG, EDA, Temp, Resp
        # On utilise ACC (3D), EDA, Temp pour correspondre a E4
        acc = chest_data['ACC']  # (n, 3)
        eda = chest_data['EDA']  # (n, 1)
        temp = chest_data['Temp']  # (n, 1)

        # Concatener les features
        # ACC a 700Hz, EDA et Temp aussi
        signal = np.hstack([acc, eda, temp])  # (n, 5)

        # Sample rate chest
        sample_rate = 700

        # Normaliser
        signal = normalize_signal(signal)

        # Creer fenetres
        X, y = create_windows(
            signal, labels,
            window_size=window_size,
            overlap=overlap,
            sample_rate=sample_rate
        )

        # Liberer memoire
        del data, chest_data, acc, eda, temp, signal, labels

        if X is not None:
            print(f"  {subject_id}: {len(X)} fenetres extraites")

        return X, y

    except Exception as e:
        print(f"Erreur traitement {subject_id}: {e}")
        del data
        return None, None


def process_subject_from_e4(subject_id, window_size=60, overlap=0.5):
    """
    Traite un sujet depuis les fichiers E4 CSV (plus leger).
    Utilise EDA et HR principalement.
    """
    print(f"\nTraitement E4 {subject_id}...")

    signals = extract_e4_features(subject_id)
    if signals is None or len(signals) == 0:
        return None, None

    # Pour E4, on n'a pas les labels directement
    # Il faudrait les extraire du .pkl ou du quest.csv
    # Pour simplifier, on retourne juste les signaux preprocesses

    try:
        # Utiliser EDA (4Hz) comme signal principal
        if "EDA" not in signals:
            return None, None

        eda = signals["EDA"]["data"]
        sample_rate = signals["EDA"]["sample_rate"]

        # Normaliser
        eda = normalize_signal(eda)

        print(f"  {subject_id}: EDA shape {eda.shape}, SR={sample_rate}Hz")

        return eda, None  # Pas de labels pour E4 seul

    except Exception as e:
        print(f"Erreur E4 {subject_id}: {e}")
        return None, None


def preprocess_wesad(use_pkl=True, window_size=60, overlap=0.5, save=True):
    """
    Preprocessing complet WESAD.

    Args:
        use_pkl: True pour utiliser .pkl (complet), False pour E4 (leger)
        window_size: taille fenetre en secondes
        overlap: overlap entre fenetres
        save: sauvegarder les resultats

    Returns:
        X_all, y_all: donnees combinees de tous les sujets
    """
    print("="*60)
    print("PREPROCESSING WESAD")
    print(f"Mode: {'PKL (complet)' if use_pkl else 'E4 (leger)'}")
    print(f"Window: {window_size}s, Overlap: {overlap*100}%")
    print("="*60)

    X_all = []
    y_all = []
    subject_ids = []

    subjects = WESAD_CONFIG["subjects"]

    for subject_id in tqdm(subjects, desc="Sujets"):
        if use_pkl:
            X, y = process_subject_from_pkl(subject_id, window_size, overlap)
        else:
            X, y = process_subject_from_e4(subject_id, window_size, overlap)

        if X is not None and y is not None:
            X_all.append(X)
            y_all.append(y)
            subject_ids.extend([subject_id] * len(X))

        # Forcer garbage collection
        import gc
        gc.collect()

    if len(X_all) == 0:
        print("Aucune donnee extraite!")
        return None, None

    # Combiner
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    print(f"\nResultat final:")
    print(f"  X shape: {X_all.shape}")
    print(f"  y shape: {y_all.shape}")
    print(f"  Labels distribution: {np.bincount(y_all)}")

    # Mapper labels vers 0, 1, 2
    label_map = {1: 0, 2: 1, 3: 2}  # baseline->0, stress->1, amusement->2
    y_all = np.array([label_map[l] for l in y_all], dtype=np.int64)

    if save:
        output_path = PROCESSED_DIR / "wesad_processed.npz"
        np.savez_compressed(
            output_path,
            X=X_all,
            y=y_all,
            subjects=np.array(subject_ids)
        )
        print(f"\nSauvegarde: {output_path}")

    return X_all, y_all


def get_data_info():
    """Affiche les informations sur les donnees WESAD disponibles."""
    print("\n" + "="*60)
    print("INFORMATIONS WESAD")
    print("="*60)

    for subject_id in WESAD_CONFIG["subjects"]:
        subject_path = WESAD_PATH / subject_id

        if not subject_path.exists():
            print(f"{subject_id}: NON TROUVE")
            continue

        pkl_exists = (subject_path / f"{subject_id}.pkl").exists()
        e4_exists = (subject_path / f"{subject_id}_E4_Data").exists()

        pkl_size = ""
        if pkl_exists:
            size_mb = (subject_path / f"{subject_id}.pkl").stat().st_size / 1e6
            pkl_size = f" ({size_mb:.0f}MB)"

        print(f"{subject_id}: PKL={'OUI'+pkl_size if pkl_exists else 'NON'}, E4={'OUI' if e4_exists else 'NON'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocessing WESAD")
    parser.add_argument("--info", action="store_true", help="Afficher infos dataset")
    parser.add_argument("--use-e4", action="store_true", help="Utiliser E4 au lieu de PKL")
    parser.add_argument("--window", type=int, default=60, help="Taille fenetre (secondes)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap (0-1)")

    args = parser.parse_args()

    if args.info:
        get_data_info()
    else:
        preprocess_wesad(
            use_pkl=not args.use_e4,
            window_size=args.window,
            overlap=args.overlap
        )
