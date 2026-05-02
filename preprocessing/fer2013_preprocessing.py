"""
Preprocessing FER2013 (Facial Expression Recognition)
Optimise pour 8GB RAM - Traitement par chunks

Images: 48x48 grayscale
Labels: 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Ajouter le path parent pour imports
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import (
    FER2013_PATH, FER2013_CONFIG, PROCESSED_DIR, CHUNK_SIZE, RANDOM_SEED
)


def parse_pixels(pixel_string):
    """
    Convertit une string de pixels en array numpy 48x48.

    Args:
        pixel_string: "70 80 82 72 ..." (2304 valeurs)

    Returns:
        numpy array (48, 48) dtype uint8
    """
    try:
        pixels = np.fromstring(pixel_string, dtype=np.uint8, sep=' ')
        if pixels.size == 48 * 48:
            return pixels.reshape(48, 48)
        else:
            return np.zeros((48, 48), dtype=np.uint8)
    except:
        return np.zeros((48, 48), dtype=np.uint8)


def normalize_image(image):
    """
    Normalise une image vers [0, 1].

    Args:
        image: numpy array (48, 48) ou (N, 48, 48)

    Returns:
        image normalisee float32
    """
    return image.astype(np.float32) / 255.0


def process_chunk(chunk):
    """
    Traite un chunk du CSV.

    Args:
        chunk: pandas DataFrame avec colonnes 'emotion' et 'pixels'

    Returns:
        X: images (n, 48, 48) float32 normalisees
        y: labels (n,) int64
    """
    images = []
    labels = []

    for _, row in chunk.iterrows():
        # Parser pixels
        img = parse_pixels(row['pixels'])
        images.append(img)

        # Label
        if 'emotion' in row:
            labels.append(int(row['emotion']))
        else:
            labels.append(-1)  # Pas de label (test set)

    X = np.stack(images, axis=0)
    X = normalize_image(X)

    y = np.array(labels, dtype=np.int64)

    return X, y


def preprocess_fer2013(csv_path=None, chunk_size=None, save=True, save_chunks=False):
    """
    Preprocessing complet FER2013 par chunks.

    Args:
        csv_path: chemin vers fer2013.csv
        chunk_size: taille des chunks (defaut depuis config)
        save: sauvegarder le resultat final
        save_chunks: sauvegarder chaque chunk separement (economise RAM)

    Returns:
        X_all, y_all si save_chunks=False
        None si save_chunks=True (donnees sauvegardees sur disque)
    """
    if csv_path is None:
        csv_path = FER2013_PATH

    if chunk_size is None:
        chunk_size = CHUNK_SIZE

    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"Fichier non trouve: {csv_path}")
        return None, None

    print("="*60)
    print("PREPROCESSING FER2013")
    print(f"Fichier: {csv_path}")
    print(f"Chunk size: {chunk_size}")
    print("="*60)

    # Compter le nombre total de lignes
    print("Comptage des lignes...")
    total_lines = sum(1 for _ in open(csv_path, 'r', encoding='utf-8')) - 1  # -1 pour header
    n_chunks = (total_lines // chunk_size) + 1
    print(f"Total: {total_lines} images en {n_chunks} chunks")

    if save_chunks:
        # Mode economie RAM: sauvegarder chaque chunk
        chunk_dir = PROCESSED_DIR / "fer2013_chunks"
        chunk_dir.mkdir(exist_ok=True)

        chunk_idx = 0
        for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunk_size),
                         total=n_chunks, desc="Chunks"):
            X, y = process_chunk(chunk)

            # Sauvegarder chunk
            chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.npz"
            np.savez_compressed(chunk_path, X=X, y=y)

            chunk_idx += 1

            # Liberer memoire
            del X, y, chunk
            import gc
            gc.collect()

        print(f"\n{chunk_idx} chunks sauvegardes dans {chunk_dir}")
        return None, None

    else:
        # Mode standard: garder tout en memoire
        X_all = []
        y_all = []

        for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunk_size),
                         total=n_chunks, desc="Chunks"):
            X, y = process_chunk(chunk)
            X_all.append(X)
            y_all.append(y)

            del chunk
            import gc
            gc.collect()

        # Combiner
        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)

        print(f"\nResultat final:")
        print(f"  X shape: {X_all.shape}")
        print(f"  y shape: {y_all.shape}")

        # Distribution des labels
        if y_all.min() >= 0:
            print(f"  Labels distribution:")
            for label_id, label_name in FER2013_CONFIG["labels"].items():
                count = np.sum(y_all == label_id)
                print(f"    {label_id} ({label_name}): {count}")

        if save:
            output_path = PROCESSED_DIR / "fer2013_processed.npz"
            np.savez_compressed(output_path, X=X_all, y=y_all)
            print(f"\nSauvegarde: {output_path}")

        return X_all, y_all


def load_fer2013_chunks(chunk_dir=None):
    """
    Charge les donnees FER2013 depuis les chunks sauvegardes.
    Utilise un generateur pour economiser la RAM.

    Args:
        chunk_dir: dossier contenant les chunks

    Yields:
        X, y pour chaque chunk
    """
    if chunk_dir is None:
        chunk_dir = PROCESSED_DIR / "fer2013_chunks"

    chunk_dir = Path(chunk_dir)

    if not chunk_dir.exists():
        print(f"Dossier non trouve: {chunk_dir}")
        return

    chunk_files = sorted(chunk_dir.glob("chunk_*.npz"))

    for chunk_path in chunk_files:
        data = np.load(chunk_path)
        yield data['X'], data['y']
        del data


def create_train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Split les donnees en train/val/test.

    Args:
        X: images
        y: labels
        train_ratio: ratio train
        val_ratio: ratio validation

    Returns:
        dict avec X_train, y_train, X_val, y_val, X_test, y_test
    """
    np.random.seed(RANDOM_SEED)

    n = len(X)
    indices = np.random.permutation(n)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return {
        'X_train': X[train_idx],
        'y_train': y[train_idx],
        'X_val': X[val_idx],
        'y_val': y[val_idx],
        'X_test': X[test_idx],
        'y_test': y[test_idx]
    }


def visualize_samples(X, y, n_samples=10, save_path=None):
    """
    Visualise quelques exemples du dataset.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib non disponible pour visualisation")
        return

    n_samples = min(n_samples, len(X))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    indices = np.random.choice(len(X), n_samples, replace=False)

    for i, idx in enumerate(indices):
        ax = axes[i]
        img = X[idx]

        # Si normalise [0,1], convertir pour affichage
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        ax.imshow(img, cmap='gray')

        label_name = FER2013_CONFIG["labels"].get(y[idx], "unknown")
        ax.set_title(f"{label_name} ({y[idx]})")
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Figure sauvegardee: {save_path}")
    else:
        plt.show()

    plt.close()


def get_data_info():
    """Affiche les informations sur les donnees FER2013."""
    print("\n" + "="*60)
    print("INFORMATIONS FER2013")
    print("="*60)

    csv_path = Path(FER2013_PATH)

    if not csv_path.exists():
        print(f"Fichier non trouve: {csv_path}")
        return

    # Taille fichier
    size_mb = csv_path.stat().st_size / 1e6
    print(f"Fichier: {csv_path.name}")
    print(f"Taille: {size_mb:.1f} MB")

    # Lire header et quelques lignes
    df_head = pd.read_csv(csv_path, nrows=5)
    print(f"Colonnes: {list(df_head.columns)}")

    # Compter les lignes
    total = sum(1 for _ in open(csv_path, 'r', encoding='utf-8')) - 1
    print(f"Total images: {total}")

    # Distribution (lecture par chunks)
    print("\nDistribution des labels:")
    label_counts = {i: 0 for i in range(7)}

    for chunk in pd.read_csv(csv_path, chunksize=5000, usecols=['emotion']):
        for label in chunk['emotion']:
            if 0 <= label <= 6:
                label_counts[label] += 1

    for label_id, count in label_counts.items():
        label_name = FER2013_CONFIG["labels"][label_id]
        pct = count / total * 100
        print(f"  {label_id} ({label_name:8s}): {count:6d} ({pct:.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocessing FER2013")
    parser.add_argument("--info", action="store_true", help="Afficher infos dataset")
    parser.add_argument("--chunks", action="store_true", help="Sauvegarder par chunks (economie RAM)")
    parser.add_argument("--chunk-size", type=int, default=500, help="Taille des chunks")
    parser.add_argument("--visualize", action="store_true", help="Visualiser exemples")

    args = parser.parse_args()

    if args.info:
        get_data_info()
    elif args.visualize:
        # Charger et visualiser
        processed_path = PROCESSED_DIR / "fer2013_processed.npz"
        if processed_path.exists():
            data = np.load(processed_path)
            visualize_samples(data['X'], data['y'],
                            save_path=PROCESSED_DIR / "fer2013_samples.png")
        else:
            print("Executez d'abord le preprocessing sans --visualize")
    else:
        preprocess_fer2013(
            chunk_size=args.chunk_size,
            save_chunks=args.chunks
        )
