# Documentation du Preprocessing
## Projet : Reconnaissance Emotionnelle Multimodale

**Date** : Février 2024
**Auteur** : [Votre nom]
**Contrainte matérielle** : 8 GB RAM

---

## Table des Matières

1. [Introduction](#1-introduction)
2. [Preprocessing WESAD (Biométrique)](#2-preprocessing-wesad-biométrique)
3. [Preprocessing FER2013 (Images)](#3-preprocessing-fer2013-images)
4. [Preprocessing GoEmotions (Texte)](#4-preprocessing-goemotions-texte)
5. [Stratégies d'Optimisation Mémoire](#5-stratégies-doptimisation-mémoire)
6. [Fichiers Générés](#6-fichiers-générés)
7. [Références](#7-références)

---

## 1. Introduction

### 1.1 Objectif du Preprocessing

Le preprocessing est une étape cruciale qui transforme les données brutes en un format exploitable par les algorithmes de deep learning. Les objectifs sont :

1. **Uniformisation** : Mettre toutes les données dans un format cohérent
2. **Normalisation** : Ramener les valeurs à des échelles comparables
3. **Segmentation** : Découper les données continues en échantillons exploitables
4. **Labellisation** : Associer chaque échantillon à une classe cible
5. **Optimisation** : Adapter le traitement aux contraintes mémoire (8 GB RAM)

### 1.2 Vue d'ensemble des Datasets

| Dataset | Type | Taille Brute | Exemples | Classes |
|---------|------|--------------|----------|---------|
| WESAD | Signaux physiologiques | ~14 GB (PKL) / 90 MB (E4) | 15 sujets | 3 (baseline, stress, amusement) |
| FER2013 | Images faciales | 301 MB | 35,887 | 7 émotions |
| GoEmotions | Texte (Reddit) | 42 MB | 211,225 | 28 émotions (multi-label) |

---

## 2. Preprocessing WESAD (Biométrique)

### 2.1 Description du Dataset

**WESAD** (Wearable Stress and Affect Detection) contient des signaux physiologiques collectés sur 15 sujets durant différentes phases émotionnelles.

**Source** : https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection

**Capteurs disponibles** :
- **RespiBAN** (poitrine) : ECG, EDA, EMG, Respiration, Température, Accéléromètre
- **Empatica E4** (poignet) : EDA, BVP, Température, Accéléromètre, HR

### 2.2 Choix des Données E4 vs PKL

#### Problème rencontré
```
Fichiers PKL : ~975 MB par sujet
Total : ~14 GB pour 15 sujets
Contrainte : 8 GB RAM
Résultat : MemoryError lors du chargement
```

#### Solution adoptée
```
Fichiers E4 (CSV) : ~6 MB par sujet
Total : ~90 MB pour 15 sujets
Résultat : Chargement sans problème
```

#### Justification
| Critère | PKL (RespiBAN) | E4 (CSV) |
|---------|----------------|----------|
| Taille | ~975 MB/sujet | ~6 MB/sujet |
| Fréquence | 700 Hz | 4-64 Hz |
| Compatibilité RAM | Non (8GB) | Oui |
| Signaux clés | Disponibles | Disponibles (EDA, HR) |
| Application réelle | Capteur médical | Montre connectée |

**Conclusion** : Les données E4 sont représentatives des wearables grand public (montres connectées), ce qui correspond mieux à l'objectif applicatif du projet.

### 2.3 Signal Utilisé : EDA (Electrodermal Activity)

#### Qu'est-ce que l'EDA ?
L'**activité électrodermale** (ou GSR - Galvanic Skin Response) mesure la conductance électrique de la peau, qui varie avec la transpiration.

#### Pourquoi l'EDA ?
| Raison | Explication |
|--------|-------------|
| **Indicateur de stress** | L'EDA augmente lors d'une activation du système nerveux sympathique (stress, excitation) |
| **Fiabilité** | Signal robuste et bien documenté dans la littérature |
| **Fréquence adaptée** | 4 Hz suffit pour capturer les variations émotionnelles (qui sont lentes, ~0.1-0.5 Hz) |
| **Légèreté** | Fichier petit (~280 KB par sujet) |

#### Caractéristiques du signal EDA
```
Fréquence d'échantillonnage : 4 Hz (4 mesures par seconde)
Unité : microSiemens (μS)
Plage typique : 0.1 - 20 μS
```

### 2.4 Traitement 1 : Extraction des Timestamps (Labels)

#### Problème
Les fichiers E4 contiennent uniquement les signaux bruts, sans indication de la phase émotionnelle.

#### Solution
Extraire les timestamps des phases depuis le fichier `quest.csv`.

#### Format du fichier quest.csv
```
# ORDER;Base;TSST;Medi 1;Fun;Medi 2;...
# START;7.08;39.55;70.19;81.25;93.38;...  (en minutes)
# END;26.32;50.3;77.1;87.47;100.15;...    (en minutes)
```

#### Mapping des phases vers les labels
| Phase | Description | Label |
|-------|-------------|-------|
| Base | Baseline (repos) | 0 |
| TSST | Trier Social Stress Test | 1 |
| Fun | Visionnage vidéos amusantes | 2 |
| Medi 1, Medi 2 | Méditation | Ignoré |
| sRead, fRead | Lecture | Ignoré |

#### Justification
- **Baseline** : État neutre de référence
- **TSST** : Protocole standardisé d'induction du stress (présentation orale + calcul mental)
- **Fun** : Induction d'émotions positives par vidéos
- **Méditation/Lecture** : Exclus car moins pertinents pour la détection stress/bien-être

### 2.5 Traitement 2 : Normalisation Z-Score

#### Problème
Les valeurs d'EDA varient fortement entre individus :
- Sujet A : 0.2 - 2.0 μS
- Sujet B : 1.0 - 15.0 μS

Un modèle entraîné sur A ne fonctionnerait pas sur B.

#### Solution : Normalisation Z-Score
```
           x - μ
x_norm = ─────────
            σ

où : μ = moyenne du signal
     σ = écart-type du signal
```

#### Exemple
```
Avant normalisation : [0.3, 0.4, 0.5, 2.1, 3.5]
Moyenne (μ) = 1.36
Écart-type (σ) = 1.31

Après normalisation : [-0.81, -0.73, -0.66, 0.56, 1.63]
```

#### Justification
| Avantage | Explication |
|----------|-------------|
| **Centrage** | Moyenne = 0 pour tous les sujets |
| **Mise à l'échelle** | Écart-type = 1 pour tous |
| **Comparabilité** | Les valeurs deviennent comparables entre sujets |
| **Convergence** | Les réseaux de neurones convergent plus vite avec des données normalisées |

### 2.6 Traitement 3 : Fenêtrage (Windowing)

#### Problème
Le signal EDA est continu (~30,000 échantillons par sujet). Les réseaux de neurones nécessitent des entrées de taille fixe.

#### Solution : Découpage en fenêtres
```
Signal continu:
[═══════════════════════════════════════════════════════]
 0s                                                   120s

Fenêtres de 10 secondes avec 50% d'overlap:
[══════════]                         → Fenêtre 1 (0-10s)
     [══════════]                    → Fenêtre 2 (5-15s)
          [══════════]               → Fenêtre 3 (10-20s)
               [══════════]          → Fenêtre 4 (15-25s)
                    ...
```

#### Paramètres choisis
| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| **Taille fenêtre** | 10 secondes | Assez long pour capturer les variations EDA (~2-5s pour un pic) |
| **Overlap** | 50% | Augmente le nombre d'échantillons, capte les transitions |
| **Samples/fenêtre** | 40 (10s × 4Hz) | Taille raisonnable pour un réseau |

#### Calcul du nombre de fenêtres
```
Pour une phase de 20 minutes (1200 secondes) :
- Sans overlap : 1200 / 10 = 120 fenêtres
- Avec 50% overlap : 1200 / 5 - 1 = 239 fenêtres
```

#### Justification de l'overlap
| Avantage | Explication |
|----------|-------------|
| **Data augmentation** | Double le nombre d'échantillons |
| **Capture des transitions** | Une fenêtre peut manquer un pic, l'autre le capture |
| **Robustesse** | Le modèle voit le même signal sous différents cadrages |

### 2.7 Résultat Final WESAD

```
Entrée :  15 sujets × ~6 MB de données E4
Sortie :  wesad_e4_processed.npz (0.24 MB)

X shape : (6725, 40, 1)
          ├── 6725 fenêtres totales
          ├── 40 échantillons par fenêtre (10s × 4Hz)
          └── 1 feature (EDA)

y shape : (6725,)
          └── Labels : 0 (baseline), 1 (stress), 2 (amusement)

Distribution :
  - Baseline :  3566 (53%)
  - Stress :    2027 (30%)
  - Amusement : 1132 (17%)
```

---

## 3. Preprocessing FER2013 (Images)

### 3.1 Description du Dataset

**FER2013** (Facial Expression Recognition 2013) contient des images de visages annotées avec 7 émotions.

**Source** : Kaggle Facial Expression Recognition Challenge

**Caractéristiques** :
- 35,887 images
- Taille : 48 × 48 pixels
- Niveaux de gris
- 7 classes d'émotions

### 3.2 Format des Données Brutes

Le fichier CSV contient :
```csv
emotion,pixels,Usage
0,"70 80 82 72 58 58 60 63...",Training
3,"151 150 147 155 148 133...",Training
```

- `emotion` : Label (0-6)
- `pixels` : 2304 valeurs (48×48) séparées par des espaces
- `Usage` : Training/PublicTest/PrivateTest

### 3.3 Traitement 1 : Lecture par Chunks

#### Problème
```
Fichier CSV : 301 MB
Chargement complet avec pandas : ~1.5 GB en RAM (expansion des strings)
Contrainte : 8 GB RAM (autres processus actifs)
```

#### Solution : Lecture par chunks
```python
# Au lieu de :
df = pd.read_csv("fer2013.csv")  # Charge tout en mémoire

# On fait :
for chunk in pd.read_csv("fer2013.csv", chunksize=500):
    process(chunk)
    save(chunk)
    del chunk  # Libère la mémoire
```

#### Paramètres
| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| **chunksize** | 500 | Compromis entre vitesse et mémoire |
| **Nombre de chunks** | 72 | 35,887 ÷ 500 ≈ 72 |

### 3.4 Traitement 2 : Conversion String → Array

#### Problème
Les pixels sont stockés comme string : `"70 80 82 72..."` (2304 nombres).

#### Solution
```python
def parse_pixels(pixel_string):
    # "70 80 82 72..." → array numpy (48, 48)
    pixels = np.fromstring(pixel_string, dtype=np.uint8, sep=' ')
    return pixels.reshape(48, 48)
```

#### Justification
| Aspect | String | Array numpy |
|--------|--------|-------------|
| Taille mémoire | ~10 KB | ~2.3 KB |
| Accès pixel | Lent (parsing) | Instantané |
| Opérations math | Impossible | Natives |
| Input CNN | Non compatible | Compatible |

### 3.5 Traitement 3 : Normalisation [0-255] → [0-1]

#### Problème
Les valeurs de pixels sont en uint8 (0-255). Les réseaux de neurones fonctionnent mieux avec des valeurs petites.

#### Solution
```python
image_normalized = image / 255.0
```

#### Exemple
```
Avant : [70, 80, 82, 150, 255, 0]
Après : [0.27, 0.31, 0.32, 0.59, 1.0, 0.0]
```

#### Justification
| Raison | Explication |
|--------|-------------|
| **Stabilité numérique** | Évite les overflow/underflow |
| **Convergence** | Gradients plus stables |
| **Standard** | Pratique courante en vision par ordinateur |
| **Plage [0-1]** | Compatible avec sigmoid, etc. |

### 3.6 Traitement 4 : Sauvegarde en .npz

#### Format choisi
```
fer2013_chunks/
├── chunk_0000.npz  → {X: (500, 48, 48), y: (500,)}
├── chunk_0001.npz
├── ...
└── chunk_0071.npz
```

#### Justification du format .npz
| Avantage | Explication |
|----------|-------------|
| **Compression** | ~3x plus petit que CSV |
| **Chargement rapide** | Format binaire natif numpy |
| **Multi-arrays** | Stocke X et y ensemble |
| **Lazy loading** | Peut charger un array à la fois |

### 3.7 Labels FER2013

| Code | Émotion | Nombre | Pourcentage |
|------|---------|--------|-------------|
| 0 | Angry | 4,953 | 13.8% |
| 1 | Disgust | 547 | 1.5% |
| 2 | Fear | 5,121 | 14.3% |
| 3 | Happy | 8,989 | 25.0% |
| 4 | Sad | 6,077 | 16.9% |
| 5 | Surprise | 4,002 | 11.2% |
| 6 | Neutral | 6,198 | 17.3% |

**Note** : Déséquilibre de classes (Disgust très minoritaire). À considérer lors de l'entraînement.

### 3.8 Résultat Final FER2013

```
Entrée :  fer2013.csv (301 MB)
Sortie :  fer2013_chunks/ (72 fichiers, ~35 MB total)

Par chunk :
  X shape : (500, 48, 48) float32 normalisé [0-1]
  y shape : (500,) int64

Total :
  35,887 images de 48×48 pixels
  7 classes d'émotions
```

---

## 4. Preprocessing GoEmotions (Texte)

### 4.1 Description du Dataset

**GoEmotions** est un dataset de Google contenant des commentaires Reddit annotés avec 28 émotions.

**Source** : https://github.com/google-research/google-research/tree/master/goemotions

**Caractéristiques** :
- 211,225 exemples
- Texte en anglais
- 28 émotions (multi-label : un texte peut avoir plusieurs émotions)

### 4.2 Format des Données Brutes

```csv
text,id,author,subreddit,...,admiration,amusement,anger,...,neutral
"That game hurt.",eew5j0j,Brdd9,nrl,...,0,0,0,...,0
"Man I love reddit.",eeibobj,MrsRobertshaw,facepalm,...,0,0,0,...,0
```

### 4.3 Traitement 1 : Lecture par Chunks

#### Problème
3 fichiers CSV totalisant 211,225 lignes.

#### Solution
```python
for chunk in pd.read_csv(csv_file, chunksize=500):
    texts, labels = process_chunk(chunk)
    save_chunk(texts, labels)
```

#### Paramètres
| Paramètre | Valeur |
|-----------|--------|
| chunksize | 500 |
| Nombre de chunks | 423 |

### 4.4 Traitement 2 : Extraction Multi-Label

#### Différence Single-Label vs Multi-Label
```
Single-label (FER2013) :
  Image → UNE émotion (ex: "happy")

Multi-label (GoEmotions) :
  Texte → PLUSIEURS émotions possibles
  "I'm so happy but also nervous!" → [joy=1, nervousness=1, autres=0]
```

#### Format des labels
```python
# 28 colonnes binaires
emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval',
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
            'pride', 'realization', 'relief', 'remorse', 'sadness',
            'surprise', 'neutral']

# Un exemple :
labels = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0]
#                                           ↑ love=1      ↑ sadness=1
```

### 4.5 Traitement 3 : Construction du Vocabulaire

#### Objectif
Créer un mapping mot → identifiant numérique pour une éventuelle utilisation sans BERT.

#### Processus
```python
# Comptage des mots
word_counts = {"the": 15234, "to": 12456, "a": 11234, ...}

# Création du vocabulaire (triés par fréquence)
vocab = {
    "[PAD]": 0,   # Token de padding
    "[UNK]": 1,   # Token pour mots inconnus
    "[CLS]": 2,   # Début de phrase (BERT style)
    "[SEP]": 3,   # Fin de phrase
    "the": 4,
    "to": 5,
    "a": 6,
    ...
}
```

#### Résultat
- **22,836 mots** dans le vocabulaire
- Sauvegardé dans `goemotions_vocab.json`

#### Note importante
Ce vocabulaire simple est fourni comme backup. En pratique, on utilisera le **tokenizer de BERT/DistilBERT** qui a son propre vocabulaire de ~30,000 tokens (incluant des sous-mots).

### 4.6 Traitement 4 : Sauvegarde Séparée Textes/Labels

#### Structure
```
goemotions_chunks/
├── chunk_0000.npz         → labels (500, 28) float32
├── chunk_0000_texts.json  → ["texte1", "texte2", ...]
├── chunk_0001.npz
├── chunk_0001_texts.json
└── ...
```

#### Justification de la séparation
| Raison | Explication |
|--------|-------------|
| **Types différents** | Labels = numpy, Textes = strings |
| **Tokenization différée** | Les textes seront tokenizés par BERT à l'entraînement |
| **Flexibilité** | Peut changer de tokenizer sans re-preprocessing |

### 4.7 Statistiques des Émotions

#### Top 10 des émotions les plus fréquentes
| Rang | Émotion | Occurrences |
|------|---------|-------------|
| 1 | neutral | ~80,000 |
| 2 | approval | ~15,000 |
| 3 | annoyance | ~12,000 |
| 4 | curiosity | ~10,000 |
| 5 | admiration | ~9,000 |
| 6 | disapproval | ~8,000 |
| 7 | amusement | ~7,000 |
| 8 | disappointment | ~6,000 |
| 9 | joy | ~5,500 |
| 10 | confusion | ~5,000 |

#### Statistiques multi-label
```
Labels par exemple :
  - Moyenne : 1.2 labels par texte
  - Maximum : 5 labels
  - Exemples sans label : ~5%
```

### 4.8 Résultat Final GoEmotions

```
Entrée :  3 fichiers CSV (42 MB total)
Sortie :
  - goemotions_chunks/ (423 × 2 fichiers)
  - goemotions_vocab.json (400 KB)

Par chunk :
  labels shape : (500, 28) float32
  texts : liste de 500 strings

Total :
  211,225 textes
  28 émotions (multi-label)
  22,836 mots dans le vocabulaire
```

---

## 5. Stratégies d'Optimisation Mémoire

### 5.1 Récapitulatif des Techniques Utilisées

| Technique | Description | Datasets |
|-----------|-------------|----------|
| **Chunking** | Traiter par petits lots | FER2013, GoEmotions |
| **Données légères** | Utiliser E4 au lieu de PKL | WESAD |
| **Garbage Collection** | `gc.collect()` après chaque chunk | Tous |
| **Types optimisés** | float32 au lieu de float64 | Tous |
| **Compression** | Format .npz compressé | Tous |
| **Générateurs** | `load_chunks()` charge à la demande | FER2013, GoEmotions |

### 5.2 Code des Générateurs

```python
def load_fer2013_chunks():
    """Charge les chunks un par un (économie RAM)."""
    for chunk_path in sorted(chunk_dir.glob("*.npz")):
        data = np.load(chunk_path)
        yield data['X'], data['y']
        del data  # Libère immédiatement
```

### 5.3 Comparaison Mémoire

| Dataset | Sans optimisation | Avec optimisation |
|---------|-------------------|-------------------|
| WESAD | ~14 GB (MemoryError) | ~50 MB |
| FER2013 | ~1.5 GB | ~100 MB par chunk |
| GoEmotions | ~800 MB | ~50 MB par chunk |

---

## 6. Fichiers Générés

### 6.1 Structure Finale

```
implementation/outputs/processed/
│
├── wesad_e4_processed.npz          # 0.24 MB
│   ├── X : (6725, 40, 1) float32   # Fenêtres EDA normalisées
│   ├── y : (6725,) int64           # Labels [0, 1, 2]
│   └── subjects : array string     # ID du sujet pour chaque fenêtre
│
├── fer2013_chunks/                 # ~35 MB total
│   ├── chunk_0000.npz              # {X: (500,48,48), y: (500,)}
│   ├── chunk_0001.npz
│   └── ... (72 fichiers)
│
├── goemotions_chunks/              # ~40 MB total
│   ├── chunk_0000.npz              # {labels: (500, 28)}
│   ├── chunk_0000_texts.json       # ["texte1", "texte2", ...]
│   └── ... (423 × 2 fichiers)
│
└── goemotions_vocab.json           # 400 KB
    └── {vocab: {...}, max_length: 128}
```

### 6.2 Comment Charger les Données

#### WESAD
```python
data = np.load("wesad_e4_processed.npz")
X, y = data['X'], data['y']
# X.shape = (6725, 40, 1)
# y.shape = (6725,)
```

#### FER2013
```python
from preprocessing.fer2013_preprocessing import load_fer2013_chunks

for X_chunk, y_chunk in load_fer2013_chunks():
    # X_chunk.shape = (500, 48, 48)
    # y_chunk.shape = (500,)
    train_on_chunk(X_chunk, y_chunk)
```

#### GoEmotions
```python
from preprocessing.goemotions_preprocessing import load_goemotions_chunks

for texts, labels in load_goemotions_chunks():
    # texts = ["texte1", "texte2", ...] (500 éléments)
    # labels.shape = (500, 28)
    train_on_chunk(texts, labels)
```

---

## 7. Références

### 7.1 Datasets

1. **WESAD** : Schmidt, P., et al. (2018). "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection." ICMI 2018.
   - https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection

2. **FER2013** : Goodfellow, I., et al. (2013). "Challenges in Representation Learning: Facial Expression Recognition Challenge." ICML 2013 Workshop.
   - https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

3. **GoEmotions** : Demszky, D., et al. (2020). "GoEmotions: A Dataset of Fine-Grained Emotions." ACL 2020.
   - https://github.com/google-research/google-research/tree/master/goemotions

### 7.2 Techniques de Preprocessing

1. **Z-Score Normalization** : Standardisation statistique classique
2. **Windowing avec Overlap** : Technique standard en traitement du signal
3. **Chunked Reading** : Pattern de lecture pour gros fichiers en Python/Pandas

### 7.3 Physiologie

1. **EDA (Electrodermal Activity)** : Boucsein, W. (2012). "Electrodermal Activity." Springer.
   - L'EDA reflète l'activité du système nerveux sympathique
   - Augmente avec le stress, l'excitation, l'engagement cognitif

---

## Annexe : Scripts de Preprocessing

| Script | Description | Commande |
|--------|-------------|----------|
| `wesad_e4_preprocessing.py` | Preprocessing WESAD (E4) | `python -m preprocessing.wesad_e4_preprocessing` |
| `fer2013_preprocessing.py` | Preprocessing FER2013 | `python -m preprocessing.fer2013_preprocessing --chunks` |
| `goemotions_preprocessing.py` | Preprocessing GoEmotions | `python -m preprocessing.goemotions_preprocessing --chunks` |

---

*Document généré dans le cadre du projet de mémoire sur la reconnaissance émotionnelle multimodale.*
