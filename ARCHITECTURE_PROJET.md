# Architecture du Projet - Reconnaissance Emotionnelle Multimodale

## Structure des dossiers

```
implementation/
│
├── ARCHITECTURE_PROJET.md          # Ce fichier
├── ARCHITECTURES_EXPLICATIONS.md   # Explications CNN-LSTM, BERT, ViT
│
├── configs/                        # Configurations
│   └── config.py                   # Hyperparametres, chemins, constantes
│
├── preprocessing/                  # Preprocessing par modalite
│   ├── __init__.py
│   ├── wesad_preprocessing.py      # Preprocessing WESAD (chunked)
│   ├── fer2013_preprocessing.py    # Preprocessing FER2013 (chunked)
│   └── goemotions_preprocessing.py # Preprocessing GoEmotions (chunked)
│
├── models/                         # Modeles deep learning
│   ├── __init__.py
│   ├── cnn_lstm_model.py           # CNN-LSTM pour biometrique
│   ├── cnn_fer_model.py            # CNN pour expressions faciales
│   └── bert_model.py               # BERT pour texte
│
├── fusion/                         # Fusion multimodale
│   ├── __init__.py
│   └── multimodal_fusion.py        # Strategies de fusion
│
├── utils/                          # Utilitaires
│   ├── __init__.py
│   ├── data_loader.py              # DataLoaders PyTorch
│   ├── metrics.py                  # Metriques evaluation
│   └── visualization.py            # Graphiques et plots
│
├── notebooks/                      # Notebooks Jupyter
│   ├── 01_exploration_wesad.ipynb
│   ├── 02_exploration_fer2013.ipynb
│   ├── 03_exploration_goemotions.ipynb
│   └── 04_training_evaluation.ipynb
│
├── outputs/                        # Resultats
│   ├── models/                     # Modeles sauvegardes (.pt)
│   ├── figures/                    # Graphiques
│   └── logs/                       # Logs d'entrainement
│
├── train_wesad.py                  # Script entrainement WESAD
├── train_fer2013.py                # Script entrainement FER2013
├── train_goemotions.py             # Script entrainement GoEmotions
├── evaluate.py                     # Evaluation des modeles
└── main.py                         # Pipeline complet
```

---

## Pipeline d'execution

```
1. PREPROCESSING (chunked pour economiser RAM)
   │
   ├── wesad_preprocessing.py
   │   └── Charge .pkl par sujet -> extrait features -> sauvegarde .npz
   │
   ├── fer2013_preprocessing.py
   │   └── Lit CSV par chunks -> convertit pixels -> sauvegarde .npz
   │
   └── goemotions_preprocessing.py
       └── Lit CSV par chunks -> tokenize -> sauvegarde tensors

2. TRAINING (modeles independants)
   │
   ├── train_wesad.py      -> outputs/models/cnn_lstm_wesad.pt
   ├── train_fer2013.py    -> outputs/models/cnn_fer2013.pt
   └── train_goemotions.py -> outputs/models/bert_goemotions.pt

3. EVALUATION & FUSION
   │
   └── evaluate.py + multimodal_fusion.py
       └── Charge les 3 modeles -> predictions -> fusion -> metriques
```

---

## Configuration (configs/config.py)

```python
# Chemins des datasets
WESAD_PATH = "../WESAD/WESAD"
FER2013_PATH = "../challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/fer2013/fer2013.csv"
GOEMOTIONS_PATH = "../goemotions"

# Hyperparametres
BATCH_SIZE = 32          # Reduire si RAM limitee (16 ou 8)
LEARNING_RATE = 1e-4
EPOCHS = 20
CHUNK_SIZE = 1000        # Pour preprocessing chunked

# Labels
WESAD_LABELS = {0: 'baseline', 1: 'stress', 2: 'amusement', 3: 'meditation'}
FER2013_LABELS = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
```

---

## Ordre d'implementation recommande

### Phase 1 : Setup et Preprocessing - COMPLETE
1. [x] Creer structure dossiers
2. [x] config.py - Configuration centralisee
3. [x] wesad_preprocessing.py - Chunked
4. [x] fer2013_preprocessing.py - Chunked
5. [x] goemotions_preprocessing.py - Chunked

### Phase 2 : Modeles individuels - COMPLETE
6. [x] cnn_lstm_model.py + train_wesad.py
7. [x] cnn_fer_model.py + train_fer2013.py
8. [x] bert_model.py + train_goemotions.py

### Phase 3 : Evaluation et Fusion - COMPLETE
9. [x] metrics.py - Accuracy, F1, Confusion matrix
10. [x] multimodal_fusion.py - Late fusion (average, weighted, voting, max)
11. [x] evaluate.py - Evaluation complete
12. [x] data_loader.py - DataLoaders PyTorch avec support chunks

### Phase 4 : Visualisation et Rapport - A FAIRE
13. [ ] visualization.py - Graphiques
14. [ ] Notebooks d'analyse
15. [ ] Rapport final

---

## Dependances (requirements.txt)

```
torch>=1.9.0
transformers>=4.0.0
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
tqdm>=4.50.0
```

Installation:
```bash
pip install torch transformers numpy pandas scikit-learn matplotlib seaborn tqdm
```

---

## Mapping des emotions entre datasets

Pour la fusion, on doit aligner les labels:

| Emotion Commune | WESAD | FER2013 | GoEmotions |
|-----------------|-------|---------|------------|
| Stress/Peur | stress | fear | fear, nervousness |
| Joie | amusement | happy | joy, amusement, excitement |
| Colere | - | angry | anger, annoyance |
| Tristesse | - | sad | sadness, disappointment |
| Neutre | baseline | neutral | neutral |
| Surprise | - | surprise | surprise |
| Degout | - | disgust | disgust |

---

## Strategies de fusion implementees

1. **Moyenne simple** : `(P_bio + P_img + P_txt) / 3`
2. **Moyenne ponderee** : Poids selon performance validation
3. **Vote majoritaire** : Classe avec le plus de votes
4. **MLP Fusion** : Reseau de neurones apprend la combinaison optimale
