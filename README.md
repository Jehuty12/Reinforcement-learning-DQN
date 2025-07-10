"# Projet d'Apprentissage par Renforcement - DQN

Ce projet implémente différents algorithmes d'apprentissage par renforcement, de l'exploration basique avec Q-Learning jusqu'aux réseaux de neurones Deep Q-Network (DQN) avancés.

## 📋 Description du Projet

Le projet est organisé en 3 étapes progressives :

### Step 1 : Frozen Lake - Q-Learning
- **Environnement** : FrozenLake (Gymnasium)
- **Algorithme** : Q-Learning avec exploration ε-greedy adaptative
- **Objectifs** : Apprendre les bases de l'apprentissage par renforcement
- **Fichiers principaux** :
  - `frozen_lake.py` : Implémentation principale avec Q-Learning adaptatif
  - `frozen_lake.ipynb` : Notebook Jupyter pour l'analyse
  - `FrozenLake_tuto.py` : Version tutoriel

### Step 2 : Cliff Walking - SARSA
- **Environnement** : CliffWalking (Gymnasium)
- **Algorithme** : SARSA (State-Action-Reward-State-Action)
- **Objectifs** : Comprendre la différence entre on-policy et off-policy
- **Fichiers principaux** :
  - `cliff.py` : Implémentation SARSA
  - `cliff_sarsa.py` : Version alternative

### Step 3 : Atari Games - Deep Q-Network (DQN)
- **Environnement** : Jeux Atari (ALE/Pong-v5 par défaut)
- **Algorithme** : DQN avec replay buffer et target network
- **Fonctionnalités avancées** :
  - Experience Replay Buffer
  - Target Network
  - Preprocessing des images
  - Prioritized Experience Replay
- **Fichiers principaux** :
  - `run.py` : Point d'entrée pour l'entraînement
  - `trainer.py` : Logique d'entraînement principal
  - `model.py` : Architecture du réseau de neurones
  - `replay_buffer.py` : Buffer d'expérience
  - `preprocessing.py` : Préprocessing des observations
  - `inference.py` : Évaluation du modèle entraîné

## 🚀 Installation et Configuration

### Prérequis
- Python 3.8+
- CUDA (optionnel, pour l'accélération GPU)

### Installation des dépendances

```bash
# Naviguer vers le dossier du projet
cd "Renforcement Learning"

# Installer les dépendances
pip install -r requirement.txt
```

### Dépendances principales
- **gymnasium** : Environnements de jeu
- **torch** : Framework de deep learning
- **numpy** : Calculs numériques
- **matplotlib** : Visualisation
- **opencv-python** : Traitement d'images
- **ale-py** : Arcade Learning Environment pour Atari

## 📚 Utilisation

### Step 1 : Frozen Lake (Q-Learning)

```bash
# Naviguer vers Step_1
cd "Step_1"

# Lancer l'entraînement basique
python frozen_lake.py

# Ou utiliser le notebook Jupyter
jupyter notebook frozen_lake.ipynb
```

### Step 2 : Cliff Walking (SARSA)

```bash
# Naviguer vers Step_2
cd "Step_2"

# Lancer l'entraînement SARSA
python cliff.py
```

### Step 3 : DQN pour Atari

```bash
# Naviguer vers Step_3
cd "Step_3"

# Entraînement avec paramètres par défaut (Pong)
python run.py

# Entraînement personnalisé
python run.py --env "ALE/Breakout-v5" --total-steps 10000000 --lr 0.0001

# Évaluation d'un modèle entraîné
python inference.py --model-path "checkpoints/dqn_pong.pth" --env "ALE/Pong-v5"
```

### Paramètres disponibles pour DQN

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| `--env` | Environnement Atari | "ALE/Pong-v5" |
| `--total-steps` | Nombre total d'étapes | 50,000,000 |
| `--buffer-cap` | Taille du replay buffer | 500,000 |
| `--batch-size` | Taille des mini-batches | 32 |
| `--gamma` | Facteur de discount | 0.99 |
| `--train-freq` | Fréquence d'entraînement | 4 |
| `--target-update` | Fréquence de mise à jour du target network | 10,000 |
| `--lr` | Taux d'apprentissage | 0.00025 |
| `--initial-replay` | Étapes avant l'entraînement | 50,000 |
| `--resume` | Reprendre depuis un checkpoint | None |
| `--save-path` | Chemin de sauvegarde | "checkpoints/dqn_breakout.pth" |

## 📊 Résultats et Métriques

Le projet génère automatiquement :
- **Graphiques de performance** : Récompenses moyennes, scores
- **Vidéos de gameplay** : Enregistrement périodique des parties
- **Checkpoints** : Sauvegarde des modèles entraînés
- **Logs d'entraînement** : Métriques détaillées

## 🏗️ Architecture du Projet

```
DQN/
├── README.md
└── Renforcement Learning/
    ├── requirement.txt
    ├── Step_1/                 # Q-Learning sur FrozenLake
    │   ├── frozen_lake.py
    │   ├── frozen_lake.ipynb
    │   └── ...
    ├── Step_2/                 # SARSA sur CliffWalking
    │   ├── cliff.py
    │   └── cliff_sarsa.py
    └── Step_3/                 # DQN pour Atari
        ├── run.py              # Point d'entrée
        ├── trainer.py          # Logique d'entraînement
        ├── model.py            # Architecture CNN
        ├── replay_buffer.py    # Experience replay
        ├── preprocessing.py    # Préprocessing images
        ├── inference.py        # Évaluation
        └── ...
```

## 🔬 Concepts Implémentés

- **Q-Learning** : Apprentissage par différence temporelle
- **ε-greedy exploration** : Balance exploration/exploitation
- **SARSA** : Algorithme on-policy
- **Deep Q-Network (DQN)** : Q-Learning avec réseaux de neurones
- **Experience Replay** : Réutilisation des expériences passées
- **Target Network** : Stabilisation de l'entraînement
- **Preprocessing** : Normalisation et redimensionnement des images

## 🎯 Environnements Supportés

### Step 1 & 2
- FrozenLake-v1
- CliffWalking-v0

### Step 3 (Atari)
- ALE/Pong-v5
- ALE/Breakout-v5
- ALE/SpaceInvaders-v5
- Tous les jeux Atari compatibles avec ALE

## 🛠️ Dépannage

### Problèmes courants

1. **Erreur CUDA** : Vérifiez l'installation de PyTorch avec support CUDA
2. **Environnement manquant** : Installez `ale-py` pour les jeux Atari
3. **Mémoire insuffisante** : Réduisez `buffer-cap` ou `batch-size`

### Performance

- **GPU recommandé** pour Step 3 (DQN) ou via un serveur cloud ( ex. Google Colab, AWS, kaggle, etc.)
- **RAM minimum** : 8GB pour les grands replay buffers
- **Temps d'entraînement** : Plusieurs heures à jours selon l'environnement

## 📄 Licence

Ce projet est à des fins éducatives et de recherche.

**Note** : Ce projet suit une progression pédagogique du simple (Q-Learning) au complexe (DQN). Il est recommandé de suivre les étapes dans l'ordre pour une meilleure compréhension." 
