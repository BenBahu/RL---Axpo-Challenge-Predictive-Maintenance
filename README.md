# Projet de Maintenance Prédictive - Détection d'Anomalies dans les Centrales Hydroélectriques

## 📋 Description

Ce projet vise à développer un système de détection d'anomalies pour les opérations de vannes dans les centrales hydroélectriques. Il se concentre sur l'analyse des séquences de fermeture de vannes et la détection de comportements anormaux à l'aide de techniques d'apprentissage automatique avancées.

Le projet est structuré en deux tâches principales :
- **Task 1** : Préprocessing des données et détermination des temps de fermeture/ouverture des vannes avec un réseau TCN (Temporal Convolutional Network)
- **Task 2** : Détection d'anomalies avec Autoencodeur et classification des types d'anomalies avec HDBSCAN

## 🏭 Contexte Industriel

Les données proviennent d'une centrale hydroélectrique (KSL) avec :
- **3 groupes de machines** : MG1, MG2, MG3
- **2 étages** : Mapragg et Sarelli
- **Signaux mesurés** :
  - Puissance active (MW)
  - Position des vannes à bille (ouvert/fermé)
  - Position des guide-vanes (%)
  - Pression d'eau en amont et en aval (bar)

## 📁 Structure du Projet

```
.
├── GroupA_Task1.ipynb              # Préprocessing et analyse des données
├── GroupA_Task2.ipynb              # Autoencodeur et HDBSCAN pour détection d'anomalies
├── GroupA_anomaliesGeneration.py   # Bibliothèque de génération d'anomalies synthétiques
├── GroupA_Report.pdf               # Rapport détaillé du projet
└── README.md                       # Ce fichier
```

## 🔧 Tâche 1 : Préprocessing et Analyse

### Objectifs
1. **Préprocessing des données** :
   - Synchronisation des signaux temporels
   - Détection et gestion des gaps dans les données
   - Segmentation des séries temporelles
   - Lissage avec moyenne mobile exponentielle (EMA)

2. **Détection des transitions** :
   - Identification des événements d'ouverture/fermeture des vannes
   - Extraction des fenêtres temporelles autour des transitions

3. **Détermination des temps de fermeture/ouverture** :
   - Utilisation d'un réseau TCN (Temporal Convolutional Network)
   - Prédiction précise des durées de transition

4. **Détection d'anomalies** :
   - Analyse des séquences de fermeture pour identifier des comportements anormaux

### Paramètres Principaux
```python
GAP_THRESHOLD_SECONDS = 3600   # Seuil pour la segmentation (1 heure)
MIN_POINTS_PER_SEGMENT = 100   # Nombre minimum de points par segment
EMA_ALPHA = 0.1                # Facteur de lissage EMA
```

### Fonctionnalités Clés
- **Analyse des gaps** : Identification des interruptions dans les données
- **Segmentation** : Division des séries temporelles en segments continus
- **Normalisation temporelle** : Alignement des signaux sur une grille temporelle uniforme
- **Détection de transitions** : Identification automatique des changements d'état des vannes

## 🤖 Tâche 2 : Détection d'Anomalies avec Autoencodeur

### Objectifs
1. **Extraction de fenêtres** :
   - Fenêtres de 360 secondes (180 avant + 180 après) centrées sur les transitions de fermeture
   - Séparation des régimes turbine (puissance > 0) et pompe (puissance ≤ 0)

2. **Entraînement d'autoencodeurs** :
   - Autoencodeur séparé pour chaque régime (turbine/pompe)
   - Réduction de dimension et reconstruction des séquences normales
   - Calcul des erreurs de reconstruction comme score d'anomalie

3. **Classification des types d'anomalies** :
   - Utilisation de HDBSCAN pour le clustering des anomalies
   - Estimation de probabilité conjointe des types d'anomalies
   - Identification de patterns d'anomalies récurrents

### Architecture
- **Données d'entraînement** : Fenêtres de fermeture normales
- **Données de test** : Fenêtres normales et anormales
- **Métrique** : Erreur de reconstruction (MSE) pour détecter les anomalies

## 🧪 Génération d'Anomalies Synthétiques

Le module `GroupA_anomaliesGeneration.py` fournit une bibliothèque complète pour générer des anomalies synthétiques dans les séquences de fermeture de vannes.

### Types d'Anomalies Implémentées

1. **Spikes (Pointes)** : `inject_closing_spikes`
   - Pointes isolées dans la séquence de fermeture
   - Amplitude configurable en multiples de l'écart-type local

2. **Level Shift (Changement de niveau)** : `inject_closing_level_shift`
   - Décalage constant de la moyenne sur un segment
   - Simule un changement de régime soudain

3. **Linear Drift (Dérive linéaire)** : `inject_closing_linear_drift`
   - Dérive linéaire progressive sur un segment
   - Simule une dégradation graduelle

4. **Variance Change (Changement de variance)** : `inject_closing_variance_change`
   - Augmentation ou diminution de la volatilité
   - Simule des bursts de bruit ou un amortissement

5. **Sinusoidal (Oscillation sinusoïdale)** : `inject_closing_sinusoidal`
   - Oscillation périodique ajoutée
   - Simule des vibrations mécaniques ou résonances

6. **Delayed Closure (Fermeture retardée)** : `inject_closing_delayed_closure`
   - Décalage temporel de la séquence de fermeture
   - Simule des retards mécaniques ou de contrôle

7. **Water Hammer Spike (Pointe de coup de bélier)** : `inject_closing_water_hammer_spike`
   - Amplification d'un pic existant
   - Simule des pics de pression dangereux

8. **Signal Dropout (Perte de signal)** : `inject_closing_signal_dropout`
   - Perte temporaire de signal (valeurs à zéro)
   - Simule des pannes de capteurs ou problèmes de communication

9. **Time Warp (Déformation temporelle)** : `inject_closing_time_warp`
   - Accélération ou ralentissement de la séquence
   - Simule une fermeture trop rapide ou trop lente

### Caractéristiques
- Toutes les anomalies sont injectées uniquement dans la **séquence de fermeture** (indices [180, 360))
- Placement biaisé vers le centre de transition (autour de l'index 200)
- Paramètres configurables pour chaque type d'anomalie
- Reproducibilité via `random_state`

## 📊 Données

### Format des Données
- **Format d'entrée** : Fichiers Parquet avec colonnes :
  - `ts` : Timestamp
  - `signal_id` : Identifiant du signal
  - `value` : Valeur mesurée

### Signaux Disponibles
- `active_power` : Puissance active (MW)
- `ball_valve_open` : Vanne ouverte (booléen)
- `ball_valve_closed` : Vanne fermée (booléen)
- `guide_vane_position` : Position des guide-vanes (%)
- `water_pressure_upstream` : Pression amont (bar)
- `water_pressure_downstream` : Pression aval (bar)

## 🚀 Utilisation

### Prérequis
```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch scipy hdbscan optuna tqdm pyarrow
```

### Exécution de Task 1
1. Ouvrir `GroupA_Task1.ipynb`
2. Configurer les paramètres de preprocessing
3. Exécuter les cellules pour :
   - Charger et mapper les signaux
   - Préprocesser les données
   - Extraire les transitions
   - Entraîner le modèle TCN
   - Détecter les anomalies

### Exécution de Task 2
1. Ouvrir `GroupA_Task2.ipynb`
2. Configurer les chemins de données (`DATA_DIR`, `OUTPUT_DIR`)
3. Exécuter les cellules pour :
   - Préprocesser les données et extraire les fenêtres
   - Entraîner les autoencodeurs (turbine et pompe)
   - Appliquer HDBSCAN pour la classification
   - Évaluer les performances

### Génération d'Anomalies
```python
from GroupA_anomaliesGeneration import inject_closing_spikes, inject_closing_level_shift

# Exemple : Injection de spikes
window_perturbed, spike_indices = inject_closing_spikes(
    window=normal_window,
    n_spikes=5,
    magnitude_range=(2.0, 5.0),
    random_state=42
)

# Exemple : Injection de level shift
window_shifted, (start, end), shift = inject_closing_level_shift(
    window=normal_window,
    segment_length=50,
    shift_factor=3.0,
    random_state=42
)
```

## 📈 Résultats

Le projet permet de :
- ✅ Préprocesser efficacement les données de capteurs industriels
- ✅ Détecter automatiquement les transitions de vannes
- ✅ Prédire les durées de fermeture/ouverture avec précision
- ✅ Identifier les anomalies dans les séquences de fermeture
- ✅ Classifier les types d'anomalies détectées
- ✅ Générer des anomalies synthétiques pour l'augmentation de données

## 📝 Notes Techniques

### Fenêtres Temporelles
- **Taille** : 360 secondes (180 avant + 180 après la transition)
- **Centrage** : Sur les événements de fermeture de vannes
- **Normalisation** : Standardisation (moyenne=0, écart-type=1)

### Régimes Opérationnels
- **Turbine** : `active_power > 0` (production d'électricité)
- **Pompe** : `active_power ≤ 0` (pompage)

### Gestion des Gaps
- Forward fill jusqu'à 5 minutes
- Gaps plus longs laissés comme NaN
- Segmentation automatique sur gaps > 1 heure

## 👥 Auteurs

Groupe A - EPFL MA3 - Machine Learning for Predictive Maintenance

## 📄 Licence

Ce projet est réalisé dans le cadre d'un cours académique à l'EPFL.

## 🔗 Références

- Rapport détaillé : `GroupA_Report.pdf`
- Documentation des notebooks : Voir les commentaires dans les cellules
