# Étude Comparative de Classificateurs pour la Base BDshape

Ce projet s'inscrit dans le cadre du Master 1 et vise à comparer différentes approches de classification appliquées au 
dataset **BDshape**. Ce jeu de données contient des formes binaires réparties en **9 classes** (animaux, avions, etc.) 
de **11 échantillons** chacune. Il présente des défis spécifiques tels que des occultations (classes 4 et 5), 
des représentations partielles (classes 2 et 3) et des distorsions (classe 6).

L'objectif est d'évaluer la performance de cinq descripteurs de formes via les approches de classification suivantes :
1.  **k-Nearest Neighbors** : Approche supervisée (avec validation Leave-One-Out).
2.  **k-Means** : Approche non-supervisée (clustering avec mapping des classes a posteriori via l'Algorithme Hongrois).
3.  **Vote Majoritaire** : Classificateur combinant les prédictions des descripteurs.

## Architecture du Projet
```
Projet/
│
├── donnees/                       # Données brutes (.E34, .F0, .F2, .GFD, .SA) toutes mélangées
│
├── donnes_numpy/               # Résultats (.npy) générés
│   ├── donnees_chargees.npy    # Dataset complet structuré (Features & Labels par méthode) crée par explore_donnees.py
│   │
│   ├── resultats_knn.npy       # Résultats de kNN crée par knn_test.py
│   ├── resultats_knn_k10.npy   # Résultats de kNN spécifiques pour k=10 (utilisé pour les courbes PR)
│   │
│   ├── resultats_kmeans.npy    # Résultats de kmeans crée par kmeans_test.py
│   │
│   └── resultats_vote.npy      # Résultats du vote majoritaire crée par vote_test.py  
│
├── src/
│   ├── chargement/             # Module de préparation des données
│   │   ├── charge_donnees.py   # Classe ChargeDonnees : parsing des fichiers .MET et organisation
│   │   └── explore_donnees.py  # Script principal : exécute le chargement, l'analyse et la sauvegarde .npy
│   │
│   ├── knn/                    # Module Classification Supervisée via k-Nearest Neighbors
│   │   ├── knn_classif.py      # Classe ClassifKNN : implémentation de l'algo kNN 
│   │   └── knn_test.py         # Script d'évaluation : Leave-One-Out et génération de rapports
│   │
│   ├── kmeans/                 # Module Classification Non-Supervisée via k-Means
│   │   ├── kmeans_classif.py   # Classe ClassifKMeans : implémentation de l'algo kmeans avec Hongrois 
│   │   └── kmeans_test.py      # Script d'évaluation : Clustering, calcul d'inertie et mapping classes
│   │
│   ├── vote_majoritaire/       # Module de Vote Majoritaire via kNN
│   │   ├── vote_classif.py     # Classe ClassifVoteMajoritaire : agrégation des prédictions de plusieurs méthodes
│   │   └── vote_test.py        # Script d'évaluation : fusion de toutes les méthodes et uniquement du top 3
│   │
│   └── utilitaires/            # Outils d'analyse 
│       └── courbe_pr.py        # Génération et sauvegarde des courbes Précision-Rappel
│
├── requirements.txt            # Liste des bibliothèques Python nécessaires
└── README.md                   # Documentation
```

## Installation

Le projet nécessite Python 3.8+.
Installez les dépendances nécessaires via le fichier fourni :

```bash
pip install -r requirements.txt
```

## Utilisation

1. Chargement des données
Lit les fichiers *.MET* dans *data/* (tous mélangés), affiche les statistiques et génère *donnees_chargees.npy*.
```bash
python src/chargement/explore_donnees.py
```

2. Lancer k-Nearest Neighbors
Exécute la validation croisée Leave-One-Out pour tous les descripteurs (par défaut k=5).
```bash
python src/knn/knn_test.py
```

3. Lancer k-Means
Effectue le clustering (par défaut k=9 classes, 10 initialisations) et calcule le taux de reconnaissance via l'algorithme Hongrois.
```bash
python src/kmeans/kmeans_test.py
```

4. Lancer le Vote Majoritaire
Combine les résultats des classifieurs pour une décision finale.
```bash
python src/vote/vote_test.py
```

## Resultats
Les scripts affichent les matrices de confusion et les scores F1 directement dans le terminal. Les fichiers persistants 
sont stockés dans le dossier *donnes_numpy/*.
