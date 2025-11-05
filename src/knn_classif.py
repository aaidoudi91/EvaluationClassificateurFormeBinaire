""" Module knn_classif.py : implémente un classificateur des k-plus-proches voisins (kNN).
Ce module propose une interface avec apprentissage, prédiction et estimation de probabilité. """

import numpy as np
from collections import Counter


class ClassifKNN:
    """ Classificateur des k-plus-proches voisins : méthode supervisée basée sur la distance entre échantillons. """

    def __init__(self, k=5, distance='euclidienne', normaliser=False):
        """ Initialise le classificateur kNN.
        :param k: Nombre de voisins à considérer.
        :param distance: Type de distance : 'euclidienne' pour L2, 'manhattan' pour L1, ou 'minkowski' avec p custom.
        :param normaliser: Si True, normalise les features. """
        self.k = k
        self.distance = distance
        self.normaliser = normaliser
        self.X_train = None  # Matrice des features d'entraînement (sera remplie dans entraine()).
        self.y_train = None  # Vecteur des étiquettes d'entraînement correspondant à X_train.
        self.moyenne = None  # Moyenne des features d'entraînement (utilisée pour normalisation).
        self.ecart_type = None  # Écart-type des features d'entraînement (utilisée pour normalisation).

    def entraine(self, X, y):
        """ Entraîne le classificateur (stocke simplement les données d'entraînement).
        :param X: Matrice des features (n_samples, n_features).
        :param y: Vecteur des labels (n_samples,). """
        self.X_train = X.copy()  # Copie pour éviter les effets de bord
        self.y_train = y.copy()

        if self.normaliser:  # Normalisation si demandée
            self.moyenne = np.mean(X, axis=0)
            self.ecart_type = np.std(X, axis=0)
            self.ecart_type[self.ecart_type == 0] = 1  # Évite division par zéro
            self.X_train = (self.X_train - self.moyenne) / self.ecart_type

    def calcule_distance(self, x1, x2, p=2):
        """ Calcule la distance de Minkowski entre deux vecteurs.
        :param x1 Vecteur à comparer.
        :param x2 Vecteur à comparer.
        :param p: Paramètre de la métrique Minkowski (2 pour euclidienne, 1 pour manhattan).
        :return Distance entre x1 et x2. """
        if p == np.inf:
            return np.max(np.abs(x1 - x2))  # Distance de Chebyshev
        return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)  # Formule Minkowski

    def predire_un(self, x):
        """ Prédit la classe d'un échantillon. (sans utilisier sklearn.neighbors.KNeighborsClassifier...)
        :param x: Vecteur de features (n_features,).
        :return tuple (classe prédite, distances aux k voisins, classes des k voisins). """
        if self.normaliser:  # Normalise si le modèle a été entraîné avec normalisation
            x = (x - self.moyenne) / self.ecart_type

        # Calcul des distances à tous les échantillons d'entraînement
        distances = []
        if self.distance == 'euclidienne':
            p = 2
        elif self.distance == 'manhattan':
            p = 1
        else:
            p = 3  # Valeur par défaut pour Minkowski

        for i in range(len(self.X_train)): # 1. Calcul des distances à tous les échantillons d'entraînement
            dist = self.calcule_distance(x, self.X_train[i], p)
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda x: x[0]) # 2. Tri par distance croissante
        k_proches = distances[:self.k] # 3. Sélection des k plus proches voisins
        k_distances = [d[0] for d in k_proches]  # Liste des distances
        k_labels = [d[1] for d in k_proches]  # Liste des classes associées

        comptage_votes = Counter(k_labels) # 4. Vote majoritaire
        classe_predite = comptage_votes.most_common(1)[0][0] # Classe majoritaire

        return classe_predite, k_distances, k_labels

    def predire(self, X):
        """ Prédit les classes pour un ensemble d'échantillons.
        :param X: Matrice des features (n_samples, n_features).
        :return Vecteur des classes prédites (n_samples,). """
        predictions = []
        for i in range(len(X)):  # Prédiction indépendante pour chaque ligne de X
            pred, _, _ = self.predire_un(X[i])
            predictions.append(pred)
        return np.array(predictions)

    def predire_proba(self, X):
        """ Prédit les probabilités d'appartenance aux classes (basées sur les votes).
         Utile pour la courbe précision-rappel.
        :param X Matrice des features (n_samples, n_features).
        :returnMatrice de probabilités (n_samples, n_classes). """
        n_samples = len(X)
        classes = np.unique(self.y_train)  # Ensemble des classes rencontrées
        n_classes = len(classes)
        probas = np.zeros((n_samples, n_classes))

        # Pour chaque échantillon, calcule la proportion de chaque classe parmi les k voisins
        for i in range(n_samples):
            _, _, k_labels = self.predire_un(X[i])
            for j, classe in enumerate(classes):
                probas[i, j] = k_labels.count(classe) / self.k

        return probas
