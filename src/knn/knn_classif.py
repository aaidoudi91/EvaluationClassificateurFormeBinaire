""" Module knn_classif.py : implémente un classificateur des k-plus-proches voisins (kNN).
Ce module propose une implémentation avec apprentissage, prédiction et estimation de probabilité. """

import numpy as np
from collections import Counter


class ClassifKNN:
    """ Classificateur des k-plus-proches voisins : méthode supervisée basée sur la distance entre échantillons.
    Le principe est d'attribuer à un point la classe majoritaire parmi ses k voisins les plus proches. """

    def __init__(self, k=5, distance='euclidienne', normaliser=False):
        """ Initialise le classificateur kNN.
        :param k: Nombre de voisins à considérer.
        :param distance 'euclidienne' pour L2, 'manhattan' pour L1, ou 'minkowski' avec p custom.
        :param normaliser si True, normalise les features. """
        self.k = k
        self.distance = distance
        self.normaliser = normaliser
        self.X_train = None  # Matrice des features d'entraînement (sera remplie dans entraine()).
        self.y_train = None  # Vecteur des étiquettes d'entraînement correspondant à X_train.
        self.moyenne = None  # Moyenne des features d'entraînement (utilisée pour normalisation).
        self.ecart_type = None  # Écart-type des features d'entraînement (utilisée pour normalisation).

    def entrainer(self, X, y):
        """ Entraîne le classificateur (stocke simplement les données d'entraînement).
        :param X: Matrice des features (n_samples, n_features).
        :param y: Vecteur des labels (n_samples,). """
        self.X_train = X.copy()  # Copie pour ne pas modifier les données originales
        self.y_train = y.copy()

        if self.normaliser:  # Normalisation si demandée
            # Calcul des statistiques sur l'ensemble d'entraînement (axis=0 : colonne par colonne)
            self.moyenne = np.mean(X, axis=0)
            self.ecart_type = np.std(X, axis=0)
            # Gestion du cas où l'écart-type est nul (colonne constante) pour éviter la division par zéro
            self.ecart_type[self.ecart_type == 0] = 1
            # Application de la normalisation Z-score : (x - mu) / sigma
            self.X_train = (self.X_train - self.moyenne) / self.ecart_type

    def calculer_distance(self, x1, x2, p=2):
        """ Calcule la distance de Minkowski entre deux vecteurs : sum(|x1 - x2|^p)^(1/p).
        :param x1 Vecteur à comparer.
        :param x2 Vecteur à comparer.
        :param p: Paramètre de la métrique Minkowski (2 pour euclidienne, 1 pour manhattan, inf=Chebyshev).
        :return Distance entre x1 et x2. """
        if p == np.inf:
            return np.max(np.abs(x1 - x2))  # Distance de Chebyshev (max des différences)
        return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)  # Formule Minkowski

    def predire_un(self, x):
        """ Prédit la classe d'un échantillon. (sans utiliser sklearn.neighbors.KNeighborsClassifier...)
        :param x: Vecteur de features (n_features,).
        :return tuple (classe_predite, k_distances, k_labels). """
        if self.normaliser:  # Normalise si le modèle a été entraîné avec normalisation
            x = (x - self.moyenne) / self.ecart_type

        if self.distance == 'euclidienne':
            p = 2
        elif self.distance == 'manhattan':
            p = 1
        else:
            p = 3  # Valeur par défaut pour Minkowski

        # 1. Calcul des distances entre x et TOUS les points d'entraînement
        distances = []
        for i in range(len(self.X_train)):
            # Compare le vecteur x avec le i-ème vecteur d'entraînement
            dist = self.calculer_distance(x, self.X_train[i], p)
            distances.append((dist, self.y_train[i]))  # Stocke (distance, classe réelle)

        # 2. Tri de la liste par distance croissante
        distances.sort(key=lambda tup: tup[0])
        # 3. Sélection des k plus proches voisins
        k_proches = distances[:self.k]
        # Séparation des distances et des labels pour le retour
        k_distances = [d[0] for d in k_proches]  # Liste des distances
        k_labels = [d[1] for d in k_proches]  # Liste des classes associées

        # 4. Vote majoritaire : la classe la plus fréquente parmi les voisins
        comptage_votes = Counter(k_labels)
        # .most_common(1) retourne une liste [(classe, compte)], on prend [0][0] pour avoir la classe
        classe_predite = comptage_votes.most_common(1)[0][0]

        return classe_predite, k_distances, k_labels

    def predire(self, X):
        """ Prédit les classes pour un ensemble d'échantillons.
        :param X: Matrice des features à prédire (n_samples, n_features).
        :return Vecteur des classes prédites (n_samples,). """
        predictions = []
        for i in range(len(X)):  # Boucle sur chaque échantillon de la matrice
            pred, _, _ = self.predire_un(X[i])  # On ignore les distances et voisins ici
            predictions.append(pred)
        return np.array(predictions)

    def predire_proba(self, X):
        """ Estime les probabilités d'appartenance aux classes (basées sur les votes).
         Utile pour la courbe précision-rappel.
        :param X Matrice des features (n_samples, n_features).
        :returnMatrice de probabilités (n_samples, n_classes). """
        n_samples = len(X)
        classes = np.unique(self.y_train)  # Liste des classes uniques connues
        n_classes = len(classes)  # Assure un ordre constant des colonnes
        probas = np.zeros((n_samples, n_classes))  # Initialisation de la matrice de sortie

        # Pour chaque échantillon, calcule la proportion de chaque classe parmi les k voisins
        for i in range(n_samples):
            _, _, k_labels = self.predire_un(X[i])  # Récupère les classes des k voisins
            for j, classe in enumerate(classes):
                # La probabilité est le ratio : (nombre de voisins de cette classe) / k
                probas[i, j] = k_labels.count(classe) / self.k

        return probas
