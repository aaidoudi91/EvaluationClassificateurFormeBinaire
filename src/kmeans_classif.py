""" Module kmeans_classif.py : implémente un classificateur non supervisé basé sur l'algorithme k-means.
Permet de regrouper les échantillons en clusters et de mapper ces clusters aux classes réelles.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class ClassifKMeans:
    """ Classificateur k-means (nuées dynamiques) via une approche non supervisée """

    def __init__(self, n_clusters=9, max_iter=100, n_init=10, graine_aleatoire=None):
        """ Initialise le classificateur k-Means.
        :param n_clusters le nombre de clusters à créer (par défaut 9, le nombre de classes de BDshape).
        :param max_iter le nombre maximal d'itérations de l’algorithme (par défaut 100).
        :param n_init le nombre d'initialisations aléatoires (on garde celle avec la plus faible inertie).
        :param graine_aleatoire la graine aléatoire pour la reproductibilité.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = graine_aleatoire
        self.centres = None  # Centres finaux des clusters
        self.labels = None  # Étiquettes de cluster pour les points d'entraînement
        self.inertie = None  # Somme des distances intra-cluster

    def initialiser_centres(self, X):
        """ Initialise les centres de clusters aléatoirement parmi les échantillons.
        :param X le tableau numpy des données (n_samples, n_features).
        :return Tableau numpy des centres initialisés (n_clusters, n_features)."""
        np.random.seed(self.random_state)
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[indices].copy()

    def assigner_clusters(self, X, centers):
        """Assigne chaque point au cluster le plus proche selon la distance euclidienne.
        :param X les données à classer (n_samples, n_features).
        :param centers actuels des clusters (n_clusters, n_features).
        :return Tableau des étiquettes de clusters pour chaque échantillon. """
        distances = np.zeros((len(X), self.n_clusters))

        for k in range(self.n_clusters):
            # Distance euclidienne entre chaque point et le centre k
            distances[:, k] = np.sqrt(np.sum((X - centers[k]) ** 2, axis=1))

        # Retourne l'indice du centre le plus proche pour chaque point
        return np.argmin(distances, axis=1)

    def mettre_a_jour_centres(self, X, labels):
        """ Met à jour les centres en calculant le barycentre des points de chaque cluster.
        (Le barycentre d’un ensemble de points est le point moyen de ces points)
        :param X les données d'entrée (n_samples, n_features).
        :param labels (étiquettes) de cluster de chaque point.
        :return Tableau des centres (n_clusters, n_features). """
        centres = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters):
            # Points appartenant au cluster k
            points_cluster = X[labels == k]
            if len(points_cluster) > 0:
                centres[k] = np.mean(points_cluster, axis=0)
            else:
                # Cluster vide : réinitialiser aléatoirement
                centres[k] = X[np.random.randint(len(X))]

        return centres

    def calculer_inertie(self, X, labels, centers):
        """ Calcule l'inertie (somme des erreurs quadratiques intra-cluster).
        :param X les données (n_samples, n_features).
        :param labels (étiquettes) de cluster associé aux échantillons.
        :param centers de clusters.
        :return: Valeur scalaire de l'inertie. """
        inertie = 0
        for k in range(self.n_clusters):
            points_cluster = X[labels == k]
            if len(points_cluster) > 0:
                inertie += np.sum((points_cluster - centers[k]) ** 2)
        return inertie

    def entrainer(self, X):
        """ Entraîne l'algorithme k-means (détaillée en annexe du sujet) sur les données.
        :param X les données d'entraînement (n_samples, n_features).
        :return: L'objet ClassifKMeans entraîné.
        """
        meilleure_inertie = np.inf
        meilleurs_centres = None
        meilleurs_labels = None

        # Essayer n_init initialisations différentes
        for init in range(self.n_init):
            # Étape 1 : Initialisation aléatoire
            centres = self.initialiser_centres(X)

            # Itérations jusqu'à convergence
            for iteration in range(self.max_iter):
                # Étape 2 : Assignation au cluster le plus proche
                labels = self.assigner_clusters(X, centres)

                # Étape 3 : Mise à jour des centres
                nouveaux_centres = self.mettre_a_jour_centres(X, labels)

                # Test de convergence : les centres ne bougent plus
                if np.allclose(centres, nouveaux_centres):
                    break

                centres = nouveaux_centres

            # Calcul de l'inertie finale
            inertie = self.calculer_inertie(X, labels, centres)

            # Garder la meilleure solution
            if inertie < meilleure_inertie:
                meilleure_inertie = inertie
                meilleurs_centres = centres
                meilleurs_labels = labels

        self.centres = meilleurs_centres
        self.labels = meilleurs_labels
        self.inertie = meilleure_inertie

        return self

    def predire(self, X):
        """ Associe de nouveaux points à leurs clusters les plus proches.
        :param X les données à prédire (n_samples, n_features).
        :return: Tableau d'entiers correspondant aux indices des clusters assignés. """
        return self.assigner_clusters(X, self.centres)


def mapper_clusters_classes(cluster_labels, labels_reels, n_clusters=9):
    """ Mappe les clusters trouvés par K-Means aux vraies classes.
    Utilise une approche de minimisation du coût (algorithme hongrois).
    :param cluster_labels les étiquettes de cluster prédit (n_samples,).
    :param labels_reels les étiquettes de classes réelles (n_samples,).
    :param n_clusters le nombre de clusters (par défaut 9).
    :return: Tuple (cluster_vers_classe, predictions) respectivement un dict {id_cluster: id_classe}
        et un tableau des classes finales prédites. """

    # Matrice de coût : -nombre de points du cluster i dans la classe j
    matrice_cout = np.zeros((n_clusters, n_clusters))

    for id_cluster in range(n_clusters):
        for id_classe in range(1, n_clusters + 1):
            masque = cluster_labels == id_cluster
            matrice_cout[id_cluster, id_classe - 1] = -np.sum(labels_reels[masque] == id_classe)

    # Trouver le mapping optimal
    indices_lignes, indices_colonnes = linear_sum_assignment(matrice_cout)

    # Créer le dictionnaire de mapping (cluster -> classe)
    cluster_vers_classe = {cluster_id: class_id + 1 for cluster_id, class_id in zip(indices_lignes, indices_colonnes)}

    # Appliquer le mapping
    predictions = np.array([cluster_vers_classe[c] for c in cluster_labels])

    return cluster_vers_classe, predictions
