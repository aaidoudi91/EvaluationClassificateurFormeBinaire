""" Module kmeans_classif.py : implémente un classificateur non supervisé basé sur l'algorithme k-means.
Permet de regrouper les échantillons en clusters et de mapper ces clusters aux classes réelles. """

import numpy as np
from scipy.optimize import linear_sum_assignment


class ClassifKMeans:
    """ Classificateur k-means via une approche non supervisée.
    L'objectif est de partitionner les données en k clusters. """

    def __init__(self, n_clusters=9, max_iter=100, n_init=10, graine_aleatoire=None):
        """ Initialise le classificateur k-Means.
        :param n_clusters le nombre de clusters à créer (par défaut 9, le nombre de classes de BDshape).
        :param max_iter le nombre maximal d'itérations de l’algorithme (par défaut 100).
        :param n_init le nombre d'initialisations aléatoires (on garde celle avec la plus faible inertie).
        :param graine_aleatoire la graine aléatoire pour la reproductibilité. """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = graine_aleatoire
        self.centres = None  # Centres finaux des clusters
        self.labels = None  # Étiquettes de cluster pour les points d'entraînement
        self.inertie = None  # Somme des carrés des distances intra-cluster (critère à minimiser)

    def initialiser_centres(self, X):
        """ Initialise les centres de clusters aléatoirement parmi les échantillons.
        :param X le tableau numpy des données (n_samples, n_features).
        :return Tableau numpy des centres initialisés. """
        np.random.seed(self.random_state)
        # Choix aléatoire de 'n_clusters' indices uniques parmi les données
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[indices].copy()

    def assigner_clusters(self, X, centres):
        """Assigne chaque point au cluster le plus proche selon la distance euclidienne.
        :param X les données à classer (n_samples, n_features).
        :param centres Coordonnées actuelles des clusters.
        :return Tableau des étiquettes de clusters pour chaque échantillon. """
        distances = np.zeros((len(X), self.n_clusters))

        for k in range(self.n_clusters):
            # Distance euclidienne entre chaque point et le centre k
            # axis=1 effectue la somme sur les dimensions pour obtenir une distance par point
            distances[:, k] = np.sqrt(np.sum((X - centres[k]) ** 2, axis=1))

        # Retourne l'indice du centre le plus proche pour chaque point
        return np.argmin(distances, axis=1)

    def mettre_a_jour_centres(self, X, labels):
        """ Recalcule la position des centres. Le nouveau centre est le barycentre (moyenne)
        des points qui lui sont assignés.
        :param X les données d'entrée (n_samples, n_features).
        :param labels (étiquettes) de cluster de chaque point.
        :return Tableau des centres (n_clusters, n_features). """
        centres = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters):
            # Extraction des points appartenant au cluster k
            points_cluster = X[labels == k]
            if len(points_cluster) > 0:
                centres[k] = np.mean(points_cluster, axis=0)  # Le nouveau centre est la moyenne des points du cluster
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
                # Somme des carrés des distances entre les points du cluster et leur centre
                inertie += np.sum((points_cluster - centers[k]) ** 2)
        return inertie

    def entrainer(self, X):
        """ Entraîne l'algorithme k-means sur les données. On conserve le modèle ayant la plus faible inertie finale.
        :param X les données d'entraînement (n_samples, n_features).
        :return: L'objet ClassifKMeans entraîné. """
        meilleure_inertie = np.inf
        meilleurs_centres = None
        meilleurs_labels = None

        # Essayer n_init initialisations différentes
        for init in range(self.n_init):
            # Étape 1 : initialisation aléatoire
            centres = self.initialiser_centres(X)

            # Itérations jusqu'à convergence
            for iteration in range(self.max_iter):
                # Étape 2 : assignation au cluster le plus proche
                labels = self.assigner_clusters(X, centres)

                # Étape 3 : mise à jour des centres
                nouveaux_centres = self.mettre_a_jour_centres(X, labels)

                # Test de convergence : les centres ne bougent plus
                if np.allclose(centres, nouveaux_centres):
                    break

                centres = nouveaux_centres

            # Étape 4 : calcul de l'inertie finale
            inertie = self.calculer_inertie(X, labels, centres)

            # Si cette tentative est la meilleure trouvée jusqu'ici, on la garde
            if inertie < meilleure_inertie:
                meilleure_inertie = inertie
                meilleurs_centres = centres
                meilleurs_labels = labels

        # Enregistrement des meilleurs paramètres dans l'objet
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
    """ Mappe les clusters trouvés par K-Means aux vraies classes via une approche de minimisation du coût
    (algorithme hongrois). Cette fonction est définie hors de la classe ClassifKMeans pour garantir
    l'intégrité de l'approche non supervisée.
    :param cluster_labels les étiquettes de cluster prédit (n_samples,).
    :param labels_reels les étiquettes de classes réelles (n_samples,).
    :param n_clusters le nombre de clusters (par défaut 9).
    :return: Tuple (cluster_vers_classe, predictions) respectivement un dict {id_cluster: id_classe}
        et un tableau des classes finales prédites. """

    # Étape 1 : construction de la matrice de coût
    # Lignes = clusters trouvés, Colonnes = classes réelles
    # On utilise des valeurs négatives car linear_sum_assignment cherche à minimiser le coût,
    # or nous voulons maximiser le nombre de correspondances.
    matrice_cout = np.zeros((n_clusters, n_clusters))

    for id_cluster in range(n_clusters):
        for id_classe in range(1, n_clusters + 1):
            # On compte combien de points du cluster 'id_cluster' appartiennent en réalité à 'id_classe'
            masque = cluster_labels == id_cluster
            # Le coût est l'opposé du nombre de correspondances
            matrice_cout[id_cluster, id_classe - 1] = -np.sum(labels_reels[masque] == id_classe)

    # Étape 2 : résolution du problème d'assignation optimale
    indices_lignes, indices_colonnes = linear_sum_assignment(matrice_cout)

    # Étape 3 : création du dictionnaire de mapping : Cluster i -> Classe j + 1
    # (On ajoute +1 car les indices de colonnes sont 0..8 mais les classes sont 1..9)
    cluster_vers_classe = {cluster_id: class_id + 1 for cluster_id, class_id in zip(indices_lignes, indices_colonnes)}

    # Étape 4 : traduction des prédictions initiales vers les classes réelles
    predictions = np.array([cluster_vers_classe[c] for c in cluster_labels])

    return cluster_vers_classe, predictions
