""" Module vote_classif.py : permet de combiner les prédictions de plusieurs méthodes de description (E34, GFD, etc.)
en utilisant une stratégie de vote majoritaire. """

import numpy as np
from collections import Counter


class ClassifVoteMajoritaire:
    """ Classificateur par vote majoritaire. Entraîne un classificateur indépendant pour chaque méthode de description,
        puis agrège leurs décisions pour la prédiction finale. """

    def __init__(self, classif_base, methodes, normaliser=True):
        """ Initialise le classificateur.
        :param classif_base la classe (elle-même, pas une instance) du classificateur à utiliser (ici, ClassifKNN).
        :param methodes la liste des noms des méthodes à combiner (ici, ['E34', 'GFD', 'SA', 'F0', 'F2']).
        :param normaliser indique si les features doivent être normalisées avant apprentissage. """
        self.classif_base = classif_base
        self.methodes = methodes
        self.normaliser = normaliser
        self.classifieurs = {}  # Dictionnaire pour stocker les modèles entraînés par méthode

    def entrainer(self, dict_donnees):
        """ Entraîne un classificateur distinct pour chaque méthode de description fournie.
        :param dict_donnees: contenant les données {methode: {'features': X, 'labels': y}}. """
        for methode in self.methodes:
            if methode not in dict_donnees:
                raise ValueError(f"Méthode {methode} non trouvée dans les données")

            X = dict_donnees[methode]['features']  # Matrice des features (n_samples, n_features)
            y = dict_donnees[methode]['labels']  # Vecteur des labels (n_samples,)

            # Créer et entraîner un classificateur pour cette méthode
            clf = self.classif_base(normaliser=self.normaliser)
            clf.entrainer(X, y)

            # Stockage du modèle et des données d'entraînement associées
            self.classifieurs[methode] = {
                'classifieurs': clf,
                'X_train': X,
                'y_train': y
            }

    def predire_un(self, indices_test):
        """ Prédit la classe d'un échantillon spécifique en utilisant ses indices dans les données d'entraînement.
        Cette méthode est utile lors de Leave-One-Out où l'on connaît l'index.
        :param indices_test: Dictionnaire {methode: index} indiquant quel échantillon tester pour chaque vue.
        :return la classe prédite par vote majoritaire. """
        votes = []

        for methode in self.methodes:
            if methode not in indices_test:
                continue

            idx = indices_test[methode]
            # Extraction de l'échantillon test depuis les données stockées [idx:idx+1]
            # Cela permet de garder la dimension 2D (1, n_features) attendue par les classifieurs
            X_test = self.classifieurs[methode]['X_train'][idx:idx + 1]

            clf = self.classifieurs[methode]['classifieurs']
            pred = clf.predire(X_test)
            votes.append(pred[0])  # Ajout du vote de cette méthode

        # Comptage des votes pour déterminer le "gagnant"
        comptes = Counter(votes)
        # most_common(1) renvoie [(classe, nombre_votes)], on récupère la classe [0][0]
        return comptes.most_common(1)[0][0]

    def predire(self, dict_echantillons_test):
        """ Prédit les classes pour un ensemble de nouveaux échantillons donnés par leurs features.
        :param dict_echantillons_test: le dictionnaire {methode: X_test} contenant les vecteurs à tester.
        On suppose ici que les échantillons sont alignés (le i-ème de E34 correspond au i-ème de GFD).
        :return: Array numpy contenant les classes prédites pour chaque échantillon. """
        # Récupération du nombre d'échantillons à tester (en regardant la première méthode disponible)
        # next(iter(...)) permet d'accéder au premier élément sans connaître sa clé
        n = len(next(iter(dict_echantillons_test.values())))
        predictions = []

        for i in range(n):  # Boucle sur chaque échantillon à prédire
            votes = []

            # Collecte des votes de chaque méthode
            for methode in self.methodes:
                # Extraction du i-ème vecteur pour la méthode courante
                X_test = dict_echantillons_test[methode][i:i + 1]
                clf = self.classifieurs[methode]['classifieurs']
                pred = clf.predire(X_test)
                votes.append(pred[0])

            # Vote majoritaire
            comptes = Counter(votes)
            # most_common(1) renvoie [(classe, nombre_votes)], on récupère la classe [0][0]
            predictions.append(comptes.most_common(1)[0][0])

        return np.array(predictions)
