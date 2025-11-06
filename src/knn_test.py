""" Module knn_test.py : évalue et compare les différentes méthodes de descripteurs en exécutant une validation
Leave-One-Out avec ClassifKNN, calcule matrice de confusion, taux global et rapport par classe, affiche les résultats
et les sauvegarde. Produit un résumé comparatif (taux et F1 macro) pour faciliter l’analyse des performances."""

import numpy as np
import pandas as pd
from knn_classif import ClassifKNN
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')


def evaluer_knn(X, y, k=5, distance='euclidienne', normaliser=True):
    """ Évalue le classificateur kNN avec validation Leave-One-Out. Chaque échantillon est tour à tour utilisé
    comme test, les autres servant à l'entraînement.
    :param X np.ndarray, matrice des features (n_samples, n_features)
    :param y np.ndarray, vecteur des labels (n_samples,)
    :param k nombre de voisins à considérer
    :param distance 'euclidienne', 'manhattan' ou 'minkowski'
    :param normaliser si True normalise les données avant apprentissage
    :return dict avec les résultats (predictions, probabilites, matrice_confusion, precision, rapport). """
    loo = LeaveOneOut()  # Validation LOO : 1 sample test, n-1 samples train
    predictions = []
    probabilites = []
    labels_vrais = []

    for indice_entrain, indice_test in loo.split(X):  # Boucle sur chaque partition LOO
        X_train, X_test = X[indice_entrain], X[indice_test]
        y_train, y_test = y[indice_entrain], y[indice_test]

        # Création et entraînement du classifieur
        classif = ClassifKNN(k=k, distance=distance, normaliser=normaliser)
        classif.entrainer(X_train, y_train)

        # Prédiction sur l’échantillon de test
        pred = classif.predire(X_test)
        proba = classif.predire_proba(X_test)

        # Stockage des résultats
        predictions.append(pred[0])  # Classe prédite
        probabilites.append(proba[0])  # Probabilités associées
        labels_vrais.append(y_test[0])  # Classe réelle

    # Conversion en tableaux numpy
    predictions = np.array(predictions)
    probabilites = np.array(probabilites)
    labels_vrais = np.array(labels_vrais)

    # Calcul des métriques de performance
    matrice_confusion = confusion_matrix(labels_vrais, predictions)
    precision = np.sum(predictions == labels_vrais) / len(labels_vrais)  # Taux de reconnaissance
    rapport = classification_report(labels_vrais, predictions,
                                   target_names=[f"Classe {i}" for i in range(1, 10)],
                                   output_dict=True)

    # Retour des résultats sous forme de dictionnaire
    return {
        'predictions': predictions,
        'probabilites': probabilites,
        'labels_vrais': labels_vrais,
        'matrice_confusion': matrice_confusion,
        'precision': precision,
        'rapport': rapport
    }


def afficher_resultats(resultats, nom_methode):
    """ Affiche les résultats de manière formatée dans le terminal.
    :param resultats produits par evaluer_knn().
    :param nom_methode du descripteur évalué. """
    print(f"\n- Résultats pour {nom_methode} :")
    print(f"\nTaux de reconnaissance : {resultats['precision'] * 100:.2f}%")

    print("\nMatrice de confusion :")
    print(resultats['matrice_confusion'])

    print("\nRapport de classification :")
    # Conversion du rapport en DataFrame pour affichage structuré
    rapport_df = pd.DataFrame(resultats['rapport']).transpose()
    print(rapport_df.to_string())


def main():
    donnees = np.load('donnees_chargees.npy', allow_pickle=True).item() # Chargement des données sauvegardées
    methodes = ['E34', 'GFD', 'SA', 'F0', 'F2']  # Liste des descripteurs testés

    print("Comparaison des méthodes de descripteurs")
    print("Configuration : k=5, distance euclidienne, normalisation activée")

    resultats_complets = {}

    for methode in methodes:  # Évaluation séquentielle de chaque méthode
        X = donnees[methode]['features']  # Matrice de features
        y = donnees[methode]['labels']  # Labels correspondants

        # Évaluation Leave-One-Out
        resultats = evaluer_knn(X, y, k=5, distance='euclidienne', normaliser=True)
        resultats_complets[methode] = resultats

        # Affichage des résultats individuels
        afficher_resultats(resultats, methode)

    # Tableau comparatif des taux de reconnaissance
    print("\nTableau récapitulatif :")
    comparaison = pd.DataFrame({
        'Méthode': methodes,
        'Taux de reconnaissance (%)': [resultats_complets[m]['precision'] * 100 for m in methodes],
        'F1-score macro': [resultats_complets[m]['rapport']['macro avg']['f1-score'] for m in methodes]
    })
    print(comparaison.to_string(index=False))

    # Sauvegarde des résultats
    np.save('resultats_knn.npy', resultats_complets, allow_pickle=True)
    print("\nRésultats sauvegardés dans 'resultats_knn.npy'")


if __name__ == "__main__":
    main()
