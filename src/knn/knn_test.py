""" Module knn_test.py : évalue et compare les différentes méthodes de descripteurs en exécutant une validation
Leave-One-Out avec ClassifKNN, calcule matrice de confusion, taux global et rapport par classe, affiche les résultats
et les sauvegarde. Produit un résumé comparatif (taux et F1 macro) pour faciliter l’analyse des performances."""

import numpy as np
import pandas as pd
from pathlib import Path
from knn_classif import ClassifKNN
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, classification_report

# Définition dynamique des chemins
REPERTOIRE_ACTUEL = Path(__file__).resolve().parent  # Dossier contenant ce script
RACINE_PROJET = REPERTOIRE_ACTUEL.parent.parent  # Remonte de deux niveaux pour trouver la racine du projet
REPERTOIRE_SORTIES = RACINE_PROJET / "donnees_numpy"  # Dossier de destination pour les fichiers .npy


def evaluer_knn(X, y, k=5, distance='euclidienne', normaliser=True):
    """ Évalue le classificateur kNN avec validation Leave-One-Out. Chaque échantillon est tour à tour utilisé
    comme test, les autres servant à l'entraînement.
    :param X (np.ndarray), matrice des features (n_samples, n_features).
    :param y (np.ndarray), vecteur des labels (n_samples,)
    :param k nombre de voisins à considérer
    :param distance 'euclidienne', 'manhattan' ou 'minkowski'
    :param normaliser si True normalise les données avant apprentissage
    :return dict avec les résultats : predictions, probabilités, matrice_confusion, precision, rapport. """
    loo = LeaveOneOut()  # Initialisation de l'objet
    predictions = []
    probabilites = []
    labels_vrais = []

    # Boucle sur chaque partition générée par LOO.
    # L'indice_entrain contient N-1 indices, indice_test contient un seul indice.
    for indice_entrain, indice_test in loo.split(X):
        # Création des sous-ensembles train/test
        X_train, X_test = X[indice_entrain], X[indice_test]
        y_train, y_test = y[indice_entrain], y[indice_test]

        # Instanciation et entraînement du classifieur sur les N-1 données
        classif = ClassifKNN(k=k, distance=distance, normaliser=normaliser)
        classif.entrainer(X_train, y_train)

        # Prédiction sur l'échantillon de test unique
        pred = classif.predire(X_test)
        proba = classif.predire_proba(X_test)

        # Stockage des résultats (le tableau est de taille 1)
        predictions.append(pred[0])  # Classe prédite
        probabilites.append(proba[0])  # Probabilités associées
        labels_vrais.append(y_test[0])  # Classe réelle

    # Conversion des listes en vecteurs NumPy pour faciliter les calculs
    predictions = np.array(predictions)
    probabilites = np.array(probabilites)
    labels_vrais = np.array(labels_vrais)

    # Calcul de la matrice de confusion (lignes = classe réelle, colonnes = classe prédite)
    matrice_confusion = confusion_matrix(labels_vrais, predictions)
    # Calcul de la précision globale
    precision = np.sum(predictions == labels_vrais) / len(labels_vrais)
    # Génération d'un rapport par classe avec précision, rappel, F1-Score
    rapport = classification_report(labels_vrais, predictions,
                                   target_names=[f"Classe {i}" for i in range(1, 10)],
                                   output_dict=True)  # output_dict=True permet de manipuler les résultats facilement

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
    """ Affiche les résultats de manière lisible dans le terminal.
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
    """ Fonction principale : charge les données, lance les évaluations comparatives et sauvegarde les résultats. """
    # Chargement des données précédemment traitées par explore_donnees.py
    fichier_donnees = REPERTOIRE_SORTIES / 'donnees_chargees.npy'
    donnees = np.load(fichier_donnees, allow_pickle=True).item()  # .item() : np.save met le dict dans un tableau 0D
    methodes = ['E34', 'GFD', 'SA', 'F0', 'F2']  # Liste des descripteurs à comparer

    print("\nComparaison des méthodes de descripteurs via kNN")
    print("Configuration : k=5, distance euclidienne, normalisation activée")

    resultats_complets = {}

    for methode in methodes:  # Itération sur chaque méthode de description d'image
        X = donnees[methode]['features']  # Les vecteurs caractéristiques
        y = donnees[methode]['labels']  # Les classes associées

        # Évaluation Leave-One-Out
        resultats = evaluer_knn(X, y, k=5, distance='euclidienne', normaliser=True)
        resultats_complets[methode] = resultats

        # Affichage des résultats individuels
        afficher_resultats(resultats, methode)

    # Tableau comparatif
    print("\nTableau récapitulatif :")
    comparaison = pd.DataFrame({
        'Méthode': methodes,
        'Taux de reconnaissance (%)': [resultats_complets[m]['precision'] * 100 for m in methodes],
        # On récupère le F1-score moyen pondéré (macro avg) pour avoir une vue d'ensemble
        'F1-score macro': [resultats_complets[m]['rapport']['macro avg']['f1-score'] for m in methodes]
    })
    print(comparaison.to_string(index=False))

    # Sauvegarde des résultats
    fichier_sortie = REPERTOIRE_SORTIES / 'resultats_knn.npy'
    np.save(fichier_sortie, resultats_complets, allow_pickle=True)
    print(f"\nRésultats sauvegardés dans : {fichier_sortie}")

if __name__ == "__main__":
    main()
