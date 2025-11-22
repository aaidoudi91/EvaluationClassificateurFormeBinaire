""" Module kmeans_test.py : exécute et évalue le clustering k-means sur plusieurs ensembles de descripteurs d’images.
Il entraîne le modèle, associe les clusters aux vraies classes, calcule les métriques de performance
et génère un tableau comparatif des résultats. """

import numpy as np
import pandas as pd
from pathlib import Path
from kmeans_classif import ClassifKMeans, mapper_clusters_classes
from sklearn.metrics import confusion_matrix, classification_report

# Définition dynamique des chemins
REPERTOIRE_ACTUEL = Path(__file__).resolve().parent  # Dossier contenant ce script
RACINE_PROJET = REPERTOIRE_ACTUEL.parent.parent  # Remonte de deux niveaux pour trouver la racine du projet
REPERTOIRE_SORTIES = RACINE_PROJET / "donnees_numpy"  # Dossier de destination pour les fichiers .npy


def evaluer_kmeans(X, y, n_clusters=9, n_init=10, normaliser=True):
    """ Évalue un modèle k-means sur un jeu de données.
    :param X ndarray la matrice des features (n_samples, n_features).
    :param y ndarray les labels réels des échantillons (pour validation uniquement).
    :param n_clusters le nombre de clusters à former (par défaut 9, comme le nombre de classes).
    :param n_init le nombre d'initialisations différentes à tester.
    :param normaliser si True normalise les données.
    :return un dictionnaire contenant les labels, prédictions, mapping, métriques et inertie. """
    if normaliser:
        # Importation locale pour ne pas charger sklearn si la normalisation n'est pas requise
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)  # Normalisation des features pour éviter l’influence d’échelles différentes

    # Entraînement du modèle k-means (non supervisé, donc sans y)
    kmeans = ClassifKMeans(n_clusters=n_clusters, n_init=n_init, graine_aleatoire=42)
    kmeans.entrainer(X)

    cluster_labels = kmeans.labels  # Récupération des étiquettes de cluster (0 à 8)

    # Mapping des clusters vers les classes réelles pour interpréter les résultats
    mapping, predictions = mapper_clusters_classes(cluster_labels, y, n_clusters)

    # Calcul de la matrice de confusion (lignes = classe réelle, colonnes = classe prédite)
    matrice_confusion = confusion_matrix(y, predictions)
    # Calcul de la précision globale
    precision = np.sum(predictions == y) / len(y)
    # Génération d'un rapport par classe avec précision, rappel, F1-Score
    rapport = classification_report(y, predictions,
                                   target_names=[f"Classe {i}" for i in range(1, 10)],
                                   output_dict=True)  # output_dict=True permet de manipuler les résultats facilement

    return {
        'cluster_labels': cluster_labels,
        'predictions': predictions,
        'mapping': mapping,
        'matrice_confusion': matrice_confusion,
        'precision': precision,
        'inertie': kmeans.inertie,
        'rapport': rapport
    }


def afficher_resultats(resultats, nom_methode):
    """ Affiche les résultats du clustering k-means de manière lisible dans le terminal.
    :param resultats un dict contenant les métriques et résultats du modèle (retourné par evaluer_kmeans).
    :param nom_methode le nom de la méthode évaluée. """
    print(f"\n- Résultats pour {nom_methode} :")
    print(f"\nTaux de reconnaissance : {resultats['precision'] * 100:.2f}%")
    print(f"Inertie: {resultats['inertie']:.2f}")

    print("\nMatrice de confusion :")
    print(resultats['matrice_confusion'])

    print("\nRapport de classification :")
    report_df = pd.DataFrame(resultats['rapport']).transpose()
    print(report_df.to_string())


def main():
    """ Fonction principale : charge les données, itère sur les descripteurs et compare les résultats. """
    # Chargement des données précédemment traitées par explore_donnees.py
    fichier_donnees = REPERTOIRE_SORTIES / 'donnees_chargees.npy'
    donnees = np.load(fichier_donnees, allow_pickle=True).item()  # .item() : np.save met le dict dans un tableau 0D
    methodes = ['E34', 'GFD', 'SA', 'F0', 'F2']  # Liste des descripteurs à comparer

    print("\nComparaison des méthodes de descripteurs via k-means")
    print("Configuration : k=9 clusters, 10 initialisations, normalisation activée")

    resultats_complets = {}

    for methode in methodes:  # Itération sur chaque méthode de description d'image
        X = donnees[methode]['features']  # Les vecteurs caractéristiques
        y = donnees[methode]['labels']  # Les classes associées

        # Lancement de l'évaluation
        resultats = evaluer_kmeans(X, y, n_clusters=9, n_init=10, normaliser=True)
        resultats_complets[methode] = resultats

        # Affichage des résultats individuels
        afficher_resultats(resultats, methode)

    # Tableau comparatif
    print("\nTableau récapitulatif :")
    comparison = pd.DataFrame({
        'Méthode': methodes,
        'Taux de reconnaissance (%)': [resultats_complets[m]['precision'] * 100 for m in methodes],
        'Inertie': [resultats_complets[m]['inertie'] for m in methodes],
        # On récupère le F1-score moyen pondéré (macro avg) pour avoir une vue d'ensemble
        'F1-score macro': [resultats_complets[m]['rapport']['macro avg']['f1-score'] for m in methodes]
    })
    print(comparison.to_string(index=False))

    # Sauvegarde des résultats
    fichier_sortie = REPERTOIRE_SORTIES / 'resultats_kmeans.npy'
    np.save(fichier_sortie, resultats_complets, allow_pickle=True)
    print(f"\nRésultats sauvegardés dans : {fichier_sortie}")


if __name__ == "__main__":
    main()
