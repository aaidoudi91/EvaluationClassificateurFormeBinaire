""" Module kmeans_test.py : exécute et évalue le clustering k-means sur plusieurs ensembles de descripteurs d’images.
Il entraîne le modèle, associe les clusters aux vraies classes, calcule les métriques de performance
et génère un tableau comparatif des résultats. """

import numpy as np
import pandas as pd
from kmeans_classif import ClassifKMeans, mapper_clusters_classes
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')


def evaluer_kmeans(X, y, n_clusters=9, n_init=10, normaliser=True):
    """ Évalue un modèle k-means sur un jeu de données.
    :param X ndarray la matrice des features.
    :param y ndarray les labels réels des échantillons.
    :param n_clusters le nombre de clusters à former (par défaut 9).
    :param n_init le nombre d'initialisations différentes à tester.
    :param normaliser si True normalise les données.
    :return: dict contenant les labels, prédictions, mapping, métriques et inertie.
    """
    if normaliser:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)  # Normalisation des features pour éviter l’influence d’échelles différentes

    # Entraînement du modèle k-means (non supervisé, donc sans y)
    kmeans = ClassifKMeans(n_clusters=n_clusters, n_init=n_init, graine_aleatoire=42)
    kmeans.entrainer(X)

    cluster_labels = kmeans.labels  # Labels de cluster assignés par k-means

    # Mapping des clusters vers les classes réelles pour interpréter les résultats
    mapping, predictions = mapper_clusters_classes(cluster_labels, y, n_clusters)

    # Calculer les métriques
    matrice_confusion = confusion_matrix(y, predictions)  # Matrice de confusion
    precision = np.sum(predictions == y) / len(y)  # Taux de reconnaissance
    rapport = classification_report(y, predictions,
                                   target_names=[f"Classe {i}" for i in range(1, 10)],
                                   output_dict=True)
    """
    print("Mapping clusters -> classes :")
    for cluster_id, class_id in sorted(mapping.items()):
        print(f"  Cluster {cluster_id} -> Classe {class_id}")"""

    return {
        'cluster_labels': cluster_labels,
        'predictions': predictions,
        'mapping': mapping,
        'matrice_confusion': matrice_confusion,
        'precision': precision,
        'inertie': kmeans.inertie,
        'rapport': rapport
    }


def afficher_resultats(results, method_name):
    """ Affiche les résultats du clustering k-means de manière formatée.
    :param results un dict contenant les métriques et résultats du modèle.
    :param method_name le nom de la méthode évaluée.
    :return: None. """
    print(f"\nRésultats pour {method_name} :")

    print(f"\nTaux de reconnaissance : {results['precision'] * 100:.2f}%")
    print(f"Inertie: {results['inertie']:.2f}")

    print("\nMatrice de confusion :")
    print(results['matrice_confusion'])

    print("\nRapport de classification :")
    report_df = pd.DataFrame(results['rapport']).transpose()
    print(report_df.to_string())


def main():
    donnees = np.load('donnees_chargees.npy', allow_pickle=True).item()
    methodes = ['E34', 'GFD', 'SA', 'F0', 'F2']

    print("Comparaison des méthodes de descripteurs")
    print("Configuration : k=9 clusters, 10 initialisations, normalisation activée")

    resultats_complets = {}

    for methode in methodes:
        X = donnees[methode]['features']
        y = donnees[methode]['labels']

        resultats = evaluer_kmeans(X, y, n_clusters=9, n_init=10, normaliser=True)
        resultats_complets[methode] = resultats
        afficher_resultats(resultats, methode)

    # Tableau comparatif
    print("\nTableau récapitulatif :")
    comparison = pd.DataFrame({
        'Méthode': methodes,
        'Taux de reconnaissance (%)': [resultats_complets[m]['precision'] * 100 for m in methodes],
        'Inertie': [resultats_complets[m]['inertie'] for m in methodes],
        'F1-score macro': [resultats_complets[m]['rapport']['macro avg']['f1-score'] for m in methodes]
    })
    print(comparison.to_string(index=False))

    # Sauvegarde
    np.save('resultats_kmeans.npy', resultats_complets, allow_pickle=True)
    print("\nRésultats sauvegardés dans 'resultats_kmeans.npy'")


if __name__ == "__main__":
    main()
