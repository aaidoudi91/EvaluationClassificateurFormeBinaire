""" Module vote_test.py : teste si la combinaison des décisions de plusieurs descripteurs permet d'obtenir
une performance supérieure à celle du meilleur descripteur pris individuellement.
Compare deux configurations : la fusion de toutes les méthodes et la fusion du "Top 3". """

import numpy as np
from pathlib import Path
import pandas as pd
from vote_classif import ClassifVoteMajoritaire
from src.knn.knn_classif import ClassifKNN
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, classification_report

# Définition dynamique des chemins
REPERTOIRE_ACTUEL = Path(__file__).resolve().parent  # Dossier contenant ce script
RACINE_PROJET = REPERTOIRE_ACTUEL.parent.parent  # Remonte de deux niveaux pour trouver la racine du projet
REPERTOIRE_SORTIES = RACINE_PROJET / "donnees_numpy"  # Dossier de destination pour les fichiers .npy


def evaluer_vote_majoritaire(dict_donnees, methodes, k=5, normaliser=True):
    """ Exécute une validation Leave-One-Out sur le classificateur par vote majoritaire.
    Pour chaque itération, le modèle entraîne N classifieurs kNN (un par méthode)
    sur N-1 échantillons et combine leurs votes pour prédire le 1 échantillon restant.
    :param dict_donnees le dictionnaire contenant les features/labels pour chaque méthode.
    :param methodes la liste des méthodes à utiliser.
    :param k pour les classifieurs kNN.
    :param normaliser le booléen pour activer la normalisation dans les kNN.
    :return: Dictionnaire contenant les prédictions, les vrais labels et les métriques.
    """
    # On récupère les labels de la première méthode (ils sont identiques pour toutes les méthodes)
    labels_vrais = dict_donnees[methodes[0]]['labels']
    predictions = []
    loo = LeaveOneOut()

    # Boucle de validation croisée (99 itérations pour 99 échantillons)
    for idx_train, idx_test in loo.split(labels_vrais):

        # Créer les ensembles d'entraînement pour chaque méthode
        donnees_train = {}
        for methode in methodes:
            X = dict_donnees[methode]['features']  # Matrice des features (n_samples, n_features)
            y = dict_donnees[methode]['labels']  # Vecteur des labels (n_samples,)

            # On ne garde que les indices d'entraînement pour construire le modèle
            donnees_train[methode] = {
                'features': X[idx_train],
                'labels': y[idx_train]
            }

        # Créer et entraîner le classificateur par vote
        # Utilisation d'une lambda qui permet à ClassifVoteMajoritaire d'instancier de nouveaux KNN à la volée
        classe_base = lambda normaliser: ClassifKNN(k=k, normaliser=normaliser)

        voteur = ClassifVoteMajoritaire(classe_base, methodes, normaliser=normaliser)
        voteur.entrainer(donnees_train)  # Entraîne les 3 ou 5 modèles KNN internes

        # Prédire pour l'échantillon test
        echantillons_test = {
            methode: dict_donnees[methode]['features'][idx_test]
            for methode in methodes
        }
        # Prédiction combinée
        prediction = voteur.predire(echantillons_test)
        predictions.append(prediction[0])

    predictions = np.array(predictions)

    # Calcul de la matrice de confusion (lignes = classe réelle, colonnes = classe prédite)
    matrice_confusion = confusion_matrix(labels_vrais, predictions)
    # Calcul de la précision globale
    precision = np.sum(predictions == labels_vrais) / len(labels_vrais)
    # Génération d'un rapport par classe avec précision, rappel, F1-Score
    rapport = classification_report(labels_vrais, predictions,
                                   target_names=[f"Classe {i}" for i in range(1, 10)],
                                   output_dict=True)  # output_dict=True permet de manipuler les résultats facilement

    return {
        'predictions': predictions,
        'labels_vrais': labels_vrais,
        'matrice_confusion': matrice_confusion,
        'precision': precision,
        'rapport': rapport,
        'methodes': methodes
    }


def afficher_resultats(results):
    """ Affiche les résultats de manière lisible dans le terminal.
    :param results le dictionnaire de résultats retourné par evaluer_vote_majoritaire(). """
    print(f"\n- Méthodes combinées : {', '.join(results['methodes'])}")
    print(f"Taux de reconnaissance : {results['precision'] * 100:.2f}%")

    print("\nMatrice de confusion :")
    print(results['matrice_confusion'])

    print("\nRapport de classification :")
    # Conversion du rapport en DataFrame pour affichage structuré
    report_df = pd.DataFrame(results['rapport']).transpose()
    print(report_df.to_string())


def main():
    """ Fonction principale : compare deux stratégies de fusion (Toutes vs Top 3)
        et affiche un comparatif avec les scores individuels. """
    # Chargement des données précédemment traitées par explore_donnees.py
    fichier_donnees = REPERTOIRE_SORTIES / 'donnees_chargees.npy'
    donnees = np.load(fichier_donnees, allow_pickle=True).item()  # .item() : np.save met le dict dans un tableau 0D

    print("\nComparaison des méthodes de descripteurs via Vote Majoritaire")

    # Premier test avec toutes les méthodes
    methodes = ['E34', 'GFD', 'SA', 'F0', 'F2']  # Liste des descripteurs à comparer
    resultats_complets = evaluer_vote_majoritaire(donnees, methodes, k=5, normaliser=True)
    afficher_resultats(resultats_complets)

    # Second test avec les trois meilleures méthodes : GFD, F0 et E34
    top3_methodes = ['GFD', 'F0', 'E34']
    resultats_top3 = evaluer_vote_majoritaire(donnees, top3_methodes, k=5, normaliser=True)
    afficher_resultats(resultats_top3)

    # Sauvegarder les résultats
    resultats_complets = {
        'methodes': resultats_complets,
        'top3_methodes': resultats_top3
    }
    fichier_sortie = REPERTOIRE_SORTIES / 'resultats_vote.npy'
    np.save(fichier_sortie, resultats_complets, allow_pickle=True)
    print(f"\nRésultats sauvegardés dans : {fichier_sortie}")


if __name__ == "__main__":
    main()
