""" Module trace_precision_rappel.py : génére des courbes précision-rappel pour les résultats kNN.
Ce script charge les résultats sauvegardés, calcule les courbes précision-rappel one-vs-rest pour chaque classe,
trace et sauvegarde les figures, et affiche les AUC par classe ainsi que la moyenne. """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


def trace_depuis_resultats(fichier_resultats, methode, k):
    """ Trace la courbe précision-rappel à partir des résultats sauvegardés.
    :param fichier_resultats: Chemin vers le fichier .npy contenant les résultats.
    :param methode: Nom de la méthode (ex: 'GFD').
    :param k: Valeur de k utilisée (ex: 10).
    :return: tuple (moy_auc, scores_auc) respectivement AUC moyen un float et scores_auc: list de floats. """

    resultats = np.load(fichier_resultats, allow_pickle=True).item() # Charger les résultats

    if methode not in resultats:  # Vérifier que la méthode est présente
        print(f"Erreur: {methode} non trouvé dans {fichier_resultats}")
        return

    donnees = resultats[methode]
    probabilities = donnees['probabilites']
    labels_vrais = donnees['labels_vrais']

    classes = np.unique(labels_vrais)  # Liste des classes uniques
    n_classes = len(classes)

    plt.figure(figsize=(10, 8))  # Création de la figure
    scores_auc = []  # Stocke les AUC par classe

    # Boucle sur chaque classe pour approche one-vs-rest
    for i, classe in enumerate(classes):
        y_vrai_binaire = (labels_vrais == classe).astype(int)  # 1 si sample appartient à la classe, 0 sinon
        y_scores = probabilities[:, i]  # Probabilité prédite pour la classe actuelle

        # Calculer précision et rappel
        precision, recall, _ = precision_recall_curve(y_vrai_binaire, y_scores)
        auc_score = auc(recall, precision)  # Calcul AUC
        scores_auc.append(auc_score)

        # Tracé de la courbe
        plt.plot(recall, precision, linewidth=2,
                 label=f'Classe {classe} (AUC={auc_score:.3f})')

    # Mise en forme
    plt.xlabel('Rappel', fontsize=12)
    plt.ylabel('Précision', fontsize=12)
    plt.title(f'Courbes Précision-Rappel - {methode} (k={k})', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # Sauvegarde de la figure
    fichier = f'resultat_pr_{methode.lower()}_k{k}.png'
    plt.savefig(fichier, dpi=300, bbox_inches='tight')
    print(f"✓ Courbe sauvegardée: {fichier}")
    plt.close()

    # Affichage des AUC
    moy_auc = np.mean(scores_auc)
    print(f"\n{methode} (k={k}):")
    print(f"  AUC moyen: {moy_auc:.3f}")
    for i, (classe, score) in enumerate(zip(classes, scores_auc)):
        print(f"  Classe {classe}: AUC = {score:.3f}")

    return moy_auc, scores_auc


def main():
    """ Génère les courbes précision-rappel pour les résultats kNN. """
    fichier_resultats = 'resultats_knn_k10.npy'  # Fichier avec k=10

    try:
        trace_depuis_resultats(fichier_resultats, 'GFD', k=10)
    except FileNotFoundError:
        print(f"\nFichier {fichier_resultats} non trouvé.")


if __name__ == "__main__":
    main()
