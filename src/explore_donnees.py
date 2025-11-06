""" Module explore_donnees.py : point d'entrée pour le chargement et la vérification du dataset BDshape.
Ce script orchestre le processus de lecture, contrôle d'intégrité, statistiques et sauvegarde. """

import numpy as np
from charge_donnees import ChargeDonnees


def main():
    rep_donnees = "../data/" # Répertoire contenant les fichiers de données, toutes mélangées
    chargeur = ChargeDonnees(rep_donnees) # Initialisation du chargeur de données

    print("\nLecture des données :")

    # Étape 1 : chargement de toutes les données disponibles
    print("\n1. Chargement des données :")
    donnees = chargeur.charger_donnees()

    # Étape 2 : vérification du nombre total d'échantillons chargés
    print("\n2. Vérification de l'intégrité :")
    total_attendu = 9 * 11  # 9 classes × 11 échantillons
    for methode in chargeur.methode:
        total_reel = len(donnees[methode]['features'])  # Nombre réel de fichiers lus
        statut = "- OK" if total_reel == total_attendu else f"- MANQUANT {total_attendu - total_reel}"
        print(f"{methode}: {total_reel}/{total_attendu} fichiers {statut}")

    # Étape 3 : vérification des dimensions des vecteurs caractéristiques
    print("\n3. Vérification des dimensions :")
    dims_attendues = {'E34': 16, 'GFD': 36, 'SA': 90, 'F0': 128, 'F2': 128}
    for methode, dim_attendues in dims_attendues.items():
        dim_reelle = donnees[methode]['features'].shape[1]  # Taille réelle du vecteur
        statut = "- OK" if dim_reelle == dim_attendues else "- ?"  # Vérification de cohérence
        print(f"{methode}: dimension {dim_reelle} (attendu: {dim_attendues}) {statut}")

    # Étape 4 : calcul et affichage des statistiques descriptives
    print("\n4. Statistiques descriptives :")
    stats = chargeur.calculer_stats(donnees)  # Génère un DataFrame des statistiques
    print(stats.to_string(index=False))  # Affiche sous forme tabulaire

    # Étape 5 : sauvegarde des données traitées pour usage ultérieur (format NumPy)
    print("\n5. Sauvegarde des données chargées :")
    np.save('donnees_chargees.npy', donnees, allow_pickle=True)  # Sauvegarde avec sérialisation d’objets
    print("Données sauvegardées dans 'donnees_chargees.npy'")

    return donnees  # Retourne la structure complète pour utilisation éventuelle


if __name__ == "__main__":
    data = main()
