""" Module explore_donnees.py : point d'entrée pour le chargement et la vérification du dataset BDshape.
Ce script orchestre le processus de lecture, contrôle d'intégrité, statistiques et sauvegarde. """

import numpy as np
from pathlib import Path
from charge_donnees import ChargeDonnees

# Définition dynamique des chemins
REPERTOIRE_ACTUEL = Path(__file__).resolve().parent  # Dossier contenant ce script
RACINE_PROJET = REPERTOIRE_ACTUEL.parent.parent  # Remonte de deux niveaux pour trouver la racine du projet
REPERTOIRE_DONNEES = RACINE_PROJET / "donnees"  # Dossier source des fichiers .MET
REPERTOIRE_SORTIES = RACINE_PROJET / "donnees_numpy"  # Dossier de destination pour les fichiers .npy

def main():
    """ Fonction principale d'exploration et de validation, exécutant dans l'ordre : chargement des données brutes,
    vérification de la cohérence, affichage des statistiques et sauvegardes des résultats.
    :return dict: Le dictionnaire complet des données chargées et structurées. """

    chargeur = ChargeDonnees(REPERTOIRE_DONNEES)  # Initialisation de la classe de chargement des données
    print("\nLecture des données :")

    # Étape 1 : chargement de toutes les données disponibles
    print("\n1. Chargement des données :")
    donnees = chargeur.charger_donnees()

    # Étape 2 : vérification du nombre total d'échantillons chargés
    print("\n2. Vérification de l'intégrité :")
    total_attendu = 9 * 11  # 9 classes × 11 échantillons
    for methode in chargeur.methode:
        total_reel = len(donnees[methode]['features'])  # Nombre réel de fichiers lus
        # Affiche OK si complet, sinon indique le nombre de fichiers manquants
        statut = "- OK" if total_reel == total_attendu else f"- MANQUANT {total_attendu - total_reel}"
        print(f"{methode}: {total_reel}/{total_attendu} fichiers {statut}")

    # Étape 3 : vérification des dimensions des vecteurs caractéristiques
    print("\n3. Vérification des dimensions :")
    # Dictionnaire des dimensions théoriques attendues par méthode de description
    dims_attendues = {'E34': 16, 'GFD': 36, 'SA': 90, 'F0': 128, 'F2': 128}
    for methode, dim_attendues in dims_attendues.items():
        dim_reelle = donnees[methode]['features'].shape[1]  # Récupération de la dimension réelle (nombre de colonnes)
        statut = "- OK" if dim_reelle == dim_attendues else "- ?"  # Vérification de cohérence
        print(f"{methode}: dimension {dim_reelle} (attendu: {dim_attendues}) {statut}")

    # Étape 4 : calcul et affichage des statistiques descriptives
    print("\n4. Statistiques descriptives :")
    stats = chargeur.calculer_stats(donnees)  # Génère un DataFrame pandas récapitulatif
    print(stats.to_string(index=False))  # Affiche le tableau sans l'index pandas

    # Étape 5 : sauvegarde des données traitées pour usage ultérieur (format NumPy)
    print("\n5. Sauvegarde des données chargées :")
    fichier_sortie = REPERTOIRE_SORTIES / 'donnees_chargees.npy'
    np.save(fichier_sortie, donnees, allow_pickle=True)  # Sauvegarde avec sérialisation d’objets
    print(f"Données sauvegardées dans : {fichier_sortie}")

    return donnees  # Retourne la structure complète pour utilisation éventuelle


if __name__ == "__main__":
    data = main()
