""" Module charge_donnees.py : gère le chargement et l'organisation du dataset BDshape.
Permet d'extraire les métadonnées des fichiers, de lire les vecteurs caractéristiques et de générer un dictionnaire. """

import os
import numpy as np
import pandas as pd


class ChargeDonnees:
    """ Charge les données des fichiers SxxNyyy.MET de la base d'imges BDshape. """

    def __init__(self, chemin):
        """ :param chemin: Chemin vers le dossier contenant les fichiers .MET. """
        self.chemin = chemin  # Dossier contenant les fichiers .MET
        self.methode = ['E34', 'GFD', 'SA', 'F0', 'F2']  # Méthodes de description
        self.nombre_classes = 9  # Nombre total de classes dans BDshape
        self.nombre_samples = 11  # Nombre d'échantillons par classe

    def analyser_fichier(self, nom_fichier):
        """ Extrait la classe et le numéro d'échantillon du nom de fichier sachant que le format est tel que
        xx = classe, yyy = échantillon dans SxxNyyy.MET.
        :param nom_fichier à analyser.
        :return (classe, sample, methode) ou None si format invalide. """
        try:
            basename = os.path.basename(nom_fichier)  # Nom de fichier sans le chemin
            parties = basename.split('.')  # Sépare le nom et l'extension
            if len(parties) != 2:  # Vérifie la validité du format
                return None

            nom, methode = parties
            if not nom.startswith('S') or 'N' not in nom:  # Format non conforme
                return None

            # Extraction des numéros de classe et d'échantillon (ex : S01N001)
            s = nom.split('N')[0]  # Partie avant N
            n = nom.split('N')[1]  # Partie après N

            classe = int(s[1:])  # Convertit "S01" -> 1
            sample = int(n)  # Convertit "001" -> 1

            return classe, sample, methode
        except Exception:
            return None # En cas d'erreur (nom mal formé)

    def charger_feature_vector(self, chemin):
        """ Charge un vecteur de caractéristiques depuis un fichier .MET.
        :param chemin vers le fichier .MET.
        :return numpy array contenant les valeurs. """
        try:
            # Lecture du fichier (une valeur numérique par ligne)
            with open(chemin, 'r') as f:
                valeurs = [float(ligne.strip()) for ligne in f if ligne.strip()]  # Ignore lignes vides
            return np.array(valeurs)  # Conversion en tableau NumPy
        except Exception as e:
            print(f"Erreur lors du chargement de {chemin}: {e}")
            return None  # Retourne None en cas d’échec

    def charger_donnees(self):
        """ Charge toutes les données pour les cinq méthodes et les organise dans un dictionnaire.
        :return dict: {method: {'features': array, 'labels': array, 'samples': list}} """
        donnees = {}

        for methode in self.methode:
            features = []  # Liste des vecteurs de caractéristiques
            labels = []  # Liste des étiquettes de classes
            samples = []  # Liste des identifiants (classe, échantillon)

            # Parcourir tous les fichiers pour cette méthode
            for classe in range(1, self.nombre_classes + 1):
                for sample in range(1, self.nombre_samples + 1):
                    nom = f"S{classe:02d}N{sample:03d}.{methode}"  # Nom du fichier attendu
                    chemin = os.path.join(self.chemin, nom)

                    if os.path.exists(chemin):  # Vérifie que le fichier existe
                        vecteur = self.charger_feature_vector(chemin)  # Lecture du vecteur
                        if vecteur is not None:
                            features.append(vecteur)  # Ajout du vecteur
                            labels.append(classe)  # Ajout de la classe associée
                            samples.append((classe, sample))  # Stocke l'identité de l'échantillon
                    else:
                        print(f"Fichier manquant: {nom}")  # Informe si un fichier est absent

            # Conversion des listes en structures NumPy pour traitement ultérieur
            donnees[methode] = {
                'features': np.array(features),
                'labels': np.array(labels),
                'samples': samples
            }

            print(f"{methode}: {len(features)} échantillons chargés, dimension: {features[0].shape if features else 'N/A'}")

        return donnees  # Renvoie le dictionnaire complet

    def calculer_stats(self, donnees):
        """ Calcule des statistiques descriptives sur les vecteurs de caractéristiques.
        :param donnees un dictionnaire contenant des données organisées par méthode.
        :return DataFrame avec les statistiques par méthode. """
        stats = []

        for methode in self.methode:
            if methode in donnees and len(donnees[methode]['features']) > 0:
                features = donnees[methode]['features']

                stats.append({
                    'Méthode': methode,
                    'Nombre échantillons': len(features),
                    'Dimension': features.shape[1],  # Taille du vecteur de caractéristiques
                    'Moyenne': np.mean(features),
                    'Écart-type': np.std(features),
                    'Min': np.min(features),
                    'Max': np.max(features)
                })

        return pd.DataFrame(stats)  # Retourne un DataFrame pandas
