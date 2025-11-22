import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# --- Chargement du fichier ---
fichier = 'donnees_chargees.npy'
data = np.load(fichier, allow_pickle=True).item()  # conversion en dict Python

# --- Paramètres ---
use_pca_preprocessing = True   # True pour réduire la dimension avant t-SNE
pca_components = 20            # Nombre de composantes PCA si utilisé
tsne_perplexity = 5            # Ajuster selon le nombre d'échantillons (ici 9x11=99)
tsne_iter = 1000               # Nombre d'itérations t-SNE

# --- Fonction pour t-SNE + affichage ---
def plot_tsne(X, y, title):
    if use_pca_preprocessing:
        pca = PCA(n_components=min(pca_components, X.shape[1]))
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X

    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, max_iter=tsne_iter, random_state=42)
    X_embedded = tsne.fit_transform(X_reduced)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y, palette='tab10', s=60)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title='Classe', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# --- Boucle sur tous les descripteurs ---
for nom in data.keys():
    X = data[nom]['features']
    y = data[nom]['labels']
    plot_tsne(X, y, f"t-SNE 2D - {nom}")

