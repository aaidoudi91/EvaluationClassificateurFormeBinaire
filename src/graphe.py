import matplotlib.pyplot as plt
import numpy as np

methods = ['E34', 'GFD', 'SA', 'F0', 'F2']
knn_scores = [62.63, 85.86, 59.60, 74.75, 55.56]
kmeans_scores = [46.46, 54.55, 49.49, 59.60, 49.49]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, knn_scores, width, label='kNN (k=5)', color='steelblue')
bars2 = ax.bar(x + width/2, kmeans_scores, width, label='k-means', color='coral')

ax.set_xlabel('Méthode de description', fontsize=12)
ax.set_ylabel('Taux de reconnaissance (%)', fontsize=12)
ax.set_title('Comparaison kNN vs k-means', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('comparison_knn_kmeans.png', dpi=300, bbox_inches='tight')
plt.close()
