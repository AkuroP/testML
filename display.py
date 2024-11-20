from sklearn.manifold import TSNE
from umap.parametric_umap import ParametricUMAP

import matplotlib.pyplot as plt
import numpy as np

def tsne_display(X, y, perplexity=30, learning_rate='auto', max_iter=None):
    """
    Visualisation des données multidimensionnelles en 2D avec TSNE.
    
    Args:
        X (array-like): Données à visualiser.
        y (array-like): Labels des données à visualiser.
        perplexity (float, optional): Contrôle le nombre de voisins considérés pour chaque point. (Par défaut: 30)
        learning_rate (float or str, optional): Taux d'apprentissage de TSNE. (Par défaut: 'auto')
        max_iter (int, optional): Nombre maximum d'itérations pour l'optimisation. (Par défaut: None)
        
    Returns:
        None: Affiche la visualisation des données sous forme de graphique 2D.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, max_iter=max_iter, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    y = np.ravel(y)
    labels = np.unique(y)

    for i in labels:
        plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=i, cmap='viridis', alpha=0.7)
        
    plt.legend()    
    plt.title("Visualisation avec TSNE")
    plt.xlabel("Composante 1")
    plt.ylabel("Composante 2")
    
    plt.savefig("tsne.png")


    plt.show()
    

def umap_display(X, y):
    """
    Visualisation des données multidimensionnelles en 2D avec UMAP, version paramétrique.
    La version paramétrique de UMAP utilise un réseau de neurones de TensorFlow pour apprendre une représentation des données.
    
    Args:
        X (array-like): Données à visualiser.
        y (array-like): Labels des données à visualiser.  

    Returns:
        None: Affiche la visualisation des données sous forme de graphique 2D.
    """
    embedder = ParametricUMAP(n_components=2, random_state=42)
    X_umap = embedder.fit_transform(X)
    
    y = np.ravel(y)
    labels = np.unique(y)

    for i in labels:
        plt.scatter(X_umap[y == i, 0], X_umap[y == i, 1], label=i, cmap='viridis', alpha=0.7)
        
    plt.legend()    
    plt.title("Visualisation avec UMAP")
    plt.xlabel("Composante 1")
    plt.ylabel("Composante 2")

    plt.savefig("umap.png")
    
    plt.show()