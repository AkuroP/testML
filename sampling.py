from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from collections import Counter

def under_sampling(X, y, sampling_strategy='auto', version=1, n_neighbors=3, n_neighbors_ver3=3):
    """
    Réduction du nombre d'instances de la classe majoritaire pour équilibrer les classes à l'aide de NearMiss.
    
    Args: 
        X (array-like): Données à équilibrer
        y (array-like): Labels associés aux données à équilibrer
        sampling_strategy (str, optional): Stratégie pour équilibrer les classes. (Par défaut: 'auto')
        version (int, optional): Version de NearMiss à utiliser. (Par défaut: 1)
        n_neighbors (int, optional): Nombre de voisins à considérer pour chaque instance. (Par défaut: 3)
        n_neighbors_ver3 (int, optional): Nombre de voisins pour la sélection de sous-ensemble de NearMiss-3. (Par défaut: 3)
        
    Returns:
        X_res (array-like): Données équilibrées
        y_res (array-like): Labels associés aux données équilibrées
    """
    print('Original dataset shape %s' % Counter(y))
    
    nm = NearMiss(sampling_strategy=sampling_strategy, version=version, n_neighbors=n_neighbors, n_neighbors_ver3=n_neighbors_ver3)
    X_res, y_res = nm.fit_resample(X, y)
    
    print('Resampled dataset shape %s' % Counter(y_res))
    
    return X_res, y_res


def over_sampling(X, y, sampling_strategy='auto', k_neighbors=5):
    """
    Augmentation du nombre d'instances de la classe minoritaire pour équilibrer les classes à l'aide de SMOTE.
    
    Args:
        X (array-like): Données à équilibrer
        y (array-like): Labels associés aux données à équilibrer
        sampling_strategy (str, optional): Stratégie pour équilibrer les classes. (Par défaut: 'auto')
        k_neighbors (int, optional): Nombre de voisins à considérer pour chaque instance. (Par défaut: 5)
        
    Returns:
        X_res (array-like): Données équilibrées
        y_res (array-like): Labels associés aux données équilibrées
    """
    print('Original dataset shape %s' % Counter(y))
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)
    
    print('Resampled dataset shape %s' % Counter(y_res))
    
    return X_res, y_res