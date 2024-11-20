from ucimlrepo import fetch_ucirepo 

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error

from display import tsne_display, umap_display
from sampling import under_sampling, over_sampling



def main(sampling_method=None):
    """
    Applique un algorithme de classification (RandomForestClassifier) sur le jeu de données Covertype.
    Utilisation de méthodes de visualisation (TSNE et UMAP) pour afficher les données en 2D.
    Possibilité d'appliquer une méthode d'équilibrage des classes: undersampling avec la méthode Nearmiss ou oversampling avec la méthode SMOTE.
    
    Args:
        sampling_method (str, optional): Méthode d'équilibrage des classes. (Par défaut: None)
        
    Returns:
        None: Affiche les métriques de performance du modèle de classification.    
    """
    covertype = fetch_ucirepo(id=31) 
    X = covertype.data.features 
    y = covertype.data.targets 
    
    if sampling_method == 'under':
        X, y = under_sampling(X, y)
    elif sampling_method == 'over':
        X, y = over_sampling(X, y)
    
    tsne_display(X, y)
    umap_display(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    print("\nCovertype dataset results:\n")
    print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))
    print("F1 Score: %.2f" % f1_score(y_test, y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Mean Absolute Error: %.2f" % mean_absolute_error(y_test, y_pred))
    
  
if __name__ == "__main__":
    main(sampling_method=None)