import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from .base import ContentBasedRecommenderBase


class KNeighborsRecommender(ContentBasedRecommenderBase):
    """Content-based recommender using K-Nearest Neighbors."""
    
    def __init__(self, seed=None, revenue_weight=1.0, n_neighbors=5, 
                 metric='euclidean', weights='uniform'):
        """
        Args:
            seed: Random seed for reproducibility
            revenue_weight: Weight for revenue optimization
            n_neighbors: Number of neighbors to consider
            metric: Distance metric ('euclidean', 'manhattan', 'cosine')
            weights: Weight function ('uniform', 'distance')
        """
        super().__init__(seed=seed, revenue_weight=revenue_weight)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.model = None
        
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train K-Nearest Neighbors model."""
        self.model = KNeighborsClassifier(
            n_neighbors=min(self.n_neighbors, len(X)),
            metric=self.metric,
            weights=self.weights,
            n_jobs=-1
        )
        
        try:
            self.model.fit(X, y)
            
            print(f"KNN model trained with {self.model.n_neighbors} neighbors")
            print(f"Using {self.metric} distance with {self.weights} weights")
            
            if len(np.unique(y)) > 1:
                train_score = self.model.score(X, y)
                print(f"Training accuracy: {train_score:.3f}")
            
        except Exception as e:
            print(f"Warning: Model training failed: {e}")
            self.model = None
    
    def _predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Predict purchase probabilities using KNN."""
        if self.model is None:
            return np.random.random(X.shape[0])
        
        try:
            probabilities = self.model.predict_proba(X)[:, 1]
            return probabilities
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            return np.random.random(X.shape[0]) 