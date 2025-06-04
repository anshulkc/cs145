import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from .base import ContentBasedRecommenderBase


class RandomForestRecommender(ContentBasedRecommenderBase):
    """Content-based recommender using Random Forest."""
    
    def __init__(self, seed=None, revenue_weight=1.0, n_estimators=100, 
                 max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 max_features='sqrt'):
        """
        Args:
            seed: Random seed for reproducibility
            revenue_weight: Weight for revenue optimization
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for splits
        """
        super().__init__(seed=seed, revenue_weight=revenue_weight)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.model = None
        self.feature_importance_ = None
        
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train Random Forest model."""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.seed,
            class_weight='balanced',
            n_jobs=-1
        )
        
        try:
            self.model.fit(X, y)
            self.feature_importance_ = self.model.feature_importances_
            
            if len(np.unique(y)) > 1:
                train_score = self.model.score(X, y)
                print(f"Training accuracy: {train_score:.3f}")
                
                try:
                    y_proba = self.model.predict_proba(X)[:, 1]
                    auc_score = roc_auc_score(y, y_proba)
                    print(f"Training AUC: {auc_score:.3f}")
                except:
                    pass
            
            if self.feature_importance_ is not None and len(self.feature_importance_) > 0:
                top_features = np.argsort(self.feature_importance_)[-5:][::-1]
                print(f"Top 5 most important features (indices): {top_features}")
                print(f"Feature importances: {self.feature_importance_[top_features]}")
            
        except Exception as e:
            print(f"Warning: Model training failed: {e}")
            self.model = None
    
    def _predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Predict purchase probabilities using Random Forest."""
        if self.model is None:
            return np.random.random(X.shape[0])
        
        try:
            probabilities = self.model.predict_proba(X)[:, 1]
            return probabilities
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            return np.random.random(X.shape[0])
    
    def get_feature_importance(self):
        """Get feature importance from the trained Random Forest model."""
        return self.feature_importance_ if self.feature_importance_ is not None else None 