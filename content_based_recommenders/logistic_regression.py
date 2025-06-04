import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .base import ContentBasedRecommenderBase


class LogisticRegressionRecommender(ContentBasedRecommenderBase):
    """Content-based recommender using Logistic Regression."""
    
    def __init__(self, seed=None, revenue_weight=1.0, regularization='l2', 
                 C=1.0, max_iter=1000):
        """
        Args:
            seed: Random seed for reproducibility
            revenue_weight: Weight for revenue optimization
            regularization: Regularization type ('l1', 'l2', 'elasticnet', or None)
            C: Inverse regularization strength
            max_iter: Maximum iterations for solver
        """
        super().__init__(seed=seed, revenue_weight=revenue_weight)
        self.regularization = regularization
        self.C = C
        self.max_iter = max_iter
        self.model = None
        
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train logistic regression model."""
        if self.regularization == 'elasticnet':
            penalty = 'elasticnet'
            l1_ratio = 0.5
            solver = 'saga'
        elif self.regularization == 'l1':
            penalty = 'l1'
            l1_ratio = None
            solver = 'liblinear'
        elif self.regularization == 'l2':
            penalty = 'l2'
            l1_ratio = None
            solver = 'liblinear'
        else:
            penalty = None
            l1_ratio = None
            solver = 'lbfgs'
        
        self.model = LogisticRegression(
            penalty=penalty,
            C=self.C,
            l1_ratio=l1_ratio,
            solver=solver,
            max_iter=self.max_iter,
            random_state=self.seed,
            class_weight='balanced'
        )
        
        try:
            self.model.fit(X, y)
            
            if len(np.unique(y)) > 1:
                train_score = self.model.score(X, y)
                print(f"Training accuracy: {train_score:.3f}")
                
                try:
                    y_proba = self.model.predict_proba(X)[:, 1]
                    auc_score = roc_auc_score(y, y_proba)
                    print(f"Training AUC: {auc_score:.3f}")
                except:
                    pass
            
        except Exception as e:
            print(f"Warning: Model training failed: {e}")
            self.model = None
    
    def _predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Predict purchase probabilities."""
        if self.model is None:
            return np.random.random(X.shape[0])
        
        try:
            probabilities = self.model.predict_proba(X)[:, 1]
            return probabilities
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            return np.random.random(X.shape[0]) 