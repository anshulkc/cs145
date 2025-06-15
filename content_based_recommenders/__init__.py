"""
Content-Based Recommenders Module

Implementations of content-based recommendation algorithms:
- LogisticRegressionRecommender: Logistic regression with feature engineering
- KNeighborsRecommender: K-nearest neighbors with distance metrics  
- RandomForestRecommender: Random forest ensemble method

Each includes revenue optimization and position bias handling.
"""

from .base import ContentBasedRecommenderBase
from .logistic_regression import LogisticRegressionRecommender
from .knn import KNeighborsRecommender
from .random_forest import MyRecommender as RandomForestRecommender

# Standard configurations for different parameter settings
CONTENT_BASED_CONFIGS = [
    {
        "name": "LogisticRegression_L2",
        "class": "LogisticRegressionRecommender",
        "parameters": {
            "seed": 42,
            "regularization": "l2",
            "C": 1.0,
            "revenue_weight": 1.0
        }
    },
    {
        "name": "LogisticRegression_L1", 
        "class": "LogisticRegressionRecommender",
        "parameters": {
            "seed": 42,
            "regularization": "l1", 
            "C": 0.1,
            "revenue_weight": 1.0
        }
    },
    {
        "name": "LogisticRegression_HighRevenue",
        "class": "LogisticRegressionRecommender", 
        "parameters": {
            "seed": 42,
            "regularization": "l2",
            "C": 1.0,
            "revenue_weight": 2.0
        }
    },
    {
        "name": "KNN_Euclidean",
        "class": "KNeighborsRecommender",
        "parameters": {
            "seed": 42,
            "n_neighbors": 5,
            "metric": "euclidean",
            "revenue_weight": 1.0
        }
    },
    {
        "name": "KNN_Manhattan",
        "class": "KNeighborsRecommender", 
        "parameters": {
            "seed": 42,
            "n_neighbors": 5,
            "metric": "manhattan",
            "revenue_weight": 1.0
        }
    },
    {
        "name": "RandomForest_Default",
        "class": "RandomForestRecommender",
        "parameters": {
            "seed": 42,
            "n_estimators": 50,
            "revenue_weight": 1.0
        }
    },
    {
        "name": "RandomForest_HighRevenue", 
        "class": "RandomForestRecommender",
        "parameters": {
            "seed": 42,
            "n_estimators": 50,
            "revenue_weight": 2.0
        }
    }
]

__all__ = [
    'ContentBasedRecommenderBase',
    'LogisticRegressionRecommender', 
    'KNeighborsRecommender',
    'RandomForestRecommender',
    'CONTENT_BASED_CONFIGS'
] 