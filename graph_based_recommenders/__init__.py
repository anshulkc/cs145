"""
Graph-Based Recommenders Module

Implementations of graph-based recommendation algorithms:
- Node2VecRecommender: Random walk-based embeddings for link prediction
- LightGCNRecommender: Simplified graph convolution for collaborative filtering  
- GCNRecommender: Graph Convolutional Networks for user-item interaction prediction

Each leverages user-item bipartite graphs for revenue optimization through link prediction.
"""

from .base import GraphBasedRecommenderBase
from .node2vec import Node2VecRecommender
from .lightgcn import LightGCNRecommender
from .gcn import GCNRecommender

# Standard configurations for different parameter settings
GRAPH_BASED_CONFIGS = [
    {
        "name": "Node2Vec_Basic",
        "class": "Node2VecRecommender",
        "parameters": {
            "embedding_dim": 64,
            "walk_length": 10,
            "num_walks": 10,
            "window_size": 5,
            "p": 1.0,
            "q": 1.0,
            "learning_rate": 0.01,
            "epochs": 50,
            "num_negative": 5,
            "batch_size": 512,
            "early_stopping_patience": 10,
            "edge_weight_strategy": "frequency",
            "revenue_weight": 1.0,
            "seed": 42
        }
    },
    {
        "name": "Node2Vec_Advanced",
        "class": "Node2VecRecommender", 
        "parameters": {
            "embedding_dim": 128,
            "walk_length": 20,
            "num_walks": 20,
            "window_size": 10,
            "p": 0.5,
            "q": 2.0,
            "learning_rate": 0.005,
            "epochs": 100,
            "num_negative": 10,
            "batch_size": 1024,
            "early_stopping_patience": 15,
            "edge_weight_strategy": "recency",
            "revenue_weight": 1.5,
            "seed": 42
        }
    },
    {
        "name": "LightGCN_Small",
        "class": "LightGCNRecommender",
        "parameters": {
            "embedding_dim": 64,
            "n_layers": 2,
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 512,
            "reg_weight": 1e-4,
            "negative_sampling_ratio": 1.0,
            "early_stopping_patience": 10,
            "graph_dropout": 0.1,
            "edge_weight_strategy": "frequency",
            "revenue_weight": 1.0,
            "seed": 42
        }
    },
    {
        "name": "LightGCN_Medium",
        "class": "LightGCNRecommender",
        "parameters": {
            "embedding_dim": 128,
            "n_layers": 3,
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 1024,
            "reg_weight": 1e-4,
            "negative_sampling_ratio": 1.5,
            "early_stopping_patience": 15,
            "graph_dropout": 0.15,
            "edge_weight_strategy": "purchase_amount",
            "revenue_weight": 1.2,
            "seed": 42
        }
    },
    {
        "name": "GCN_Basic",
        "class": "GCNRecommender",
        "parameters": {
            "embedding_dim": 64,
            "hidden_dims": [128, 64],
            "dropout": 0.3,
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 512,
            "reg_weight": 1e-4,
            "negative_sampling_ratio": 1.0,
            "early_stopping_patience": 10,
            "use_batch_norm": True,
            "graph_dropout": 0.1,
            "edge_weight_strategy": "frequency",
            "revenue_weight": 1.0,
            "seed": 42
        }
    },
    {
        "name": "GCN_Deep",
        "class": "GCNRecommender",
        "parameters": {
            "embedding_dim": 128,
            "hidden_dims": [256, 128, 64],
            "dropout": 0.5,
            "learning_rate": 0.0005,
            "epochs": 100,
            "batch_size": 1024,
            "reg_weight": 1e-3,
            "negative_sampling_ratio": 2.0,
            "early_stopping_patience": 15,
            "use_batch_norm": True,
            "graph_dropout": 0.2,
            "edge_weight_strategy": "recency",
            "revenue_weight": 1.3,
            "seed": 42
        }
    }
]

__all__ = [
    'GraphBasedRecommenderBase',
    'Node2VecRecommender', 
    'LightGCNRecommender',
    'GCNRecommender',
    'GRAPH_BASED_CONFIGS'
] 