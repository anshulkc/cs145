"""
Sequence-Based Recommenders Module

Implementations of sequence-based recommendation algorithms:
- AutoRegressiveRecommender: N-gram autoregressive models for sequential patterns
- RNNRecommender: RNN/GRU implementations for sequence modeling
- TransformerRecommender: Transformer/self-attention architectures

Each includes sequence curation, revenue optimization, and temporal pattern modeling.
"""

from .base import SequenceBasedRecommenderBase
from .autoregressive import AutoRegressiveRecommender
from .rnn_lstm import RNNRecommender, GRURecommender
from .transformer import TransformerRecommender

# Standard configurations for different parameter settings
SEQUENCE_BASED_CONFIGS = [
    {
        "name": "AutoRegressive_Order1",
        "class": "AutoRegressiveRecommender",
        "parameters": {
            "seed": 42,
            "order": 1,
            "smoothing_alpha": 0.1,
            "sequence_length": 10,
            "revenue_weight": 1.0
        }
    },
    {
        "name": "AutoRegressive_Order2",
        "class": "AutoRegressiveRecommender",
        "parameters": {
            "seed": 42,
            "order": 2,
            "smoothing_alpha": 0.1,
            "sequence_length": 20,
            "revenue_weight": 1.0
        }
    },
    {
        "name": "AutoRegressive_Order3",
        "class": "AutoRegressiveRecommender",
        "parameters": {
            "seed": 42,
            "order": 3,
            "smoothing_alpha": 0.01,
            "sequence_length": 30,
            "revenue_weight": 1.5
        }
    },
    {
        "name": "RNN_Basic",
        "class": "RNNRecommender",
        "parameters": {
            "seed": 42,
            "hidden_size": 64,
            "num_layers": 1,
            "dropout": 0.2,
            "sequence_length": 20,
            "embedding_dim": 32,
            "revenue_weight": 1.0
        }
    },
    {
        "name": "GRU_Efficient",
        "class": "GRURecommender",
        "parameters": {
            "seed": 42,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.3,
            "sequence_length": 25,
            "embedding_dim": 64,
            "revenue_weight": 1.5
        }
    },
    {
        "name": "Transformer_Small",
        "class": "TransformerRecommender",
        "parameters": {
            "seed": 42,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dropout": 0.2,
            "sequence_length": 20,
            "revenue_weight": 1.0
        }
    },
    {
        "name": "Transformer_Medium",
        "class": "TransformerRecommender",
        "parameters": {
            "seed": 42,
            "d_model": 128,
            "nhead": 8,
            "num_layers": 4,
            "dropout": 0.3,
            "sequence_length": 30,
            "revenue_weight": 1.5
        }
    },
    {
        "name": "Transformer_Large",
        "class": "TransformerRecommender",
        "parameters": {
            "seed": 42,
            "d_model": 256,
            "nhead": 8,
            "num_layers": 6,
            "dropout": 0.4,
            "sequence_length": 50,
            "revenue_weight": 2.0
        }
    }
]

__all__ = [
    'SequenceBasedRecommenderBase',
    'AutoRegressiveRecommender',
    'RNNRecommender',
    'GRURecommender',
    'TransformerRecommender',
    'SEQUENCE_BASED_CONFIGS'
] 