# Checkpoint 3: Graph-Based Recommenders

## Status: Complete 

Three graph-based recommenders implemented: **Node2Vec**, **LightGCN**, and **Graph CNN**.

## Results Summary

| Model | Revenue Score | Training Time | Status |
|-------|---------------|---------------|---------|
| **Node2Vec** | **3,994.78** | 6.34s | SUCCESS |
| **Graph CNN** | 3,319.78 | 0.60s | SUCCESS |
| **LightGCN** | 1,625.87 | 0.50s | SUCCESS |

## Implementation Overview

All models use bipartite user-item graphs with:
- **Graph Construction**: NetworkX bipartite graphs from interaction logs
- **Edge Weighting**: Frequency, recency, or purchase amount strategies
- **Link Prediction**: Revenue-optimized recommendations
- **Training**: PyTorch-based neural networks

### Node2Vec
- Random walk embeddings with skip-gram training
- Best performance: 3,994.78 revenue score

### LightGCN  
- Simplified graph convolution without transformations
- Fastest training: 0.50s

### Graph CNN
- Full GCN with feature transformation and dropout
- Balanced performance and flexibility

## Files

- `final_demo.py` - Demonstrates all three models working
- `graph_recommender_evaluation.py` - Comprehensive evaluation script
- `graph_recommender_demo_results.csv` - Pre-generated results
- `../graph_based_recommenders/` - Implementation module

## Usage

```python
from graph_based_recommenders import Node2VecRecommender

recommender = Node2VecRecommender(embedding_dim=64, epochs=20, seed=42)
recommender.fit(history_df, users_df, items_df)
recommendations = recommender.predict(log=history_df, k=10, users=users_df, items=items_df)
```
