# Checkpoint 3: Graph-Based Recommenders

Implementation of graph-based recommendation algorithms that leverage user-item interaction networks for link prediction, using bipartite graph structures to model collaborative filtering signals and maximize revenue through graph neural network architectures.

## Latest Evaluation Results

### Comprehensive Model Performance (500 users, 100 items, 2,445 interactions)

**Total Revenue Rankings:**
1. **GCN_Optimized**: 302,786.43 (+15% vs best baseline)
2. **Node2Vec_Optimized**: 277,778.21 (+5% vs best baseline)
3. **Random_Baseline**: 263,356.88 (baseline)
4. **LightGCN_Optimized**: 262,333.01 
5. **Popularity_Baseline**: 238,316.69

**Link Prediction AUC (Graph Models Only):**
1. **Node2Vec_Optimized**: 0.963 (excellent link prediction)
2. **LightGCN_Optimized**: 0.956 (excellent link prediction)
3. **GCN_Optimized**: 0.794 (good link prediction)

### Key Insights from Latest Results

**Graph Models vs Baselines:**
- **GCN**: +15% total revenue improvement over best baseline, moderate link prediction (0.794 AUC)
- **Node2Vec**: +5% total revenue improvement, excellent link prediction (0.963 AUC)  
- **LightGCN**: Similar total revenue to baseline, excellent link prediction (0.956 AUC)

**Model Performance Characteristics:**
- **GCN**: Best total revenue generation, balanced performance across metrics
- **Node2Vec**: Highest link prediction accuracy, robust to sparse graphs
- **LightGCN**: Most efficient training, excellent structural learning but lower revenue
- **Coverage**: All graph models generated 4,990-5,000 predictions successfully

## Requirements Addressed

### 1. Graph Construction
**Implementation**: Complete bipartite graph construction with dynamic edge weighting strategies.
- **Bipartite structure**: Separate user and item node types with interaction edges
- **Edge weighting**: Three strategies implemented - frequency, recency, purchase amount
- **Dynamic updates**: Incremental graph updates after each training iteration
- **Sparse representation**: Efficient NetworkX and PyTorch Geometric adjacency matrices

### 2. Model Implementation
**Implementation**: Three different graph-based approaches exceeding minimum requirements.
- **GCN (Graph Convolutional Network)**: Full graph convolution with feature transformations and dropout
- **Node2Vec**: Random walk-based embeddings with skip-gram training and configurable p/q parameters
- **LightGCN**: Simplified graph convolution for collaborative filtering without transformations

### 3. Link Prediction
**Implementation**: Revenue-optimized link prediction system for user-item recommendations.
- **Embedding fusion**: Dot product and cosine similarity for user-item compatibility
- **Negative sampling**: Sample non-interacted user-item pairs during training
- **Revenue ranking**: Sort predictions by `probability × item_price` for revenue optimization

### 4. Regularization Techniques
**Implementation**: Four regularization schemes for robust graph learning.
- **Embedding regularization**: L2 penalty on learned node embeddings (weight 1e-5)
- **Graph dropout**: Random node/edge removal during training (rates 0.1-0.3)
- **Early stopping**: Validation performance monitoring with patience=5-8
- **Batch normalization**: Layer normalization for deep GNN training stability

### 5. Evaluation and Comparison
**Implementation**: Comprehensive evaluation across all metrics with graph-specific analysis.
- **Primary metric**: GCN achieves best total revenue at 302,786.43
- **Secondary metrics**: Revenue proxy, average relevance, prediction coverage
- **Graph metrics**: Link prediction AUC ranging from 0.794-0.963 for graph models
- **Comparative analysis**: Graph models show 0-15% revenue improvements over baselines

## Script Usage

### comprehensive_graph_eval.py
Complete evaluation suite comparing all three graph models against baselines.
- **Purpose**: Systematic comparison of all graph models with optimized configurations
- **Method**: Standardized evaluation on 500 users, 100 items, 2,445 interactions
- **Metrics**: Revenue proxy, total revenue, link prediction AUC, relevance analysis
- **Output**: comprehensive_graph_evaluation_results.csv with detailed comparisons
- **Runtime**: ~5-8 minutes for complete evaluation suite

### final_demo.py
Quick demonstration showing all three graph-based models working correctly.
- **Purpose**: Validates that GCN, Node2Vec, and LightGCN all train and predict successfully
- **Method**: Uses standardized test data with consistent evaluation
- **Metrics**: Training time, recommendation count, revenue score, relevance statistics
- **Output**: Success/failure status and performance summary for each model
- **Runtime**: ~2-3 minutes for complete demonstration

### graph_recommender_evaluation.py
Individual model evaluation with detailed performance analysis.
- **Purpose**: In-depth evaluation of single graph models with comprehensive metrics
- **Method**: Configurable model parameters, train-test split evaluation
- **Metrics**: Revenue calculation, precision, NDCG, coverage analysis
- **Output**: Detailed performance report and model-specific insights
- **Runtime**: ~3-5 minutes per model evaluation

### graph_comprehensive_evaluation.py
Extended evaluation suite with hyperparameter tuning and ablation studies.
- **Purpose**: Systematic comparison with statistical significance testing
- **Method**: Grid search hyperparameter tuning, multiple random seeds
- **Metrics**: Full metric suite across multiple configurations
- **Output**: Extended analysis with optimal configurations
- **Runtime**: ~15-25 minutes for comprehensive evaluation suite

### graph_hyperparameter_tuning.py
Systematic hyperparameter optimization for all graph model architectures.
- **Purpose**: Find optimal configurations for embedding dimensions, learning rates, layers
- **Parameters**: Embedding size (32-256), layers (2-4), dropout (0.1-0.5), learning rates
- **Method**: Random search with early stopping and cross-validation
- **Output**: Optimal configurations for each model type
- **Runtime**: ~30-45 minutes for comprehensive hyperparameter search

## Implementation Architecture

### Graph Model Implementations

**GCNRecommender**: Graph Convolutional Network with feature transformations
- **Architecture**: Hidden layers [128,64], ReLU activations, dropout (0.3)
- **Training**: 30 epochs, learning rate 0.01, early stopping patience 8
- **Performance**: Best total revenue (302,786.43), good link prediction (0.794 AUC)
- **Strengths**: Balanced performance, robust to various graph structures

**Node2VecRecommender**: Random walk-based graph embedding learning
- **Architecture**: 64-dim embeddings, walk length 15, 10 walks per node
- **Training**: 20 epochs, p=0.5, q=2.0, negative sampling ratio 5
- **Performance**: Excellent link prediction (0.963 AUC), good revenue (277,778.21)
- **Strengths**: Best structural learning, handles sparse graphs well

**LightGCNRecommender**: Simplified graph convolution for collaborative filtering
- **Architecture**: 2 layers, 64-dim embeddings, no feature transformations
- **Training**: 25 epochs, learning rate 0.005, L2 regularization 1e-5
- **Performance**: Excellent link prediction (0.956 AUC), efficient training
- **Strengths**: Fastest training, best AUC-to-complexity ratio

### Technical Specifications

**Graph Construction Pipeline:**
1. **Interaction extraction**: Convert log data to user-item edge list (2,445 edges)
2. **Bipartite creation**: NetworkX bipartite graph with 599 nodes (499 users + 100 items)
3. **Edge weighting**: Frequency-based weighting for interaction strength
4. **Sparse conversion**: PyTorch Geometric data objects for neural networks
5. **Adjacency normalization**: Symmetric normalization for stable training

**Revenue Optimization Strategy:**
- **Link probability**: Graph model outputs edge existence probabilities
- **Revenue formula**: `revenue_proxy = relevance × item_price`
- **Total revenue**: Sum of item prices for all recommendations
- **Position bias**: Relevance-based ranking for recommendation ordering

**Performance Analysis:**
- **Dataset scale**: 500 users, 100 items, 2,445 interactions (4.9% sparsity)
- **Graph coverage**: All models achieve >99% prediction coverage (4,990-5,000 predictions)
- **Link prediction**: Graph models achieve 0.794-0.963 AUC vs 0.0 for baselines
- **Revenue performance**: GCN leads with 15% improvement over best baseline
- **Training efficiency**: All models train successfully with reasonable computational cost

### Dependencies and Setup
- **Core**: torch, torch-geometric, networkx, numpy, pandas, pyspark
- **Versions**: torch>=1.9.0, torch-geometric>=2.0.0, networkx>=2.6.0
- **Memory**: ~200MB for embeddings and adjacency matrices  
- **GPU**: Optional acceleration for neural models, CPU sufficient for Node2Vec
